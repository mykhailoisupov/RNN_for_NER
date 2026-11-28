"""Posts of public Kyiv Telegram channels, labelled by LLM agents after scripts/labeling_guide.md ->
data/kyiv_posts.csv: whole labelled posts that build_dataset adds to train and val as they are.

    python -m scripts.kyiv_posts fetch      # t.me/s/<channel> -> data/external/telegram/<channel>.jsonl
    python -m scripts.kyiv_posts chunks     # 5,000 of them -> chunks/NNN.jsonl, 200 posts each, for the agents
    python -m scripts.kyiv_posts collect    # chunks + labels/NNN.jsonl of the agents -> data/kyiv_posts.csv

An agent reads a chunk ({"id", "text"} per line) and writes labels/<same name>: {"id", "locations": [exact substrings]}.
LLM-written posts go the same way from chunks/synth-NNN.jsonl. Test is dated August-September 2023; those months are
left out, and a post equal to a test post is dropped.
"""
import argparse
import html
import json
import re
import time
import urllib.request
from collections import Counter
from datetime import date

import pandas as pd

from scripts.build_dataset import WORD, normalize_text
from src.config import DATA_DIR, EXTERNAL_DIR
from src.label_policy import POLICY, relabel
from src.tags import aligned
from src.tokenizer import prepare, tokenize

DIR = EXTERNAL_DIR / "telegram"
CHANNELS = [                               # public handles with a web preview, active in 2023
    # the channels whose signatures test posts carry
    "darnicalive", "huevyi_kiev", "kyiv_novyny_24", "obolonlife", "lossolomas_kyiv", "semenovatut", "u_now", "kyivoperat",
    # city-wide Kyiv news
    "kievreal1", "truexakyiv", "vitaliy_klitschko", "kyivsegodna", "kyiv_live", "KyivNOW", "kyivtv",
    # districts of Kyiv and towns around it
    "darnycya_info", "obolon_info", "podil_info", "svyatoshyn_info", "borshagivka_online", "vyshneve_news", "irpinnews",
    "buchainfo", "boryspilinfo", "fastivinfo", "fastivnews", "bila_tserkva_online", "kyivregion_info",
]
#  Left out after adversarial validation against test (BiGRU AUC of one channel, 2026-11-28; the kept ones 0.66-0.86,
#  their mix 0.65, train posts 0.76-0.77): kyivpasstrans 0.99, kievinfo_kyiv 0.99, kyiv_golovne 0.99 (one template or
#  one signature line in every post), borshchagovkanews 0.97, buchanews 0.94, KyivCityOfficial 0.91.
START, END = date(2023, 6, 1), date(2024, 3, 31)
TEST_MONTHS = {(2023, 8), (2023, 9)}
PAGES = 60                                 # pages of 20 posts per channel, spread evenly over the period
TOTAL, CHUNK = 5000, 200
MESSAGE = re.compile(r'data-post="[^"/]+/(\d+)"(.*?)<time datetime="(\d{4})-(\d\d)-(\d\d)', re.S)
TEXT = re.compile(r'<div class="tgme_widget_message_text js-message_text[^>]*>(.*?)</div>', re.S)


def page(channel, before):
    """The 20 posts before id `before`: [(id, date, text)], oldest first; a post without text has text ''."""
    request = urllib.request.Request(f"https://t.me/s/{channel}?before={before}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        page_html = response.read().decode("utf-8", "replace")
    time.sleep(1)
    posts = []
    for number, body, year, month, day in MESSAGE.findall(page_html):
        text = TEXT.search(body)
        text = re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", "\n", text.group(1))) if text else ""
        posts.append((int(number), date(int(year), int(month), int(day)), html.unescape(text).strip()))
    return posts


def first_id_after(channel, day, newest):
    """The smallest `before` whose page ends after `day`: binary search over post ids, which grow with time."""
    low, high = 1, newest
    while low < high:
        middle = (low + high) // 2
        posts = page(channel, middle)
        if posts and posts[-1][1] > day:
            high = middle
        else:
            low = middle + 1
    return low


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        path = DIR / f"{channel}.jsonl"
        if path.exists():
            continue
        newest = page(channel, 10 ** 9)[-1][0]
        low, high = first_id_after(channel, START, newest), first_id_after(channel, END, newest)
        posts = {}
        for before in range(high, low, -max(20, (high - low) // PAGES)):
            for number, day, text in page(channel, before):
                if text and START <= day <= END and (day.year, day.month) not in TEST_MONTHS:
                    posts[number] = {"id": f"{channel}/{number}", "date": str(day), "text": text}
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for _, p in sorted(posts.items())))
        print(f"{channel}: ids {low}-{high}, {len(posts)} posts", flush=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chunks(seed=0):
    posts = pd.DataFrame([p for channel in CHANNELS for p in read_jsonl(DIR / f"{channel}.jsonl")])
    posts["channel"] = posts.id.str.split("/").str[0]
    posts["norm"] = posts.text.map(normalize_text)
    test = set(pd.read_csv(DATA_DIR / "test.csv").text.map(normalize_text))
    words = posts.text.map(lambda t: len(WORD.findall(t)))
    posts = posts[words.between(5, 450) & ~posts.norm.isin(test)].drop_duplicates("norm")
    per_channel = posts.groupby("channel").sample(frac=1, random_state=seed).groupby("channel").head(TOTAL // len(CHANNELS))
    rest = posts.drop(per_channel.index)
    picked = pd.concat([per_channel, rest.sample(min(len(rest), TOTAL - len(per_channel)), random_state=seed)])
    picked = picked.sample(frac=1, random_state=seed)
    (DIR / "chunks").mkdir(exist_ok=True)
    for i in range(0, len(picked), CHUNK):
        rows = picked.iloc[i:i + CHUNK][["id", "text"]].to_dict("records")
        (DIR / "chunks" / f"{i // CHUNK:03d}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(picked)} posts of {len(posts)} in {-(-len(picked) // CHUNK)} chunks\n{picked.channel.value_counts().to_string()}")


def collect():
    rows, stats, names = [], Counter(), Counter()
    for chunk in sorted((DIR / "chunks").glob("*.jsonl")):
        labels = DIR / "labels" / chunk.name
        if not labels.exists():
            stats["chunks without labels"] += 1
            continue
        locations = {row["id"]: row["locations"] for row in read_jsonl(labels)}
        for post in read_jsonl(chunk):
            if post["id"] not in locations:
                stats["posts without a label row"] += 1
                continue
            text, spans = post["text"], []
            for name in locations[post["id"]]:
                found = [m.span() for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text)] if name.strip() else []
                stats["names not in the post"] += not found
                spans += found
            text = prepare(text)   # same length: the spans hold
            spans = relabel(text, spans, POLICY)
            if not aligned(tokenize(text), spans):
                stats["posts dropped: span cuts a word"] += 1
                continue
            names.update(text[a:b] for a, b in spans)
            stats["posts"] += 1
            stats["posts without names"] += not spans
            rows.append({"text": text, "loc_markers": str(spans), "kind": "synth" if chunk.name.startswith("synth") else "kyiv"})
    pd.DataFrame(rows).to_csv(DATA_DIR / "kyiv_posts.csv", index=False)
    print("\n".join(f"{key:34} {value:>7,}" for key, value in stats.items()))
    print("top names:", ", ".join(f"{name} {n}" for name, n in names.most_common(20)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "chunks", "collect"])
    {"fetch": fetch, "chunks": chunks, "collect": collect}[parser.parse_args().command]()
