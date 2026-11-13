"""Build data/normalized/: clean and relabel uk/ru sentences, then glue them into posts shaped like test.

    python -m scripts.build_dataset    # cleaning is cached; --rebuild redoes it

Run from the project root.
"""
import argparse
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.postprocess import in_org_name
from src.text import encode, parse_spans, prepare

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "normalized"

RU_SHARE, VAL_RU_SHARE = 0.20, 0.08
VAL_PCT, VAL_PCT_RARE = 5, 20             # % of uk sentences sent to val: ordinary / with Kyiv or a street
UPSAMPLE_KYIV, UPSAMPLE_STREET = 4, 4
HOUSE_NUMBER_P = 0.5
N_FOLDS = 5
RU_DOC_MOD = 10                           # keep full rows for every 10th ru document, enough for 20% of train
SEPARATORS, SEPARATOR_P = ["\n\n", "\n", " "], [0.7, 0.2, 0.1]

# Shares of test posts, measured on GPT labels of test.
BIN_NAMES = ["0-15", "16-30", "31-60", "61-120", "121+"]
BIN_LOW, BIN_HIGH = np.array([1, 16, 31, 61, 121]), np.array([15, 30, 60, 120, 250])
TARGET_WORDS = np.array([0.220, 0.239, 0.245, 0.201, 0.094])
TARGET_EMPTY = np.array([0.4286, 0.2281, 0.2393, 0.1667, 0.1556])
TARGET_PROFILE = {"names 0": 25.6, "names 1": 35.6, "names 2": 20.3, "names 3-4": 11.9, "names 5+": 6.5,
                  **{f"words {n}": 100 * p for n, p in zip(BIN_NAMES, TARGET_WORDS)},
                  "kyiv": 41.1, "street": 9.6, "multiline": 85.0, "median words": 36, "ru": 8.0}

STATS = Counter()

# Also imported by notebooks/eda, so the EDA numbers and the build use one definition.
STREET =r"(?:вул\.|ул\.|вулиц[яіюе]|улиц[аыуеу]|проспект(?:і|у|ом|а|е)?|просп\.|провул(?:ок|ку)|переул(?:ок|ке)|бульвар(?:і|у|а|е)?|бул\.|площ(?:а|і|у|адь|ади)|набережн\w*)"
PROBES = {
    "Ukraine": re.compile(r"\b(?:Україн(?:а|и|і|у|ою)|Украин(?:а|ы|е|у|ой))\b", re.I),
    "Kyiv": re.compile(r"\b(?:Київ|Києва|Києві|Києву|Києвом|Киев|Киева|Киеве|Киеву|Киевом)\b", re.I),
}
WORD = re.compile(r"\w+(?:['-]\w+)*")
KYIV_MENTION = re.compile(r"київ|києв|киев", re.I)


def clean_spans(text, spans):
    """Merge touching spans, trim edge whitespace, drop spans without letters or digits."""
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    cleaned = []
    for a, b in merged:
        chunk = text[a:b]
        a, b = a + len(chunk) - len(chunk.lstrip()), b - (len(chunk) - len(chunk.rstrip()))
        if a < b and any(c.isalnum() for c in text[a:b]):
            cleaned.append((a, b))
    return cleaned


def normalize_text(text):
    """Deduplication key only; NFKC changes length, so it never reaches the model."""
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def normalised_loc_signature(text, spans):
    return tuple(sorted((normalize_text(text[:a]), normalize_text(text[a:b]), normalize_text(text[b:]))
                        for a, b in spans if normalize_text(text[a:b])))


def features(text, spans):
    labels = {text[a:b] for a, b in spans}
    return {"words": len(WORD.findall(text)), "labels": len(labels),
            "kyiv": bool(KYIV_MENTION.search(text)),
            "street": any(re.search(STREET, s, re.I) for s in labels)}


# --- relabelling one sentence, same rules for uk and ru -------------------------------
TCH = re.compile(r"\s[-–—]\s[^-–—]{1,40}\s[-–—]\s*TCH\.?\s*$")
# ponytail: only unambiguous forms. `площадь России`, `вулиці Києва`, `набережная Невы` are not streets,
# so those type words are left out and a few true streets keep the old boundary. Add a name-frequency
# guard if street recall matters more than this.
TYPE_BEFORE = re.compile(r"(?<!\w)(?:вул\.|ул\.|вулиц(?:я|ю|ею)|улиц(?:а|е|у|ей)|просп\.|проспект(?:і|у|ом|а|е)?"
                         r"|провул(?:ок|ку|ка)|переул(?:ок|ке|ка)|бульвар(?:і|у|а|е|ом)?|бул\.|пл\.|площади"
                         r"|набережной)\s*$", re.I)
# Singular only: `районах` after a list of names is not part of one name.
AREA_AFTER = re.compile(r"\s+(?:район(?:і|у|ом|а|е)?|област(?:ь|і|и|ю|ью)|обл\.|р-н|кра(?:й|я|е|ю|ем)"
                        r"|республ[іи]к(?:а|и|е|у|ою|ой)|республіці)(?!\w)", re.I)
ADJECTIVE_END = re.compile(r"(?:ськ|цьк|зьк|ск|цк)\w{1,3}$")   # `Донецької`, not `района` or `Росією`


def fix_spans(text, loc, org=()):
    """Prepared sentence + raw LOC/ORG spans -> (text, spans), or None to drop the row."""
    assert all(0 <= a < b <= len(text) for a, b in [*loc, *org])
    cut = TCH.search(text) if "TCH" in text else None   # `... - Київ - TCH.` is a site tag, not text
    if cut:
        text = text[:cut.start()]
        loc, org = ([s for s in spans if s[1] <= len(text)] for spans in (loc, org))
        STATS["TCH footer cut"] += 1
    if not text.strip():
        STATS["rows dropped: empty text"] += 1
        return None
    spans = clean_spans(text, loc)
    STATS["rows changed by clean_spans"] += spans != list(loc)
    relabelled = []
    for a, b in spans:
        if b < len(text) and text[b - 1].isdigit() and text[b].isdigit():   # `Галицька, 9|3`
            while b < len(text) and text[b].isdigit():
                b += 1
            STATS["house number completed"] += 1
        before = TYPE_BEFORE.search(text, max(0, a - 35), a)   # ru leaves the street type outside
        if before:
            a = before.start()
            STATS["street type joined"] += 1
        after = AREA_AFTER.match(text, b) if ADJECTIVE_END.search(text, a, b) else None
        if after:
            b = after.end()
            STATS["area type joined"] += 1
        relabelled.append((a, b))
    for m in PROBES["Kyiv"].finditer(text):
        if not any(c < m.end() and m.start() < d for c, d in [*relabelled, *org]):
            relabelled.append(m.span())
            STATS["Kyiv labelled"] += 1
    outside_org = [(a, b) for a, b in relabelled if not any(c <= a and b <= d for c, d in org)]
    STATS["LOC inside ORG removed"] += len(relabelled) - len(outside_org)
    spans = clean_spans(text, outside_org)
    if spans and encode(text, spans) is None:   # shifted offsets or glued words, not fixable
        STATS["rows dropped: span cuts a word"] += 1
        return None
    return text, spans


# --- sentence tables ---------------------------------------------------------------------
def _crc(value):
    return zlib.crc32(str(value).encode())


def load_sentences(lang, chunksize=500_000):
    """Clean every row. Copies and label conflicts are judged on the whole corpus, not on the kept part."""
    extra = ["doc_id", "sent_id"] if lang == "ru" else []
    kept, norm_hashes, label_hashes = [], [], []
    for chunk in pd.read_csv(DATA / f"{lang}_geo_dataset.csv", chunksize=chunksize,
                             usecols=["text", "loc_markers", "org_markers", *extra]):
        STATS["rows in"] += len(chunk)
        rows = [fix_spans(prepare(t), parse_spans(loc), parse_spans(org))
                for t, loc, org in zip(chunk.text, chunk.loc_markers, chunk.org_markers)]
        chunk = chunk.loc[[r is not None for r in rows], extra].assign(
            text=[r[0] for r in rows if r], spans=[r[1] for r in rows if r])
        chunk["norm"] = chunk.text.map(normalize_text)
        chunk = chunk[chunk.norm.ne("")]
        # hash() is stable within one run, which is all the conflict search needs.
        norm_hashes.append(chunk.norm.map(hash).to_numpy(np.int64))
        label_hashes.append(np.array([hash(normalised_loc_signature(t, s))
                                      for t, s in zip(chunk.text, chunk.spans)], np.int64))
        kept.append(chunk[chunk.doc_id % RU_DOC_MOD == 0] if lang == "ru" else chunk)
    corpus = pd.DataFrame({"norm": np.concatenate(norm_hashes), "labels": np.concatenate(label_hashes)})
    versions = corpus.groupby("norm").labels.nunique()
    conflicts = versions.index[versions > 1]
    STATS["extra copies in corpus"] = len(corpus) - len(versions)
    STATS["groups with conflicting labels"] = len(conflicts)
    STATS["rows in conflicting groups"] = int(corpus.norm.isin(conflicts).sum())

    df = pd.concat(kept, ignore_index=True)
    df = df[~df.norm.map(hash).isin(conflicts)].drop_duplicates("norm").reset_index(drop=True)
    df = df.join(pd.DataFrame([features(t, s) for t, s in zip(df.text, df.spans)]))
    if lang == "ru":
        key = df.doc_id.map(_crc)
        in_val = key % 20 == 0
    else:   # a larger share of Kyiv and street sentences makes val test-like without repeats
        key = df.norm.map(_crc)
        in_val = key % 100 < np.where(df.kyiv | df.street, VAL_PCT_RARE, VAL_PCT)
    df["split"] = np.where(in_val, "val", "train")
    df["fold"] = key // 100 % N_FOLDS
    assert not df.norm.duplicated().any()   # one copy per sentence: no leak between splits or folds
    STATS["sentences kept"] = len(df)
    return df


# --- posts ------------------------------------------------------------------------------
STREET_START = re.compile(rf"{STREET}\s", re.I)


def add_house_number(text, spans, rng):
    """`вул. Хрещатик` -> `вул. Хрещатик, 28` inside the label: train has almost no addresses."""
    for i, (a, b) in enumerate(spans):
        ends_word = text[b - 1].isalpha() and (b == len(text) or not text[b].isalnum())
        if STREET_START.match(text, a, b) and ends_word and not re.match(r",?\s*\d", text[b:b + 4]):
            number = f", {rng.integers(1, 200)}"
            shifted = [(c, d + len(number)) if j == i else (c + len(number), d + len(number)) if c >= b else (c, d)
                       for j, (c, d) in enumerate(spans)]
            return text[:b] + number + text[b:], shifted
    return text, spans


def upsample(sent, rng):
    sent = sent.reset_index(drop=True)
    repeats = np.where(sent.street, UPSAMPLE_STREET, np.where(sent.kyiv, UPSAMPLE_KYIV, 1))
    out = sent.loc[sent.index.repeat(repeats)]
    numbered = out.index.duplicated() & out.street.to_numpy() & (rng.random(len(out)) < HOUSE_NUMBER_P)
    pairs = [add_house_number(t, s, rng) if n else (t, s) for t, s, n in zip(out.text, out.spans, numbered)]
    return out.assign(text=[p[0] for p in pairs], spans=[p[1] for p in pairs]).reset_index(drop=True)


def compose(sent, rng, words_p, by_doc=False):
    """Glue consecutive sentences into posts whose length in words follows `words_p`."""
    if by_doc:   # ru: neighbouring sentences of one document, documents in random order
        order = pd.Series(rng.permutation(sent.doc_id.nunique()), index=sent.doc_id.unique())
        sent = sent.assign(_order=sent.doc_id.map(order)).sort_values(["_order", "sent_id"])
        doc = sent.doc_id.to_numpy()
    else:        # uk has no documents
        sent = sent.iloc[rng.permutation(len(sent))]
        doc = None
    texts, spans, words = sent.text.to_numpy(), sent.spans.to_numpy(), sent.words.to_numpy()
    n = len(sent)
    bins = rng.choice(len(words_p), size=n, p=words_p)
    targets = rng.integers(BIN_LOW[bins], BIN_HIGH[bins] + 1)
    separators = rng.choice(len(SEPARATORS), size=n, p=SEPARATOR_P)
    posts, i = [], 0
    while i < n:
        target, parts, post_spans, total, length, first = targets[len(posts)], [], [], 0, 0, i
        while True:
            if parts:
                parts.append(SEPARATORS[separators[i]])
                length += len(parts[-1])
            parts.append(texts[i])
            post_spans += [(a + length, b + length) for a, b in spans[i]]
            length += len(texts[i])
            total += words[i]
            i += 1
            # stop at the end of a document, or when one more sentence would move away from the target
            if i == n or (doc is not None and doc[i] != doc[i - 1]) or abs(total + words[i] - target) >= abs(total - target):
                break
        text = "".join(parts)
        posts.append((text, post_spans, total, len({text[a:b] for a, b in post_spans}), i - first))
    posts = pd.DataFrame(posts, columns=["text", "spans", "words", "labels", "sentences"])
    posts["bin"] = np.searchsorted(BIN_HIGH[:-1], posts.words)
    return posts


def match_test(posts, rng):
    """The only place where data is removed for having no names.

    Per length bin: keep every post with names and as many empty posts as test has.
    """
    keep = np.ones(len(posts), bool)
    empty, bins = posts.labels.to_numpy() == 0, posts.bin.to_numpy()
    for b, share in enumerate(TARGET_EMPTY):
        candidates = np.flatnonzero(empty & (bins == b))
        wanted = round(int((~empty & (bins == b)).sum()) * share / (1 - share))
        if len(candidates) > wanted:
            keep[rng.choice(candidates, len(candidates) - wanted, replace=False)] = False
    return posts[keep]


def build_posts(sent, rng, by_doc=False):
    """compose + match_test; candidate lengths are corrected so the kept posts follow TARGET_WORDS."""
    words_p = TARGET_WORDS / TARGET_WORDS.sum()
    for _ in range(3):
        candidates = compose(sent, rng, words_p, by_doc)
        posts = match_test(candidates, rng)
        got = np.bincount(posts.bin, minlength=len(words_p)) / len(posts)
        words_p = words_p * np.clip(TARGET_WORDS / np.maximum(got, 1e-3), 0.25, 4)
        words_p /= words_p.sum()
    dropped = candidates.drop(posts.index)
    STATS["empty sentences in"] += int(sent.labels.eq(0).sum())
    STATS["empty sentences removed"] += int(dropped.sentences.sum())
    STATS["empty sentences kept in empty posts"] += int(posts.sentences[posts.labels.eq(0)].sum())
    STATS["empty posts removed"] += len(dropped)
    STATS["empty posts kept"] += int(posts.labels.eq(0).sum())
    return posts[["text", "spans"]]


def profile(posts):
    f = pd.DataFrame([features(t, s) for t, s in zip(posts.text, posts.spans)])
    names = pd.cut(f.labels, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    words = np.bincount(np.searchsorted(BIN_HIGH[:-1], f.words), minlength=len(BIN_NAMES)) / len(f)
    return {**{f"names {k}": 100 * v for k, v in names.value_counts(normalize=True, sort=False).items()},
            **{f"words {n}": 100 * w for n, w in zip(BIN_NAMES, words)},
            "kyiv": 100 * f.kyiv.mean(), "street": 100 * f.street.mean(),
            "multiline": 100 * posts.text.str.contains("\n").mean(), "median words": f.words.median(),
            "ru": 100 * posts.lang.eq("ru").mean(), "posts": len(posts)}


# --- build ------------------------------------------------------------------------------
def _report(title):
    print(f"\n{title}")
    for key, value in STATS.items():
        print(f"  {key:38} {value:>10,}")
    STATS.clear()


def build(rebuild=False, drop_ukraine=False, ru_share=RU_SHARE, seed=0):
    OUT.mkdir(exist_ok=True)
    sentences = {}
    for lang in ("uk", "ru"):   # stage 1: cleaning, cached because ru takes minutes
        path = OUT / f"{lang}_sentences.parquet"
        if rebuild or not path.exists():
            load_sentences(lang).to_parquet(path)
            _report(f"{lang}: cleaning")
        sent = pd.read_parquet(path)
        sent["spans"] = [[(int(a), int(b)) for a, b in spans] for spans in sent.spans]
        # Label policies here, not in cleaning, so they share one cache.
        # Test does not label a city/country inside an organisation name (`поліція Києва`), LB +0.0073.
        before = sum(map(len, sent.spans))
        sent["spans"] = [[s for s in spans if not in_org_name(t, *s)] for t, spans in zip(sent.text, sent.spans)]
        STATS["LOC in org name removed"] += before - sum(map(len, sent.spans))
        if drop_ukraine:
            sent["spans"] = [[s for s in spans if not PROBES["Ukraine"].fullmatch(t[s[0]:s[1]])]
                             for t, spans in zip(sent.text, sent.spans)]
        sent["labels"] = [len({t[a:b] for a, b in spans}) for t, spans in zip(sent.text, sent.spans)]
        _report(f"{lang}: label policy")
        sentences[lang] = sent

    rng = np.random.default_rng(seed)
    result = {}
    for split, share in (("train", ru_share), ("val", VAL_RU_SHARE)):   # stage 2: posts
        parts = {}
        for lang, sent in sentences.items():
            sent = sent[sent.split.eq(split)]
            folds = []
            # per fold, so one post never mixes sentences of two folds
            for fold, group in sent.groupby("fold") if split == "train" else [(-1, sent)]:
                if split == "train" and lang == "uk":
                    group = upsample(group, rng)
                folds.append(build_posts(group, rng, by_doc=lang == "ru").assign(lang=lang, fold=fold))
            parts[lang] = pd.concat(folds, ignore_index=True)
            _report(f"{split} {lang}: removal of data without names")
        wanted_ru = round(len(parts["uk"]) * share / (1 - share))
        if wanted_ru > len(parts["ru"]):
            print(f"  only {len(parts['ru']):,} ru posts for {wanted_ru:,} wanted: lower --ru-share or RU_DOC_MOD")
        parts["ru"] = parts["ru"].sample(n=min(wanted_ru, len(parts["ru"])), random_state=seed)
        posts = pd.concat(parts.values()).sample(frac=1, random_state=seed).reset_index(drop=True)
        assert all(encode(t, s) is not None for t, s in zip(posts.text, posts.spans))
        result[split] = posts
        posts.drop(columns=[] if split == "train" else ["fold"]).to_parquet(OUT / f"{split}.parquet")

    table = pd.DataFrame({"test": TARGET_PROFILE, **{name: profile(posts) for name, posts in result.items()}})
    print("\nProfile, % of posts\n" + table.round(1).to_string())
    return result


def upload():
    """Log train/val parquet as W&B artifact `normalized`; colab.setup() downloads the latest one."""
    import wandb

    from src.config import WANDB_ENTITY, WANDB_PROJECT

    with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type="dataset") as run:
        artifact = wandb.Artifact("normalized", type="dataset")
        for split in ("train", "val"):
            artifact.add_file(str(OUT / f"{split}.parquet"))
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="redo the cleaning stage even if cached")
    parser.add_argument("--drop-ukraine", action="store_true", help="do not label Україна/Украина")
    parser.add_argument("--ru-share", type=float, default=RU_SHARE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upload", action="store_true", help="log train/val parquet to W&B for Colab")
    args = parser.parse_args()
    build(args.rebuild, args.drop_ukraine, args.ru_share, args.seed)
    if args.upload:
        upload()
