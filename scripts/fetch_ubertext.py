"""UberText-NER-Silver (lang-uk, Apache-2.0): sentences of 264 public Telegram channels with silver IOB labels ->
data/tg_geo_dataset.csv in the shape of uk_geo_dataset.csv, so build_dataset relabels it like uk.

    python -m scripts.fetch_ubertext [--shards 2]

Only sentences with a LOC are kept: uk and ru have enough sentences without names. Labels differ from test in the
type words (`селі X` has LOC `X`), `столиці` (O) and cities in posts (`мер Києва` is LOC); fix_spans relabels all three.
"""
import argparse
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "external" / "ubertext-ner-silver"
URL = "https://huggingface.co/datasets/lang-uk/UberText-NER-Silver/resolve/main/data/train-{:05d}-of-00009.parquet"
NO_SPACE_BEFORE = set(",.;:!?)]»%…”")
NO_SPACE_AFTER = set("([«„“")
APOSTROPHES = set("'’ʼ")


def detokenize(tokens, tags):
    """Tokens with IOB tags -> text and {"LOC": [(a, b)], "ORG": [...]}; the corpus keeps no spacing, so it is guessed.
    ponytail: a spaced `Київ - столиця` comes out as `Київ-столиця`; the corpus splits every hyphen, compounds win."""
    text, spans, open_type = "", {"LOC": [], "ORG": []}, None
    quoted = glue_next = False
    for token, tag in zip(tokens, tags):
        glue = not text or glue_next or token in NO_SPACE_BEFORE
        glue_next = token in NO_SPACE_AFTER
        if token in ("-", *APOSTROPHES) and text[-1:].isalpha():
            glue = glue_next = True                     # `Івано - Франківськ`, `Солом ' янська`
        elif token == '"':
            glue, glue_next, quoted = quoted or not text, not quoted, not quoted
        start = len(text) + (0 if glue else 1)
        text += token if glue else " " + token
        kind = tag[2:]
        if tag.startswith("I-") and kind == open_type:
            spans[kind][-1] = (spans[kind][-1][0], len(text))
        elif kind in spans:
            spans[kind].append((start, len(text)))
            open_type = kind
        else:
            open_type = None
    return text, spans


def sentences(path):
    tokens, tags = [], []
    for row in pd.read_parquet(path).text:
        if row.strip(" \n"):
            token, tag = row.rstrip("\n").rsplit(" ", 1)
            tokens.append(token)
            tags.append(tag)
        elif tokens:
            yield tokens, tags
            tokens, tags = [], []
    if tokens:
        yield tokens, tags


def fetch(shards):
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(shards):
        path = CACHE / f"train-{i:05d}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(i), path)
        for tokens, tags in sentences(path):
            if "B-LOC" in tags:
                text, spans = detokenize(tokens, tags)
                rows.append({"text": text, "loc_markers": spans["LOC"], "org_markers": spans["ORG"]})
        print(f"shard {i}: {len(rows):,} sentences with a LOC so far", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "tg_geo_dataset.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=2)
    fetch(parser.parse_args().shards)
