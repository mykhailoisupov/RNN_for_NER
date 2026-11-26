"""Files -> Post: train/val built by build_dataset, sample/test as the competition gives them, tokenized and tagged."""
import sys
from typing import NamedTuple

import pandas as pd

from src.config import DATA_DIR, NORMALIZED_DIR
from src.tags import parse_spans, spans_to_tags
from src.tokenizer import tokenize

EVAL_SETS = ("val", "sample")    # scored every epoch; the order of val_dataloader
SPLITS = (*EVAL_SETS, "test")  # what load_eval returns and a model predicts


class Post(NamedTuple):
    """A post as its file gives it, tokenized and tagged."""
    text: str
    tokens: list | None   # Token offsets, to cut answers from the text; None for train, which never cuts any
    words: list           # token texts, one str object per distinct word
    labels: list          # O/B/I of every word
    spans: list           # gold (start, end) character spans; what the transformer labels subwords by
    locations: list       # gold answers


def _posts(texts, spans, keep_tokens=True):
    """Train keeps no Token objects: 30M of them took ~5 GB, and only answers need offsets."""
    posts = []
    for text, pairs in zip(texts, spans):
        tokens = tokenize(text)
        pairs = [(int(a), int(b)) for a, b in pairs]
        posts.append(Post(text, tokens if keep_tokens else None, [sys.intern(t.text) for t in tokens],
                          spans_to_tags(tokens, pairs), pairs, [text[a:b] for a, b in pairs]))
    return posts


def load_train(train_size, seed):
    """Posts from build_dataset; train_size 0 is all of them."""
    train = pd.read_parquet(NORMALIZED_DIR / "train.parquet", columns=["text", "spans"])
    if train_size:
        train = train.sample(min(train_size, len(train)), random_state=seed)
    return _posts(train["text"], train["spans"], keep_tokens=False)


def load_eval():
    """val from build_dataset, sample and test as the competition gives them."""
    val = pd.read_parquet(NORMALIZED_DIR / "val.parquet", columns=["text", "spans"])
    sample = pd.read_csv(DATA_DIR / "labeling_sample.csv")
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["text"])
    return {
        "val": _posts(val["text"], val["spans"]),
        "sample": _posts(sample["text"], sample["loc_markers"].map(parse_spans)),
        "test": _posts(test["text"], [[]] * len(test)),
    }


def load_test_ids():
    """text_id of every test post, in the order of load_eval: the submission's key."""
    return pd.read_csv(DATA_DIR / "test.csv", usecols=["text_id"])["text_id"].tolist()


def load_val_langs():
    """Source (uk, ru, tg) of every val post, in the order of load_eval: val F1 per source."""
    return pd.read_parquet(NORMALIZED_DIR / "val.parquet", columns=["lang"])["lang"].tolist()
