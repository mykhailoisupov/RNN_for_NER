"""Post -> windows of subwords: input ids, O/B/I labels from the gold character spans, offsets. The tokenizer reads
the whole prepared text, as in pretraining, not our words one by one; offsets hold in the original text, since
prepare() keeps its length.

    python -m src.transformer.dataset     # check: labels -> spans -> F1 against gold
"""
from functools import cache

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.tags import B, I, IGNORE, O
from src.tokenizer import prepare

CHUNK = 10_000   # posts per tokenizer call


@cache
def load_tokenizer(name):
    """A fast tokenizer: only it returns offset_mapping."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    assert tokenizer.is_fast, f"{name} has no fast tokenizer"
    return tokenizer


def subword_labels(offsets, spans):
    """O/B/I of every subword (a, b) of a window. B on the first piece that holds a span's start, not on the first
    piece in the window: a window may begin inside a name, and XLM-R puts a lone `▁` (49, 50) before `Англ` (49, 53)."""
    labels = [O] * len(offsets)
    for s, e in spans:
        begun = False                                   # the span's B is given
        for k, (a, b) in enumerate(offsets):
            if a < e and b > s:                         # the subword overlaps the span
                labels[k] = B if a <= s < b and not begun else I
                begun = True
    return labels


def encode(posts, tokenizer, max_length, stride, keep_offsets=True):
    """Windows of every post: input_ids, labels, post (its index), offsets. A post longer than max_length gives
    several, `stride` subwords shared. Train keeps no offsets: for 440k posts they take gigabytes."""
    windows = []
    for first in range(0, len(posts), CHUNK):
        chunk = posts[first:first + CHUNK]
        enc = tokenizer([prepare(p.text) for p in chunk], max_length=max_length, stride=stride, truncation=True,
                        return_overflowing_tokens=True, return_offsets_mapping=True)
        for k, i in enumerate(enc["overflow_to_sample_mapping"]):
            real = [j for j, seq in enumerate(enc.sequence_ids(k)) if seq is not None]   # not <s>, </s>
            offsets = np.array(enc["offset_mapping"][k], np.int32)
            labels = np.full(len(offsets), IGNORE, np.int8)
            labels[real] = subword_labels([tuple(o) for o in offsets[real].tolist()], chunk[i].spans)
            window = {"input_ids": np.array(enc["input_ids"][k], np.int32), "labels": labels, "post": first + i}
            if keep_offsets:
                window["offsets"] = offsets
            windows.append(window)
    return windows


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]


# --- check: what labelling subwords loses by itself -------------------------------------------------------------
def _spans_from_labels(window):
    spans = []
    for (a, b), label in zip(window["offsets"].tolist(), window["labels"].tolist()):
        if label == B or (label == I and not spans):
            spans.append([a, b])
        elif label == I:
            spans[-1][1] = b
    return [tuple(s) for s in spans]


def _check(name="xlm-roberta-base", max_length=512, stride=128, val_posts=3000):
    """Gold spans -> labels -> spans, scored by the metric on posts of one window: ≈ 1.0 on sample and val."""
    from src.dataset import load_eval
    from src.metric import counts_per_post, scores

    tokenizer = load_tokenizer(name)
    posts = load_eval()
    for split, items in (("sample", posts["sample"]), ("val", posts["val"][:val_posts])):
        windows = encode(items, tokenizer, max_length, stride)
        per_post = np.bincount([w["post"] for w in windows], minlength=len(items))
        single = [w for w in windows if per_post[w["post"]] == 1]
        gold = [items[w["post"]].locations for w in single]
        predicted = [[items[w["post"]].text[a:b] for a, b in _spans_from_labels(w)] for w in single]
        print(f"{split}: {len(items)} posts, {len(windows)} windows, {int((per_post > 1).sum())} posts split; "
              f"oracle F1 {scores(counts_per_post(gold, predicted))['f1']:.4f}")

    post = next(p for p in posts["sample"] if p.locations)
    window = encode([post], tokenizer, max_length, stride)[0]
    print("\ngold:", post.locations, "\nsubwords not O:")
    for token, label in zip(tokenizer.convert_ids_to_tokens(window["input_ids"].tolist()), window["labels"]):
        if label != O:
            print(f"  {token:15} {'OBI'[label] if label != IGNORE else '-100'}")


if __name__ == "__main__":
    _check()
