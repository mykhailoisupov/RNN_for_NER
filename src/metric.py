"""The competition metric: micro precision/recall/F1 over the sets of normalised strings of every post."""
import re
import string

import numpy as np


def hard_processing(text):
    """Verbatim copy of the competition metric's normalisation (the missing `f` is theirs)."""
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\b\w\b\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def counts_per_post(y_true, y_pred):
    """tp, fp, fn of every post over sets of hard_processing'ed strings; shape (posts, 3)."""
    assert len(y_true) == len(y_pred)
    rows = []
    for t, p in zip(y_true, y_pred):
        t, p = {hard_processing(s) for s in t}, {hard_processing(s) for s in p}
        rows.append((len(t & p), len(p - t), len(t - p)))
    return np.array(rows, dtype=float).reshape(-1, 3)


def precision_recall_f1(counts):
    """Precision, recall, F1 of summed tp/fp/fn, shape (..., 3). Competition edge cases:
    nothing predicted -> precision 1 if nothing is gold else 0; nothing gold -> recall 1."""
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    precision = (tp + (tp + fp == 0) * (fn == 0)) / np.maximum(tp + fp, 1)
    recall = (tp + (tp + fn == 0)) / np.maximum(tp + fn, 1)
    return precision, recall, 2 * precision * recall / np.maximum(precision + recall, 1e-12)


def scores(counts):
    """Precision, recall, F1 of per-post counts (posts, 3), as floats by name."""
    return dict(zip(("precision", "recall", "f1"), map(float, precision_recall_f1(counts.sum(0)))))


def bootstrap_delta(base, new, n=1000, seed=0):
    """95% interval of F1(new) - F1(base) over posts resampled together (a paired bootstrap)."""
    idx = np.random.default_rng(seed).integers(0, len(base), (n, len(base)))
    return np.quantile(precision_recall_f1(new[idx].sum(1))[2] - precision_recall_f1(base[idx].sum(1))[2], [0.025, 0.975])
