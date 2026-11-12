"""Descriptive test fingerprints for future validation design; no split selection."""
import re

import numpy as np
import pandas as pd

from scripts.label_analysis import STREET
from scripts.test_analysis import words


def feature_row(text, labels):
    labels = set(labels)
    return {"chars": len(text), "words": len(words(text)), "labels": len(labels),
            "kyiv": bool(re.search(r"київ|києв|киев", text, re.I)),
            "street": any(re.search(STREET, s, re.I) for s in labels)}


def source_features(frame):
    data = {k: np.empty(len(frame), dtype=bool if k in ["kyiv", "street"] else np.int32)
            for k in ["chars", "words", "labels", "kyiv", "street", "clean_labels"]}
    for pos, (text, spans, cleaned) in enumerate(zip(frame.text, frame.loc_markers, frame.spans)):
        row = feature_row(text, (text[a:b] for a, b in spans))
        row["clean_labels"] = len({text[a:b] for a, b in cleaned})
        for key, value in row.items(): data[key][pos] = value
    return pd.DataFrame(data, index=frame.index, copy=False)


def sentence_ranges(text):
    # Same character length, so source offsets remain valid after decoding literal \n.
    view = text.replace("\\n", " \n")
    for match in list(re.finditer(r"\b(?:вул|ул|м|г|ст|р|ім|просп|обл|т\.д)\.", view, re.I)):
        a, b = match.span()
        view = view[:a] + view[a:b].replace(".", "\uffff") + view[b:]
    start = 0
    for sep in re.finditer(r"\s*\n+\s*|(?<=[.!?…])\s+(?=[А-ЯІЇЄҐA-Z0-9«\"])", view):
        if words(text[start:sep.start()]): yield start, sep.start()
        start = sep.end()
    if words(text[start:]): yield start, len(text)


def pseudo_sentences(predictions):
    rows = []
    for row in predictions.itertuples():
        spans = {s: [(m.start(), m.end()) for m in re.finditer(re.escape(s), row.text)] for s in row.locations}
        for number, (a, b) in enumerate(sentence_ranges(row.text)):
            definite, possible = [], []
            for s, hits in spans.items():
                overlapping = [(c, d) for c, d in hits if c < b and a < d]
                if len(hits) == 1 and overlapping and a <= hits[0][0] and hits[0][1] <= b:
                    definite.append(s)
                elif overlapping:
                    possible.append(s)
            text = row.text[a:b]
            record = feature_row(text, definite)
            record.update(post_key=row.key, sentence=number, text=text, start=a, end=b,
                          labels_upper=len(set(definite + possible)), uncertain=bool(possible))
            rows.append(record)
    return pd.DataFrame(rows)


def profile(frame, label_col="labels"):
    n = frame[label_col]
    return {"rows": len(frame), "zero_labels_pct": 100 * n.eq(0).mean(),
            "labels_total": int(n.sum()), "labels_per_row_mean": n.mean(),
            "labels_p50": n.median(), "labels_p90": n.quantile(.9), "labels_max": n.max(),
            "words_p50": frame.words.median(), "words_p90": frame.words.quantile(.9),
            "kyiv_rows_pct": 100 * frame.kyiv.mean(), "street_label_rows_pct": 100 * frame.street.mean()}


def strata(frame):
    labels = pd.cut(frame.labels, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    lengths = pd.cut(frame.words, [-1, 15, 30, 60, 120, np.inf], labels=["0-15", "16-30", "31-60", "61-120", "121+"])
    return pd.Series(list(zip(labels.astype(str), lengths.astype(str), frame.kyiv, frame.street)), index=frame.index)


def distributions(frame, label_col="labels"):
    labels = pd.cut(frame[label_col], [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    lengths = pd.cut(frame.words, [-1, 15, 30, 60, 120, np.inf], labels=["0-15", "16-30", "31-60", "61-120", "121+"])
    rows = []
    for name, values in {"labels": labels, "words": lengths, "kyiv": frame.kyiv, "street": frame.street}.items():
        for value, count in values.value_counts(sort=False).items():
            rows.append({"feature": name, "bin": str(value), "rows": int(count), "fraction": count / len(frame)})
    return pd.DataFrame(rows)


def distribution_distances(tables, target):
    rows = []
    reference = tables[target]
    for name, table in tables.items():
        for feature in reference.feature.unique():
            a = reference.loc[reference.feature.eq(feature)].set_index("bin").fraction
            b = table.loc[table.feature.eq(feature)].set_index("bin").fraction
            keys = a.index.union(b.index)
            distance = .5 * (a.reindex(keys, fill_value=0) - b.reindex(keys, fill_value=0)).abs().sum()
            rows.append({"target": target, "dataset": name, "feature": feature, "total_variation": distance})
    return pd.DataFrame(rows)
