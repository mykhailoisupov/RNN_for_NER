"""Training-data cleaning review and model-independent span recovery limits."""
from collections import Counter

import numpy as np
import pandas as pd


def cleaning_review(frame, clean_fn, seed=0, per_kind=8):
    rng = np.random.default_rng(seed)
    controls = set(rng.choice(frame.index, min(128, len(frame)), replace=False))
    groups = {k: [] for k in ["merge", "trim", "punctuation_only", "reorder", "other_change",
                             "unchanged_with_LOC", "unchanged_without_LOC"]}
    counts = Counter(rows=len(frame))
    for index, text, before, after in zip(frame.index, frame.text, frame.loc_markers, frame.spans):
        if before == after:
            if index in controls:
                groups["unchanged_with_LOC" if before else "unchanged_without_LOC"].append(index)
            continue
        counts["changed_rows"] += 1
        assert all(0 <= a < b <= len(text) and any(c.isalnum() for c in text[a:b]) for a,b in after)
        assert clean_fn(text, after) == after
        raw_chars = {i for a,b in before for i in range(a,b) if text[i].isalnum()}
        clean_chars = {i for a,b in after for i in range(a,b) if text[i].isalnum()}
        counts["rows_losing_alphanumeric_coverage"] += bool(raw_chars - clean_chars)
        ordered = sorted(before)
        flags = {"merge": any(a <= max(e for _,e in ordered[:i]) for i,(a,b) in enumerate(ordered) if i),
                 "trim": any(text[a:b] != text[a:b].strip() for a,b in before),
                 "punctuation_only": any(not any(c.isalnum() for c in text[a:b]) for a,b in before),
                 "reorder": before != ordered}
        if not any(flags.values()): flags["other_change"] = True
        for kind, flag in flags.items():
            if flag:
                groups[kind].append(index)
                counts[kind + "_rows"] += 1
    review = []
    for kind, indices in groups.items():
        for index in rng.choice(indices, min(per_kind, len(indices)), replace=False):
            row = frame.loc[index]
            review.append({"source_index": index, "reason": kind, "text": row.text,
                           "before_spans": row.loc_markers, "after_spans": row.spans,
                           "before_strings": [row.text[a:b] for a,b in row.loc_markers],
                           "after_strings": [row.text[a:b] for a,b in row.spans],
                           "review_decision": "", "review_note": ""})
    return dict(counts), pd.DataFrame(review)


def span_recovery(frame, tokenize_fn, bio_fn, settings, span_col="spans"):
    counters = {name: Counter() for name,_,_ in settings}
    for text, spans in zip(frame.text, frame[span_col]):
        offsets = tokenize_fn(text)
        tags, _ = bio_fn(offsets, spans)
        predictions, i = set(), 0
        while i < len(tags):
            if tags[i] == "O": i += 1; continue
            j = i + 1
            while j < len(tags) and tags[j] == "I": j += 1
            predictions.add((offsets[i][0], offsets[j-1][1]))
            i = j
        gold = set(spans)
        for name, size, stride in settings:
            if size is None: windows = [(0, len(offsets))]
            elif stride is None: windows = [(0, min(size, len(offsets)))]
            else:
                windows = []
                for start in range(0, len(offsets), stride):
                    windows.append((start, min(start + size, len(offsets))))
                    if start + size >= len(offsets): break
            c = counters[name]
            c["texts"] += 1
            c["gold_spans"] += len(gold)
            c["not_representable_by_BIO"] += len(gold - predictions)
            c["full_sequence_extra_spans"] += len(predictions - gold)
            for a,b in gold & predictions:
                hits = [k for k,(s,e) in enumerate(offsets) if s < b and a < e]
                recovered = bool(hits) and any(start <= hits[0] and hits[-1] < end for start,end in windows)
                c["recoverable_spans"] += recovered
                c["lost_to_length_or_windows"] += not recovered
    result = pd.DataFrame(counters).T.fillna(0).astype(int)
    result["oracle_span_recall_pct"] = 100*result.recoverable_spans/result.gold_spans.replace(0,np.nan)
    assert (result.gold_spans == result.not_representable_by_BIO + result.recoverable_spans + result.lost_to_length_or_windows).all()
    return result
