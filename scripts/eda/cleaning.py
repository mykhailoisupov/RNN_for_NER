"""Cleaning review, span recovery limits and near-duplicate search for the training sentences."""
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH


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


def _canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _features(text, spans, shingle_size):
    matches = list(re.finditer(r"\w+(?:['’ʼ-]\w+)*", text))
    tokens = tuple(_canonical(m.group()) for m in matches)
    tags = ["O"] * len(tokens)
    for a, b in sorted(spans):
        hit = [i for i, m in enumerate(matches) if m.start() < b and a < m.end()]
        if not hit or matches[hit[0]].start() != a or matches[hit[-1]].end() != b:
            return None
        if any(tags[i] != "O" for i in hit):
            return None
        for i in hit:
            tags[i] = "B" if i == hit[0] else "I"
    shingles = {tokens[i:i + shingle_size] for i in range(len(tokens) - shingle_size + 1)}
    return tokens, tuple(tags), shingles


def _labels_agree(left, right):
    # Matching entity names alone cannot distinguish different occurrences of a name.
    for op, a, b, c, d in SequenceMatcher(None, left[0], right[0], autojunk=False).get_opcodes():
        if op == "equal":
            if left[1][a:b] != right[1][c:d]:
                return False
        elif any(t != "O" for t in left[1][a:b] + right[1][c:d]):
            return False
    return True


def deduplicate_near(frame, protected_norms=(), *, threshold=0.90,
                     shingle_size=3, min_tokens=6):
    """Return retained rows, a removal audit and counts; never mutate input.

    Search only within identical ordered LOC-name sequences. Short sentences,
    disputed exact groups and spans not aligned to word boundaries are retained.
    Every removed row must match a retained representative directly.
    """
    if not 0 < threshold <= 1 or not 1 <= shingle_size <= min_tokens:
        raise ValueError("Invalid Jaccard threshold or shingle/token lengths")
    if not frame.index.is_unique:
        raise ValueError("Source row indices must be unique for the audit")
    protected_norms = set(protected_norms)
    groups = defaultdict(list)
    counts = Counter(input_rows=len(frame))
    texts, spans = frame.text.to_numpy(), frame.spans.to_numpy()
    indices = frame.index.to_numpy()
    for pos, (text, markers, norm) in enumerate(zip(texts, spans, frame.norm)):
        if any(not 0 <= a < b <= len(text) for a, b in markers):
            raise ValueError(f"Invalid span at source index {indices[pos]}")
        if norm in protected_norms:
            counts["protected_exact_conflicts"] += 1
            continue
        if not markers:
            raise ValueError("Expected sentences with at least one LOC span")
        key = tuple(_canonical(text[a:b]) for a, b in sorted(markers))
        groups[key].append(pos)

    template = MinHash(num_perm=128, seed=0)
    removed, audit = set(), []
    for positions in groups.values():
        if len(positions) == 1:
            counts["singleton_LOC_sequences"] += 1
            continue
        # Partitioning by LOC sequence bounds memory and preserves different place names.
        lsh = MinHashLSH(threshold=threshold, num_perm=128, params=(16, 8))
        retained_features = {}
        for pos in positions:
            features = _features(texts[pos], spans[pos], shingle_size)
            if features is None:
                counts["unaligned_spans_retained"] += 1
                continue
            if len(features[0]) < min_tokens:
                counts["short_sentences_retained"] += 1
                continue
            counts["rows_searched"] += 1
            sketch = template.copy()
            sketch.update_batch(["\x1f".join(s).encode("utf-8") for s in sorted(features[2])])
            for candidate in sorted(lsh.query(sketch)):
                counts["candidate_pairs_checked"] += 1
                previous = retained_features[candidate]
                similarity = len(features[2] & previous[2]) / len(features[2] | previous[2])
                if similarity < threshold:
                    continue
                if not _labels_agree(previous, features):
                    counts["label_disagreements_retained"] += 1
                    continue
                removed.add(pos)
                audit.append((indices[pos], indices[candidate], similarity))
                break
            else:
                lsh.insert(pos, sketch)
                retained_features[pos] = features

    audit = pd.DataFrame(audit, columns=["removed_index", "kept_index", "jaccard"])
    result = frame.iloc[[pos for pos in range(len(frame)) if pos not in removed]].copy()
    counts.update(removed_near_duplicates=len(removed), retained_rows=len(result))
    assert len(frame) == len(result) + len(audit)
    assert audit.kept_index.isin(result.index).all()
    assert not audit.removed_index.isin(result.index).any()
    return result, audit, dict(counts)
