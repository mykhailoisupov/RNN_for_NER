"""Conservative near-duplicate experiment for LOC-labelled training sentences."""

import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd
from datasketch import MinHash, MinHashLSH


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
