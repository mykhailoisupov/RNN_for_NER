"""Entity merging ported from rampart, src/ner/classifier.ts (mergeBioTokens, repairSpans and helpers), run on
Viterbi tags.

A name gets the mean probability of its words' tags. Names of at least MIN_SCORE are kept; weaker ones of at least
EXTEND_SCORE survive only when a chain of connectors (space, apostrophe, hyphen, an initial's period) joins them to a
kept one. Neighbours across a connector are merged. Rampart's rescue of capitalised particles and other thresholds
lost on val: docs/historic/postprocessing.md.
"""
import re

from src.tags import B, O

MIN_SCORE = 0.4        # anchors: spans at least this confident
EXTEND_SCORE = 0.15    # weaker spans survive only next to an anchor
CONNECTOR_RE = re.compile(r"[\s'’.-]*")


def merge_bio_tokens(tokens):
    """tokens: (tag, start, end, score) of the non-O words in order -> spans {start, end, score}, score = mean.
    Viterbi puts no I after O, so a span is a B and the I words right after it."""
    groups = []
    for tag, start, end, score in tokens:
        if groups and tag != B:
            groups[-1][1] = end
            groups[-1][2].append(score)
        else:
            groups.append([start, end, [score]])
    return [{"start": a, "end": b, "score": sum(s) / len(s)} for a, b, s in groups]


def is_initial_char(raw, idx):
    """A lone capital letter with a non-letter to its left (the `F` in `John F.`)."""
    if not 0 <= idx < len(raw) or not raw[idx].isupper():
        return False
    return idx == 0 or not raw[idx - 1].isalpha()


def can_bridge(raw, a, b):
    left, right = (a, b) if a["start"] <= b["start"] else (b, a)
    gap = raw[left["end"]:right["start"]]
    if not CONNECTOR_RE.fullmatch(gap):
        return False
    # A period bridges fragments only across an initial ("J." + "R."), not after a full word.
    return "." not in gap or is_initial_char(raw, left["end"] - 1)


def merge_adjacent_connectors(raw, spans):
    merged = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if merged and can_bridge(raw, merged[-1], span):
            merged[-1] = {**merged[-1], "end": max(merged[-1]["end"], span["end"])}
        else:
            merged.append(dict(span))
    return merged


def repair_spans(raw, spans):
    """Spans of at least MIN_SCORE and the weaker ones a chain of connectors joins to them, neighbours merged."""
    kept = [s for s in spans if s["score"] >= MIN_SCORE]
    candidates = [s for s in spans if s["score"] < MIN_SCORE]
    while moved := [c for c in candidates if any(can_bridge(raw, c, k) for k in kept)]:
        kept += moved
        candidates = [c for c in candidates if c not in moved]
    return merge_adjacent_connectors(raw, kept)


def rampart_spans(raw, offsets, probs, tags):
    """Viterbi tags + their probabilities -> (start, end) spans after rampart's merge and repair."""
    tokens = [(t, a, b, float(probs[k, t])) for k, (t, (a, b)) in enumerate(zip(tags, offsets)) if t != O]
    spans = [s for s in merge_bio_tokens(tokens) if s["score"] >= EXTEND_SCORE]
    return [(s["start"], s["end"]) for s in repair_spans(raw, spans)]
