"""Word probabilities -> spans: Viterbi that never starts a name on I, rampart's merge of the names it tags, then the
repairs of what the two get wrong."""
import re

import numpy as np

from src.label_policy import trim_edges
from src.tags import B, I, O, TAGS, tags_to_spans


# --- Viterbi ----------------------------------------------------------------------------------------------------
def viterbi(probs):
    """Best tag path of one post [words, tags] with O->I and start->I forbidden, so no span starts on I (argmax gave
    `Михайлівського` + `собору` around an O)."""
    logp = np.log(np.maximum(probs, 1e-9))
    ban = np.zeros((len(TAGS), len(TAGS)))
    ban[O, I] = -np.inf
    score = logp[0].copy()
    score[I] = -np.inf
    back = []
    for row in logp[1:]:
        step = score[:, None] + ban              # [prev, cur]
        back.append(step.argmax(0))
        score = step.max(0) + row
    path = [int(score.argmax())]
    for prev in reversed(back):
        path.append(int(prev[path[-1]]))
    return path[::-1]


# --- rampart ----------------------------------------------------------------------------------------------------
# Entity merging ported from rampart, src/ner/classifier.ts (mergeBioTokens, repairSpans and helpers), run on Viterbi
# tags.
#
# A name gets the mean probability of its words' tags. Names of at least MIN_SCORE are kept; weaker ones of at least
# EXTEND_SCORE survive only when a chain of connectors (space, apostrophe, hyphen, an initial's period) joins them to
# a kept one. Neighbours across a connector are merged. Rampart's rescue of capitalised particles and other
# thresholds lost on val: docs/historic/postprocessing.md.
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


# --- span repairs -----------------------------------------------------------------------------------------------
def split_at_newlines(text, spans):
    """A name never runs over a line break: rampart joins `Межовій\\n\\nКиївщина`, two lines of a list."""
    out = []
    for a, b in spans:
        cuts = [k for k in range(a, b) if text[k] == "\n"]
        out += [(c, d) for c, d in zip([a] + [k + 1 for k in cuts], cuts + [b]) if c < d]
    return out


SPAN_REPAIRS = [split_at_newlines, trim_edges]   # what the decoder gets wrong, on every split


# --- probabilities -> spans -------------------------------------------------------------------------------------
def decode(text, tokens, probs, rampart=True, repairs=SPAN_REPAIRS):
    """(start, end) spans of one post from its word probabilities [words, tags]. Rampart joins neighbours across a
    space, hyphen, apostrophe or an initial's period and drops weak names with no confident neighbour."""
    if not len(probs):
        return []
    tags = viterbi(probs)
    if rampart:
        spans = rampart_spans(text, [(t.start, t.end) for t in tokens], probs, tags)
    else:
        spans = tags_to_spans(tokens, tags)
    for repair in repairs:
        spans = repair(text, spans)
    return spans
