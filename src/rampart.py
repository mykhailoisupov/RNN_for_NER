"""Entity merging ported from rampart, src/ner/classifier.ts (mergeBioTokens, repairSpans and helpers).

Source: https://github.com/nationaldesignstudio/rampart
Copyright 2026 National Design Studio, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes: TypeScript -> Python; one entity type (LOC), so the label checks are dropped; tokens are whole words, so
there is no `##` subword rule; no token windows (the RNN reads the whole post) and no hyphen-to-space folding before
the model (ours was trained on hyphens). The capitalised-particle rescue is for person names in rampart and stays
off for LOC unless `rescue=True`. The result goes back to word tags, so our postprocessing runs on it unchanged.

Input is an independent per-word classification, as transformers.js returns it: O words are dropped, the rest
carry their tag and its probability. Because O words are gone, `B O I` merges into one span across the O word.
"""
import re

MIN_SCORE = 0.4        # anchors: spans at least this confident
EXTEND_SCORE = 0.15    # weaker spans survive only next to an anchor
MAX_ITERS = 32
CONNECTOR_RE = re.compile(r"[\s'’.-]*")
# Python's re has no \p{Lu}/\p{L}: capitals of the Latin and Cyrillic scripts in our data, and any letter.
UPPER = "[A-ZÀ-ÖØ-ÞА-ЯЁІЇЄҐЎ]"
PARTICLE = rf"{UPPER}(?:[^\W\d_]|['’]){{0,3}}"
LEFT_PARTICLE_RE = re.compile(rf"({PARTICLE})([\s'’.-]{{1,3}})$")
RIGHT_PARTICLE_RE = re.compile(rf"([\s'’.-]{{1,3}})({PARTICLE})")
O, B, I = 0, 1, 2


def merge_bio_tokens(tokens):
    """tokens: (tag, start, end, score) of the non-O words in order -> spans {start, end, score}, score = mean."""
    out, current = [], None
    for tag, start, end, score in tokens:
        if current is not None and tag != B:
            current["end"] = end
            current["score"] += score
            current["count"] += 1
        else:
            if current is not None:
                out.append({"start": current["start"], "end": current["end"], "score": current["score"] / current["count"]})
            current = {"start": start, "end": end, "score": score, "count": 1}
    if current is not None:
        out.append({"start": current["start"], "end": current["end"], "score": current["score"] / current["count"]})
    return out


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
            previous = merged[-1]
            merged[-1] = {**previous, "end": max(previous["end"], span["end"]), "score": max(previous["score"], span["score"])}
        else:
            merged.append(dict(span))
    return merged


def rescue_capitalized_particles(raw, span, spans, self_index):
    """Extend a span over an untagged capitalised particle of up to 4 letters (`Van`, `J.`), never into a neighbour."""
    left_bound, right_bound = 0, len(raw)
    for i, other in enumerate(spans):
        if i == self_index:
            continue
        if left_bound < other["end"] <= span["start"]:
            left_bound = other["end"]
        if span["end"] <= other["start"] < right_bound:
            right_bound = other["start"]
    start, end = span["start"], span["end"]
    left = LEFT_PARTICLE_RE.search(raw[:start])
    if left and ("." not in left[2] or len(left[1]) == 1) and start - len(left[0]) >= left_bound:
        start -= len(left[0])
    right = RIGHT_PARTICLE_RE.match(raw, end)
    if right and "." not in right[1] and right.end() <= right_bound:
        end = right.end()
    return span if (start, end) == (span["start"], span["end"]) else {**span, "start": start, "end": end}


def repair_spans(raw, spans, anchor_score=MIN_SCORE, extend_score=EXTEND_SCORE, rescue=False):
    kept = [dict(s) for s in spans if s["score"] >= anchor_score]
    candidates = [dict(s) for s in spans if extend_score <= s["score"] < anchor_score]
    for _ in range(MAX_ITERS):
        changed = False
        for i in range(len(candidates) - 1, -1, -1):
            if any(can_bridge(raw, candidates[i], span) for span in kept):
                kept.append(candidates.pop(i))
                changed = True
        merged = merge_adjacent_connectors(raw, kept)
        if [(s["start"], s["end"]) for s in merged] != [(s["start"], s["end"]) for s in kept]:
            changed = True
        kept = merged
        if rescue:
            for i in range(len(kept)):
                repaired = rescue_capitalized_particles(raw, kept[i], kept, i)
                if repaired is not kept[i]:
                    kept[i] = repaired
                    changed = True
        if not changed:
            break
    return sorted(kept, key=lambda s: (s["start"], s["end"]))


def rampart_tags(raw, offsets, probs, tags, min_score=MIN_SCORE, extend_score=EXTEND_SCORE, rescue=False):
    """Word tags (O/B/I ids) + their probabilities -> word tags after rampart's merge and repair."""
    tokens = [(t, a, b, float(probs[k, t])) for k, (t, (a, b)) in enumerate(zip(tags, offsets)) if t != O]
    spans = [s for s in merge_bio_tokens(tokens) if s["score"] >= extend_score and s["end"] > s["start"]]
    out = [O] * len(offsets)
    for span in repair_spans(raw, spans, min_score, extend_score, rescue):
        inside = [k for k, (a, b) in enumerate(offsets) if a < span["end"] and b > span["start"]]
        for n, k in enumerate(inside):
            out[k] = B if n == 0 else I
    return out


if __name__ == "__main__":
    import numpy as np

    def run(text, words, tags, scores, **kw):
        offsets, pos = [], 0
        for w in words:
            pos = text.index(w, pos)
            offsets.append((pos, pos + len(w)))
            pos += len(w)
        probs = np.zeros((len(words), 3))
        probs[np.arange(len(words)), tags] = scores
        return rampart_tags(text, offsets, probs, tags, **kw)

    # O words are dropped, so B O I is one span across the O word.
    assert run("вул. Велика Васильківська", ["вул", ".", "Велика", "Васильківська"], [B, O, I, I], [.9, .9, .9, .9]) == [B, I, I, I]
    # A weak span survives next to an anchor across a space, not across a comma.
    assert run("Київ Пасажирський", ["Київ", "Пасажирський"], [B, B], [.9, .2]) == [B, I]
    assert run("Київ, Львів", ["Київ", ",", "Львів"], [B, O, B], [.9, .9, .2]) == [B, O, O]
    # Two anchors joined by a hyphen become one span.
    assert run("Івано - Франківськ", ["Івано", "-", "Франківськ"], [B, O, B], [.9, .9, .9]) == [B, I, I]
    # A period bridges only after an initial.
    assert run("Київ. Львів", ["Київ", ".", "Львів"], [B, O, B], [.9, .9, .9]) == [B, O, B]
    assert run("м. Т. Шевченка", ["м", ".", "Т", ".", "Шевченка"], [O, O, B, O, B], [.9, .9, .9, .9, .9]) == [O, O, B, I, I]
    # Below EXTEND_SCORE a span is dropped even next to an anchor.
    assert run("Київ Пасажирський", ["Київ", "Пасажирський"], [B, B], [.9, .1]) == [B, O]
    # A lone weak span with no anchor is dropped.
    assert run("у Києві", ["у", "Києві"], [O, B], [.9, .3]) == [O, O]
    # Rescue (off for LOC by default): an untagged capitalised particle joins, a lowercase word does not.
    assert run("Нью Йорк", ["Нью", "Йорк"], [O, B], [.9, .9]) == [O, B]
    assert run("Нью Йорк", ["Нью", "Йорк"], [O, B], [.9, .9], rescue=True) == [B, I]
    assert run("у Йорк", ["у", "Йорк"], [O, B], [.9, .9], rescue=True) == [O, B]
    print("rampart port: ok")
