"""Word tags O, B, I and the character spans they stand for; also the `(a, b)` spans of a labelled CSV."""
import re

TAGS = ["O", "B", "I"]       # one entity type: a location
O, B, I = range(len(TAGS))
SPAN = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_spans(value):
    """`[(3, 8), (12, 17)]` as a CSV cell keeps it -> [(3, 8), (12, 17)]."""
    return [(int(a), int(b)) for a, b in SPAN.findall(str(value))]


def spans_to_tags(tokens, spans):
    """A tag per token; a token already taken by an earlier, overlapping span keeps its tag."""
    tags = [O] * len(tokens)
    for a, b in spans:
        inside = [i for i, t in enumerate(tokens) if t.start < b and t.end > a]
        for k, i in enumerate(inside):
            if tags[i] == O:
                tags[i] = B if k == 0 else I
    return tags


def tags_to_spans(tokens, tags):
    """A stray I after O starts a span of its own; Viterbi never gives one."""
    spans, i = [], 0
    while i < len(tags):
        if tags[i] == O:
            i += 1
            continue
        j = i + 1
        while j < len(tags) and tags[j] == I:
            j += 1
        spans.append((tokens[i].start, tokens[j - 1].end))
        i = j
    return spans


def aligned(tokens, spans):
    """Do the spans survive spans -> tags -> spans: on token edges, none overlapping?"""
    return tags_to_spans(tokens, spans_to_tags(tokens, spans)) == sorted(spans)
