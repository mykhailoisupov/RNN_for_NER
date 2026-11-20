"""Word tags O, B, I and the character spans they stand for."""
TAGS = ["O", "B-LOC", "I-LOC"]
O, B, I = range(len(TAGS))


def spans_to_tags(tokens, spans):
    """A tag per token; a token already inside an earlier span keeps its tag."""
    tags = [O] * len(tokens)
    for a, b in spans:
        inside = [i for i, t in enumerate(tokens) if t.start < b and t.end > a]
        for k, i in enumerate(inside):
            if tags[i] == O:
                tags[i] = B if k == 0 else I
    return tags


def tags_to_spans(tokens, tags):
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
