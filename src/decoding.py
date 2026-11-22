"""Word probabilities -> spans: Viterbi that never starts a name on I, then rampart's span repair."""
import numpy as np

from src.rampart import rampart_spans
from src.tags import I, O, TAGS, tags_to_spans


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


def decode(text, tokens, probs, rampart=True):
    """(start, end) spans of one post from its word probabilities [words, tags]. Rampart joins neighbours across a
    space, hyphen, apostrophe or an initial's period and drops weak names with no confident neighbour."""
    if not len(probs):
        return []
    tags = viterbi(probs)
    if rampart:
        return rampart_spans(text, [(t.start, t.end) for t in tokens], probs, tags)
    return tags_to_spans(tokens, tags)
