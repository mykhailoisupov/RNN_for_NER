"""Subword probabilities of the windows -> [words, 3] of every post: what decode, submission and the ensemble with the
RNNs read. A subword of two windows takes the one where it is further from an edge; a word, its first subword: a
mean would blur B, since a name's later pieces are labelled I (oracle F1 on val 0.94 -> 0.75).

    python -m src.transformer.word_probs  # check: gold labels as probabilities -> words -> answers -> F1
"""
import numpy as np

from src.tags import IGNORE, O, TAGS

NO_SUBWORD = np.eye(len(TAGS))[O]   # a newline: the tokenizer gives it no piece


def stitch(windows, probs):
    """The windows of one post and their [subwords, 3] -> offsets and probabilities of its distinct subwords, in text
    order. Special tokens (label IGNORE) dropped."""
    best = {}
    for window, p in zip(windows, probs):
        real = np.flatnonzero(window["labels"] != IGNORE)
        for rank, j in enumerate(real):
            context = min(rank, len(real) - 1 - rank)   # subwords between it and the nearer edge
            key = tuple(window["offsets"][j])
            if key not in best or context > best[key][0]:
                best[key] = context, p[j]
    keys = sorted(best)
    return np.array(keys, np.int32).reshape(-1, 2), np.array([best[k][1] for k in keys]).reshape(-1, len(TAGS))


def to_words(tokens, offsets, probs):
    """Our words (Token start, end) <- the first subword that overlaps each."""
    out = np.tile(NO_SUBWORD, (len(tokens), 1))
    for k, token in enumerate(tokens):
        inside = (offsets[:, 0] < token.end) & (offsets[:, 1] > token.start)
        if inside.any():
            out[k] = probs[inside][0]
    return out

def post_probs(posts, windows, probs):
    """[words, 3] of every post from the windows of encode() and their probabilities, in the same order."""
    by_post = [([], []) for _ in posts]
    for window, p in zip(windows, probs):
        by_post[window["post"]][0].append(window)
        by_post[window["post"]][1].append(p[:len(window["input_ids"])])   # the collator's padding off
    return [to_words(post.tokens, *stitch(w, p)) for post, (w, p) in zip(posts, by_post)]


# --- check: gold labels through the whole answers path ----------------------------------------------------------
def _check(name="xlm-roberta-base", val_posts=3000):
    """Oracle F1 as src.rnn.train counts it, from subwords: the same with one window per post and with many."""
    from src.dataset import load_eval
    from src.metric import counts_per_post, scores
    from src.submission import gold_and_predicted
    from src.transformer.dataset import encode, load_tokenizer

    tokenizer = load_tokenizer(name)
    posts = load_eval()
    for split, items in (("sample", posts["sample"]), ("val", posts["val"][:val_posts])):
        words = [np.eye(len(TAGS))[p.labels] for p in items]
        print(f"{split}: words {scores(counts_per_post(*gold_and_predicted(items, words)))['f1']:.4f}", end="")
        for max_length, stride in ((512, 128), (64, 16)):
            windows = encode(items, tokenizer, max_length, stride)
            gold = [np.eye(len(TAGS))[np.where(w["labels"] == IGNORE, O, w["labels"])] for w in windows]
            f1 = scores(counts_per_post(*gold_and_predicted(items, post_probs(items, windows, gold))))["f1"]
            print(f" | windows {max_length}/{stride} {f1:.4f}", end="")
        print()


if __name__ == "__main__":
    _check()
