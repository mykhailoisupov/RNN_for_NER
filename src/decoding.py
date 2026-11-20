"""Word probabilities -> word tags: Viterbi that never starts a name on I, then rampart's span repair."""
import torch

from src.rampart import rampart_tags
from src.tags import I, O, TAGS


def viterbi(logits, lengths):
    """Best tag path per sequence with O->I and start->I forbidden, so no span starts on I (argmax gave
    `Михайлівського` + `собору` around an O). Returns one list of tag ids per sequence, cut to its length."""
    logp = logits.float().log_softmax(-1)
    ban = torch.zeros(len(TAGS), len(TAGS), device=logp.device)
    ban[O, I] = float("-inf")
    score = logp[:, 0].clone()
    score[:, I] = float("-inf")
    back = []
    for t in range(1, logp.size(1)):
        best, prev = (score.unsqueeze(2) + ban).max(1)          # [batch, prev, cur] -> [batch, cur]
        back.append(prev)
        alive = (t < lengths.to(logp.device)).unsqueeze(1)
        score = torch.where(alive, best + logp[:, t], score)     # padding keeps the score of the last token
    back = torch.stack(back, 1).tolist() if back else []
    paths = []
    for b, n in enumerate(lengths.tolist()):
        tag = int(score[b].argmax())
        path = [tag]
        for t in range(n - 1, 0, -1):
            tag = back[b][t - 1][tag]
            path.append(tag)
        paths.append(path[::-1])
    return paths


def decode(text, tokens, probs, rampart=True):
    """Tags of one post from its word probabilities [words, 3]. Rampart joins neighbours across a space, hyphen,
    apostrophe or an initial's period and drops weak names with no confident neighbour."""
    if not len(probs):
        return []
    tags = viterbi(torch.log(torch.tensor(probs).clamp_min(1e-9))[None], torch.tensor([len(probs)]))[0]
    if rampart:
        tags = rampart_tags(text, [(t.start, t.end) for t in tokens], probs, tags)
    return tags
