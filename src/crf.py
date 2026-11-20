"""Linear-chain CRF over the O/B/I tags.

It learns how likely each tag is after each other tag, trains on the score of the whole tag sequence of a post, and
gives the probability of every tag of every word given the whole post (forward-backward). Decoding, rampart and the
ensemble read those probabilities as they read softmax before.
"""
import torch
from torch import nn

from src.tags import I, O, TAGS

BANNED = -1e4   # O -> I and start -> I: a name never starts on I


class CRF(nn.Module):
    def __init__(self):
        super().__init__()
        n = len(TAGS)
        self.transitions = nn.Parameter(torch.zeros(n, n))   # [from, to]
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))
        ban, start_ban = torch.zeros(n, n), torch.zeros(n)
        ban[O, I] = start_ban[I] = BANNED
        self.register_buffer("ban", ban)
        self.register_buffer("start_ban", start_ban)

    def _alphas(self, emissions, mask):
        """log-sum of the scores of every tag path up to step t that ends in each tag: [batch, steps, tags].
        Past the end of a post the last real step is carried on."""
        transitions = self.transitions + self.ban
        alpha = self.start + self.start_ban + emissions[:, 0]
        alphas = [alpha]
        for t in range(1, emissions.size(1)):
            step = torch.logsumexp(alpha.unsqueeze(2) + transitions, 1) + emissions[:, t]
            alpha = torch.where(mask[:, t, None], step, alpha)
            alphas.append(alpha)
        return torch.stack(alphas, 1)

    def _betas(self, emissions, mask):
        """log-sum of the scores of every tag path from step t to the end of the post, without step t itself."""
        transitions = self.transitions + self.ban
        end = self.end.expand(emissions.size(0), -1)
        beta = end
        betas = [beta]
        for t in range(emissions.size(1) - 2, -1, -1):
            step = torch.logsumexp(transitions + (emissions[:, t + 1] + beta).unsqueeze(1), 2)
            beta = torch.where(mask[:, t + 1, None], step, end)
            betas.append(beta)
        return torch.stack(betas[::-1], 1)

    def nll(self, emissions, tags, mask):
        """Negative log-likelihood of the gold tag sequences, per word, so the loss keeps the scale of cross-entropy."""
        emissions = emissions.float()
        transitions = self.transitions + self.ban
        emitted = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
        score = (self.start + self.start_ban)[tags[:, 0]] + emitted[:, 0]
        score = score + ((transitions[tags[:, :-1], tags[:, 1:]] + emitted[:, 1:]) * mask[:, 1:]).sum(1)
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end[last]
        log_z = torch.logsumexp(self._alphas(emissions, mask)[:, -1] + self.end, -1)
        return (log_z - score).sum() / mask.sum()

    def marginals(self, emissions, mask):
        """P(tag of word t | whole post): [batch, steps, tags]; rows past the end of a post are meaningless."""
        emissions = emissions.float()
        alphas = self._alphas(emissions, mask)
        log_z = torch.logsumexp(alphas[:, -1] + self.end, -1)
        return (alphas + self._betas(emissions, mask) - log_z[:, None, None]).exp()
