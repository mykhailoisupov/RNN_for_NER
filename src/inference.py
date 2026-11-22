"""Checkpoint -> word probabilities of every post, and the .npz they are kept in (float16 rows + lengths per split).

A post longer than the window is cut into overlapping windows (the last one ends at the end of the post) and each
word's probabilities are averaged over the windows that cover it. 256/64: a bit under the longest train post
(352 words), overlap as rampart's NER_TOKEN_OVERLAP. LB 0.66180 -> 0.66277; 128-word windows gave 0.66084.
"""
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import SPLITS, BucketSampler, NERDataset, collate
from src.model import TaggerModule
from src.tags import TAGS
from src.vocab import saved_vocab

WINDOW, OVERLAP = 256, 64


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path):
    """A checkpoint saved by src.train -> (model, vocab); the data is not needed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    vocab = saved_vocab(cfg.vocab, checkpoint["vocab"])
    model = TaggerModule(cfg, vocab)
    model.load_state_dict(checkpoint["state_dict"])
    return model, vocab


@torch.no_grad()
def _word_probs(model, examples, batch_size, on):
    model.eval().to(on or device())
    sampler = BucketSampler([len(ex.ids) for ex in examples], batch_size, shuffle=False)
    probs = [None] * len(examples)
    for batch in DataLoader(NERDataset(examples), batch_sampler=sampler, collate_fn=collate):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        for row, i in zip(model.probs(batch).cpu().numpy(), batch["idx"].tolist()):
            probs[i] = row[:len(examples[i].words)]
    return probs


def windows(n, size, overlap):
    """Word ranges of length `size`, stepping size - overlap; the last one ends at the end of the post."""
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size, size - overlap)) + [n - size]
    return [(s, s + size) for s in starts]


def predict_probs(model, examples, batch_size, window=WINDOW, overlap=OVERLAP, on=None):
    """[words, 3] probabilities per example; posts longer than `window` words are read in windows."""
    pieces, owners = [], []
    for i, ex in enumerate(examples):
        if len(ex.words) <= window:
            pieces.append(ex)
            owners.append((i, 0))
            continue
        for a, b in windows(len(ex.words), window, overlap):
            pieces.append(replace(ex, tokens=ex.tokens[a:b], words=ex.words[a:b], locations=[], ids=ex.ids[a:b],
                                  labels=ex.labels[a:b], chars=ex.chars[a:b], shapes=ex.shapes[a:b]))
            owners.append((i, a))
    sums = [np.zeros((len(ex.words), len(TAGS))) for ex in examples]
    counts = [np.zeros(len(ex.words)) for ex in examples]
    for probs, (i, a) in zip(_word_probs(model, pieces, batch_size, on), owners):
        sums[i][a:a + len(probs)] += probs
        counts[i][a:a + len(probs)] += 1
    return [s / np.maximum(c, 1)[:, None] for s, c in zip(sums, counts)]


def save_probs(path, probs):
    arrays = {}
    for split, rows in probs.items():
        arrays[split] = np.concatenate(rows).astype(np.float16)
        arrays[f"{split}_lengths"] = np.array([len(r) for r in rows])
    np.savez_compressed(path, **arrays)


def load_probs(path):
    data = np.load(path)
    return {
        split: np.split(data[split].astype(np.float32), np.cumsum(data[f"{split}_lengths"])[:-1])
        for split in SPLITS if split in data.files
    }
