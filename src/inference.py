"""Checkpoint -> word probabilities of every post, and the .npz they are kept in (float16 rows + lengths per split).

Posts are read whole, as ValScore reads val. 256/64-word windows (LB 0.66180 -> 0.66277 on 2026-11-20, when train
posts had at most 352 tokens) went when train got posts of up to 615; code at commit 9d4aa46.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import config_from_dict
from src.dataset import SPLITS, BucketSampler, NERDataset, collate
from src.model import TaggerModule
from src.vocab import saved_vocab


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path):
    """A checkpoint saved by src.train -> (model, vocab); the data is not needed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(checkpoint["config"])
    vocab = saved_vocab(cfg.vocab, checkpoint["vocab"])
    model = TaggerModule(cfg, vocab)
    model.load_weights(checkpoint["state_dict"])
    return model, vocab


@torch.no_grad()
def predict_probs(model, examples, batch_size):
    """[words, 3] probabilities of every example, each post read whole."""
    model.eval().to(device())
    sampler = BucketSampler([len(ex.ids) for ex in examples], batch_size, shuffle=False)
    probs = [None] * len(examples)
    for batch in DataLoader(NERDataset(examples), batch_sampler=sampler, collate_fn=collate):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        for row, i in zip(model.probs(batch).cpu().numpy(), batch["idx"].tolist()):
            probs[i] = row[:len(examples[i].words)]
    return probs


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
