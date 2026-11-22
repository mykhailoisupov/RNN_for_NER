"""Checkpoint of src.train -> word probabilities of every post, and the .npz they are kept in (float16 rows + lengths
per split), which src.ensemble averages.

    python -m src.predict models/<run>/best.ckpt --out models/<run>/probs.npz

Posts are read whole, as ValScore reads val. 256/64-word windows (LB 0.66180 -> 0.66277 on 2026-11-20, when train
posts had at most 352 tokens) went when train got posts of up to 615; code at commit 9d4aa46.
"""
import argparse

import numpy as np
import torch

from src.dataset import SPLITS, build_examples, load_eval, loader
from src.model import TaggerModule


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model, examples):
    """[words, 3] probabilities of every example, each post read whole."""
    model.eval().to(device())
    probs = [None] * len(examples)
    for batch in loader(examples, model.cfg.data.eval_batch_size):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = TaggerModule.load(args.checkpoint)
    posts = load_eval()
    save_probs(args.out, {split: predict_probs(model, build_examples(posts[split], model.vocab)) for split in SPLITS})
    print("written", args.out)
