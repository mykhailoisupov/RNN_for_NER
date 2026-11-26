"""Checkpoint of src.rnn.train -> word probabilities of every post -> the .npz of src.ensemble.save_probs.

    python -m src.rnn.predict models/<run>/best.ckpt --out models/<run>/probs.npz

Posts are read whole, as ValScore reads val. 256/64-word windows (LB 0.66180 -> 0.66277 on 2026-11-20, when train
posts had at most 352 tokens) went when train got posts of up to 615; code at commit 9d4aa46.
"""
import argparse

import torch

from src.dataset import SPLITS, load_eval
from src.device import device
from src.ensemble import save_probs
from src.rnn.dataset import build_examples, loader
from src.rnn.model import TaggerModule


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = TaggerModule.load(args.checkpoint)
    posts = load_eval()
    save_probs(args.out, {split: predict_probs(model, build_examples(posts[split], model.vocab)) for split in SPLITS})
    print("written", args.out)
