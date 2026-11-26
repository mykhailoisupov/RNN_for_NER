"""Run directory of src.transformer.train -> word probabilities of every post -> the .npz of src.ensemble.save_probs,
the same as the RNNs give.

    python -m src.transformer.predict models/<run> --out models/<run>/probs.npz
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.cli_config import config_from_dict
from src.dataset import SPLITS, load_eval
from src.device import device
from src.ensemble import save_probs
from src.transformer.collator import Collator
from src.transformer.config import Config
from src.transformer.dataset import WindowDataset, encode, load_tokenizer
from src.transformer.models.token_classifier import load_model
from src.transformer.word_probs import post_probs

CONFIG = "run_config.json"   # the Config of a run, next to its weights


def load_run(run_dir):
    """Fine-tuned model, its tokenizer and Config."""
    cfg = config_from_dict(json.loads((Path(run_dir) / CONFIG).read_text()), Config)
    return load_model(str(run_dir)), load_tokenizer(str(run_dir)), cfg


@torch.no_grad()
def predict_probs(model, tokenizer, posts, cfg: Config):
    """[words, 3] of every post."""
    windows = encode(posts, tokenizer, cfg.data.max_length, cfg.data.stride)
    batches = DataLoader(WindowDataset(windows), cfg.data.eval_batch_size, collate_fn=Collator(tokenizer.pad_token_id))
    model.eval().to(device())
    probs = []
    for batch in batches:
        batch.pop("labels")
        logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits
        probs += list(logits.float().softmax(-1).cpu().numpy())
    return post_probs(posts, windows, probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, tokenizer, cfg = load_run(args.run_dir)
    posts = load_eval()
    save_probs(args.out, {split: predict_probs(model, tokenizer, posts[split], cfg) for split in SPLITS})
    print("written", args.out)
