"""Checkpoint of src.train -> word probabilities on val/sample/test, the .npz src.ensemble averages.

    python -m src.predict models/<run>/best.ckpt --out models/<run>/probs.npz
"""
import argparse

from src.dataset import SPLITS, build_examples, load_eval
from src.inference import load_model, predict_probs, save_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, vocab = load_model(args.checkpoint)
    posts = load_eval()
    save_probs(args.out, {split: predict_probs(model, build_examples(posts[split], vocab), model.cfg.data.eval_batch_size)
                          for split in SPLITS})
    print("written", args.out)
