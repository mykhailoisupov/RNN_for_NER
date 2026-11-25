"""Several .npz of word probabilities -> their weighted average -> answers -> the submission csv (+ `_raw`). The .npz
(float16 rows + lengths per split) is written by save_probs, the same for every model.

    python -m src.ensemble models/<run>/probs.npz models/<run>/probs.npz ... --name <name>
"""
import argparse

import numpy as np
import pandas as pd

from src.config import SUBMISSION_DIR
from src.dataset import EVAL_SETS, SPLITS, load_eval, load_test_ids
from src.metric import counts_per_post, scores
from src.submission import gold_and_predicted, write_submissions


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


def average(members, weights, split):
    return [sum(w * p for w, p in zip(weights, posts)) / sum(weights) for posts in zip(*(m[split] for m in members))]


def ensemble_probs(members, weights, posts):
    """Weighted average per split; a split is skipped when some .npz were made on other posts or by another tokenizer
    (val changes with every build_dataset). test must match: after a tokenizer change, remake the .npz with
    the model's predict (src.rnn.predict)."""
    probs = {}
    for split in SPLITS:
        want = [len(post.words) for post in posts[split]]
        if all(split in m and [len(r) for r in m[split]] == want for m in members):
            probs[split] = average(members, weights, split)
        else:
            print(f"{split}: skipped, .npz made on other data")
    assert "test" in probs, "test probs do not match test.csv"
    return probs


def run(paths, weights, name):
    members = [load_probs(p) for p in paths]
    posts, test_ids = load_eval(), load_test_ids()
    probs = ensemble_probs(members, weights, posts)

    row = {}
    for split in [s for s in EVAL_SETS if s in probs]:
        c = counts_per_post(*gold_and_predicted(posts[split], probs[split]))
        row.update({f"{split}_{k}": v for k, v in scores(c).items()})
    table = pd.DataFrame([row])
    SUBMISSION_DIR.mkdir(exist_ok=True)
    table.to_csv(SUBMISSION_DIR / f"{name}_scores.csv", index=False)
    print(table.round(4).to_string(index=False))

    for path in write_submissions(name, test_ids, posts["test"], probs["test"]):
        print("written", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probs", nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.probs, args.weights or [1.0] * len(args.probs), args.name)
