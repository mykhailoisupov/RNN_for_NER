"""Several .npz of word probabilities -> their weighted average -> answers -> the submission csv (+ `_raw`).

    python -m src.ensemble models/<run>/probs.npz models/<run>/probs.npz ... --name <name>
"""
import argparse

import pandas as pd

from src.config import SUBMISSION_DIR
from src.dataset import SPLITS, load_eval, load_test_ids
from src.metric import counts_per_post, scores
from src.predict import load_probs
from src.submission import answers, write_submissions


def average(members, weights, split):
    return [sum(w * p for w, p in zip(weights, posts)) / sum(weights) for posts in zip(*(m[split] for m in members))]


def ensemble_probs(members, weights, examples):
    """Weighted average per split; a split is skipped when some .npz were made on other posts or by another tokenizer
    (val changes with every build_dataset). test must match: after a tokenizer change, remake the .npz with
    src.predict."""
    probs = {}
    for split in SPLITS:
        want = [len(ex.words) for ex in examples[split]]
        if all(split in m and [len(r) for r in m[split]] == want for m in members):
            probs[split] = average(members, weights, split)
        else:
            print(f"{split}: skipped, .npz made on other data")
    assert "test" in probs, "test probs do not match test.csv"
    return probs


def run(paths, weights, name):
    members = [load_probs(p) for p in paths]
    examples, test_ids = load_eval(), load_test_ids()
    probs = ensemble_probs(members, weights, examples)

    row = {}
    for split in [s for s in ("val", "sample") if s in probs]:
        pred = [answers(ex, p, policy=[]) for ex, p in zip(examples[split], probs[split])]
        c = counts_per_post([ex.locations for ex in examples[split]], pred)
        row.update({f"{split}_{k}": v for k, v in scores(c).items()})
    table = pd.DataFrame([row])
    SUBMISSION_DIR.mkdir(exist_ok=True)
    table.to_csv(SUBMISSION_DIR / f"{name}_scores.csv", index=False)
    print(table.round(4).to_string(index=False))

    for path in write_submissions(name, test_ids, list(zip(examples["test"], probs["test"]))):
        print("written", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probs", nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.probs, args.weights or [1.0] * len(args.probs), args.name)
