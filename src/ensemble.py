"""Several .npz of word probabilities -> their weighted average -> answers -> the submission csv (+ `_raw`).

    python -m src.ensemble models/ensemble-probs-w256o64/{b1yzs79t,lr5hudf6,e4ob4l0j,fi9xjw14}.npz --name win256
"""
import argparse

import pandas as pd

from src.dataset import SPLITS, build_examples, load_eval, load_test_ids
from src.inference import load_probs
from src.metric import counts_per_post, precision_recall_f1
from src.project import SUBMISSION_DIR
from src.submission import answers, write_submission
from src.vocab import NoVocab


def average(members, weights, split):
    rows = [member[split] for member in members]
    assert all(len(r) == len(rows[0]) for r in rows)
    return [sum(w * r[i] for w, r in zip(weights, rows)) / sum(weights) for i in range(len(rows[0]))]


def eval_examples():
    posts = load_eval()
    return {split: build_examples(posts[split], NoVocab()) for split in SPLITS}


def ensemble_probs(members, weights, examples):
    """Weighted average per split; a split is skipped when some .npz were made on other posts
    (val changes with every build_dataset). test.csv never changes, so test must match."""
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
    examples, test_ids = eval_examples(), load_test_ids()
    probs = ensemble_probs(members, weights, examples)

    row = {}
    for split in [s for s in ("val", "sample") if s in probs]:
        pred = [answers(ex, p, policy=[]) for ex, p in zip(examples[split], probs[split])]
        c = counts_per_post([ex.locations for ex in examples[split]], pred)
        row.update(zip((f"{split}_p", f"{split}_r", f"{split}_f1"), map(float, precision_recall_f1(c.sum(0)))))
    scores = pd.DataFrame([row])
    SUBMISSION_DIR.mkdir(exist_ok=True)
    scores.to_csv(SUBMISSION_DIR / f"{name}_scores.csv", index=False)
    print(scores.round(4).to_string(index=False))

    test = list(zip(examples["test"], probs["test"]))
    # `_raw`: without the test policy, to see which submission rules the relabelled train has made redundant
    for suffix, rows in (("", [answers(ex, p) for ex, p in test]), ("_raw", [answers(ex, p, policy=[]) for ex, p in test])):
        path = SUBMISSION_DIR / f"{name}{suffix}.csv"
        write_submission(path, test_ids, rows)
        print("written", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probs", nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.probs, args.weights or [1.0] * len(args.probs), args.name)
