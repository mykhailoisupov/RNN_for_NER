"""Several .npz of word probabilities -> their weighted average -> answers -> the submission csv (+ `_raw`).

    python -m src.ensemble models/ensemble-probs-w256o64/{b1yzs79t,lr5hudf6,e4ob4l0j,fi9xjw14}.npz --name win256
"""
import argparse

import pandas as pd

from src.config import SUBMISSION_DIR
from src.dataset import build_examples, load_eval
from src.inference import SPLITS, load_probs
from src.metric import counts_per_post, precision_recall_f1
from src.submission import answers, write_submission
from src.vocab import NoVocab


def average(members, weights, split):
    rows = [member[split] for member in members]
    assert all(len(r) == len(rows[0]) for r in rows)
    return [sum(w * r[i] for w, r in zip(weights, rows)) / sum(weights) for i in range(len(rows[0]))]


def eval_examples():
    rows, test_ids = load_eval()
    return {split: build_examples(rows[split], NoVocab()) for split in SPLITS}, test_ids


def run(paths, weights, name):
    members = [load_probs(p) for p in paths]
    examples, test_ids = eval_examples()
    probs = {split: average(members, weights, split) for split in SPLITS if all(split in m for m in members)}
    for split, rows in probs.items():
        assert [len(r) for r in rows] == [len(ex.word_starts) for ex in examples[split]], f"{split}: probs do not match local data"

    row = {}
    for split in ("val", "sample"):
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
