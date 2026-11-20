import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import SUBMISSION_DIR
from src.dataset import BucketSampler, NERDataset, build_examples, collate, load_eval
from src.metric import counts_per_post, precision_recall_f1
from src.submission import answers, write_submission
from src.vocab import NoVocab

SPLITS = ("val", "sample", "test")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def word_probs(model, examples, batch_size, on=None):
    model.eval().to(on or device())
    sampler = BucketSampler([len(ex.ids) for ex in examples], batch_size, shuffle=False)
    probs = [None] * len(examples)
    for batch in DataLoader(NERDataset(examples), batch_sampler=sampler, collate_fn=collate):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        for row, i in zip(model(batch).float().softmax(-1).cpu().numpy(), batch["idx"].tolist()):
            probs[i] = row[examples[i].word_starts]
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
