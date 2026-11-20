import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import SUBMISSION_DIR
from src.data import BucketSampler, NERDataset, WordEncoder, collate, encode, load_splits, predicted_locations
from src.metric import counts_per_post, precision_recall_f1
from src.model import viterbi
from src.postprocess import test_policy
from src.rampart import rampart_tags

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


def decode(rows):
    """Viterbi over the averaged word probabilities."""
    return [viterbi(torch.log(torch.tensor(r).clamp_min(1e-9))[None], torch.tensor([len(r)]))[0] if len(r) else [] for r in rows]


def eval_examples():
    splits, test_ids = load_splits(train_size=1, seed=0)
    return {split: encode(splits[split], WordEncoder()) for split in SPLITS}, test_ids


def write_submission(path, test_ids, locations):
    pd.DataFrame({
        "text_id": test_ids,
        "locations": [json.dumps(r, ensure_ascii=False) for r in locations],
    }).sort_values("text_id").to_csv(path, index=False)


def run(paths, weights, name, rampart=False):
    members = [load_probs(p) for p in paths]
    examples, test_ids = eval_examples()
    probs = {split: average(members, weights, split) for split in SPLITS if all(split in m for m in members)}
    for split, rows in probs.items():
        assert [len(r) for r in rows] == [len(ex.word_starts) for ex in examples[split]], f"{split}: probs do not match local data"

    def locations(split):
        tags = decode(probs[split])
        if rampart:   # rampart's span repair on the Viterbi tags: neighbours joined by a space or hyphen become one name
            tags = [rampart_tags(ex.text, [(a, b) for _, a, b in ex.words], p, t)
                    for ex, p, t in zip(examples[split], probs[split], tags)]
        return [predicted_locations(ex, t, guards=True) for ex, t in zip(examples[split], tags)]

    row = {}
    for split in ("val", "sample"):
        c = counts_per_post([ex.locations for ex in examples[split]], locations(split))
        row.update(zip((f"{split}_p", f"{split}_r", f"{split}_f1"), map(float, precision_recall_f1(c.sum(0)))))
    grid = pd.DataFrame([row])
    SUBMISSION_DIR.mkdir(exist_ok=True)
    grid.to_csv(SUBMISSION_DIR / f"{name}_grid.csv", index=False)
    print(grid.round(4).to_string(index=False))

    raw = locations("test")
    policy = [test_policy(ex.text, r) for ex, r in zip(examples["test"], raw)]
    for suffix, rows in (("", policy), ("_raw", raw)):
        path = SUBMISSION_DIR / f"{name}{suffix}.csv"
        write_submission(path, test_ids, rows)
        print("written", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probs", nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--name", required=True)
    parser.add_argument("--rampart", action="store_true", help="rampart span repair after Viterbi (LB 0.64739 -> 0.66180)")
    args = parser.parse_args()
    run(args.probs, args.weights or [1.0] * len(args.probs), args.name, args.rampart)
