import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import SUBMISSION_DIR
from src.data import BucketSampler, NERDataset, WordEncoder, collate, comp_metric, encode, load_splits, predicted_locations
from src.model import viterbi
from src.postprocess import hard_processing, test_policy

SPLITS = ("val", "sample", "test")


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def word_probs(model, examples, batch_size):
    model.eval().to(device())
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


def decode(rows, o_weight=1.0):
    """Threshold on the O class: a word is O only if o_weight * P(O) >= max(P(B), P(I)).
    o_weight < 1 gives more names (recall up), > 1 fewer (precision up); 1 is plain Viterbi."""
    scale = torch.tensor([o_weight, 1.0, 1.0])   # TAGS = O, B-LOC, I-LOC
    return [
        viterbi(torch.log((torch.tensor(r) * scale).clamp_min(1e-9))[None], torch.tensor([len(r)]))[0] if len(r) else []
        for r in rows
    ]


def argmax_tags(rows):
    """Each word on its own: `B O I` holes and I without B are possible."""
    return [r.argmax(1).tolist() for r in rows]


def strict(tags):
    """I with no B/I before it becomes O (strict IOB2)."""
    out = []
    for tag in tags:
        out.append(0 if tag == 2 and (not out or out[-1] == 0) else tag)
    return out


def merge_adjacent(tags):
    """B right after a name continues it: `B B` -> `B I`."""
    return [2 if tag == 1 and k and tags[k - 1] != 0 else tag for k, tag in enumerate(tags)]


def eval_examples():
    splits, test_ids = load_splits(train_size=1, seed=0)
    return {split: encode(splits[split], WordEncoder()) for split in SPLITS}, test_ids


def write_submission(path, test_ids, locations):
    pd.DataFrame({
        "text_id": test_ids,
        "locations": [json.dumps(r, ensure_ascii=False) for r in locations],
    }).sort_values("text_id").to_csv(path, index=False)


def post_counts(y_true, y_pred):
    """tp, fp, fn of every post, counted as comp_metric counts them."""
    rows = []
    for t, p in zip(y_true, y_pred):
        t, p = {hard_processing(s) for s in t}, {hard_processing(s) for s in p}
        rows.append((len(t & p), len(p - t), len(t - p)))
    return np.array(rows)


def f1(counts):
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def bootstrap_delta(base, new, n=1000, seed=0):
    """95% interval of F1(new) - F1(base) over posts resampled together (a paired bootstrap)."""
    idx = np.random.default_rng(seed).integers(0, len(base), (n, len(base)))
    return np.quantile(f1(new[idx].sum(1)) - f1(base[idx].sum(1)), [0.025, 0.975])


def run(paths, weights, name, o_weights=(1.0,)):
    members = [load_probs(p) for p in paths]
    examples, test_ids = eval_examples()
    probs = {split: average(members, weights, split) for split in SPLITS if all(split in m for m in members)}
    for split, rows in probs.items():
        assert [len(r) for r in rows] == [len(ex.word_starts) for ex in examples[split]], f"{split}: probs do not match local data"

    def locations(split, o_weight):
        return [predicted_locations(ex, t, guards=True) for ex, t in zip(examples[split], decode(probs[split], o_weight))]

    grid, counts = [], {}
    for o_weight in sorted({*o_weights, 1.0}):
        row = {"o_weight": o_weight}
        for split in ("val", "sample"):
            gold, pred = [ex.locations for ex in examples[split]], locations(split, o_weight)
            row.update(zip((f"{split}_p", f"{split}_r", f"{split}_f1"), comp_metric(gold, pred)))
            if split == "val":
                counts[o_weight] = post_counts(gold, pred)
        grid.append(row)
    grid = pd.DataFrame(grid)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    grid.to_csv(SUBMISSION_DIR / f"{name}_grid.csv", index=False)
    print(grid.round(4).to_string(index=False))

    best = float(grid.o_weight[grid.val_f1.idxmax()])   # chosen on val only; sample is a check
    low, high = bootstrap_delta(counts[1.0], counts[best])
    raw, base = locations("test", best), locations("test", 1.0)
    changed = sum(set(a) != set(b) for a, b in zip(raw, base))
    print(f"best o_weight {best}: val F1 {f1(counts[1.0].sum(0)):.4f} -> {f1(counts[best].sum(0)):.4f}, "
          f"delta 95% CI [{low:+.4f}, {high:+.4f}]; test posts changed: {changed} of {len(raw)}")
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
    parser.add_argument("--o-weights", type=float, nargs="+", default=[1.0],
                        help="grid for the O-class threshold, best val F1 wins; 1 is plain Viterbi")
    args = parser.parse_args()
    run(args.probs, args.weights or [1.0] * len(args.probs), args.name, args.o_weights)
