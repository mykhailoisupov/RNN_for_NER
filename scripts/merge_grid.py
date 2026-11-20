"""Entity merging compared on the saved ensemble probabilities: argmax / Viterbi, BIO repair rules, rampart, ablations.

    python -m scripts.merge_grid models/ensemble-probs/b1yzs79t.npz ...   # writes submission/merge_grid.csv

Chosen on val F1; sample is a check. Writes merge-best.csv and merge-rampart.csv (+ _raw) for Kaggle.
"""
import argparse
import itertools

import pandas as pd

from src.config import SUBMISSION_DIR
from src.data import predicted_locations
from src.metric import bootstrap_delta, counts_per_post, precision_recall_f1
from src.ensemble import average, decode, eval_examples, load_probs, write_submission
from src.postprocess import test_policy
from src.rampart import EXTEND_SCORE, MIN_SCORE, rampart_tags

BASE = "viterbi"


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


def rampart(decoded, **kw):
    def merge(ex, probs, tags):
        offsets = [(a, b) for _, a, b in ex.words]
        return rampart_tags(ex.text, offsets, probs, tags, **kw)
    return decoded, merge


def configs():
    """name -> (decoder, merge(example, probs, tags) -> tags, guards, glue)."""
    keep = lambda ex, p, t: t
    out = {}
    for decoder, (rule, fn) in itertools.product(["viterbi", "argmax"], [("", keep), ("+strict", strict), ("+merge_adjacent", merge_adjacent)]):
        out[decoder + rule] = (decoder, fn if fn is keep else lambda ex, p, t, fn=fn: fn(t), True, True)
    for anchor, extend in itertools.product([0.3, 0.4, 0.5, 0.6], [0.1, 0.15, 0.25]):
        out[f"rampart({anchor},{extend})"] = (*rampart("argmax", min_score=anchor, extend_score=extend), True, True)
    out["rampart+rescue"] = (*rampart("argmax", min_score=MIN_SCORE, extend_score=EXTEND_SCORE, rescue=True), True, True)
    out["rampart on viterbi"] = (*rampart("viterbi", min_score=MIN_SCORE, extend_score=EXTEND_SCORE), True, True)
    return out


def bad_sequences(tags_by_post):
    """I with O or nothing before it, and `x O I` holes."""
    orphan = sum(t == 2 and (k == 0 or tags[k - 1] == 0) for tags in tags_by_post for k, t in enumerate(tags))
    holes = sum(tags[k - 2] != 0 and tags[k - 1] == 0 and tags[k] == 2 for tags in tags_by_post for k in range(2, len(tags)))
    return orphan, holes


def main(paths, name):
    members = [load_probs(p) for p in paths]
    examples, test_ids = eval_examples()
    probs = {split: average(members, [1.0] * len(members), split) for split in ("val", "sample", "test")}
    for split, rows in probs.items():
        assert [len(r) for r in rows] == [len(ex.word_starts) for ex in examples[split]], f"{split}: probs do not match local data"
    decoded = {split: {"viterbi": decode(rows), "argmax": argmax_tags(rows)} for split, rows in probs.items()}
    for split in ("val", "test"):
        print(f"argmax on {split}: orphan I {bad_sequences(decoded[split]['argmax'])[0]}, "
              f"`x O I` holes {bad_sequences(decoded[split]['argmax'])[1]}")

    def locations(split, config):
        decoder, merge, guards, glue = config
        return [predicted_locations(ex, merge(ex, p, t), guards=guards, glue=glue)
                for ex, p, t in zip(examples[split], probs[split], decoded[split][decoder])]

    def score(label, config, rows, counts):
        row = {"config": label}
        for split in ("val", "sample"):
            c = counts_per_post([ex.locations for ex in examples[split]], locations(split, config))
            row.update(zip((f"{split}_p", f"{split}_r", f"{split}_f1"), map(float, precision_recall_f1(c.sum(0)))))
            if split == "val":
                counts[label] = c
        rows.append(row)
        print(" | ".join(f"{v:.4f}" if isinstance(v, float) else v for v in row.values()), flush=True)

    grid, rows, counts = configs(), [], {}
    for label, config in grid.items():
        score(label, config, rows, counts)
    best = max(rows, key=lambda r: r["val_f1"])["config"]
    decoder, merge, _, _ = grid[best]
    for label, config in ((f"{best} -guards", (decoder, merge, False, True)), (f"{best} -glue_hyphens", (decoder, merge, True, False))):
        grid[label] = config
        score(label, config, rows, counts)

    table = pd.DataFrame(rows)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    table.to_csv(SUBMISSION_DIR / f"{name}_grid.csv", index=False)
    print(table.sort_values("val_f1", ascending=False).round(4).to_string(index=False))

    base_test = locations("test", grid[BASE])
    best_rampart = table[table.config.str.startswith("rampart")].sort_values("val_f1").config.iloc[-1]
    best = table.sort_values("val_f1").config.iloc[-1]
    for tag, label in (("best", best), ("rampart", best_rampart)):
        low, high = bootstrap_delta(counts[BASE], counts[label])
        raw = locations("test", grid[label])
        changed = sum(set(a) != set(b) for a, b in zip(raw, base_test))
        print(f"{tag}: {label}  val F1 {precision_recall_f1(counts[BASE].sum(0))[2]:.4f} -> {precision_recall_f1(counts[label].sum(0))[2]:.4f}, "
              f"delta 95% CI [{low:+.4f}, {high:+.4f}]; test posts changed: {changed} of {len(raw)}")
        policy = [test_policy(ex.text, r) for ex, r in zip(examples["test"], raw)]
        for suffix, locs in (("", policy), ("_raw", raw)):
            write_submission(SUBMISSION_DIR / f"{name}-{tag}{suffix}.csv", test_ids, locs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probs", nargs="+")
    parser.add_argument("--name", default="merge")
    args = parser.parse_args()
    main(args.probs, args.name)
