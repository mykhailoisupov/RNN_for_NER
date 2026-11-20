"""Ablations of the answers path on saved ensemble probabilities: without rampart, without each span repair.

    python -m scripts.compare_decoders models/ensemble-probs-w256o64/*.npz   # writes submission/compare_decoders.csv

Val decides, sample is a check; every row is compared with the submission path (rampart + all repairs) by a paired
bootstrap on val. argmax, BIO repair rules and rampart's threshold grid lost to this path in 661b9dc.
"""
import argparse

import pandas as pd

from src.config import SUBMISSION_DIR
from src.ensemble import average, eval_examples, load_probs
from src.metric import bootstrap_delta, counts_per_post, precision_recall_f1
from src.submission import SPAN_REPAIRS, answers

BASE = "rampart"
CONFIGS = {
    BASE: {},
    "viterbi without rampart": {"rampart": False},
    **{f"rampart -{r.__name__}": {"repairs": [x for x in SPAN_REPAIRS if x is not r]} for r in SPAN_REPAIRS},
}


def main(paths):
    members = [load_probs(p) for p in paths]
    examples, _ = eval_examples()
    probs = {split: average(members, [1.0] * len(members), split) for split in ("val", "sample")}
    rows, counts = [], {}
    for label, kw in CONFIGS.items():
        row = {"config": label}
        for split in ("val", "sample"):
            pred = [answers(ex, p, policy=[], **kw) for ex, p in zip(examples[split], probs[split])]
            c = counts_per_post([ex.locations for ex in examples[split]], pred)
            row.update(zip((f"{split}_p", f"{split}_r", f"{split}_f1"), map(float, precision_recall_f1(c.sum(0)))))
            if split == "val":
                counts[label] = c
        low, high = bootstrap_delta(counts[BASE], counts[label])
        row.update(val_delta_low=low, val_delta_high=high)
        rows.append(row)
        print(" | ".join(f"{v:+.4f}" if isinstance(v, float) else v for v in row.values()), flush=True)
    table = pd.DataFrame(rows)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    table.to_csv(SUBMISSION_DIR / "compare_decoders.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probs", nargs="+")
    main(parser.parse_args().probs)
