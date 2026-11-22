"""Ablations of the answers path on saved ensemble probabilities: without rampart, without each span repair,
without each test-policy step.

    python -m scripts.compare_decoders models/<run>/probs.npz ...   # writes submission/compare_decoders.csv

Val decides, sample is a check; every row is compared with the submission path (rampart + all repairs) by a paired
bootstrap on val (on sample when the .npz predate the current val). `test_changed` is the number of test posts a
step changes: 0 means the step is dead weight for these models. The .npz must match the current tokenizer and test.
Earlier results (O threshold, argmax, BIO repair rules, rampart's threshold grid, the policy ablation on LB):
docs/historic/postprocessing.md.
"""
import argparse

import pandas as pd

from src.config import SUBMISSION_DIR
from src.dataset import EVAL_SETS, load_eval
from src.decoding import SPAN_REPAIRS
from src.ensemble import ensemble_probs
from src.label_policy import POLICY
from src.metric import bootstrap_delta, counts_per_post, scores
from src.predict import load_probs
from src.submission import answers, gold_and_predicted

BASE = "rampart"
CONFIGS = {
    BASE: {},
    "viterbi without rampart": {"rampart": False},
    **{f"rampart -{r.__name__}": {"repairs": [x for x in SPAN_REPAIRS if x is not r]} for r in SPAN_REPAIRS},
}
#  val and sample are scored without the label policy, so its steps have no score there: only test_changed
POLICY_CONFIGS = {f"policy -{r.__name__}": {"policy": [x for x in POLICY if x is not r]} for r in dict.fromkeys(POLICY)}


def main(paths):
    members = [load_probs(p) for p in paths]
    examples = load_eval()
    probs = ensemble_probs(members, [1.0] * len(members), examples)
    scored = [s for s in EVAL_SETS if s in probs]   # val is gone when the .npz predate the current val
    test = list(zip(examples["test"], probs["test"]))
    base_test = [set(answers(ex, p)) for ex, p in test]

    def test_changed(kw):
        return sum(set(answers(ex, p, **kw)) != b for (ex, p), b in zip(test, base_test))

    rows, counts = [], {}
    for label, kw in CONFIGS.items():
        row = {"config": label}
        for split in scored:
            c = counts_per_post(*gold_and_predicted(examples[split], probs[split], **kw))
            row.update({f"{split}_{k}": v for k, v in scores(c).items()})
            if split == scored[0]:
                counts[label] = c
        low, high = bootstrap_delta(counts[BASE], counts[label])
        row.update(delta_low=low, delta_high=high, test_changed=test_changed(kw))
        rows.append(row)
        print(" | ".join(f"{v:+.4f}" if isinstance(v, float) else str(v) for v in row.values()), flush=True)
    rows += [{"config": label, "test_changed": test_changed(kw)} for label, kw in POLICY_CONFIGS.items()]
    table = pd.DataFrame(rows)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    table.to_csv(SUBMISSION_DIR / "compare_decoders.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probs", nargs="+")
    main(parser.parse_args().probs)
