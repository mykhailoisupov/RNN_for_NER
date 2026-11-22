"""Model output -> answers in the form of the ORIGINAL text, and the submission csv.

The model reads `prepare(text)`, which has the same length as `text`. Token offsets found on the prepared text
are therefore valid in the original, and an answer is cut from the original: its apostrophes, spaces and case
reach the submission untouched.
"""
import json

import pandas as pd

from src.decoding import decode
from src.label_policy import POLICY, relabel, trim_edges
from src.metric import hard_processing
from src.project import SUBMISSION_DIR
from src.tokenizer import prepare


def split_at_newlines(text, spans):
    """A name never runs over a line break: rampart joins `Межовій\\n\\nКиївщина`, two lines of a list."""
    out = []
    for a, b in spans:
        cuts = [k for k in range(a, b) if text[k] == "\n"]
        out += [(c, d) for c, d in zip([a] + [k + 1 for k in cuts], cuts + [b]) if c < d]
    return out


SPAN_REPAIRS = [split_at_newlines, trim_edges]   # what the decoder gets wrong, on every split


def answers(example, probs, rampart=True, repairs=SPAN_REPAIRS, policy=POLICY):
    """Word probabilities of one post -> its answers: tags, spans, repairs, the label policy (not for val and sample:
    `policy=[]` scores the model alone), strings. The metric compares sets; an empty string is a false positive."""
    text = example.text
    spans = decode(text, example.tokens, probs, rampart)
    for repair in repairs:
        spans = repair(text, spans)
    if policy:
        spans = relabel(prepare(text), spans, policy)   # the text train was relabelled on: no footers, plain spaces
    return list(dict.fromkeys(s for s in (text[a:b] for a, b in spans) if hard_processing(s)))


def write_submissions(name, test_ids, test):
    """submission/<name>.csv and <name>_raw.csv from (example, word probabilities) of every test post. `_raw`:
    without the test policy, to see which submission rules the relabelled train has made redundant."""
    SUBMISSION_DIR.mkdir(exist_ok=True)
    paths = SUBMISSION_DIR / f"{name}.csv", SUBMISSION_DIR / f"{name}_raw.csv"
    for path, policy in zip(paths, (POLICY, [])):
        rows = [answers(ex, p, policy=policy) for ex, p in test]
        pd.DataFrame({"text_id": test_ids, "locations": [json.dumps(r, ensure_ascii=False) for r in rows]}
                     ).sort_values("text_id").to_csv(path, index=False)
    return paths
