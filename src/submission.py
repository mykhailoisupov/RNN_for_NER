"""Model output -> answers in the form of the ORIGINAL text, and the submission csv.

The model reads `prepare(text)`, which has the same length as `text`. Token offsets found on the prepared text
are therefore valid in the original, and an answer is cut from the original: its apostrophes, spaces and case
reach the submission untouched.
"""
import json
import re

import pandas as pd

from src.decoding import decode
from src.label_policy import (CAPITAL, HOUSE, HOUSE_REST, STREET, STREET_ANY, STREET_FORM, STREET_LEFT, in_org_name,
                              is_common_noun, settlement_start, trim)
from src.metric import hard_processing
from src.tags import tags_to_spans
from src.text import prepare


# --- span repairs: (text, spans) -> spans --------------------------------------------------------------------
def split_at_brackets(text, spans):
    """A round bracket is never inside a name: `Левицького 18а (Курчатова` -> `Левицького 18а` + `Курчатова`."""
    prepared = prepare(text)
    out = []
    for a, b in spans:
        cuts = [k for k in range(a, b) if prepared[k] in "()"]
        out += [(c, d) for c, d in zip([a] + [k + 1 for k in cuts], cuts + [b]) if c < d]
    return out


def trim_edges(text, spans):
    return [trim(text, a, b) for a, b in spans]


def drop_common_nouns(text, spans):
    return [(a, b) for a, b in spans if not is_common_noun(text[a:b])]


SPAN_REPAIRS = [split_at_brackets, trim_edges, drop_common_nouns]


# --- test policy: (text, answers) -> answers, only for the submission ----------------------------------------
def _occurrences(text, s):
    return [m.start() for m in re.finditer(re.escape(s), text)]


def extend_addresses(text, answers):
    """Test labels addresses with the street type and the house number (`вул. Солом'янська, 28`); the metric drops
    digits but keeps the comma, so `Лукʼяненка` != `Лукʼяненка, 3`. LB +0.004."""
    out = []
    for loc in answers:
        new = loc
        for k in _occurrences(text, loc):
            if any(o != loc and loc in o and any(j <= k < j + len(o) for j in _occurrences(text, o)) for o in answers):
                continue                                         # this mention is inside a longer answer
            a, b = k, k + len(loc)
            left = STREET_LEFT.search(text[max(0, a - 15):a])
            if left and not STREET_ANY.search(loc):
                a -= len(left.group())                           # `Мілютенка` -> `вул. Мілютенка`
            house = HOUSE.match(text, b)
            street = (left or STREET_ANY.search(loc) or STREET_FORM.search(loc)
                      or re.search(r"(?:" + STREET + r")\s*" + re.escape(loc), text, re.I)
                      or (house and re.search(r"[^\d,\s]", house.group()[1:])))
            if street and house and not re.search(r"\d", loc):
                b = house.end()                                  # `Лукʼяненка` -> `Лукʼяненка, 3`
            elif re.search(r",\s*\d+$", loc) and (rest := HOUSE_REST.match(text, b)):
                b = rest.end()                                   # `Відрадний, 69` -> `Відрадний, 69/1`
            new = text[a:b]
            break
        out.append(new)
    # `Бориспільська` next to `Бориспільська, 9` and never alone in the text: a cut copy of the address
    cut = {o for o in out if _occurrences(text, o) and all(
        any(p.startswith(o + ",") and text.startswith(p, k) for p in out) for k in _occurrences(text, o))}
    return list(dict.fromkeys(o for o in out if o not in cut))


def add_capital_mentions(text, answers):
    """«столиця» is a location in test (LB +0.006)."""
    return list(dict.fromkeys(answers + [text[m.start():m.end()] for m in CAPITAL.finditer(prepare(text))]))


def drop_org_names(text, answers):
    """A city/country in an organisation name is not a location, when every its mention is one (LB +0.0073)."""
    return [loc for loc in answers
            if not (k := _occurrences(text, loc)) or not all(in_org_name(text, i, i + len(loc)) for i in k)]


def extend_settlements(text, answers):
    """Every mention of an answer, a village/station with its type word (LB +0.0058); not inside a longer word
    (`Киев` in `Киев-...`)."""
    out = []
    for loc in answers:
        whole = [k for k in _occurrences(text, loc) if not re.match(r"[\w-]", text[k + len(loc):k + len(loc) + 1])]
        out += [text[settlement_start(text, k):k + len(loc)] for k in whole] or [loc]
    return list(dict.fromkeys(out))


TEST_POLICY = [extend_addresses, add_capital_mentions, drop_org_names, extend_settlements]


def answers(example, probs, rampart=True, repairs=SPAN_REPAIRS, policy=TEST_POLICY):
    """Word probabilities of one post -> its answers: tags, spans, repairs, strings, then the test policy.
    The metric compares sets, and an empty string counts as a false positive."""
    text = example.text
    spans = tags_to_spans(example.tokens, decode(text, example.tokens, probs, rampart))
    for repair in repairs:
        spans = repair(text, spans)
    out = list(dict.fromkeys(s for s in (text[a:b] for a, b in spans) if hard_processing(s)))
    for step in policy:
        out = step(text, out)
    return out


def write_submission(path, test_ids, rows):
    pd.DataFrame({
        "text_id": test_ids,
        "locations": [json.dumps(r, ensure_ascii=False) for r in rows],
    }).sort_values("text_id").to_csv(path, index=False)
