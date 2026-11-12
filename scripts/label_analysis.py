"""Surface-form and boundary diagnostics; predictions are not treated as gold."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from scripts.test_analysis import comparison_text, words

STREET = r"(?:вул\.|ул\.|вулиц[яіюе]|улиц[аыуеу]|проспект(?:і|у|ом|а|е)?|просп\.|провул(?:ок|ку)|переул(?:ок|ке)|бульвар(?:і|у|а|е)?|бул\.|площ(?:а|і|у|адь|ади)|набережн\w*)"
AREA = r"(?:район\w*|област\w*|республ[іи]к\w*)"
TYPE = rf"(?:{STREET}|{AREA}|міст(?:о|а|і|ом)|город(?:а|е|ом)?|село|селі|селище|станці[яїю]|станци[яию]|метро|м\.|г\.)"
PREFIX = re.compile(rf"^{TYPE}(?:\s+|$)", re.I)
SUFFIX = re.compile(rf"\s+{TYPE}$", re.I)
STREET_INSIDE = re.compile(rf"^{STREET}(?:\s+|$)", re.I)
STREET_BEFORE = re.compile(rf"(?:^|\s){STREET}\s*$", re.I)


def core_form(label):
    value = comparison_text(label).strip(' «»"')
    value = PREFIX.sub("", value)
    value = SUFFIX.sub("", value)
    value = re.sub(r",?\s+\d+[\w/\-]*\s*$", "", value)
    return value.strip(' ,«»"')


def form_profile(counts):
    totals = Counter()
    lengths, weights = [], []
    for label, count in counts.items():
        s = label.strip()
        n = len(words(s))
        lengths.append(n); weights.append(count)
        flags = {
            "multiword_pct": n > 1, "digits_pct": any(c.isdigit() for c in s),
            "type_prefix_pct": bool(PREFIX.search(s)), "type_suffix_pct": bool(SUFFIX.search(s)),
            "street_prefix_pct": bool(STREET_INSIDE.search(s)),
            "edge_space_pct": label != s,
            "edge_punctuation_pct": bool(s) and (not s[0].isalnum() or not s[-1].isalnum()),
            "no_letters_or_digits_pct": not any(c.isalnum() for c in s),
            "hyphen_pct": "-" in s, "apostrophe_pct": any(c in s for c in "'’ʼ`"),
        }
        for k, flag in flags.items(): totals[k] += flag * count
    total = sum(counts.values())
    return {"labels": total, "unique_forms": len(counts),
            "mean_words": np.average(lengths, weights=weights) if total else np.nan,
            **{k: 100 * value / total if total else np.nan for k, value in totals.items()}}


def boundary_profile(records):
    totals = Counter()
    for text, a, b in records:
        label = text[a:b]
        inside = bool(STREET_INSIDE.search(label.strip()))
        outside = bool(STREET_BEFORE.search(text[max(0, a - 35):a]))
        totals["located_labels"] += 1
        totals["street_type_inside"] += inside
        totals["street_type_just_before"] += outside and not inside
        totals["mid_word_boundaries"] += (a > 0 and text[a-1].isalnum() and label[0].isalnum()) or (b < len(text) and text[b].isalnum() and label[-1].isalnum())
    return dict(totals)


def same_core_variants(source_counts, test_counts):
    target = {core_form(s) for s in test_counts} - {""}
    lookup = {}
    for name, counts in source_counts.items():
        grouped = {}
        for label, n in counts.items():
            core = core_form(label)
            if core in target: grouped.setdefault(core, Counter())[label] += n
        lookup[name] = grouped
    rows = []
    for label, n in test_counts.items():
        core = core_form(label)
        if not core: continue
        for name, grouped in lookup.items():
            candidates = grouped.get(core, {})
            if not candidates: continue
            variants = Counter(candidates).most_common(5)
            rows.append({"test_prediction": label, "test_posts": n, "core": core, "source": name,
                         "exact_form_seen": comparison_text(label) in {comparison_text(s) for s in candidates},
                         "source_variants": variants,
                         "different_boundary_style": any(bool(PREFIX.search(label)) != bool(PREFIX.search(s)) or bool(SUFFIX.search(label)) != bool(SUFFIX.search(s)) for s in candidates)})
    return pd.DataFrame(rows)
