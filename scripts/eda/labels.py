"""Label form, boundaries and annotation policy of the sources, sample and GPT-labelled test."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from scripts.build_dataset import PROBES, STREET

from .test_vs_sources import comparison_text, words


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


FORM_PATTERNS = {
    "street_address": rf"(?:{STREET}|узвіз|шосе)",
    "district_region": r"район|област|\bобл\.|\bр-н\b|щин[аіиу]$",
    "station_metro": r"метро|станці|станци|вокзал",
    "building_complex": r"\b(?:ТЦ|ТРЦ|ЖК|БЦ)\b|стадіон|стадион|аеропорт|аэропорт|університет|университет",
    "natural_feature": r"річк|\bрек[аиуы]\b|мор[ея]|озер|остр[іо]в|заток|залив|лагун|парк|ліс|\bлес\b",
    "Ukraine_surface": r"^укра[їи]н(?:а|и|ы|і|е|у|ою|ой)$",
    "Kyiv_surface": r"^(?:київ|києва|києві|києву|києвом|киев|киева|киеве|киеву|киевом)$",
}


def type_profile(counts):
    patterns = {k: re.compile(p, re.I) for k, p in FORM_PATTERNS.items()}
    totals = Counter()
    for label, n in counts.items():
        hits = [k for k, rx in patterns.items() if rx.search(label)]
        for k in hits: totals[k] += n
        totals["no_pattern_match"] += n * (not hits)
    denominator = sum(counts.values())
    return pd.DataFrame([{"pattern": k, "labels": totals[k], "all_labels": denominator,
                          "percent": 100 * totals[k] / denominator if denominator else 0}
                         for k in [*patterns, "no_pattern_match"]])


def context_probes(rows, *, org_available, prediction_mode=False):
    """rows yields (text, LOC spans, ORG spans, unresolved prediction spans)."""
    summary = Counter(rows=0, LOC_spans=0, LOC_inside_ORG=0,
                      LOC_crossing_ORG=0, labels_in_footer=0,
                      LOC_ending_inside_number=0)
    mentions = {name: Counter() for name in PROBES}
    examples = []
    example_counts = Counter()
    for text, locs, orgs, unresolved in rows:
        summary["rows"] += 1
        # Preserve offsets while preventing the literal 'n' in '\n' joining words.
        search_text = text.replace("\\n", " \n")
        separators = list(re.finditer(r"\\n|\n", text))
        last_start = separators[-1].end() if separators else 0
        footer_start = last_start if re.search(r"Надіслати новину|пряма трансляція|Подписаться|Підписатися|Підписатись", text[last_start:], re.I) else None
        tch = re.search(r"TCH\.?\s*$", text)
        for a, b in locs:
            summary["LOC_spans"] += 1
            inside = any(c <= a and b <= d for c, d in orgs)
            overlap = any(c < b and a < d for c, d in orgs)
            summary["LOC_inside_ORG"] += inside
            summary["LOC_crossing_ORG"] += overlap and not inside
            in_footer = footer_start is not None and a >= footer_start
            summary["labels_in_footer"] += in_footer
            cut_number = b < len(text) and text[b - 1].isdigit() and text[b].isdigit()
            summary["LOC_ending_inside_number"] += cut_number
            for kind, hit in [("LOC_inside_ORG", inside), ("footer_LOC", in_footer), ("number_cut", cut_number), ("TCH_row_LOC", bool(tch))]:
                if hit and example_counts[kind] < 3:
                    examples.append({"kind": kind, "LOC": text[a:b], "context": text[max(0,a-50):a] + "[" + text[a:b] + "]" + text[b:b+50]})
                    example_counts[kind] += 1
        for name, rx in PROBES.items():
            for match in rx.finditer(search_text):
                a, b = match.span()
                counts = mentions[name]
                counts["occurrences"] += 1
                selected = any(c <= a and b <= d for c, d in locs)
                ambiguous = any(c <= a and b <= d for c, d in unresolved)
                in_org = any(c <= a and b <= d for c, d in orgs)
                status = "LOC" if selected else "unresolved_GPT_occurrence" if ambiguous else "inside_ORG_only" if in_org else "not_LOC"
                counts[status] += 1
                if status != "LOC" and example_counts[(name,status)] < 2:
                    examples.append({"kind": name + ":" + status, "LOC": match.group(), "context": text[max(0,a-50):a] + "[" + match.group() + "]" + text[b:b+50]})
                    example_counts[(name,status)] += 1
    summary["ORG_available"] = org_available
    summary["prediction_mode"] = prediction_mode
    return dict(summary), pd.DataFrame([{"name": k, **v} for k,v in mentions.items()]), pd.DataFrame(examples)
