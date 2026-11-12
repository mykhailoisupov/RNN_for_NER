"""Annotation-policy probes, with explicit heuristic and ambiguity counts."""
import re
from collections import Counter

import pandas as pd

from scripts.label_analysis import STREET

FORM_PATTERNS = {
    "street_address": rf"(?:{STREET}|узвіз|шосе)",
    "district_region": r"район|област|\bобл\.|\bр-н\b|щин[аіиу]$",
    "station_metro": r"метро|станці|станци|вокзал",
    "building_complex": r"\b(?:ТЦ|ТРЦ|ЖК|БЦ)\b|стадіон|стадион|аеропорт|аэропорт|університет|университет",
    "natural_feature": r"річк|\bрек[аиуы]\b|мор[ея]|озер|остр[іо]в|заток|залив|лагун|парк|ліс|\bлес\b",
    "Ukraine_surface": r"^укра[їи]н(?:а|и|ы|і|е|у|ою|ой)$",
    "Kyiv_surface": r"^(?:київ|києва|києві|києву|києвом|киев|киева|киеве|киеву|киевом)$",
}
PROBES = {
    "Ukraine": re.compile(r"\b(?:Україн(?:а|и|і|у|ою)|Украин(?:а|ы|е|у|ой))\b", re.I),
    "Kyiv": re.compile(r"\b(?:Київ|Києва|Києві|Києву|Києвом|Киев|Киева|Киеве|Киеву|Киевом)\b", re.I),
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
