"""Label form, boundaries and annotation policy of the sources, sample and regex-labelled test."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from scripts.build_dataset import DISTRICT, STATS, STREET, fix_spans
from src.label_policy import KYIV, NAME_WORD
from src.text import prepare

from .test_vs_sources import comparison_text, words

PROBES = {   # names whose mentions context_probes counts: labelled, inside an ORG only, not labelled
    "Ukraine": re.compile(r"\b(?:Україн(?:а|и|і|у|ою)|Украин(?:а|ы|е|у|ой))\b", re.I),
    "Kyiv": KYIV,
}

# Chosen on a grid (min 2/3/5/10, share 0/.3/.5/.7, ru forms, generic patterns): F1 sample 0.716 / uk val 0.842.
LEXICON_MIN, LEXICON_SHARE, LEXICON_SCAN = 2, 0.5, 300_000
EDGE = r"(?<![\w'’ʼ-])", r"(?![\w'’ʼ-])"
STREET_NAME = re.compile(rf"(?<!\w)(?i:{STREET})\s*{NAME_WORD}(?:\s+{NAME_WORD})?")   # `вул. Хрещатик`
SETTLEMENT_NAME = re.compile(rf"(?<!\w)(?i:сел(?:о|а|і|ом|е)|селищ(?:е|а|і|ем)|смт|метро|станці(?:я|ї|ю|єю))"
                             rf"\s+[«\"]?({NAME_WORD})")                                # `селі Забуччя`


def regex_labeller(sent):
    """Labelled sentences (text, spans) -> text -> LOC spans, for test, which has no labels.

    Capitalised forms labelled LEXICON_MIN+ times and in LEXICON_SHARE+ of their occurrences (counted on the first
    LEXICON_SCAN sentences), found by one regex, plus street and settlement patterns; then fix_spans, as for train.
    """
    counts = Counter(t[a:b] for t, spans in zip(sent.text, sent.spans) for a, b in spans)
    names = {n for n, c in counts.items() if c >= LEXICON_MIN and n[:1].isupper()}

    def compile_names(names):
        return re.compile(rf"{EDGE[0]}(?:{'|'.join(map(re.escape, sorted(names, key=len, reverse=True)))}){EDGE[1]}")

    seen, hit = Counter(), Counter()
    rx = compile_names(names)
    for t, spans in zip(sent.text[:LEXICON_SCAN], sent.spans[:LEXICON_SCAN]):
        for m in rx.finditer(t):
            seen[m.group()] += 1
            hit[m.group()] += any(a <= m.start() and m.end() <= b for a, b in spans)
    rx = compile_names(n for n in names if hit[n] >= LEXICON_SHARE * seen[n])   # `Шахтар`, `Тимошенко` drop out

    def label(text):
        text = prepare(text)
        spans = [m.span() for m in rx.finditer(text)]
        spans += [m.span() for m in STREET_NAME.finditer(text)] + [m.span(1) for m in SETTLEMENT_NAME.finditer(text)]
        out = fix_spans(text, spans)
        STATS.clear()
        return out[1] if out else []
    return label


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


FORM_PATTERNS = {
    "street_address": rf"(?:{STREET}|узвіз|шосе)",
    "district_region": DISTRICT,
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


def context_probes(rows, *, org_available):
    """rows yields (text, LOC spans, ORG spans)."""
    summary = Counter(rows=0, LOC_spans=0, LOC_inside_ORG=0,
                      LOC_crossing_ORG=0, labels_in_footer=0,
                      LOC_ending_inside_number=0)
    mentions = {name: Counter() for name in PROBES}
    examples = []
    example_counts = Counter()
    for text, locs, orgs in rows:
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
                in_org = any(c <= a and b <= d for c, d in orgs)
                status = "LOC" if selected else "inside_ORG_only" if in_org else "not_LOC"
                counts[status] += 1
                if status != "LOC" and example_counts[(name,status)] < 2:
                    examples.append({"kind": name + ":" + status, "LOC": match.group(), "context": text[max(0,a-50):a] + "[" + match.group() + "]" + text[b:b+50]})
                    example_counts[(name,status)] += 1
    summary["ORG_available"] = org_available
    return dict(summary), pd.DataFrame([{"name": k, **v} for k,v in mentions.items()]), pd.DataFrame(examples)
