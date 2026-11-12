"""Model tags -> submission strings in the form of the ORIGINAL text.

The model reads `prepare(text)`, which has the same length as `text`. Token offsets
found on the prepared text are therefore valid in the original, and the answer is cut
from the original: its apostrophes, spaces and case reach the submission untouched.
"""
import re
import string

from src.text import prepare, tags_to_spans

#  Compound names split on a hyphen are glued (Києво-Печерська, Івано-Франківськ, Нью-Делі). A plain hyphen between
#  two names is a route (`Київ-Одеса`, `Брест-Ковель`): two answers, so only these left parts glue.
COMPOUND_LEFT = re.compile(r"\w*(?:ськ|цьк|зьк|ев|єв|ов|ан)о|Нью|Сан|Санкт|Сен|Усть|Ріо|Лос|Буенос")
QUOTES = {"«": "»", "“": "”", "„": "“"}
EDGES = re.compile(r"[\W_]*(.*?)[\W_]*", re.S)
CAPITAL = re.compile(r"(?i)\bстолиц(?:я|і|ю|ею)\b")


def hard_processing(text):
    """Verbatim copy of the competition metric's normalisation (the missing `f` is theirs)."""
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\b\w\b\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def glue_hyphens(text, offsets, tags):
    tags = list(tags)
    for k in range(1, len(offsets) - 1):
        (pa, pb), (a, b), (na, _) = offsets[k - 1], offsets[k], offsets[k + 1]
        if (text[a:b] in ("-", "–") and pb == a and b == na and tags[k - 1] != "O" and tags[k + 1] != "O"
                and COMPOUND_LEFT.fullmatch(text[pa:pb])):
            tags[k] = tags[k + 1] = "I"
    return tags


def split_brackets(text, offsets, tags):
    """A round bracket is never inside a name: `Левицького 18а (Курчатова` -> `Левицького 18а` + `Курчатова`."""
    tags = list(tags)
    for k, (a, b) in enumerate(offsets):
        if tags[k] != "O" and text[a:b] in ("(", ")"):
            tags[k] = "O"
            if k + 1 < len(tags) and tags[k + 1] == "I":
                tags[k + 1] = "B"
    return tags


def to_submission(original_text, offsets, tags):
    """`offsets` come from tokenize(prepare(original_text)); the strings come from the original."""
    prepared = prepare(original_text)
    tags = split_brackets(prepared, offsets, glue_hyphens(prepared, offsets, tags))
    out = {}
    for a, b in tags_to_spans(offsets, tags):
        m = EDGES.fullmatch(original_text, a, b)        # the metric scores `Київ,` as a miss
        a, b = m.start(1), m.end(1)
        label = original_text[a:b]
        for open_, close in QUOTES.items():             # ...but `метро «Лісова»` keeps its closing quote
            if label.count(open_) > label.count(close) and original_text[b:b + 1] == close:
                label += close
        if hard_processing(label):                      # an empty string counts as a false positive
            out[label] = None                           # the metric compares sets
    return list(out)


def add_capital(original_text, locations):
    """Test policy, not the model: test labels «столиця» as a location (LB +0.006), the training sources don't.
    Only for the submission, so val/f1 keeps measuring the model against its own labels."""
    found = [original_text[m.start():m.end()] for m in CAPITAL.finditer(prepare(original_text))]
    return list(dict.fromkeys(locations + found))


