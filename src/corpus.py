"""Sentences of the source corpora (uk/ru/tg geo datasets): their dedup key, their labels as test labels them, and
the label features the test profile is measured by. Shared by scripts.build_dataset and notebooks/eda."""
import re
import unicodedata
import zlib
from collections import Counter

from src.label_policy import CAPITAL, KYIV, LOWERABLE, TRAIN_POLICY, clean_spans, relabel
from src.tags import aligned
from src.text import tokenize

STREET = r"(?:вул\.|ул\.|вулиц[яіюе]|улиц[аыуеу]|проспект(?:і|у|ом|а|е)?|просп\.|провул(?:ок|ку)|переул(?:ок|ке)|бульвар(?:і|у|а|е)?|бул\.|площ(?:а|і|у|адь|ади)|набережн\w*)"
DISTRICT = r"район|област|\bобл\.|\bр-н\b|щин[аіиу]$"   # `Дарницькому районі`, `Київщина`
TCH = re.compile(r"\s[-–—]\s[^-–—]{1,40}\s[-–—]\s*TCH\.?\s*$")


def crc(value):
    """A hash stable across runs, unlike hash(): splits and folds by it."""
    return zlib.crc32(str(value).encode())


def normalize_text(text):
    """Deduplication key only; NFKC changes length, so it never reaches the model."""
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def fix_spans(text, loc, org=(), stats=None):
    """Prepared sentence + raw LOC/ORG spans -> (text, spans) labelled like test, or None to drop the row.
    `stats` counts what every step changed."""
    stats = Counter() if stats is None else stats
    assert all(0 <= a < b <= len(text) for a, b in [*loc, *org])
    cut = TCH.search(text) if "TCH" in text else None   # `... - Київ - TCH.` is a site tag, not text
    if cut:
        text = text[:cut.start()]
        loc, org = ([s for s in spans if s[1] <= len(text)] for spans in (loc, org))
        stats["TCH footer cut"] += 1
    if not text.strip():
        stats["rows dropped: empty text"] += 1
        return None
    spans = clean_spans(text, loc)
    stats["rows changed by clean_spans"] += spans != list(loc)
    relabelled = relabel(text, spans, TRAIN_POLICY, stats)
    #  A label inside an ORG span is dropped (`Динамо Київ`); a Kyiv or «столиця» mention the policy adds, if it touches
    #  one at all: ORG offsets are shifted now and then (`Укрістгаз (Ки`).
    added = [s for s in relabelled if s not in spans and (KYIV.fullmatch(text, *s) or CAPITAL.fullmatch(text, *s))]
    spans = [(a, b) for a, b in relabelled
             if not any(c <= a and b <= d or (a, b) in added and c < b and a < d for c, d in org)]
    stats["LOC inside ORG removed"] += len(relabelled) - len(spans)
    if crc(text) % 5 == 0:      # `москви`, `рф`: test writes them in lowercase, train never does; 20% of sentences
        for a, b in spans:
            if LOWERABLE.fullmatch(text, a, b):
                text = text[:a] + text[a:b].lower() + text[b:]
                stats["LOC lowercased"] += 1
    if spans and not aligned(tokenize(text), spans):   # shifted offsets or glued words, not fixable
        stats["rows dropped: span cuts a word"] += 1
        return None
    return text, spans
