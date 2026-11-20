"""Text the model sees and how labels map onto it; shared by training, inference and the dataset build.

Every function here keeps the text length, so offsets found on prepared text cut the original.
"""
import re
import unicodedata
from typing import NamedTuple


class _CharTable(dict):
    """Lazy str.translate table: spaces, emoji and invisible chars -> ' ', apostrophes -> "'"."""

    def __missing__(self, cp):
        category = unicodedata.category(chr(cp))
        if category in ("Zs", "Zl", "Zp", "Cf") or 0xFE00 <= cp <= 0xFE0F or cp == 0x20E3:
            out = " "
        elif cp >= 0x2190 and category in ("So", "Sk", "Cn", "Co"):   # keeps № ° © below U+2190
            out = " "
        else:
            out = cp
        self[cp] = out
        return out


_CHARS = _CharTable({ord(c): "'" for c in "’ʼ`"} | {ord("#"): " "})
#  URLs and @handles become spaces, train has none; placeholder tokens if augmentation adds them
_URL = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+|@\w+")
_HOMOGLYPHS = str.maketrans("aceiopxyABCEHIKMOPTX", "асеіорхуАВСЕНІКМОРТХ")
_LATIN, _CYRILLIC = re.compile("[a-zA-Z]"), re.compile("[Ѐ-ӿ]")
_MIXED = re.compile("[a-zA-Z][Ѐ-ӿ]|[Ѐ-ӿ][a-zA-Z]")
_LETTERS = re.compile(r"[^\W\d_]+")
FOOTER_RE = re.compile(r"(?i)підписат(ись|ися)|подписаться|надіслати новину|запропонувати новину"
                       r"|пряма трансляція|життя.оболонь|^@\w+$")


def _fix_homoglyphs(match):
    word = match.group()
    if not (_LATIN.search(word) and _CYRILLIC.search(word)):
        return word
    fixed = word.translate(_HOMOGLYPHS)
    return word if _LATIN.search(fixed) else fixed   # Livе keeps its real Latin letters


def normalize(text):
    out = _URL.sub(lambda m: " " * len(m.group()), text).translate(_CHARS)
    if _MIXED.search(out):
        out = _LETTERS.sub(_fix_homoglyphs, out)
    assert len(out) == len(text)
    return out


def blank_footers(text):
    """Channel-signature lines -> spaces: names in them are not answers."""
    # Raw line length, not stripped: normalize() turns a URL into spaces and must not change the verdict.
    return "\n".join(" " * len(line) if len(line) < 80 and FOOTER_RE.search(line.strip()) else line
                     for line in text.split("\n"))


def prepare(text):
    return normalize(blank_footers(text))


TOKEN = re.compile(r"\w+(?:'\w+)*|\n+|[^\w\s]")
SPAN = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_spans(value):
    return [(int(a), int(b)) for a, b in SPAN.findall(str(value))]


class Token(NamedTuple):
    text: str
    start: int
    end: int


def tokenize(text):
    """Tokens of prepare(text); prepare keeps the length, so their offsets also cut the original."""
    return [Token(m.group(), *m.span()) for m in TOKEN.finditer(prepare(text))]
