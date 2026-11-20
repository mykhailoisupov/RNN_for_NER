"""Text -> tokens with offsets, as the model reads it.

prepare() blanks Telegram noise (emoji, links, channel signatures) one character for one, so the text keeps its
length: token offsets cut the same span of the original, and answers keep its spelling.
Train, val, test and the label policy all read text through prepare().
"""
import re
import unicodedata
from typing import NamedTuple


# --- Cleaning: every replacement is one character for one -----------------------------------------------------

class _CharMap(dict):
    """str.translate table, filled on first sight of a character: noise -> ' ', anything else stays."""

    def __missing__(self, cp):
        category = unicodedata.category(chr(cp))
        # spaces, invisible format chars, emoji modifiers; U+3164 is a blank that Unicode counts as a letter
        if category in ("Zs", "Zl", "Zp", "Cf") or 0xFE00 <= cp <= 0xFE0F or cp in (0x20E3, 0x3164):
            out = " "
        elif cp >= 0x2190 and category in ("So", "Sk", "Cn", "Co"):   # arrows, emoji; № ° © lie below and stay
            out = " "
        else:
            out = cp
        self[cp] = out
        return out


# test writes ’ and ʼ, train ': one apostrophe for all; `#` of a hashtag is not part of a name
_CHAR_MAP = _CharMap({ord(c): "'" for c in "’ʼ`"} | {ord("#"): " "})
_URL = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+|@\w+")   # links and @handles: train has none
# Latin letters that look Cyrillic: `Kиїв` typed with a Latin K
_HOMOGLYPHS = str.maketrans("aceiopxyABCEHIKMOPTX", "асеіорхуАВСЕНІКМОРТХ")
_LATIN, _CYRILLIC = re.compile("[a-zA-Z]"), re.compile("[Ѐ-ӿ]")
_MIXED = re.compile("[a-zA-Z][Ѐ-ӿ]|[Ѐ-ӿ][a-zA-Z]")   # Latin next to Cyrillic: the text has a mixed word
_LETTERS = re.compile(r"[^\W\d_]+")


def _fix_homoglyphs(match):
    """A word of both scripts -> Cyrillic, unless real Latin letters remain (`Livе` stays)."""
    word = match.group()
    if not (_LATIN.search(word) and _CYRILLIC.search(word)):
        return word
    fixed = word.translate(_HOMOGLYPHS)
    return word if _LATIN.search(fixed) else fixed


def normalize(text):
    out = _URL.sub(lambda m: " " * len(m.group()), text).translate(_CHAR_MAP)
    if _MIXED.search(out):              # the word-by-word pass only for the few texts that need it
        out = _LETTERS.sub(_fix_homoglyphs, out)
    assert len(out) == len(text)        # the offsets depend on it
    return out


# --- Channel signatures: the names in them are not answers ------------------------------------------------------

_FOOTER = re.compile(r"(?i)підписат(ись|ися)|подписаться|надіслати новину|запропонувати новину"
                     r"|пряма трансляція|життя.оболонь|^@\w+$")


def blank_footers(text):
    """A short line that asks to subscribe or send news -> spaces of its length."""
    # raw length, not stripped: a URL that normalize() blanks later must not change the verdict
    return "\n".join(" " * len(line) if len(line) < 80 and _FOOTER.search(line.strip()) else line
                     for line in text.split("\n"))


def prepare(text):
    """The text the model reads; same length as `text`."""
    return normalize(blank_footers(text))


# --- Tokens ---------------------------------------------------------------------------------------------------

TOKEN = re.compile(r"\w+(?:'\w+)*|\n+|[^\w\s]")   # a word (`пам'ять` whole), a run of line breaks, one other char


class Token(NamedTuple):
    text: str     # as prepared: `пам'ять`, not `памʼять`
    start: int    # offsets hold in the original text as well
    end: int


def tokenize(text):
    return [Token(m.group(), *m.span()) for m in TOKEN.finditer(prepare(text))]
