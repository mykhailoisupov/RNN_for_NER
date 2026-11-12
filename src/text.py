"""Text the model sees and how labels map onto it; shared by training, inference and the dataset build.

Every function here keeps the text length, so offsets found on prepared text cut the original.
"""
import re
import unicodedata


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


def tokenize(text):
    return [m.span() for m in TOKEN.finditer(text)]


def spans_to_bio(offsets, spans):
    tags, misaligned = ["O"] * len(offsets), 0
    for a, b in spans:
        hit = [i for i, (s, e) in enumerate(offsets) if s < b and a < e]
        if not hit or offsets[hit[0]][0] != a or offsets[hit[-1]][1] != b:
            misaligned += 1
        if hit:
            tags[hit[0]] = "B"
            for i in hit[1:]:
                tags[i] = "I"
    return tags, misaligned


def tags_to_spans(offsets, tags):
    spans, i = [], 0
    while i < len(tags):
        if tags[i] == "O":
            i += 1
            continue
        j = i + 1
        while j < len(tags) and tags[j] == "I":
            j += 1
        spans.append((offsets[i][0], offsets[j - 1][1]))
        i = j
    return spans


def encode(text, spans):
    """-> (tokens, offsets, BIO tags), or None when the spans do not survive spans -> BIO -> spans."""
    offsets = tokenize(text)
    tags, _ = spans_to_bio(offsets, spans)
    if tags_to_spans(offsets, tags) != sorted(spans):
        return None
    return [text[a:b] for a, b in offsets], offsets, tags
