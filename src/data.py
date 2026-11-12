"""Normalise, clean and relabel uk/ru sentences, then glue them into test-like posts.

    python -m src.data            # build data/normalized/{uk,ru}_sentences, train, val
    python -m src.data --check    # self-check only, nothing is written

Run from the project root. "step N" in comments refers to docs/dataset_preparation.md.
"""
import argparse
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.label_analysis import STREET  # same street definition as the EDA numbers
from scripts.target_analysis import PROBES

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "normalized"

# --- knobs -------------------------------------------------------------------
RU_SHARE, VAL_RU_SHARE = 0.20, 0.08       # share of ru posts in train / val (test has ~8%)
VAL_PCT, VAL_PCT_RARE = 5, 20             # % of uk sentences sent to val: ordinary / Kyiv or street
UPSAMPLE_KYIV, UPSAMPLE_STREET = 4, 4     # copies of such sentences in uk train
HOUSE_NUMBER_P = 0.5                      # share of street copies that get ", <number>" (step 22)
N_FOLDS = 5
RU_DOC_MOD = 10                           # keep full rows for every 10th ru document
SEPARATORS, SEPARATOR_P = ["\n\n", "\n", " "], [0.7, 0.2, 0.1]

# Test profile, EDA 6.7-6.8. Bins by words: 0-15, 16-30, 31-60, 61-120, 121+.
BIN_NAMES = ["0-15", "16-30", "31-60", "61-120", "121+"]
BIN_LOW, BIN_HIGH = np.array([1, 16, 31, 61, 121]), np.array([15, 30, 60, 120, 250])
TARGET_WORDS = np.array([0.220, 0.239, 0.245, 0.201, 0.094])
TARGET_EMPTY = np.array([0.4286, 0.2281, 0.2393, 0.1667, 0.1556])   # posts without names per bin
TARGET_PROFILE = {"names 0": 25.6, "names 1": 35.6, "names 2": 20.3, "names 3-4": 11.9, "names 5+": 6.5,
                  **{f"words {n}": 100 * p for n, p in zip(BIN_NAMES, TARGET_WORDS)},
                  "kyiv": 41.1, "street": 9.6, "multiline": 85.0, "median words": 36, "ru": 8.0}

STATS = Counter()  # per-step counters, printed by the build


# --- 1. normalisation, always 1:1 by characters (steps 31-39) ------------------
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
# ponytail: URLs and @handles become spaces, train has none; placeholder tokens if augmentation adds them
_URL = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+|@\w+")
_HOMOGLYPHS = str.maketrans("aceiopxyABCEHIKMOPTX", "асеіорхуАВСЕНІКМОРТХ")
_LATIN, _CYRILLIC = re.compile("[a-zA-Z]"), re.compile("[Ѐ-ӿ]")
_MIXED = re.compile("[a-zA-Z][Ѐ-ӿ]|[Ѐ-ӿ][a-zA-Z]")
_LETTERS = re.compile(r"[^\W\d_]+")
FOOTER_RE = re.compile(r"(?i)підписат(ись|ися)|подписаться|надіслати новину|запропонувати новину"
                       r"|пряма трансляція|життя.оболонь|^@\w+$")   # EDA 6.10


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
    assert len(out) == len(text)   # offsets in the result index the original text
    return out


def blank_footers(text):
    """Channel-signature lines -> spaces (steps 15, 48)."""
    # Raw line length, not stripped: normalize() turns a URL into spaces and must not change the verdict.
    return "\n".join(" " * len(line) if len(line) < 80 and FOOTER_RE.search(line.strip()) else line
                     for line in text.split("\n"))


def prepare(text):
    """What the model sees. Same length as `text`, so predicted offsets cut the original."""
    return normalize(blank_footers(text))


# --- 2. tokens and tags (steps 40-41); helpers moved from EDA cell 3 -------------
TOKEN = re.compile(r"\w+(?:'\w+)*|\n+|[^\w\s]")   # line break is a token (step 35)
SPAN = re.compile(r"\((\d+),\s*(\d+)\)")
WORD = re.compile(r"\w+(?:['-]\w+)*")
KYIV_MENTION = re.compile(r"київ|києв|киев", re.I)


def parse_spans(value):
    return [(int(a), int(b)) for a, b in SPAN.findall(str(value))]


def tokenize(text):
    return [m.span() for m in TOKEN.finditer(text)]


def clean_spans(text, spans):
    # Merges spans that touch or overlap, trims edge whitespace, drops spans without letters or digits.
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    cleaned = []
    for a, b in merged:
        chunk = text[a:b]
        a, b = a + len(chunk) - len(chunk.lstrip()), b - (len(chunk) - len(chunk.rstrip()))
        if a < b and any(c.isalnum() for c in text[a:b]):
            cleaned.append((a, b))
    return cleaned


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


def normalize_text(text):
    """Deduplication key only; NFKC changes length, so it never reaches the model."""
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def normalised_loc_signature(text, spans):
    return tuple(sorted((normalize_text(text[:a]), normalize_text(text[a:b]), normalize_text(text[b:]))
                        for a, b in spans if normalize_text(text[a:b])))


def features(text, spans):
    labels = {text[a:b] for a, b in spans}
    return {"words": len(WORD.findall(text)), "labels": len(labels),
            "kyiv": bool(KYIV_MENTION.search(text)),
            "street": any(re.search(STREET, s, re.I) for s in labels)}


# --- 3. cleaning and relabelling of one sentence, same for uk and ru -------------
TCH = re.compile(r"\s[-–—]\s[^-–—]{1,40}\s[-–—]\s*TCH\.?\s*$")
# ponytail: only unambiguous forms. `площадь России`, `вулиці Києва`, `набережная Невы` are not streets,
# so those type words are left out and a few true streets keep the old boundary. Add a name-frequency
# guard if street recall matters more than this.
TYPE_BEFORE = re.compile(r"(?<!\w)(?:вул\.|ул\.|вулиц(?:я|ю|ею)|улиц(?:а|е|у|ей)|просп\.|проспект(?:і|у|ом|а|е)?"
                         r"|провул(?:ок|ку|ка)|переул(?:ок|ке|ка)|бульвар(?:і|у|а|е|ом)?|бул\.|пл\.|площади"
                         r"|набережной)\s*$", re.I)
# Singular only: `районах` after a list of names is not part of one name.
AREA_AFTER = re.compile(r"\s+(?:район(?:і|у|ом|а|е)?|област(?:ь|і|и|ю|ью)|обл\.|р-н|кра(?:й|я|е|ю|ем)"
                        r"|республ[іи]к(?:а|и|е|у|ою|ой)|республіці)(?!\w)", re.I)
ADJECTIVE_END = re.compile(r"(?:ськ|цьк|зьк|ск|цк)\w{1,3}$")   # `Донецької`, not `района` or `Росією`


def fix_spans(text, loc, org=()):
    """Normalised sentence + raw LOC/ORG spans -> (text, spans), or None to drop the row."""
    assert all(0 <= a < b <= len(text) for a, b in [*loc, *org])                # step 1
    cut = TCH.search(text) if "TCH" in text else None                           # step 6
    if cut:
        text = text[:cut.start()]
        loc, org = ([s for s in spans if s[1] <= len(text)] for spans in (loc, org))
        STATS["TCH footer cut"] += 1
    if not text.strip():
        STATS["rows dropped: empty text"] += 1
        return None
    spans = clean_spans(text, loc)                                              # step 2
    STATS["rows changed by clean_spans"] += spans != list(loc)
    relabelled = []
    for a, b in spans:
        if b < len(text) and text[b - 1].isdigit() and text[b].isdigit():       # step 5
            while b < len(text) and text[b].isdigit():
                b += 1
            STATS["house number completed"] += 1
        before = TYPE_BEFORE.search(text, max(0, a - 35), a)                    # steps 16-17
        if before:
            a = before.start()
            STATS["street type joined"] += 1
        after = AREA_AFTER.match(text, b) if ADJECTIVE_END.search(text, a, b) else None   # step 18
        if after:
            b = after.end()
            STATS["area type joined"] += 1
        relabelled.append((a, b))
    for m in PROBES["Kyiv"].finditer(text):                                     # step 20
        if not any(c < m.end() and m.start() < d for c, d in [*relabelled, *org]):
            relabelled.append(m.span())
            STATS["Kyiv labelled"] += 1
    outside_org = [(a, b) for a, b in relabelled if not any(c <= a and b <= d for c, d in org)]   # step 21
    STATS["LOC inside ORG removed"] += len(relabelled) - len(outside_org)
    spans = clean_spans(text, outside_org)
    if spans and encode(text, spans) is None:                                   # step 4
        STATS["rows dropped: span cuts a word"] += 1
        return None
    return text, spans


# --- 4. sentence tables ------------------------------------------------------------
def _crc(value):
    return zlib.crc32(str(value).encode())


def load_sentences(lang, chunksize=500_000):
    """Clean every row; conflicts and copies are judged on the whole corpus (steps 8-12)."""
    extra = ["doc_id", "sent_id"] if lang == "ru" else []
    kept, norm_hashes, label_hashes = [], [], []
    for chunk in pd.read_csv(DATA / f"{lang}_geo_dataset.csv", chunksize=chunksize,
                             usecols=["text", "loc_markers", "org_markers", *extra]):
        STATS["rows in"] += len(chunk)
        rows = [fix_spans(prepare(t), parse_spans(loc), parse_spans(org))
                for t, loc, org in zip(chunk.text, chunk.loc_markers, chunk.org_markers)]
        chunk = chunk.loc[[r is not None for r in rows], extra].assign(
            text=[r[0] for r in rows if r], spans=[r[1] for r in rows if r])
        chunk["norm"] = chunk.text.map(normalize_text)
        chunk = chunk[chunk.norm.ne("")]
        # hash() is stable within one run, which is all the conflict search needs.
        norm_hashes.append(chunk.norm.map(hash).to_numpy(np.int64))
        label_hashes.append(np.array([hash(normalised_loc_signature(t, s))
                                      for t, s in zip(chunk.text, chunk.spans)], np.int64))
        kept.append(chunk[chunk.doc_id % RU_DOC_MOD == 0] if lang == "ru" else chunk)
    corpus = pd.DataFrame({"norm": np.concatenate(norm_hashes), "labels": np.concatenate(label_hashes)})
    versions = corpus.groupby("norm").labels.nunique()
    conflicts = versions.index[versions > 1]
    STATS["extra copies in corpus"] = len(corpus) - len(versions)
    STATS["groups with conflicting labels"] = len(conflicts)
    STATS["rows in conflicting groups"] = int(corpus.norm.isin(conflicts).sum())

    df = pd.concat(kept, ignore_index=True)
    df = df[~df.norm.map(hash).isin(conflicts)].drop_duplicates("norm").reset_index(drop=True)
    df = df.join(pd.DataFrame([features(t, s) for t, s in zip(df.text, df.spans)]))
    if lang == "ru":
        key = df.doc_id.map(_crc)
        in_val = key % 20 == 0
    else:   # a larger share of Kyiv and street sentences makes val test-like without repeats
        key = df.norm.map(_crc)
        in_val = key % 100 < np.where(df.kyiv | df.street, VAL_PCT_RARE, VAL_PCT)
    df["split"] = np.where(in_val, "val", "train")
    df["fold"] = key // 100 % N_FOLDS
    assert not df.norm.duplicated().any()   # one copy per sentence: no leak between splits or folds
    STATS["sentences kept"] = len(df)
    return df


# --- 5. posts ------------------------------------------------------------------------
STREET_START = re.compile(rf"{STREET}\s", re.I)


def add_house_number(text, spans, rng):
    """`вул. Хрещатик` -> `вул. Хрещатик, 28` inside the label (step 22)."""
    for i, (a, b) in enumerate(spans):
        if STREET_START.match(text, a, b) and not text[b - 1].isdigit() and not re.match(r",?\s*\d", text[b:b + 4]):
            number = f", {rng.integers(1, 200)}"
            shifted = [(c, d + len(number)) if j == i else (c + len(number), d + len(number)) if c >= b else (c, d)
                       for j, (c, d) in enumerate(spans)]
            return text[:b] + number + text[b:], shifted
    return text, spans


def upsample(sent, rng):
    """Repeat Kyiv and street sentences (steps 26-27); street copies get house numbers."""
    sent = sent.reset_index(drop=True)
    repeats = np.where(sent.street, UPSAMPLE_STREET, np.where(sent.kyiv, UPSAMPLE_KYIV, 1))
    out = sent.loc[sent.index.repeat(repeats)]
    numbered = out.index.duplicated() & out.street.to_numpy() & (rng.random(len(out)) < HOUSE_NUMBER_P)
    pairs = [add_house_number(t, s, rng) if n else (t, s) for t, s, n in zip(out.text, out.spans, numbered)]
    return out.assign(text=[p[0] for p in pairs], spans=[p[1] for p in pairs]).reset_index(drop=True)


def compose(sent, rng, words_p, by_doc=False):
    """Glue consecutive sentences into posts whose length follows `words_p` (step 24)."""
    if by_doc:   # ru: neighbouring sentences of one document, documents in random order
        order = pd.Series(rng.permutation(sent.doc_id.nunique()), index=sent.doc_id.unique())
        sent = sent.assign(_order=sent.doc_id.map(order)).sort_values(["_order", "sent_id"])
        doc = sent.doc_id.to_numpy()
    else:        # uk has no documents
        sent = sent.iloc[rng.permutation(len(sent))]
        doc = None
    texts, spans, words = sent.text.to_numpy(), sent.spans.to_numpy(), sent.words.to_numpy()
    n = len(sent)
    bins = rng.choice(len(words_p), size=n, p=words_p)
    targets = rng.integers(BIN_LOW[bins], BIN_HIGH[bins] + 1)
    separators = rng.choice(len(SEPARATORS), size=n, p=SEPARATOR_P)
    posts, i = [], 0
    while i < n:
        target, parts, post_spans, total, length, first = targets[len(posts)], [], [], 0, 0, i
        while True:
            if parts:
                parts.append(SEPARATORS[separators[i]])
                length += len(parts[-1])
            parts.append(texts[i])
            post_spans += [(a + length, b + length) for a, b in spans[i]]
            length += len(texts[i])
            total += words[i]
            i += 1
            # stop at the end of a document, or when one more sentence would move away from the target
            if i == n or (doc is not None and doc[i] != doc[i - 1]) or abs(total + words[i] - target) >= abs(total - target):
                break
        text = "".join(parts)
        posts.append((text, post_spans, total, len({text[a:b] for a, b in post_spans}), i - first))
    posts = pd.DataFrame(posts, columns=["text", "spans", "words", "labels", "sentences"])
    posts["bin"] = np.searchsorted(BIN_HIGH[:-1], posts.words)
    return posts


def match_test(posts, rng):
    """The only place where data is removed for having no names (steps 14, 25).

    Per length bin: keep every post with names and as many empty posts as test has.
    """
    keep = np.ones(len(posts), bool)
    empty, bins = posts.labels.to_numpy() == 0, posts.bin.to_numpy()
    for b, share in enumerate(TARGET_EMPTY):
        candidates = np.flatnonzero(empty & (bins == b))
        wanted = round(int((~empty & (bins == b)).sum()) * share / (1 - share))
        if len(candidates) > wanted:
            keep[rng.choice(candidates, len(candidates) - wanted, replace=False)] = False
    return posts[keep]


def build_posts(sent, rng, by_doc=False):
    """compose + match_test, with candidate lengths corrected so the kept posts follow TARGET_WORDS."""
    words_p = TARGET_WORDS / TARGET_WORDS.sum()
    for _ in range(3):
        candidates = compose(sent, rng, words_p, by_doc)
        posts = match_test(candidates, rng)
        got = np.bincount(posts.bin, minlength=len(words_p)) / len(posts)
        words_p = words_p * np.clip(TARGET_WORDS / np.maximum(got, 1e-3), 0.25, 4)
        words_p /= words_p.sum()
    dropped = candidates.drop(posts.index)
    empty = sent.labels.eq(0).sum()
    STATS["empty sentences in"] += int(empty)
    STATS["empty sentences removed"] += int(dropped.sentences.sum())
    STATS["empty sentences kept in empty posts"] += int(posts.sentences[posts.labels.eq(0)].sum())
    STATS["empty posts removed"] += len(dropped)
    STATS["empty posts kept"] += int(posts.labels.eq(0).sum())
    return posts[["text", "spans"]]


def profile(posts):
    f = pd.DataFrame([features(t, s) for t, s in zip(posts.text, posts.spans)])
    names = pd.cut(f.labels, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    words = np.bincount(np.searchsorted(BIN_HIGH[:-1], f.words), minlength=len(BIN_NAMES)) / len(f)
    return {**{f"names {k}": 100 * v for k, v in names.value_counts(normalize=True, sort=False).items()},
            **{f"words {n}": 100 * w for n, w in zip(BIN_NAMES, words)},
            "kyiv": 100 * f.kyiv.mean(), "street": 100 * f.street.mean(),
            "multiline": 100 * posts.text.str.contains("\n").mean(), "median words": f.words.median(),
            "ru": 100 * posts.lang.eq("ru").mean(), "posts": len(posts)}


# --- 6. build ------------------------------------------------------------------------
def _report(title):
    print(f"\n{title}")
    for key, value in STATS.items():
        print(f"  {key:38} {value:>10,}")
    STATS.clear()


def build(rebuild=False, drop_ukraine=False, ru_share=RU_SHARE, seed=0):
    OUT.mkdir(exist_ok=True)
    sentences = {}
    for lang in ("uk", "ru"):                                      # stage 1: cleaning, cached
        path = OUT / f"{lang}_sentences.parquet"
        if rebuild or not path.exists():
            load_sentences(lang).to_parquet(path)
            _report(f"{lang}: cleaning")
        sent = pd.read_parquet(path)
        sent["spans"] = [[(int(a), int(b)) for a, b in spans] for spans in sent.spans]
        if drop_ukraine:                                           # step 19, variant B
            sent["spans"] = [[s for s in spans if not PROBES["Ukraine"].fullmatch(t[s[0]:s[1]])]
                             for t, spans in zip(sent.text, sent.spans)]
            sent["labels"] = [len({t[a:b] for a, b in spans}) for t, spans in zip(sent.text, sent.spans)]
        sentences[lang] = sent

    rng = np.random.default_rng(seed)
    result = {}
    for split, share in (("train", ru_share), ("val", VAL_RU_SHARE)):   # stage 2: posts
        parts = {}
        for lang, sent in sentences.items():
            sent = sent[sent.split.eq(split)]
            folds = []
            for fold, group in sent.groupby("fold") if split == "train" else [(-1, sent)]:
                if split == "train" and lang == "uk":
                    group = upsample(group, rng)
                folds.append(build_posts(group, rng, by_doc=lang == "ru").assign(lang=lang, fold=fold))
            parts[lang] = pd.concat(folds, ignore_index=True)
            _report(f"{split} {lang}: removal of data without names")
        wanted_ru = round(len(parts["uk"]) * share / (1 - share))
        if wanted_ru > len(parts["ru"]):
            print(f"  only {len(parts['ru']):,} ru posts for {wanted_ru:,} wanted: lower --ru-share or RU_DOC_MOD")
        parts["ru"] = parts["ru"].sample(n=min(wanted_ru, len(parts["ru"])), random_state=seed)
        posts = pd.concat(parts.values()).sample(frac=1, random_state=seed).reset_index(drop=True)
        assert all(encode(t, s) is not None for t, s in zip(posts.text, posts.spans))   # spans == BIO == spans
        result[split] = posts
        posts.drop(columns=[] if split == "train" else ["fold"]).to_parquet(OUT / f"{split}.parquet")

    table = pd.DataFrame({"test": TARGET_PROFILE, **{name: profile(posts) for name, posts in result.items()}})
    print("\nProfile, % of posts (test: EDA 6.7)\n" + table.round(1).to_string())
    return result


# --- self-check ----------------------------------------------------------------------
def check():
    def labels(result):
        return [result[0][a:b] for a, b in result[1]]

    # 1:1 normalisation on real posts
    texts = pd.concat([pd.read_csv(DATA / "test.csv").text, pd.read_csv(DATA / "labeling_sample.csv").text])
    for text in texts:
        seen = prepare(text)
        assert len(seen) == len(text) and prepare(seen) == seen
        for a, b in tokenize(seen):   # a token cut from the original differs only in replaced characters
            assert all(o == s or o in "’ʼ`" or o.translate(_HOMOGLYPHS) == s for o, s in zip(text[a:b], seen[a:b]))
    assert normalize("🫶Київ, вул.\xa0Хрещатик, 28") == " Київ, вул. Хрещатик, 28"
    assert normalize("Укрaїнa Livе п’ять #Київ t.me/x") == "Україна Livе п'ять  Київ " + " " * 6
    assert blank_footers("Новина\nПідписатись на канал") == "Новина\n" + " " * 20
    assert [t for t in encode("Київ\n\nм. Львів", [(0, 4), (6, 14)])[2]] == ["B", "O", "B", "I", "I"]

    # relabelling
    t = "ДТП на улице Королева в Улан-Удэ."
    assert labels(fix_spans(t, [(13, 21), (24, 32)])) == ["улице Королева", "Улан-Удэ"]
    t = "Общая площадь России огромна."
    assert labels(fix_spans(t, [(14, 20)])) == ["России"]
    t = "Кафе на пл. Галицькій, 93 у Львові."
    assert labels(fix_spans(t, [(8, 24), (28, 34)])) == ["пл. Галицькій, 93", "Львові"]
    t = "У Шевченківському районі та в Печерському й Дарницькому районах Києва."
    assert labels(fix_spans(t, [(2, 17), (30, 41), (64, 69)])) == ["Шевченківському районі", "Печерському", "Києва"]
    t = "У супермаркеті Києва погрожували покупцю - Київ - TCH."
    assert fix_spans(t, [(15, 20), (43, 47)]) == ("У супермаркеті Києва погрожували покупцю", [(15, 20)])
    assert fix_spans("Вилетіли до Гонконга одразу.", [(12, 24)]) is None
    assert labels(fix_spans("Сьогодні в києві дощ.", [])) == ["києві"]
    assert labels(fix_spans("Клуб Динамо Київ переміг.", [(12, 16)], [(5, 16)])) == []

    # gluing keeps every label under its span, including after house numbers are inserted
    rng = np.random.default_rng(0)
    sent = pd.DataFrame({"text": [f"Сталося на вул. Тестова{'абвгд'[i % 5]} у місті." if i % 3 == 0 else f"Речення номер {i} без назв."
                                  for i in range(600)]})
    sent["spans"] = [[(11, len(t) - 9)] if i % 3 == 0 else [] for i, t in enumerate(sent.text)]
    sent = sent.join(pd.DataFrame([features(t, s) for t, s in zip(sent.text, sent.spans)]))
    posts = build_posts(upsample(sent, rng), rng)
    found = [text[a:b] for text, spans in zip(posts.text, posts.spans) for a, b in spans]
    assert found and all(re.fullmatch(r"вул\. Тестова[абвгд](, \d+)?", s) for s in found)
    assert any("," in s for s in found) and all(encode(t, s) for t, s in zip(posts.text, posts.spans))
    by_bin = posts.assign(bin=np.searchsorted(BIN_HIGH[:-1], [features(t, s)["words"] for t, s in zip(posts.text, posts.spans)]),
                          empty=[not s for s in posts.spans]).groupby("bin").empty.mean()
    assert (by_bin <= TARGET_EMPTY[by_bin.index] + 0.05).all()   # empty posts never exceed the test share
    STATS.clear()
    print("data.py: all checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="run the self-check and exit")
    parser.add_argument("--rebuild", action="store_true", help="redo the cleaning stage even if cached")
    parser.add_argument("--drop-ukraine", action="store_true", help="do not label Україна/Украина (step 19, variant B)")
    parser.add_argument("--ru-share", type=float, default=RU_SHARE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.check:
        check()
    else:
        build(args.rebuild, args.drop_ukraine, args.ru_share, args.seed)
