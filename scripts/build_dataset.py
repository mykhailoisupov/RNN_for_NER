"""Build data/normalized/: clean and relabel uk/ru sentences, then glue them into posts shaped like test.

    python -m scripts.build_dataset    # cleaning and labels are cached; --rebuild redoes them

Run from the project root.
"""
import argparse
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.corpus import DISTRICT, STREET, crc, fix_spans, normalize_text, parse_spans
from src.label_policy import REGION_ADJ
from src.project import DATA_DIR, NORMALIZED_DIR, WANDB_ENTITY, WANDB_PROJECT
from src.tags import aligned
from src.tokenizer import prepare, tokenize

RU_SHARE, VAL_RU_SHARE = 0.20, 0.08
TG_SHARE = 0.25                           # posts from Telegram sentences (scripts.fetch_ubertext), train and val
VAL_PCT, VAL_PCT_RARE = 5, 10             # % of uk sentences sent to val: ordinary / with Kyiv or a street
HELD_PCT, HELD_MAX = 10, 50               # % of the name stems of at most 50 sentences that are held out for val
FOOTER_SHARE = 0.05                       # posts that end with a blank line, left of a channel signature (test: 31%)
SHORT_LINE_SHARE = 0.05                   # uk/tg posts that get a Telegram line of 1-3 words (test: 30% have one)
UPSAMPLE_KYIV, UPSAMPLE_STREET = 4, 4
HOUSE_NUMBER_P = 0.5
CAPS_NAME_SHARE = 0.01                    # names in capitals (`КРЕЩАТИК`); test has ~3 in 477 posts, train had none
CAPS_SENTENCE_SHARE, CAPS_WORD_SHARE = 0.005, 0.05   # posts with a sentence / one plain word in capitals; not fitted
                                                     # to the profile rows `caps line`, `caps words / 1000`
NAME_LINES_SHARE = 0.03                   # posts with names that get a list/heading of bare names: ~0.05 a post, as test
RARE_MIN, RARE_SHARE = 50, 0.03           # a form seen 50+ times and labelled in < 3% of them is not a place
N_FOLDS = 5
RU_DOC_MOD = 10                           # keep full rows for every 10th ru document, enough for 20% of train
SEPARATORS, SEPARATOR_P = ["\n\n", "\n", " "], [0.7, 0.2, 0.1]

# Target shares of test posts, from notebooks/EDA.ipynb as run on 2026-11-20 (test labelled by regex_labeller):
# names per post from 6.6, posts and empty posts per length bin from 6.8, kyiv and street from 6.7; district is
# features() over the same labels.
BIN_NAMES = ["0-15", "16-30", "31-60", "61-120", "121+"]
BIN_LOW, BIN_HIGH = np.array([1, 16, 31, 61, 121]), np.array([15, 30, 60, 120, 250])
TARGET_WORDS = np.array([0.220, 0.239, 0.245, 0.201, 0.094])
TARGET_EMPTY = np.array([0.6476, 0.3684, 0.3504, 0.1458, 0.0667])
TARGET_PROFILE = {"names 0": 35.2, "names 1": 31.4, "names 2": 17.2, "names 3-4": 10.9, "names 5+": 5.2,
                  **{f"words {n}": 100 * p for n, p in zip(BIN_NAMES, TARGET_WORDS)},
                  "kyiv": 41.1, "street": 10.7, "district": 14.5, "multiline": 85.0, "median words": 36,
                  #  what test has, not a target: FOOTER_SHARE and SHORT_LINE_SHARE are kept low on purpose
                  "ends with newline": 36.9, "short line": 23.9,
                  "caps line": 1.7, "caps words / 1000": 6.0, "ru": 8.0}

STATS = Counter()

WORD = re.compile(r"\w+(?:['-]\w+)*")
FORM = re.compile(r"(?<![\w'’ʼ-])[А-ЯІЇЄҐA-Z][\w'’ʼ]+(?![\w'’ʼ-])")   # one capitalised word
KYIV_MENTION = re.compile(r"київ|києв|киев", re.I)
CAPS_WORD = re.compile(r"(?<!\w)[A-ZА-ЯІЇЄҐЁ]{4,}(?!\w)")


def caps_line(text):
    """Has the post a line of 8+ letters with no lowercase one?"""
    return any(sum(c.isalpha() for c in line) >= 8 and line == line.upper() for line in text.split("\n"))


def short_line(text):
    """Has the post a line of 1-3 words (`Слава ЗСУ!`, `Київ:`)?"""
    return any(0 < len(WORD.findall(line)) <= 3 for line in text.split("\n"))


def stem(name):
    """`Баришівка`, `Баришівці` -> `баришів`: one key for the case forms of a name."""
    return " ".join(w[:-2] or w for w in name.lower().split())


def normalised_loc_signature(text, spans):
    return tuple(sorted((normalize_text(text[:a]), normalize_text(text[a:b]), normalize_text(text[b:]))
                        for a, b in spans if normalize_text(text[a:b])))


def features(text, spans):
    labels = {text[a:b] for a, b in spans}
    return {"words": len(WORD.findall(text)), "labels": len(labels),
            "kyiv": bool(KYIV_MENTION.search(text)),
            "street": any(re.search(STREET, s, re.I) for s in labels),
            "district": any(re.search(DISTRICT, s, re.I) for s in labels)}


# --- sentence tables ---------------------------------------------------------------------
def load_sentences(lang, chunksize=500_000):
    """Clean every row. Copies and label conflicts are judged on the whole corpus, not on the kept part."""
    extra = ["doc_id", "sent_id"] if lang == "ru" else []
    kept, norm_hashes, label_hashes = [], [], []
    labelled, seen = Counter(), Counter()
    for chunk in pd.read_csv(DATA_DIR / f"{lang}_geo_dataset.csv", chunksize=chunksize,
                             usecols=["text", "loc_markers", "org_markers", *extra]):
        STATS["rows in"] += len(chunk)
        rows = [fix_spans(prepare(t), parse_spans(loc), parse_spans(org), STATS)
                for t, loc, org in zip(chunk.text, chunk.loc_markers, chunk.org_markers)]
        for text, spans in filter(None, rows):
            forms = list(FORM.finditer(text, 1))      # inside any span counts: `Незалежності` in `майдані Незалежності`
            seen.update(m.group() for m in forms)
            labelled.update(m.group() for m in forms if any(a <= m.start() and m.end() <= b for a, b in spans))
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
    # People, clubs, organisations labelled as places now and then (`Тимошенко`, `Шахтар`, `ООН`): drop the label.
    rare = {w for w, n in labelled.items()
            if seen[w] >= RARE_MIN and n < RARE_SHARE * seen[w] and not REGION_ADJ.fullmatch(w)}
    before = df.spans.map(len).sum()
    df["spans"] = [[(a, b) for a, b in spans if t[a:b] not in rare] for t, spans in zip(df.text, df.spans)]
    STATS["rare forms"] = len(rare)
    STATS["LOC of a rare form removed"] = int(before - df.spans.map(len).sum())
    df = df.join(pd.DataFrame([features(t, s) for t, s in zip(df.text, df.spans)]))
    if lang == "ru":
        key = df.doc_id.map(crc)
        in_val = key % 20 == 0
    else:
        #  Names held out for val: every sentence with a rare name stem goes there, so a third of the val names are
        #  new to the model, as in test (a split by sentence alone leaves 10%). Frequent names (`Київ`) stay in both.
        #  ponytail: stems are counted per source, a name rare in uk and frequent in tg stays seen; count over all
        #  sources in build() if the `val names that train does not label` share it prints falls short.
        stems = [{stem(t[a:b]) for a, b in s} for t, s in zip(df.text, df.spans)]
        count = Counter(x for row in stems for x in row)
        held = {x for x, n in count.items() if n <= HELD_MAX and crc(x) % 100 < HELD_PCT}
        has_held = np.array([not held.isdisjoint(row) for row in stems])
        STATS["name stems held out for val"] = len(held)
        STATS["sentences with a held-out name"] = int(has_held.sum())
        key = df.norm.map(crc)
        #  a larger share of Kyiv and street sentences makes val test-like without repeats
        in_val = has_held | (key % 100 < np.where(df.kyiv | df.street, VAL_PCT_RARE, VAL_PCT))
    df["split"] = np.where(in_val, "val", "train")
    df["fold"] = key // 100 % N_FOLDS
    assert not df.norm.duplicated().any()   # one copy per sentence: no leak between splits or folds
    STATS["sentences kept"] = len(df)
    return df


# --- posts ------------------------------------------------------------------------------
STREET_START = re.compile(rf"{STREET}\s", re.I)


def add_house_number(text, spans, rng):
    """`вул. Хрещатик` -> `вул. Хрещатик, 28` inside the label: train has almost no addresses."""
    for i, (a, b) in enumerate(spans):
        ends_word = text[b - 1].isalpha() and (b == len(text) or not text[b].isalnum())
        if STREET_START.match(text, a, b) and ends_word and not re.match(r",?\s*\d", text[b:b + 4]):
            number = f", {rng.integers(1, 200)}"
            shifted = [(c, d + len(number)) if j == i else (c + len(number), d + len(number)) if c >= b else (c, d)
                       for j, (c, d) in enumerate(spans)]
            return text[:b] + number + text[b:], shifted
    return text, spans


def upsample(sent, rng):
    sent = sent.reset_index(drop=True)
    repeats = np.where(sent.street, UPSAMPLE_STREET, np.where(sent.kyiv, UPSAMPLE_KYIV, 1))
    out = sent.loc[sent.index.repeat(repeats)]
    numbered = out.index.duplicated() & out.street.to_numpy() & (rng.random(len(out)) < HOUSE_NUMBER_P)
    pairs = [add_house_number(t, s, rng) if n else (t, s) for t, s, n in zip(out.text, out.spans, numbered)]
    return out.assign(text=[p[0] for p in pairs], spans=[p[1] for p in pairs]).reset_index(drop=True)


def compose(sent, rng, words_p, by_doc=False):
    """Glue consecutive sentences into posts whose length in words follows `words_p`."""
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
    """The only place where data is removed for having no names.

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
    """compose + match_test; candidate lengths are corrected so the kept posts follow TARGET_WORDS."""
    words_p = TARGET_WORDS / TARGET_WORDS.sum()
    for _ in range(3):
        candidates = compose(sent, rng, words_p, by_doc)
        posts = match_test(candidates, rng)
        got = np.bincount(posts.bin, minlength=len(words_p)) / len(posts)
        words_p = words_p * np.clip(TARGET_WORDS / np.maximum(got, 1e-3), 0.25, 4)
        words_p /= words_p.sum()
    dropped = candidates.drop(posts.index)
    STATS["empty sentences in"] += int(sent.labels.eq(0).sum())
    STATS["empty sentences removed"] += int(dropped.sentences.sum())
    STATS["empty sentences kept in empty posts"] += int(posts.sentences[posts.labels.eq(0)].sum())
    STATS["empty posts removed"] += len(dropped)
    STATS["empty posts kept"] += int(posts.labels.eq(0).sum())
    return posts[["text", "spans"]]


# --- augmentation -----------------------------------------------------------------------
def _attach(text, pairs, block, block_spans, rng):
    """Put a block before or after the post, a blank line between them."""
    if rng.random() < 0.5:
        shift = len(text) + 2
        return text + "\n\n" + block, pairs + [(a + shift, b + shift) for a, b in block_spans]
    shift = len(block) + 2
    return block + "\n\n" + text, block_spans + [(a + shift, b + shift) for a, b in pairs]


#  Lines a model should leave alone. Not `Слава Україні!`: train labels `Україні` everywhere else, and with the slogan
#  as a negative the v1 models stopped finding it in 5 of the 9 test posts that end with it.
SLOGANS = ["Героям слава!", "Слава ЗСУ!", "Слава нації!"]
NAME = re.compile(r"[А-ЯІЇЄҐA-Z][^\d\n]*")
FIRST_WORD = re.compile(r"[А-ЯІЇЄҐ][а-яіїєґ']{2,}")
#  An oblique-case ending: test lists names in the nominative (`Фастів - 12`), so `Фастові`, `Києва` stay out.
#  ponytail: genitive `Одеси` passes (same ending as plural `Черкаси`); pymorphy3 if list recall does not grow.
OBLIQUE = re.compile(r"(?:і|ї|у|ю|ом|ем|єм|ою|ею|ах|ях|ова|ева|єва|ого|ої|ій|ому|ім|ни|ви|пи)$")   # `України`, `Росії`


def name_block(rng, names, weights, negatives):
    """As in test: 30% a list `Чорнобиль - 19` of 3-8 lines (10% of them no name), 70% one heading `Київщина:`
    (half of them no name: `Сьогодні:`, `Слава Україні!`). No emoji: prepare() turns them into spaces anyway."""
    is_list = rng.random() < 0.3
    sep = rng.choice([" - ", " – ", " — ", ": "])
    block, spans = "", []
    for _ in range(rng.randint(3, 8) if is_list else 1):
        block += "\n" if block else ""
        number = f"{sep}{rng.randint(1, 99)}"
        if rng.random() < (0.1 if is_list else 0.5):
            word = rng.choice(negatives)
            block += (word + number if is_list else rng.choice(SLOGANS) if rng.random() < 0.3 else word + ":")
            continue
        name = rng.choices(names, weights)[0]
        lead, tail = ("", number) if is_list else rng.choice([("", ":"), ("- ", ""), ("", "")])
        spans.append((len(block) + len(lead), len(block) + len(lead) + len(name)))
        block += lead + name + tail
    return block, spans


def add_name_lines(texts, spans, share, seed):
    """Add a block of bare names to `share` of the posts that have names. Names (nominative-looking, drawn by how
    often they are labelled) and negatives (capitalised first words outside labels: `Сьогодні`) come from these
    posts, so val gets no train names."""
    counts = Counter(text[a:b] for text, pairs in zip(texts, spans) for a, b in pairs
                     if NAME.fullmatch(text, a, b) and len(text[a:b].split()) <= 3 and not OBLIQUE.search(text[a:b]))
    names = sorted(n for n, k in counts.items() if k >= 2)
    weights = [counts[n] for n in names]
    negatives = sorted({m.group() for text, pairs in zip(texts, spans) if (m := FIRST_WORD.match(text))
                        and not any(a < m.end() for a, _ in pairs)} - set(names))
    rng = random.Random(seed)
    out_texts, out_spans = [], []
    for text, pairs in zip(texts, spans):
        pairs = [(int(a), int(b)) for a, b in pairs]
        if pairs and names and rng.random() < share:
            text, pairs = _attach(text, pairs, *name_block(rng, names, weights, negatives), rng)
        out_texts.append(text)
        out_spans.append(pairs)
    return out_texts, out_spans


def add_short_lines(texts, spans, pool, share, seed):
    """A Telegram sentence of 1-3 words with its labels (`Далі буде…`, `Феодосія.`) before or after `share` of the
    posts; `pool` holds (text, spans) of this split only."""
    rng = random.Random(seed)
    out_texts, out_spans = [], []
    for text, pairs in zip(texts, spans):
        if pool and rng.random() < share:
            text, pairs = _attach(text, pairs, *rng.choice(pool), rng)
        out_texts.append(text)
        out_spans.append(pairs)
    return out_texts, out_spans


def add_blank_footer(texts, share, seed):
    """`text\\n\\n      `: what blank_footers leaves of the channel signature under a test post. Labels do not move."""
    rng = random.Random(seed)
    return [text + "\n\n" + " " * rng.randint(15, 55) if rng.random() < share else text for text in texts]


LINE = re.compile(r"[^\n]+")
PLAIN_WORD = re.compile(r"[^\W\d_]{4,}")


def add_caps(texts, spans, name_share, sentence_share, word_share, seed):
    """Capitals as Telegram writes them, labels untouched: a name (`КРЕЩАТИК`, share per name), a sentence
    (`ЦЕ ПІДПАЛ - КУПУЄМ ТРАЛ!`, per post), one plain word outside the names (`це ДУЖЕ важливо`, per post), so that
    capitals alone do not make a name."""
    rng = random.Random(seed)
    out = []
    for text, pairs in zip(texts, spans):
        cuts = [span for span in pairs if rng.random() < name_share]
        lines = [m.span() for m in LINE.finditer(text)]   # posts glue sentences mostly by \n
        if lines and rng.random() < sentence_share:
            cuts.append(rng.choice(lines))
        if rng.random() < word_share:
            words = [m.span() for m in PLAIN_WORD.finditer(text) if not m.group().isupper()
                     and not any(a < m.end() and m.start() < b for a, b in pairs)]
            cuts += rng.sample(words, min(1, len(words)))
        for span in cuts:
            text = _upper(text, span)
        out.append(text)
    return out


def _upper(text, span):
    a, b = span
    upper = text[a:b].upper()
    return text[:a] + upper + text[b:] if len(upper) == b - a else text   # `ß` -> `SS` would shift the labels


def profile(posts):
    f = pd.DataFrame([features(t, s) for t, s in zip(posts.text, posts.spans)])
    names = pd.cut(f.labels, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    words = np.bincount(np.searchsorted(BIN_HIGH[:-1], f.words), minlength=len(BIN_NAMES)) / len(f)
    return {**{f"names {k}": 100 * v for k, v in names.value_counts(normalize=True, sort=False).items()},
            **{f"words {n}": 100 * w for n, w in zip(BIN_NAMES, words)},
            "kyiv": 100 * f.kyiv.mean(), "street": 100 * f.street.mean(), "district": 100 * f.district.mean(),
            "multiline": 100 * posts.text.str.contains("\n").mean(), "median words": f.words.median(),
            "ends with newline": 100 * posts.text.str.contains(r"\n *$").mean(),
            "short line": 100 * posts.text.map(short_line).mean(),
            "caps line": 100 * posts.text.map(caps_line).mean(),
            "caps words / 1000": 1000 * posts.text.str.count(CAPS_WORD.pattern).sum() / f.words.sum(),
            "ru": 100 * posts.lang.eq("ru").mean(), "tg": 100 * posts.lang.eq("tg").mean(), "posts": len(posts)}


# --- build ------------------------------------------------------------------------------
def _report(title):
    print(f"\n{title}")
    for key, value in STATS.items():
        print(f"  {key:38} {value:>10,}")
    STATS.clear()


def build(rebuild=False, ru_share=RU_SHARE, tg_share=TG_SHARE, seed=0):
    NORMALIZED_DIR.mkdir(exist_ok=True)
    sentences = {}
    for lang in ("uk", "ru", "tg"):   # stage 1: cleaning and labels, cached because ru takes minutes
        path = NORMALIZED_DIR / f"{lang}_sentences.parquet"
        if lang == "tg" and not (DATA_DIR / "tg_geo_dataset.csv").exists() and not path.exists():
            print("no data/tg_geo_dataset.csv (python -m scripts.fetch_ubertext): posts without Telegram sentences")
            continue
        if rebuild or not path.exists():
            load_sentences(lang).to_parquet(path)
            _report(f"{lang}: cleaning")
        sent = pd.read_parquet(path)
        sent["spans"] = [[(int(a), int(b)) for a, b in spans] for spans in sent.spans]
        sentences[lang] = sent

    rng = np.random.default_rng(seed)
    result = {}
    for split, ru in (("train", ru_share), ("val", VAL_RU_SHARE)):   # stage 2: posts
        parts = {}
        for lang, sent in sentences.items():
            sent = sent[sent.split.eq(split)]
            folds = []
            # per fold, so one post never mixes sentences of two folds
            for fold, group in sent.groupby("fold") if split == "train" else [(-1, sent)]:
                if split == "train" and lang == "uk":
                    group = upsample(group, rng)
                folds.append(build_posts(group, rng, by_doc=lang == "ru").assign(lang=lang, fold=fold))
            posts = pd.concat(folds, ignore_index=True)
            # Test has lists and headings of bare names (`Чорнобиль - 19`); names come from this split only.
            texts, spans = add_name_lines(posts.text.tolist(), posts.spans.tolist(), NAME_LINES_SHARE, seed)
            if lang != "ru" and "tg" in sentences:   # a Ukrainian line in a Russian post would be made up
                tg = sentences["tg"]
                short = tg[tg.split.eq(split) & tg.words.between(1, 3)]
                texts, spans = add_short_lines(texts, spans, list(zip(short.text, short.spans)), SHORT_LINE_SHARE, seed + 1)
            texts = add_caps(texts, spans, CAPS_NAME_SHARE, CAPS_SENTENCE_SHARE, CAPS_WORD_SHARE, seed)
            texts = add_blank_footer(texts, FOOTER_SHARE, seed + 2)
            parts[lang] = posts.assign(text=texts, spans=spans)
            _report(f"{split} {lang}: removal of data without names")
        shares = {lang: share for lang, share in (("ru", ru), ("tg", tg_share)) if lang in parts}
        for lang, share in shares.items():   # shares of all posts; uk is the rest
            wanted = round(len(parts["uk"]) * share / (1 - sum(shares.values())))
            if wanted > len(parts[lang]):
                print(f"  only {len(parts[lang]):,} {lang} posts for {wanted:,} wanted: lower --{lang}-share")
            parts[lang] = parts[lang].sample(n=min(wanted, len(parts[lang])), random_state=seed)
        posts = pd.concat(parts.values()).sample(frac=1, random_state=seed).reset_index(drop=True)
        assert all(aligned(tokenize(t), s) for t, s in zip(posts.text, posts.spans))
        result[split] = posts
        posts.drop(columns=[] if split == "train" else ["fold"]).to_parquet(NORMALIZED_DIR / f"{split}.parquet")

    table = pd.DataFrame({"test": TARGET_PROFILE, **{name: profile(posts) for name, posts in result.items()}})
    print("\nProfile, % of posts\n" + table.round(1).to_string())
    names = {split: Counter(t[a:b] for t, spans in zip(posts.text, posts.spans) for a, b in spans)
             for split, posts in result.items()}
    unseen = sum(n for name, n in names["val"].items() if name not in names["train"])
    print(f"\nval names that train does not label: {100 * unseen / sum(names['val'].values()):.1f}% (test: 23-28%)")
    return result


def upload():
    """Log train/val parquet as W&B artifact `normalized`; colab.setup() downloads the latest one."""
    import wandb

    with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type="dataset") as run:
        artifact = wandb.Artifact("normalized", type="dataset")
        for split in ("train", "val"):
            artifact.add_file(str(NORMALIZED_DIR / f"{split}.parquet"))
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="redo cleaning and labels even if cached")
    parser.add_argument("--ru-share", type=float, default=RU_SHARE)
    parser.add_argument("--tg-share", type=float, default=TG_SHARE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upload", action="store_true", help="log train/val parquet to W&B for Colab")
    args = parser.parse_args()
    build(args.rebuild, args.ru_share, args.tg_share, args.seed)
    if args.upload:
        upload()
