"""How test differs from the source corpora: vocabulary, language, length and label profile."""
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.corpus import STREET


def comparison_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().replace("\\n", "\n").split())


def words(text):
    return re.findall(r"\w+(?:['’ʼ-]\w+)*", comparison_text(text))


def shingles(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def scan_corpus(frame, test, name):
    """Scan every source row; exact matches and exhaustive shared-trigram retrieval.

    Candidate rows need at least two shared unique trigrams. Similarity is then
    calculated exactly; best scores refer to this candidate set, not all strings.
    """
    test_words = [words(t) for t in test.text]
    test_ids = test.text_id.to_numpy()
    test_word_strings = [f" {' '.join(t)} " for t in test_words]
    test_shingles = [shingles(t) for t in test_words]
    inverted = defaultdict(list)
    for i, grams in enumerate(test_shingles):
        for gram in grams:
            inverted[gram].append(i)
    vocab = set(w for ts in test_words for w in ts)
    observed = Counter()
    key_functions = {
        "raw": lambda t: t,
        "NFKC_case_space": comparison_text,
        "NFKC_without_punctuation": lambda t: re.sub(r"\W+", " ", comparison_text(t)).strip(),
    }
    test_keys = {}
    empty_test_keys = Counter()
    for method, function in key_functions.items():
        lookup = defaultdict(list)
        for i, text in enumerate(test.text):
            key = function(text)
            if key.strip():
                lookup[key].append(i)
            else:
                empty_test_keys[method] += 1
        test_keys[method] = lookup
    exact_counts, exact_test_ids, empty_source_keys = Counter(), defaultdict(set), Counter()
    best, containment = {}, {}
    sizes = np.empty((len(frame), 2), dtype=np.int32)
    flags = Counter()
    emoji = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    for pos, row in enumerate(frame[["text"]].itertuples()):
        text = row.text
        norm = comparison_text(text)
        tokens = words(text)
        sizes[pos] = len(text), len(tokens)
        flags["emoji_like_pct"] += bool(emoji.search(text))
        flags["linebreak_pct"] += "\n" in text or "\\n" in text
        flags["url_pct"] += bool(re.search(r"https?://|t\.me/", text))
        flags["handle_pct"] += bool(re.search(r"@\w+", text))
        flags["uk_specific_letters_pct"] += bool(re.search("[іїєґ]", norm))
        flags["ru_specific_letters_pct"] += bool(re.search("[ыэёъ]", norm))
        observed.update(w for w in tokens if w in vocab)
        keys = {"raw": text, "NFKC_case_space": norm,
                "NFKC_without_punctuation": re.sub(r"\W+", " ", norm).strip()}
        for method, key in keys.items():
            if not key.strip():
                empty_source_keys[method] += 1
                continue
            hits = test_keys[method].get(key, ())
            if hits:
                exact_counts[method] += 1
                exact_test_ids[method].update(hits)
        grams = shingles(tokens)
        shared = Counter(i for gram in grams for i in inverted.get(gram, ()))
        for i, common in shared.items():
            if common < 2:
                continue
            jaccard = common / (len(grams) + len(test_shingles[i]) - common)
            coverage = common / len(grams)
            record = {"dataset": name, "test_id": int(test_ids[i]),
                      "source_index": int(row.Index), "jaccard": jaccard,
                      "source_shingle_coverage": coverage, "shared_trigrams": common,
                      "source_words": len(tokens), "source_text": text}
            if i not in best or jaccard > best[i]["jaccard"]:
                best[i] = record
            if len(tokens) >= 6 and coverage >= .9:
                # A long post may contain a whole training sentence despite low Jaccard.
                record = {**record, "literal_word_sequence":
                          f" {' '.join(tokens)} " in test_word_strings[i]}
                if i not in containment or common > containment[i]["shared_trigrams"]:
                    containment[i] = record
        if (pos + 1) % 1_000_000 == 0:
            print(f"{name}: scanned {pos + 1:,} / {len(frame):,} rows", flush=True)
    profile = {"rows": len(frame)}
    for col, label in enumerate(["chars", "words"]):
        for q in [50, 90, 99]:
            profile[f"{label}_p{q}"] = float(np.percentile(sizes[:, col], q))
    profile.update({k: round(100 * v / len(frame), 2) for k, v in flags.items()})
    exact = pd.DataFrame([
        {"dataset": name, "method": m, "matching_source_rows": exact_counts[m],
         "matched_test_posts": len(exact_test_ids[m]), "empty_test_keys_excluded": empty_test_keys[m],
         "empty_source_keys_excluded": empty_source_keys[m]} for m in key_functions
    ])
    token_count = sum(map(len, test_words))
    coverage = {"dataset": name, "test_word_types": len(vocab),
                "unseen_test_word_types": len(vocab - observed.keys()),
                "test_token_coverage_pct": 100 * sum(w in observed for ts in test_words for w in ts) / token_count if token_count else np.nan}
    return {"profile": profile, "exact": exact, "nearest": pd.DataFrame(best.values()),
            "containment": pd.DataFrame(containment.values()), "coverage": coverage,
            "unseen_words": sorted(vocab - observed.keys())}


def language_profile(texts):
    from lingua import Language, LanguageDetectorBuilder
    detector = LanguageDetectorBuilder.from_languages(
        Language.UKRAINIAN, Language.RUSSIAN, Language.ENGLISH,
        Language.POLISH, Language.BELARUSIAN,
    ).build()
    rows = []
    for text in texts:
        scores = detector.compute_language_confidence_values(text.replace("\\n", "\n"))
        top, second = scores[:2]
        rows.append({"language": top.language.iso_code_639_1.name.lower(),
                     "score": top.value, "margin": top.value - second.value,
                     "uncertain": top.value < .8 or top.value - second.value < .2})
    return pd.DataFrame(rows)


def feature_row(text, labels):
    labels = set(labels)
    return {"chars": len(text), "words": len(words(text)), "labels": len(labels),
            "kyiv": bool(re.search(r"київ|києв|киев", text, re.I)),
            "street": any(re.search(STREET, s, re.I) for s in labels)}


def source_features(frame):
    data = {k: np.empty(len(frame), dtype=bool if k in ["kyiv", "street"] else np.int32)
            for k in ["chars", "words", "labels", "kyiv", "street", "clean_labels"]}
    for pos, (text, spans, cleaned) in enumerate(zip(frame.text, frame.loc_markers, frame.spans)):
        row = feature_row(text, (text[a:b] for a, b in spans))
        row["clean_labels"] = len({text[a:b] for a, b in cleaned})
        for key, value in row.items(): data[key][pos] = value
    return pd.DataFrame(data, index=frame.index, copy=False)


def sentence_ranges(text):
    # Same character length, so source offsets remain valid after decoding literal \n.
    view = text.replace("\\n", " \n")
    for match in list(re.finditer(r"\b(?:вул|ул|м|г|ст|р|ім|просп|обл|т\.д)\.", view, re.I)):
        a, b = match.span()
        view = view[:a] + view[a:b].replace(".", "\uffff") + view[b:]
    start = 0
    for sep in re.finditer(r"\s*\n+\s*|(?<=[.!?…])\s+(?=[А-ЯІЇЄҐA-Z0-9«\"])", view):
        if words(text[start:sep.start()]): yield start, sep.start()
        start = sep.end()
    if words(text[start:]): yield start, len(text)


def pseudo_sentences(posts):
    """posts: text, key, spans -> one row per sentence with the labels inside it."""
    rows = []
    for row in posts.itertuples():
        for number, (a, b) in enumerate(sentence_ranges(row.text)):
            text = row.text[a:b]
            record = feature_row(text, (row.text[c:d] for c, d in row.spans if a <= c and d <= b))
            record.update(post_key=row.key, sentence=number, text=text, start=a, end=b)
            rows.append(record)
    return pd.DataFrame(rows)


def profile(frame, label_col="labels"):
    n = frame[label_col]
    return {"rows": len(frame), "zero_labels_pct": 100 * n.eq(0).mean(),
            "labels_total": int(n.sum()), "labels_per_row_mean": n.mean(),
            "labels_p50": n.median(), "labels_p90": n.quantile(.9), "labels_max": n.max(),
            "words_p50": frame.words.median(), "words_p90": frame.words.quantile(.9),
            "kyiv_rows_pct": 100 * frame.kyiv.mean(), "street_label_rows_pct": 100 * frame.street.mean()}


def strata(frame):
    labels = pd.cut(frame.labels, [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    lengths = pd.cut(frame.words, [-1, 15, 30, 60, 120, np.inf], labels=["0-15", "16-30", "31-60", "61-120", "121+"])
    return pd.Series(list(zip(labels.astype(str), lengths.astype(str), frame.kyiv, frame.street)), index=frame.index)


def distributions(frame, label_col="labels"):
    labels = pd.cut(frame[label_col], [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", "5+"])
    lengths = pd.cut(frame.words, [-1, 15, 30, 60, 120, np.inf], labels=["0-15", "16-30", "31-60", "61-120", "121+"])
    rows = []
    for name, values in {"labels": labels, "words": lengths, "kyiv": frame.kyiv, "street": frame.street}.items():
        for value, count in values.value_counts(sort=False).items():
            rows.append({"feature": name, "bin": str(value), "rows": int(count), "fraction": count / len(frame)})
    return pd.DataFrame(rows)


def distribution_distances(tables, target):
    rows = []
    reference = tables[target]
    for name, table in tables.items():
        for feature in reference.feature.unique():
            a = reference.loc[reference.feature.eq(feature)].set_index("bin").fraction
            b = table.loc[table.feature.eq(feature)].set_index("bin").fraction
            keys = a.index.union(b.index)
            distance = .5 * (a.reindex(keys, fill_value=0) - b.reindex(keys, fill_value=0)).abs().sum()
            rows.append({"target": target, "dataset": name, "feature": feature, "total_variation": distance})
    return pd.DataFrame(rows)
