"""Word -> ids the model looks up: fastText rows, BPEmb pieces, or nothing (Flair reads the words themselves)."""
import gzip
import urllib.request
from collections import Counter

import numpy as np

from src.config import EMBEDDINGS_DIR

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.vec.gz"
PAD, UNK = 0, 1


def fasttext_vectors():
    source = EMBEDDINGS_DIR / "cc.uk.300.vec.gz"
    words_path = EMBEDDINGS_DIR / "cc.uk.300.words.txt"
    matrix_path = EMBEDDINGS_DIR / "cc.uk.300.npy"
    if not matrix_path.exists():
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            print("downloading", FASTTEXT_URL)
            urllib.request.urlretrieve(FASTTEXT_URL, source)
        with gzip.open(source, "rt", encoding="utf-8") as f:
            count, dim = map(int, next(f).split())
            matrix = np.lib.format.open_memmap(matrix_path.with_suffix(".tmp.npy"), "w+", np.float32, (count, dim))
            words = []
            for i, line in enumerate(f):
                word, _, values = line.rstrip("\n").partition(" ")
                words.append(word)
                matrix[i] = np.array(values.split(), dtype=np.float32)
            matrix.flush()
        words_path.write_text("\n".join(words), encoding="utf-8")
        matrix_path.with_suffix(".tmp.npy").rename(matrix_path)
    words = words_path.read_text(encoding="utf-8").split("\n")
    return {word: i for i, word in enumerate(words)}, np.load(matrix_path, mmap_mode="r")


class FastTextVocab:
    """{word: row}; vectors are zeros for a saved vocabulary, whose rows come with the checkpoint."""

    def __init__(self, words, vectors=None):
        self.words = words
        self.vectors = np.zeros((len(words) + 2, 300), dtype=np.float32) if vectors is None else vectors

    def encode(self, word):
        return [self.words.get(word, UNK)]


def fasttext_vocab(splits, min_freq):
    """Words of every split that cc.uk.300 knows or train has `min_freq` times; unknown ones get random rows."""
    index, matrix = fasttext_vectors()
    counts = Counter(t.text for _, tokens, _ in splits["train"] for t in tokens)
    seen = {t.text for rows in splits.values() for _, tokens, _ in rows for t in tokens}

    def row(word):
        # prepare() turns every apostrophe into ', which cc.uk.300 never has: it spells such words
        # without one (Соломянському, памяті), a few with ʼ (памʼять).
        for form in (word, word.replace("'", "ʼ"), word.replace("'", "")):
            for f in (form, form.lower()):
                if f in index:
                    return index[f]
        return None

    vocab = sorted(w for w in seen if row(w) is not None or counts[w] >= min_freq)
    words = {word: i + 2 for i, word in enumerate(vocab)}
    rng = np.random.default_rng(0)
    vectors = np.zeros((len(vocab) + 2, matrix.shape[1]), dtype=np.float32)
    for word, i in words.items():
        r = row(word)
        vectors[i] = matrix[r] if r is not None else rng.normal(0, 0.1, matrix.shape[1])
    print(f"fastText vocab: {len(vocab)}, pretrained: {sum(row(w) is not None for w in vocab)}")
    return FastTextVocab(words, vectors)


class BpeVocab:
    def __init__(self, vocab_size):
        from bpemb import BPEmb

        self.bpemb = BPEmb(lang="uk", vs=vocab_size, dim=300, cache_dir=EMBEDDINGS_DIR)
        self.vectors = np.vstack([np.zeros((1, 300)), self.bpemb.vectors]).astype(np.float32)

    def encode(self, word):
        return [i + 1 for i in self.bpemb.encode_ids(word)]


class NoVocab:
    vectors = None

    def encode(self, word):
        return [UNK]


def build_vocab(cfg, splits):
    if cfg.embedding == "fasttext":
        return fasttext_vocab(splits, cfg.min_freq)
    if cfg.embedding == "bpemb":
        return BpeVocab(cfg.bpe_vocab_size)
    if cfg.embedding == "flair":
        return NoVocab()
    raise ValueError(f"unknown embedding: {cfg.embedding}")


def saved_vocab(cfg, words):
    """The vocabulary of a checkpoint: its fastText words, or rebuilt from the config."""
    if cfg.embedding == "fasttext":
        return FastTextVocab(words)
    return build_vocab(cfg, None)
