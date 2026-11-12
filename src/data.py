import gzip
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import List

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.config import DATA_DIR, EMBEDDINGS_DIR, EVAL_SETS, TAGS, Config
from src.postprocess import hard_processing, to_submission
from src.text import parse_spans, prepare, tokenize

NORMALIZED_DIR = DATA_DIR / "normalized"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uk.300.vec.gz"
TAG_ID = {tag: i for i, tag in enumerate(TAGS)}
PAD, UNK, IGNORE = 0, 1, -100


@dataclass
class Example:
    text: str
    words: list
    locations: list
    ids: list
    labels: list
    word_starts: list


def split_words(text):
    """Tokens of prepare(text); prepare keeps the length, so offsets also cut the original."""
    prepared = prepare(text)
    return [(prepared[a:b], a, b) for a, b in tokenize(prepared)]


def spans_to_tags(words, spans):
    tags = ["O"] * len(words)
    for a, b in spans:
        inside = [i for i, (_, start, end) in enumerate(words) if start < b and end > a]
        for k, i in enumerate(inside):
            if tags[i] == "O":
                tags[i] = "B-LOC" if k == 0 else "I-LOC"
    return tags


def comp_metric(y_true: List[List[str]], y_pred: List[List[str]]):
    assert len(y_true) == len(y_pred)
    tp, fp, fn, p,  = 0.0, 0.0, 0.0, 0.0

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        y_true_sample = set([hard_processing(s) for s in y_true_sample])
        y_pred_sample = set([hard_processing(s) for s in y_pred_sample])

        tp += len(y_true_sample & y_pred_sample)
        fp += len(y_pred_sample - y_true_sample)
        fn += len(y_true_sample - y_pred_sample)
        p += len(y_true_sample)

    if tp + fp == 0:
        if p == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = tp/(tp + fp)

    if tp + fn == 0:
        if p == 0:
            recall = 1.0
        else:
            recall = 0.0
    else:
        recall = tp/(tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 8)


def load_splits(train_size, seed):
    """train/val: posts from scripts/build_dataset.py; sample/test: raw competition files."""
    train = pd.read_parquet(NORMALIZED_DIR / "train.parquet", columns=["text", "spans"])
    if train_size:
        train = train.sample(min(train_size, len(train)), random_state=seed)
    val = pd.read_parquet(NORMALIZED_DIR / "val.parquet", columns=["text", "spans"])
    sample = pd.read_csv(DATA_DIR / "labeling_sample.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    def rows(texts, spans):
        return [(text, split_words(text), [(int(a), int(b)) for a, b in s]) for text, s in zip(texts, spans)]

    splits = {
        "train": rows(train["text"], train["spans"]),
        "val": rows(val["text"], val["spans"]),
        "sample": rows(sample["text"], sample["loc_markers"].map(parse_spans)),
        "test": rows(test["text"], [[]] * len(test)),
    }
    return splits, test["text_id"].tolist()


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


class FastTextEncoder:
    def __init__(self, splits, min_freq):
        index, matrix = fasttext_vectors()
        counts = Counter(word for _, words, _ in splits["train"] for word, _, _ in words)
        seen = {word for rows in splits.values() for _, words, _ in rows for word, _, _ in words}

        def row(word):
            # prepare() turns every apostrophe into ', which cc.uk.300 never has: it spells such words
            # without one (Соломянському, памяті), a few with ʼ (памʼять).
            for form in (word, word.replace("'", "ʼ"), word.replace("'", "")):
                for f in (form, form.lower()):
                    if f in index:
                        return index[f]
            return None

        vocab = sorted(w for w in seen if row(w) is not None or counts[w] >= min_freq)
        self.vocab = {word: i + 2 for i, word in enumerate(vocab)}
        rng = np.random.default_rng(0)
        self.vectors = np.zeros((len(vocab) + 2, matrix.shape[1]), dtype=np.float32)
        for word, i in self.vocab.items():
            r = row(word)
            self.vectors[i] = matrix[r] if r is not None else rng.normal(0, 0.1, matrix.shape[1])
        print(f"fastText vocab: {len(vocab)}, pretrained: {sum(row(w) is not None for w in vocab)}")

    def encode(self, word):
        return [self.vocab.get(word, UNK)]


class BPEmbEncoder:
    def __init__(self, vocab_size):
        from bpemb import BPEmb

        self.bpemb = BPEmb(lang="uk", vs=vocab_size, dim=300, cache_dir=EMBEDDINGS_DIR)
        self.vectors = np.vstack([np.zeros((1, 300)), self.bpemb.vectors]).astype(np.float32)

    def encode(self, word):
        return [i + 1 for i in self.bpemb.encode_ids(word)]


class WordEncoder:
    vectors = None

    def encode(self, word):
        return [UNK]


def build_encoder(cfg, splits):
    if cfg.embedding == "fasttext":
        return FastTextEncoder(splits, cfg.min_freq)
    if cfg.embedding == "bpemb":
        return BPEmbEncoder(cfg.bpe_vocab_size)
    if cfg.embedding == "flair":
        return WordEncoder()
    raise ValueError(f"unknown embedding: {cfg.embedding}")


def encode(rows, encoder):
    cache, examples = {}, []
    for text, words, spans in rows:
        tags = spans_to_tags(words, spans)
        ids, labels, word_starts = [], [], []
        for (word, _, _), tag in zip(words, tags):
            if word not in cache:
                cache[word] = encoder.encode(word) or [UNK]
            pieces = cache[word]
            inner = "O" if tag == "O" else "I-LOC"
            word_starts.append(len(ids))
            ids += pieces
            labels += [TAG_ID[tag]] + [TAG_ID[inner]] * (len(pieces) - 1)
        if not ids:
            ids, labels = [UNK], [IGNORE]
        locations = [text[a:b] for a, b in spans]
        examples.append(Example(text, words, locations, ids, labels, word_starts))
    return examples


def predicted_locations(example, piece_tags):
    tags = [TAGS[piece_tags[i]][0] for i in example.word_starts]   # B-LOC -> B, as src.text expects
    offsets = [(start, end) for _, start, end in example.words]
    return to_submission(example.text, offsets, tags)


def score(examples, predictions):
    y_true = [examples[i].locations for i, _ in predictions]
    y_pred = [predicted_locations(examples[i], tags) for i, tags in predictions]
    return comp_metric(y_true, y_pred)


class NERDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        words = [word for word, _, _ in example.words]
        return i, torch.tensor(example.ids), torch.tensor(example.labels), words


def collate(batch):
    idx, ids, labels, words = zip(*batch)
    return {
        "idx": torch.tensor(idx),
        "words": list(words),
        "ids": pad_sequence(ids, batch_first=True, padding_value=PAD),
        "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE),
        "lengths": torch.tensor([len(x) for x in ids]),
    }


class NERDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.examples = None

    def setup(self, stage=None):
        if self.examples is not None:
            return
        splits, self.test_ids = load_splits(self.cfg.train_size, self.cfg.seed)
        encoder = build_encoder(self.cfg, splits)
        self.vectors = encoder.vectors
        self.examples = {name: encode(rows, encoder) for name, rows in splits.items()}

    def oracle_f1(self):
        return {
            f"{name}/oracle_f1": score(self.examples[name], [(i, ex.labels) for i, ex in enumerate(self.examples[name])])
            for name in EVAL_SETS
        }

    def loader(self, name, shuffle=False):
        size = self.cfg.batch_size if shuffle else self.cfg.batch_size * 4
        return DataLoader(NERDataset(self.examples[name]), batch_size=size, shuffle=shuffle, collate_fn=collate)

    def train_dataloader(self):
        return self.loader("train", shuffle=True)

    def val_dataloader(self):
        return [self.loader(name) for name in EVAL_SETS]

    def predict_dataloader(self):
        return self.loader("test")
