"""Files -> Example -> batches: train/val built by build_dataset, sample/test as the competition gives them."""
import random
from dataclasses import dataclass

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.config import Config
from src.project import DATA_DIR, NORMALIZED_DIR
from src.tags import I, O, spans_to_tags
from src.text import parse_spans, tokenize
from src.vocab import PAD, UNK, build_vocab

IGNORE = -100
EVAL_SETS = ("val", "sample")    # scored every epoch; the order of val_dataloader
SPLITS = (*EVAL_SETS, "test")  # what load_eval returns and inference predicts


@dataclass
class Example:
    text: str
    tokens: list
    locations: list
    ids: list
    labels: list
    word_starts: list


def _rows(texts, spans):
    return [(text, tokenize(text), [(int(a), int(b)) for a, b in s]) for text, s in zip(texts, spans)]


def load_train(train_size, seed):
    """Posts from build_dataset; train_size 0 is all of them."""
    train = pd.read_parquet(NORMALIZED_DIR / "train.parquet", columns=["text", "spans"])
    if train_size:
        train = train.sample(min(train_size, len(train)), random_state=seed)
    return _rows(train["text"], train["spans"])


def load_eval():
    """val from build_dataset, sample and test as the competition gives them; also the test ids and the source
    (uk, ru, tg) of every val post."""
    val = pd.read_parquet(NORMALIZED_DIR / "val.parquet", columns=["text", "spans", "lang"])
    sample = pd.read_csv(DATA_DIR / "labeling_sample.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    rows = {
        "val": _rows(val["text"], val["spans"]),
        "sample": _rows(sample["text"], sample["loc_markers"].map(parse_spans)),
        "test": _rows(test["text"], [[]] * len(test)),
    }
    return rows, test["text_id"].tolist(), val["lang"].tolist()


def build_examples(rows, vocab):
    cache, examples = {}, []
    for text, tokens, spans in rows:
        tags = spans_to_tags(tokens, spans)
        ids, labels, word_starts = [], [], []
        for token, tag in zip(tokens, tags):
            if token.text not in cache:
                cache[token.text] = vocab.encode(token.text) or [UNK]
            pieces = cache[token.text]
            word_starts.append(len(ids))
            ids += pieces
            labels += [tag] + [O if tag == O else I] * (len(pieces) - 1)
        if not ids:
            ids, labels = [UNK], [IGNORE]
        locations = [text[a:b] for a, b in spans]
        examples.append(Example(text, tokens, locations, ids, labels, word_starts))
    return examples


class NERDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        words = [t.text for t in example.tokens]
        return i, torch.tensor(example.ids), torch.tensor(example.labels), words


class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle, pool=100):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = batch_size * pool if shuffle else len(lengths)

    def __iter__(self):
        order = np.random.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for i in range(0, len(order), self.pool):
            chunk = order[i:i + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[j:j + self.batch_size].tolist() for j in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        full, rest = divmod(len(self.lengths), self.pool)
        return full * -(-self.pool // self.batch_size) + -(-rest // self.batch_size)


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
        rows, self.test_ids, self.val_lang = load_eval()
        rows = {"train": load_train(self.cfg.data.train_size, self.cfg.seed), **rows}
        self.vocab = build_vocab(self.cfg.vocab, rows)
        self.vectors = self.vocab.vectors
        self.examples = {name: build_examples(r, self.vocab) for name, r in rows.items()}

    def loader(self, name, shuffle=False):
        examples = self.examples[name]
        size = self.cfg.data.batch_size if shuffle else self.cfg.data.eval_batch_size
        sampler = BucketSampler([len(ex.ids) for ex in examples], size, shuffle)
        return DataLoader(NERDataset(examples), batch_sampler=sampler, collate_fn=collate)

    def train_dataloader(self):
        return self.loader("train", shuffle=True)

    def val_dataloader(self):
        return [self.loader(name) for name in EVAL_SETS]
