"""Post -> Example -> batches: the RNN inputs of a post (vocabulary ids, char ids, word shapes)."""
import random
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.dataset import EVAL_SETS, load_eval, load_test_ids, load_train, load_val_langs
from src.rnn.config import Config
from src.rnn.vocab import MAX_WORD_CHARS, PAD, UNK, build_vocab, encode_chars, word_shape
from src.tags import IGNORE


@dataclass
class Example:
    """The model inputs of a post: numpy arrays, one row per word."""
    text: str
    tokens: list | None   # see src.dataset.Post
    words: list
    locations: list
    ids: np.ndarray       # int32 vocabulary ids
    labels: np.ndarray    # int8 O/B/I; IGNORE for the UNK of an empty post
    chars: np.ndarray     # uint8 [words, MAX_WORD_CHARS] char ids, PAD-filled
    shapes: np.ndarray    # int8 word_shape


def _char_row(word):
    row = np.zeros(MAX_WORD_CHARS, np.uint8)
    ids = encode_chars(word)
    row[:len(ids)] = ids
    return row


def build_examples(posts, vocab):
    """One id, tag, char ids and shape per word. An empty post gets one UNK, which the loss counts as O."""
    cache, examples = {}, []
    for post in posts:
        for word in post.words:
            if word not in cache:
                cache[word] = vocab.encode(word), _char_row(word), word_shape(word)
        ids, chars, shapes = zip(*([cache[w] for w in post.words] or [(UNK, np.zeros(MAX_WORD_CHARS, np.uint8), PAD)]))
        examples.append(Example(post.text, post.tokens, post.words, post.locations, np.array(ids, np.int32),
                                np.array(post.labels or [IGNORE], np.int8), np.stack(chars), np.array(shapes, np.int8)))
    return examples


class NERDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        return i, example.ids, example.labels, example.words, example.chars, example.shapes


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


def _pad(arrays, value):
    return pad_sequence([torch.from_numpy(a).long() for a in arrays], batch_first=True, padding_value=value)


def collate(batch):
    idx, ids, labels, words, chars, shapes = zip(*batch)
    width = max(1, max(int((c != PAD).sum(1).max()) for c in chars))   # the longest word of the batch
    char_ids = torch.zeros(len(chars), max(map(len, ids)), width, dtype=torch.long)
    for k, c in enumerate(chars):
        char_ids[k, :len(c)] = torch.from_numpy(c[:, :width])
    return {
        "idx": torch.tensor(idx),
        "words": list(words),
        "ids": _pad(ids, PAD),
        "labels": _pad(labels, IGNORE),
        "lengths": torch.tensor([len(x) for x in ids]),
        "chars": char_ids,
        "shapes": _pad(shapes, PAD),
    }


def loader(examples, batch_size, shuffle=False):
    """Batches of posts of similar length; shuffled for training, in a fixed order otherwise."""
    sampler = BucketSampler([len(ex.ids) for ex in examples], batch_size, shuffle)
    return DataLoader(NERDataset(examples), batch_sampler=sampler, collate_fn=collate)


class NERDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.examples = None

    def setup(self, stage=None):
        if self.examples is not None:
            return
        self.test_ids, self.val_lang = load_test_ids(), load_val_langs()
        posts = {"train": load_train(self.cfg.data.train_size, self.cfg.seed), **load_eval()}
        self.vocab = build_vocab(self.cfg.vocab, posts)
        self.examples = {name: build_examples(p, self.vocab) for name, p in posts.items()}

    def train_dataloader(self):
        return loader(self.examples["train"], self.cfg.data.batch_size, shuffle=True)

    def val_dataloader(self):
        return [loader(self.examples[name], self.cfg.data.eval_batch_size) for name in EVAL_SETS]
