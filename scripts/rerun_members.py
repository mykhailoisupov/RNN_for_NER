"""Re-run the four ensemble members on val/sample/test with overlapping windows (256 words, overlap 64).

    python -m scripts.rerun_members [--members b1yzs79t ...]
    python -m src.ensemble models/ensemble-probs-w256o64/{b1yzs79t,lr5hudf6,e4ob4l0j,fi9xjw14}.npz --rampart --name win256

A post longer than the window is cut into windows (step 192, the last one ends at the end of the post) and each
word's probabilities are averaged over the windows that cover it; shorter posts are as whole. Writes
models/ensemble-probs-w256o64/{run}.npz in the format src.ensemble reads. Each member first reproduces its saved
whole-post probabilities (models/ensemble-probs/{run}.npz), or the model or its vocabulary loaded wrong.
LB: 0.66180 -> 0.66277 (one test post changed; 128-word windows gave 0.66084).
"""
import argparse
import json

import numpy as np
import torch

from src.config import MODELS_DIR, Config
from src.data import UNK, Example, FastTextEncoder, WordEncoder, encode, load_splits
from src.ensemble import SPLITS, load_probs, save_probs, word_probs
from src.model import Tagger

WINDOW, OVERLAP = 256, 64   # a bit under the longest train post (352 words); overlap as rampart's NER_TOKEN_OVERLAP
OUT = MODELS_DIR / f"ensemble-probs-w{WINDOW}o{OVERLAP}"

# run -> (checkpoint, embedding, saved fastText vocabulary or None to build it from today's data).
# b1yzs79t predates the apostrophe lookup in FastTextEncoder (d1a6fa4), so today's code builds a different
# vocabulary; its vocab.json was built at its training commit 1da6a94 on normalized:v0 (473 127 rows, as the ckpt).
MEMBERS = {
    "b1yzs79t": ("norm-fasttext-bilstm-train_size=0-epochs=10-patience=2-b1yzs79t/best.ckpt", "fasttext",
                 MODELS_DIR / "norm-fasttext-bilstm-train_size=0-epochs=10-patience=2-b1yzs79t/vocab.json"),
    "e4ob4l0j": ("v1-fasttext-bilstm-train_size=0-epochs=10-patience=2-e4ob4l0j/model.ckpt", "fasttext", None),
    "lr5hudf6": ("norm-flair-bilstm-train_size=0-epochs=10-patience=2-lr5hudf6/model.ckpt", "flair", None),
    "fi9xjw14": ("v1-flair-bilstm-train_size=0-epochs=10-patience=2-fi9xjw14/model.ckpt", "flair", None),
}


def windows(n, size, overlap):
    """Word ranges of length `size`, stepping size - overlap; the last one ends at the end of the post."""
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size, size - overlap)) + [n - size]
    return [(s, s + size) for s in starts]


def window_examples(examples, size, overlap):
    subs, owners = [], []
    for i, ex in enumerate(examples):
        assert ex.word_starts == list(range(len(ex.words))), "one model token per word"
        for a, b in windows(len(ex.words), size, overlap):
            subs.append(Example(ex.text, ex.words[a:b], [], ex.ids[a:b], ex.labels[a:b], list(range(b - a))))
            owners.append((i, a))
    return subs, owners


def merge_windows(examples, sub_probs, owners):
    """Average each word's probabilities over the windows that cover it."""
    sums = [np.zeros((len(ex.word_starts), 3)) for ex in examples]
    counts = [np.zeros(len(ex.word_starts)) for ex in examples]
    for probs, (i, a) in zip(sub_probs, owners):
        sums[i][a:a + len(probs)] += probs
        counts[i][a:a + len(probs)] += 1
    return [s / np.maximum(c, 1)[:, None] for s, c in zip(sums, counts)]


class SavedVocab:
    """A fastText vocabulary saved as {word: row}; the vectors themselves come from the checkpoint."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.vectors = np.zeros((len(vocab) + 2, 300), dtype=np.float32)

    def encode(self, word):
        return [self.vocab.get(word, UNK)]


def load_member(run, eval_rows):
    path, embedding, vocab = MEMBERS[run]
    cfg = Config(embedding=embedding, train_size=0)
    if embedding != "fasttext":
        encoder = WordEncoder()
    elif vocab is not None:
        encoder = SavedVocab(json.loads(vocab.read_text(encoding="utf-8")))
    else:
        splits, _ = load_splits(0, cfg.seed)
        encoder = FastTextEncoder(splits, cfg.min_freq)
        del splits
    state = torch.load(MODELS_DIR / path, map_location="cpu", weights_only=False)["state_dict"]
    if encoder.vectors is not None:
        rows = state["model.embedder.embedding.weight"].shape[0]
        assert encoder.vectors.shape[0] == rows, f"{run}: vocabulary {encoder.vectors.shape[0]} != checkpoint {rows}"
    model = Tagger(cfg, encoder.vectors)
    model.load_state_dict(state)
    # Flair's character LM on 64 of the longest posts needs ~30 GB on MPS; 8 posts per batch are enough on CPU.
    batch_size = cfg.eval_batch_size if embedding == "fasttext" else 8
    return model, {split: encode(eval_rows[split], encoder) for split in SPLITS}, batch_size


def main(members):
    eval_rows, _ = load_splits(train_size=1, seed=0)
    OUT.mkdir(exist_ok=True)
    for run in members:
        model, examples, batch_size = load_member(run, eval_rows)
        # Flair's character LM fills MPS memory (a new graph per sequence length) and is slow on CPU, so it runs on
        # CPU and only where it has to: the sample check and the posts a window cuts.
        flair = MEMBERS[run][1] == "flair"
        on = torch.device("cpu") if flair else None
        saved = load_probs(MODELS_DIR / "ensemble-probs" / f"{run}.npz")
        full = {split: saved[split] if flair and split != "sample" else word_probs(model, examples[split], batch_size, on)
                for split in SPLITS}
        diff = max(np.abs(np.concatenate(full[s]) - np.concatenate(saved[s])).max() for s in SPLITS)
        print(f"{run} whole posts vs saved ensemble-probs: max |diff| {diff:.4f}", flush=True)
        assert diff < 2e-2, f"{run}: does not reproduce the saved probabilities"
        probs = {}
        for split in SPLITS:   # posts no longer than the window are the same as whole
            long = [i for i, ex in enumerate(examples[split]) if len(ex.words) > WINDOW]
            subs, owners = window_examples([examples[split][i] for i in long], WINDOW, OVERLAP)
            sub_probs = word_probs(model, subs, batch_size, on) if subs else []   # sample has no long posts
            probs[split] = list(full[split])
            for i, p in zip(long, merge_windows([examples[split][i] for i in long], sub_probs, owners)):
                probs[split][i] = p
            print(f"{run} {split}: {len(long)} posts cut into windows", flush=True)
        save_probs(OUT / f"{run}.npz", probs)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="+", default=list(MEMBERS))
    main(parser.parse_args().members)
