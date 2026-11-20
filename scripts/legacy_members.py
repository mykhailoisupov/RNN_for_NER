"""Word probabilities of the four ensemble members, trained before checkpoints kept their config and vocabulary.

    python -m scripts.legacy_members [--members b1yzs79t ...] [--out DIR]

Writes DIR/{run}.npz (default models/ensemble-probs-w256o64, already saved there; not needed for a submission).
"""
import argparse
import json
from pathlib import Path

import torch

from src.config import MODELS_DIR, Config
from src.dataset import build_examples, load_eval, load_train
from src.inference import SPLITS, predict_probs, save_probs
from src.model import TaggerModule
from src.vocab import FastTextVocab, NoVocab, fasttext_vocab

OUT = MODELS_DIR / "ensemble-probs-w256o64"
# run -> (checkpoint, embedding, saved fastText vocabulary or None to build it from today's data).
# b1yzs79t predates the apostrophe lookup in fasttext_vocab (d1a6fa4), so today's code builds a different
# vocabulary; its vocab.json was built at its training commit 1da6a94 on normalized:v0 (473 127 rows, as the ckpt).
MEMBERS = {
    "b1yzs79t": ("norm-fasttext-bilstm-train_size=0-epochs=10-patience=2-b1yzs79t/best.ckpt", "fasttext",
                 MODELS_DIR / "norm-fasttext-bilstm-train_size=0-epochs=10-patience=2-b1yzs79t/vocab.json"),
    "e4ob4l0j": ("v1-fasttext-bilstm-train_size=0-epochs=10-patience=2-e4ob4l0j/model.ckpt", "fasttext", None),
    "lr5hudf6": ("norm-flair-bilstm-train_size=0-epochs=10-patience=2-lr5hudf6/model.ckpt", "flair", None),
    "fi9xjw14": ("v1-flair-bilstm-train_size=0-epochs=10-patience=2-fi9xjw14/model.ckpt", "flair", None),
}


def load_member(run, eval_rows):
    path, embedding, saved = MEMBERS[run]
    cfg = Config(embedding=embedding, train_size=0)
    if embedding != "fasttext":
        vocab = NoVocab()
    elif saved is not None:
        vocab = FastTextVocab(json.loads(saved.read_text(encoding="utf-8")))
    else:
        vocab = fasttext_vocab({"train": load_train(0, cfg.seed), **eval_rows}, cfg.min_freq)
    state = torch.load(MODELS_DIR / path, map_location="cpu", weights_only=False)["state_dict"]
    if vocab.vectors is not None:
        rows = state["model.embedder.embedding.weight"].shape[0]
        assert vocab.vectors.shape[0] == rows, f"{run}: vocabulary {vocab.vectors.shape[0]} != checkpoint {rows}"
    model = TaggerModule(cfg, vocab)
    model.load_state_dict(state)
    return model, vocab


def main(members, out):
    eval_rows, *_ = load_eval()
    out.mkdir(exist_ok=True)
    for run in members:
        model, vocab = load_member(run, eval_rows)
        # Flair's character LM fills MPS memory (a new graph per sequence length): CPU, 8 posts per batch.
        flair = MEMBERS[run][1] == "flair"
        on, batch_size = (torch.device("cpu"), 8) if flair else (None, model.cfg.eval_batch_size)
        save_probs(out / f"{run}.npz", {split: predict_probs(model, build_examples(eval_rows[split], vocab), batch_size, on=on)
                                        for split in SPLITS})
        print("written", out / f"{run}.npz", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="+", default=list(MEMBERS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.members, args.out)
