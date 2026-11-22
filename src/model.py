"""The tagger: word vectors (fastText, Flair, characters, shape) -> BiLSTM -> tag scores -> CRF, and its training.

A checkpoint keeps the config as a dict and every weight but Flair's, which is frozen and comes from the Hub.
"""
from dataclasses import asdict
from functools import partial

import lightning as L
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from src.config import Config, ModelConfig, VocabConfig
from src.crf import CRF
from src.dataset import EVAL_SETS
from src.tags import TAGS
from src.vocab import CHARS, PAD, SHAPES


RNNS = {"lstm": nn.LSTM, "gru": nn.GRU}
CHAR_DIM = 30     # char embedding and CNN filters (Ma & Hovy, 2016)
CASING_DIM = 8
OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "radam": torch.optim.RAdam,
    "sgd": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
}


class LookupEmbedder(nn.Module):
    """fastText rows of the vocabulary, frozen unless the config says otherwise."""

    def __init__(self, vectors, freeze):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(vectors), freeze=freeze, padding_idx=PAD
        )
        self.output_dim = vectors.shape[1]

    def forward(self, batch):
        return self.embedding(batch["ids"])


class FlairEmbedder(nn.Module):
    """Frozen forward and backward Ukrainian character LMs: the state after each word, as Flair embeds it."""

    def __init__(self):
        super().__init__()
        # flair is the optional [flair] extra: imported only when a model uses it
        from flair.embeddings import FlairEmbeddings
        from huggingface_hub import hf_hub_download
        def load(repo):
            return FlairEmbeddings(hf_hub_download(repo, "best-lm.pt")).lm

        self.forward_lm = load("dchaplinsky/flair-uk-forward")
        self.backward_lm = load("dchaplinsky/flair-uk-backward")
        assert self.forward_lm.is_forward_lm and not self.backward_lm.is_forward_lm
        for p in self.parameters():
            p.requires_grad = False
        self.output_dim = self.forward_lm.hidden_size * 2

    def train(self, mode=True):
        super().train(mode)
        self.forward_lm.eval()
        self.backward_lm.eval()
        return self

    @torch.no_grad()
    def forward(self, batch):
        import flair
        flair.device = batch["ids"].device
        words = batch["words"]
        texts = [" ".join(w) for w in words]

        fwd = self.forward_lm.get_representation(texts, "\n", " ")
        bwd = self.backward_lm.get_representation(texts, "\n", " ")

        rows, cols, f, b = [], [], [], []
        for i, sentence in enumerate(words):
            start = 0
            for j, word in enumerate(sentence):
                end = start + len(word)
                rows.append(i)
                cols.append(j)
                f.append(end + 1)
                b.append(len(texts[i]) + 1 - start)
                start = end + 1
        rows, cols, f, b = (torch.tensor(x, device=fwd.device) for x in (rows, cols, f, b))
        out = torch.zeros(len(words), batch["ids"].size(1), self.output_dim, device=fwd.device, dtype=fwd.dtype)
        out[rows, cols] = torch.cat([fwd[f, rows], bwd[b, rows]], dim=-1)
        return out


class CharCNN(nn.Module):
    """Letters of a word -> CHAR_DIM numbers: endings (`-івка`, `-щина`), capitals, apostrophes, unknown words.
    The max runs over the word's own letters only, so a word gets the same numbers whatever else is in the batch."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(CHARS) + 2, CHAR_DIM, padding_idx=PAD)
        self.conv = nn.Conv1d(CHAR_DIM, CHAR_DIM, kernel_size=3, padding=1)
        self.output_dim = CHAR_DIM

    def forward(self, batch):
        chars = batch["chars"]                                                  # [batch, steps, letters]
        b, t, n = chars.shape
        chars = chars.view(b * t, n)
        x = self.conv(self.embedding(chars).transpose(1, 2))                    # [words, dim, letters]
        pad = (chars == PAD).unsqueeze(1)
        out = x.masked_fill(pad, float("-inf")).max(2).values
        out = out.masked_fill(pad.all(2), 0.0)    # padding steps of a post: 0, not -inf (word dropout makes -inf * 0)
        return out.view(b, t, CHAR_DIM)


class CasingEmbedder(nn.Module):
    """word_shape (lower, Title, UPPER, digit, ...) -> CASING_DIM numbers."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(SHAPES), CASING_DIM, padding_idx=PAD)
        self.output_dim = CASING_DIM

    def forward(self, batch):
        return self.embedding(batch["shapes"])


class StackedEmbedder(nn.Module):
    """Every part gives each token a vector; the token gets them glued end to end."""

    def __init__(self, parts):
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self.output_dim = sum(p.output_dim for p in parts)

    def forward(self, batch):
        return torch.cat([p(batch) for p in self.parts], -1)


def build_embedder(vectors, cfg: VocabConfig):
    """fastText rows (when the vocabulary has vectors) and/or Flair, then characters and word shape."""
    parts = [LookupEmbedder(vectors, cfg.freeze_embeddings)] if vectors is not None else []
    if "flair" in cfg.embedding:
        parts.append(FlairEmbedder())
    return StackedEmbedder(parts + [CharCNN(), CasingEmbedder()])


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, cfg: ModelConfig):
        super().__init__()
        self.rnn = RNNS[cfg.rnn](
            input_dim,
            cfg.hidden_size,
            cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.rnn_dropout if cfg.num_layers > 1 else 0.0,
        )
        directions = 2 if cfg.bidirectional else 1
        self.output_dim = directions * cfg.hidden_size

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out


def lengths_mask(batch):
    """[batch, steps]: True on the real tokens of every post."""
    lengths = batch["lengths"].to(batch["ids"].device)
    return torch.arange(batch["ids"].size(1), device=lengths.device) < lengths[:, None]


class Tagger(nn.Module):
    """Stacked word vectors -> word dropout -> BiLSTM -> per-tag scores of every token, which the CRF reads."""

    def __init__(self, vectors, cfg: Config):
        super().__init__()
        self.embedder = build_embedder(vectors, cfg.vocab)
        self.word_dropout = cfg.model.word_dropout
        self.embedding_dropout = nn.Dropout(cfg.model.embedding_dropout)
        self.encoder = RNNEncoder(self.embedder.output_dim, cfg.model)
        self.rnn_dropout = nn.Dropout(cfg.model.rnn_dropout)
        self.head = nn.Linear(self.encoder.output_dim, len(TAGS))
        self.crf = CRF()

    def forward(self, batch):
        x = self.embedder(batch)
        if self.training and self.word_dropout:   # whole words: a name must be found from its context too
            x = x * torch.bernoulli(x.new_full((*x.shape[:2], 1), 1 - self.word_dropout))
        out = self.encoder(self.embedding_dropout(x), batch["lengths"])
        return self.head(self.rnn_dropout(out))


class TaggerModule(L.LightningModule):
    """Training of the Tagger. A checkpoint also keeps the config and the fastText words, so src.inference.load_model
    rebuilds the model without the data."""

    def __init__(self, cfg: Config, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab_words = getattr(vocab, "words", None)
        self.model = Tagger(vocab.vectors, cfg)

    def forward(self, batch):
        return self.model(batch)

    def loss(self, emissions, batch):
        """CRF loss; IGNORE (-100, the only token of an empty post) counts as O."""
        return self.model.crf.nll(emissions, batch["labels"].clamp_min(0), lengths_mask(batch))

    def probs(self, batch):
        """P(tag) of every token given the whole post: what decoding, rampart and the ensemble read."""
        return self.model.crf.marginals(self(batch), lengths_mask(batch))

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch["idx"]))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """-> token probabilities, for the F1 callback in src.train."""
        emissions = self(batch)
        name = EVAL_SETS[dataloader_idx]
        self.log(f"{name}/loss", self.loss(emissions, batch), add_dataloader_idx=False, batch_size=len(batch["idx"]))
        return self.model.crf.marginals(emissions, lengths_mask(batch))

    def _flair_prefixes(self):
        return tuple(f"{name}." for name, module in self.named_modules() if isinstance(module, FlairEmbedder))

    def on_save_checkpoint(self, checkpoint):
        """Config as a dict (new fields take their defaults on load), the fastText words, no Flair weights."""
        checkpoint["config"] = asdict(self.cfg)
        checkpoint["vocab"] = self.vocab_words
        flair = self._flair_prefixes()
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith(flair)}

    def load_weights(self, state_dict):
        """Weights of a checkpoint; only Flair's may be missing, as on_save_checkpoint leaves them out."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        flair = self._flair_prefixes()
        wrong = [k for k in missing if not k.startswith(flair)] + list(unexpected)
        if wrong:
            raise RuntimeError(f"checkpoint does not match the model: {wrong[:10]}")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = self.cfg.optim
        optimizer = OPTIMIZERS[opt.optimizer](params, lr=opt.lr, weight_decay=opt.weight_decay)
        if opt.scheduler == "none":
            return optimizer
        if opt.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": self.cfg.monitor}}
        steps = self.trainer.estimated_stepping_batches
        if opt.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=steps)
        elif opt.scheduler == "onecycle":
            scheduler = OneCycleLR(optimizer, max_lr=opt.lr, total_steps=steps)
        else:
            raise ValueError(f"unknown scheduler: {opt.scheduler}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
