import dataclasses
from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from src.config import EVAL_SETS, Config
from src.dataset import IGNORE
from src.vocab import PAD
from src.tags import TAGS


RNNS = {"lstm": nn.LSTM, "gru": nn.GRU}
OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "radam": torch.optim.RAdam,
    "sgd": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
}


def build_embedder(vectors, cfg: Config):
    if cfg.embedding in ("fasttext", "bpemb"):
        return LookupEmbedder(vectors, cfg)
    if cfg.embedding == "flair":
        return FlairEmbedder(cfg)
    raise ValueError(f"unknown embedding: {cfg.embedding}")


class Tagger(nn.Module):
    def __init__(self, vectors, cfg: Config):
        super().__init__()
        self.embedder = build_embedder(vectors, cfg)
        self.encoder = RNNEncoder(self.embedder.output_dim, cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(self.encoder.output_dim, len(TAGS))

    def forward(self, batch):
        x = self.dropout(self.embedder(batch))
        out = self.encoder(x, batch["lengths"])
        return self.head(self.dropout(out))


class LookupEmbedder(nn.Module):
    def __init__(self, vectors, cfg: Config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(vectors), freeze=cfg.freeze_embeddings, padding_idx=PAD
        )
        self.output_dim = vectors.shape[1]

    def forward(self, batch):
        return self.embedding(batch["ids"])


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, cfg: Config):
        super().__init__()
        self.rnn = RNNS[cfg.rnn](
            input_dim,
            cfg.hidden_size,
            cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        directions = 2 if cfg.bidirectional else 1
        self.output_dim = directions * cfg.hidden_size

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out
    

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

    def loss(self, logits, labels):
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch["labels"])
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch["idx"]))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """-> token probabilities, for the F1 callback in src.train."""
        logits = self(batch)
        name = EVAL_SETS[dataloader_idx]
        self.log(f"{name}/loss", self.loss(logits, batch["labels"]), add_dataloader_idx=False, batch_size=len(batch["idx"]))
        return logits.float().softmax(-1)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = dataclasses.asdict(self.cfg)
        checkpoint["vocab"] = self.vocab_words

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = OPTIMIZERS[self.cfg.optimizer](params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if self.cfg.scheduler == "none":
            return optimizer
        if self.cfg.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": self.cfg.monitor}}
        steps = self.trainer.estimated_stepping_batches
        if self.cfg.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=steps)
        elif self.cfg.scheduler == "onecycle":
            scheduler = OneCycleLR(optimizer, max_lr=self.cfg.lr, total_steps=steps)
        else:
            raise ValueError(f"unknown scheduler: {self.cfg.scheduler}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class FlairEmbedder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
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
