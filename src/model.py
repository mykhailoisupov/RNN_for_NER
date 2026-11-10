import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import EVAL_SETS, TAGS, Config
from src.data import IGNORE, PAD, score


RNNS = {"lstm": nn.LSTM, "gru": nn.GRU}


class RNNTagger(nn.Module):
    def __init__(self, vectors, cfg: Config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(vectors), freeze=cfg.freeze_embeddings, padding_idx=PAD
        )
        self.rnn = RNNS[cfg.rnn](
            vectors.shape[1],
            cfg.hidden_size,
            cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        directions = 2 if cfg.bidirectional else 1
        self.head = nn.Linear(directions * cfg.hidden_size, len(TAGS))

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        return self.head(self.dropout(out))


class Tagger(L.LightningModule):
    def __init__(self, cfg: Config, vectors):
        super().__init__()
        self.cfg = cfg
        self.model = RNNTagger(vectors, cfg)
        self.outputs = {}

    def forward(self, batch):
        return self.model(batch["ids"], batch["lengths"])

    def loss(self, logits, labels):
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)

    def predictions(self, logits, batch):
        tags = logits.argmax(-1)
        return [
            (i, row[:n].tolist())
            for i, row, n in zip(batch["idx"].tolist(), tags, batch["lengths"].tolist())
        ]

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch["labels"])
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch["idx"]))
        return loss

    def on_validation_epoch_start(self):
        self.outputs = {name: [] for name in EVAL_SETS}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        name = EVAL_SETS[dataloader_idx]
        logits = self(batch)
        self.log(f"{name}/loss", self.loss(logits, batch["labels"]), add_dataloader_idx=False, batch_size=len(batch["idx"]))
        self.outputs[name] += self.predictions(logits, batch)

    def on_validation_epoch_end(self):
        for name, predictions in self.outputs.items():
            f1 = score(self.trainer.datamodule.examples[name], predictions)
            self.log(f"{name}/f1", f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self.predictions(self(batch), batch)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
