from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from src.config import EVAL_SETS, TAGS, Config
from src.data import IGNORE, PAD, score


O_TAG, I_TAG = TAGS.index("O"), TAGS.index("I-LOC")


def viterbi(logits, lengths):
    """Best tag path per sequence with O->I and start->I forbidden, so no span starts on I (argmax gave
    `Михайлівського` + `собору` around an O). Returns one list of tag ids per sequence, cut to its length."""
    logp = logits.float().log_softmax(-1)
    ban = torch.zeros(len(TAGS), len(TAGS), device=logp.device)
    ban[O_TAG, I_TAG] = float("-inf")
    score = logp[:, 0].clone()
    score[:, I_TAG] = float("-inf")
    back = []
    for t in range(1, logp.size(1)):
        best, prev = (score.unsqueeze(2) + ban).max(1)          # [batch, prev, cur] -> [batch, cur]
        back.append(prev)
        alive = (t < lengths.to(logp.device)).unsqueeze(1)
        score = torch.where(alive, best + logp[:, t], score)     # padding keeps the score of the last token
    back = torch.stack(back, 1).tolist() if back else []
    paths = []
    for b, n in enumerate(lengths.tolist()):
        tag = int(score[b].argmax())
        path = [tag]
        for t in range(n - 1, 0, -1):
            tag = back[b][t - 1][tag]
            path.append(tag)
        paths.append(path[::-1])
    return paths


RNNS = {"lstm": nn.LSTM, "gru": nn.GRU}
OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "radam": torch.optim.RAdam,
    "sgd": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
}


def build_embedder(vectors, cfg: Config):
    if cfg.embedding in ("fasttext", "bpemb"):
        return FastTextEmbedder(vectors, cfg)
    if cfg.embedding == "flair":
        return FlairEmbedder(cfg)
    raise ValueError(f"unknown embedding: {cfg.embedding}")


class RNNTagger(nn.Module):
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


class FastTextEmbedder(nn.Module):
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
    

class Tagger(L.LightningModule):
    def __init__(self, cfg: Config, vectors):
        super().__init__()
        self.cfg = cfg
        self.model = RNNTagger(vectors, cfg)
        self.outputs = {}

    def forward(self, batch):
        return self.model(batch)

    def loss(self, logits, labels):
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)

    def predictions(self, logits, batch):
        return list(zip(batch["idx"].tolist(), viterbi(logits, batch["lengths"])))

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
