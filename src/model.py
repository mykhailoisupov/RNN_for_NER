import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import EVAL_SETS, TAGS, Config
from src.data import IGNORE, PAD, score


RNNS = {"lstm": nn.LSTM, "gru": nn.GRU}


def build_embedder(vectors, cfg: Config):
    if cfg.embedding in ("fasttext", "bpemb"):
        return FastTextEmbedder(vectors, cfg)  # обидва: номер → вектор з таблиці
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

    @torch.no_grad()
    def forward(self, batch):
        import flair
        flair.device = batch["ids"].device
        words = batch["words"]
        texts = [" ".join(w) for w in words]

        fwd = self.forward_lm.get_representation(texts, "\n", " ")
        bwd = self.backward_lm.get_representation(texts, "\n", " ")

        out = torch.zeros(len(words), batch["ids"].size(1), self.output_dim, device=fwd.device)
        for i, sentence in enumerate(words):
            start = 0
            for j, word in enumerate(sentence):
                end = start + len(word)
                f = end + 1
                b = len(texts[i]) + 1 - start
                out[i, j] = torch.cat([fwd[f, i], bwd[b, i]])
                start = end + 1
        return out
