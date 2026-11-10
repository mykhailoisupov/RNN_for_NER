from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submission"

WANDB_ENTITY = "whoresandbaddies"
WANDB_PROJECT = "rnn-for-ner"

TAGS = ["O", "B-LOC", "I-LOC"]
EVAL_SETS = ["val", "sample"]


@dataclass
class Config:
    embedding: str = "fasttext"
    bpe_vocab_size: int = 50000
    train_size: int = 200_000
    min_freq: int = 2
    freeze_embeddings: bool = True
    rnn: str = "lstm"
    bidirectional: bool = True
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 128
    epochs: int = 5
    grad_clip: float = 1.0
    monitor: str = "val/f1"
    seed: int = 42
    accelerator: str = "auto"
    wandb_mode: str = "online"
    name: str = ""
