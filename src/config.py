"""Where the project's things live, on disk and online, and the hyperparameters of a run, grouped by the component
that reads them. Shared by src and scripts."""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized"     # train/val of scripts.build_dataset
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submission"

WANDB_ENTITY = "whoresandbaddies"
WANDB_PROJECT = "rnn-for-ner"
KAGGLE_COMPETITION = "kse-ua-location-extraction-2026"


@dataclass
class VocabConfig:      # src.vocab, model.build_embedder
    embedding: str = "fasttext+flair"   # fasttext | flair | fasttext+flair
    min_freq: int = 2
    freeze_embeddings: bool = True


@dataclass
class DataConfig:       # NERDataModule, predict_probs
    train_size: int = 0             # 0: every post
    batch_size: int = 128
    eval_batch_size: int = 64


@dataclass
class ModelConfig:      # Tagger, RNNEncoder
    rnn: str = "lstm"               # lstm | gru
    bidirectional: bool = True
    hidden_size: int = 256
    num_layers: int = 2
    embedding_dropout: float = 0.3  # on the stacked word vectors
    rnn_dropout: float = 0.3        # between RNN layers and before the head
    word_dropout: float = 0.05      # share of whole word vectors zeroed in training


@dataclass
class OptimConfig:      # TaggerModule.configure_optimizers, Trainer
    optimizer: str = "adamw"        # adamw | adam | radam | sgd
    scheduler: str = "plateau"      # none | plateau | cosine | onecycle; plateau halves the LR after an epoch
                                    # without val gain, and goes with early stopping (onecycle plans all epochs)
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0


@dataclass
class Config:
    vocab: VocabConfig = field(default_factory=VocabConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    epochs: int = 10
    patience: int = 2               # epochs without val gain before stopping; 0: no early stopping
    monitor: str = "val/f1"
    seed: int = 42
    accelerator: str = "auto"
    wandb_mode: str = "online"
    name: str = ""


def config_from_dict(values, cls=Config):
    """Config of a checkpoint (asdict): fields it lacks take their defaults, fields since removed are dropped."""
    known = {f.name: f for f in fields(cls)}
    return cls(**{k: config_from_dict(v, known[k].type) if is_dataclass(known[k].type) else v
                  for k, v in values.items() if k in known})
