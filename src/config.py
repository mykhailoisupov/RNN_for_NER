"""Hyperparameters of a run, grouped by the component that reads them."""
from dataclasses import dataclass, field


@dataclass
class VocabConfig:      # src.vocab, model.build_embedder
    embedding: str = "fasttext+flair"   # fasttext | bpemb | flair | fasttext+flair
    bpe_vocab_size: int = 50000
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
    optimizer: str = "adamw"        # adamw 
    scheduler: str = "onecycle"         # none | plateau | cosine | onecycle
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
    patience: int = 0               # 0: no early stopping
    monitor: str = "val/f1"
    seed: int = 42
    accelerator: str = "auto"
    wandb_mode: str = "online"
    name: str = ""

