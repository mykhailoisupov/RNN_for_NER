"""Hyperparameters of a transformer run, grouped by the component that reads them; one CLI flag per field."""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "xlm-roberta-base"  # HF hub encoder; its tokenizer too


@dataclass
class DataConfig:       # src.transformer.dataset, collator
    train_size: int = 0             # 0: every post
    val_size: int = 0               # 0: every post; fewer for a quick run
    max_length: int = 512           # subwords per window, at most the model's limit
    stride: int = 128               # subwords two windows of a long post share
    batch_size: int = 16
    eval_batch_size: int = 32


@dataclass
class OptimConfig:      # HF TrainingArguments
    optimizer: str = "adamw_torch"  # any TrainingArguments.optim
    scheduler: str = "linear"       # any TrainingArguments.lr_scheduler_type
    lr: float = 3e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    epochs: int = 3
    seed: int = 42
    wandb_mode: str = "online"
    name: str = ""
