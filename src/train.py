import argparse
import dataclasses
import json

import lightning as L
import pandas as pd
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import MODELS_DIR, SUBMISSION_DIR, WANDB_ENTITY, WANDB_PROJECT, Config
from src.data import NERDataModule, predicted_locations
from src.model import Tagger


def run(cfg: Config):
    L.seed_everything(cfg.seed)
    name = cfg.name or f"{cfg.embedding}-{'bi' if cfg.bidirectional else ''}{cfg.rnn}"

    data = NERDataModule(cfg)
    data.setup()
    model = Tagger(cfg, data.vectors)

    logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=name,
        config=dataclasses.asdict(cfg),
        mode=cfg.wandb_mode,
    )
    logger.watch(model.model, log="gradients", log_freq=200)
    logger.experiment.summary.update(data.oracle_f1())

    run_dir = MODELS_DIR / f"{name}-{logger.version}"
    checkpoint = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=cfg.monitor, mode="max")
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor()],
        gradient_clip_val=cfg.grad_clip,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=data)

    batches = trainer.predict(model, datamodule=data, ckpt_path="best")
    predictions = [p for batch in batches for p in batch]
    examples = data.examples["test"]
    submission = pd.DataFrame({
        "text_id": [data.test_ids[i] for i, _ in predictions],
        "locations": [json.dumps(predicted_locations(examples[i], tags), ensure_ascii=False) for i, tags in predictions],
    }).sort_values("text_id")
    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"{name}-{logger.version}.csv"
    submission.to_csv(path, index=False)

    logger.experiment.summary["best/" + cfg.monitor] = checkpoint.best_model_score.item()
    wandb.finish()
    print("submission:", path)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(Config):
        kind = type(field.default)
        if kind is bool:
            parser.add_argument(f"--{field.name}", type=lambda v: v.lower() in {"1", "true", "yes"}, default=field.default)
        else:
            parser.add_argument(f"--{field.name}", type=kind, default=field.default)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
