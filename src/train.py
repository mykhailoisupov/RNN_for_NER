import argparse
import dataclasses

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import MODELS_DIR, SUBMISSION_DIR, WANDB_ENTITY, WANDB_PROJECT, Config
from src.dataset import NERDataModule
from src.ensemble import SPLITS, save_probs, word_probs
from src.model import Tagger
from src.submission import answers, write_submission

NAME_SKIP = {"embedding", "rnn", "bidirectional", "name", "wandb_mode", "accelerator"}


def run_name(cfg: Config):
    if cfg.name:
        return cfg.name
    base = f"{cfg.embedding}-{'bi' if cfg.bidirectional else ''}{cfg.rnn}"
    changes = [
        f"{f.name}={getattr(cfg, f.name)}"
        for f in dataclasses.fields(cfg)
        if f.name not in NAME_SKIP and getattr(cfg, f.name) != f.default
    ]
    return "-".join([base, *changes])


def run(cfg: Config):
    L.seed_everything(cfg.seed)
    name = run_name(cfg)

    data = NERDataModule(cfg)
    data.setup()
    model = Tagger(cfg, data.vectors)

    logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=name,
        config=dataclasses.asdict(cfg),
        mode=cfg.wandb_mode,
        log_model="all",
    )
    logger.watch(model.model, log="gradients", log_freq=200)
    logger.experiment.summary.update(data.oracle_f1())

    run_dir = MODELS_DIR / f"{name}-{logger.version}"
    checkpoint = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=cfg.monitor, mode="max")
    callbacks = [checkpoint, LearningRateMonitor()]
    if cfg.patience:
        callbacks.append(EarlyStopping(monitor=cfg.monitor, mode="max", patience=cfg.patience))
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.grad_clip,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=data)

    model = Tagger.load_from_checkpoint(checkpoint.best_model_path, cfg=cfg, vectors=data.vectors)
    word = {split: word_probs(model, data.examples[split], cfg.eval_batch_size) for split in SPLITS}
    probs = run_dir / "probs.npz"
    save_probs(probs, word)
    test = list(zip(data.examples["test"], word["test"]))
    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"{name}.csv"
    # `_raw`: without the test policy, to see which submission rules the relabelled train has made redundant
    raw_path = SUBMISSION_DIR / f"{name}_raw.csv"
    write_submission(path, data.test_ids, [answers(ex, p) for ex, p in test])
    write_submission(raw_path, data.test_ids, [answers(ex, p, policy=[]) for ex, p in test])

    logger.experiment.summary["best/" + cfg.monitor] = checkpoint.best_model_score.item()
    outputs = wandb.Artifact(f"submission-{logger.version}", type="submission")
    for file in (path, raw_path, probs):
        outputs.add_file(str(file))
    logger.experiment.log_artifact(outputs)
    wandb.finish()
    print("submission:", path)
    print("weights:", checkpoint.best_model_path)
    try:   # Colab: the browser saves the best checkpoint; locally it stays in models/
        from google.colab import files

        files.download(checkpoint.best_model_path)
    except Exception:
        pass
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
