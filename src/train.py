import argparse
import dataclasses

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import EVAL_SETS, MODELS_DIR, SUBMISSION_DIR, WANDB_ENTITY, WANDB_PROJECT, Config
from src.dataset import NERDataModule
from src.inference import SPLITS, load_model, predict_probs, save_probs
from src.metric import counts_per_post, hard_processing, precision_recall_f1
from src.model import TaggerModule
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


def strings(examples, predictions):
    """Gold and predicted names of (index, word probabilities) pairs; the submission path without the test policy."""
    return ([examples[i].locations for i, _ in predictions],
            [answers(examples[i], probs, policy=[]) for i, probs in predictions])


def prf(y_true, y_pred):
    return dict(zip(("precision", "recall", "f1"), map(float, precision_recall_f1(counts_per_post(y_true, y_pred).sum(0)))))


def score(examples, predictions):
    return prf(*strings(examples, predictions))


def oracle_f1(examples):
    """F1 of the gold tags through the answers path: what tags lose by themselves."""
    return {f"{name}/oracle_f1": score(examples[name], [(i, np.eye(3)[np.array(ex.labels)[ex.word_starts]])
                                                       for i, ex in enumerate(examples[name])])["f1"]
            for name in EVAL_SETS}


class ValScore(L.Callback):
    """val/sample P/R/F1 of the probabilities TaggerModule.validation_step returns. For val also F1 of the names train
    labels (`val/f1_seen`), of the names it does not (`val/f1_unseen`) and of every source (`val/f1_tg`)."""

    def on_fit_start(self, trainer, module):
        self.seen = {hard_processing(s) for ex in trainer.datamodule.examples["train"] for s in ex.locations}

    def on_validation_epoch_start(self, trainer, module):
        self.outputs = {name: [] for name in EVAL_SETS}

    def on_validation_batch_end(self, trainer, module, probs, batch, batch_idx, dataloader_idx=0):
        name = EVAL_SETS[dataloader_idx]
        examples = trainer.datamodule.examples[name]
        for row, i in zip(probs.cpu().numpy(), batch["idx"].tolist()):
            self.outputs[name].append((i, row[examples[i].word_starts]))

    def on_validation_epoch_end(self, trainer, module):
        data = trainer.datamodule
        for name, predictions in self.outputs.items():
            y_true, y_pred = strings(data.examples[name], predictions)
            for metric, value in prf(y_true, y_pred).items():
                module.log(f"{name}/{metric}", value, prog_bar=metric == "f1")
            if name != "val":
                continue
            for label, known in (("seen", True), ("unseen", False)):   # a name is filtered in gold and predicted alike
                keep = [[[s for s in row if (hard_processing(s) in self.seen) == known] for row in rows]
                        for rows in (y_true, y_pred)]
                module.log(f"val/f1_{label}", prf(*keep)["f1"])
            lang = [data.val_lang[i] for i, _ in predictions]
            for label in sorted(set(lang)):
                rows = [k for k, source in enumerate(lang) if source == label]
                module.log(f"val/f1_{label}", prf([y_true[k] for k in rows], [y_pred[k] for k in rows])["f1"])


def run(cfg: Config):
    L.seed_everything(cfg.seed)
    name = run_name(cfg)

    data = NERDataModule(cfg)
    data.setup()
    model = TaggerModule(cfg, data.vocab)

    logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=name,
        config=dataclasses.asdict(cfg),
        mode=cfg.wandb_mode,
        log_model="all",
    )
    logger.watch(model.model, log="gradients", log_freq=200)
    logger.experiment.summary.update(oracle_f1(data.examples))

    run_dir = MODELS_DIR / f"{name}-{logger.version}"
    checkpoint = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=cfg.monitor, mode="max")
    callbacks = [ValScore(), checkpoint, LearningRateMonitor()]
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

    model, _ = load_model(checkpoint.best_model_path)
    word = {split: predict_probs(model, data.examples[split], cfg.eval_batch_size) for split in SPLITS}
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
