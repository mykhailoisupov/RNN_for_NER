import argparse
import dataclasses

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import Config
from src.dataset import EVAL_SETS, SPLITS, NERDataModule
from src.inference import load_model, predict_probs, save_probs
from src.metric import counts_per_post, hard_processing, precision_recall_f1
from src.model import TaggerModule
from src.project import MODELS_DIR, SUBMISSION_DIR, WANDB_ENTITY, WANDB_PROJECT
from src.submission import answers, write_submission
from src.tags import TAGS

NAME_SKIP = {"embedding", "rnn", "bidirectional", "name", "wandb_mode", "accelerator"}


def dotted_fields(cfg: Config):
    """(`model.hidden_size`, value) for every field of the Config and of its groups: the CLI flags and run names."""
    for key, value in dataclasses.asdict(cfg).items():
        if isinstance(value, dict):
            yield from ((f"{key}.{k}", v) for k, v in value.items())
        else:
            yield key, value


def run_name(cfg: Config):
    if cfg.name:
        return cfg.name
    base = f"{cfg.vocab.embedding}-{'bi' if cfg.model.bidirectional else ''}{cfg.model.rnn}"
    default = dict(dotted_fields(Config()))
    changes = [f"{key.split('.')[-1]}={value}" for key, value in dotted_fields(cfg)
               if key.split(".")[-1] not in NAME_SKIP and value != default[key]]
    return "-".join([base, *changes])


def strings(examples, predictions):
    """Gold and predicted names of (index, word probabilities) pairs; the submission path without the test policy."""
    return ([examples[i].locations for i, _ in predictions],
            [answers(examples[i], probs, policy=[]) for i, probs in predictions])


def prf(y_true, y_pred):
    return dict(zip(("precision", "recall", "f1"), map(float, precision_recall_f1(counts_per_post(y_true, y_pred).sum(0)))))


def oracle_f1(examples):
    """F1 of the gold tags through the answers path: what tags lose by themselves."""
    return {f"{name}/oracle_f1": prf(*strings(examples[name], [(i, np.eye(len(TAGS))[ex.labels[:len(ex.words)]])
                                                              for i, ex in enumerate(examples[name])]))["f1"]
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
            self.outputs[name].append((i, row[:len(examples[i].words)]))

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
            counts = counts_per_post(y_true, y_pred)
            lang = np.array([data.val_lang[i] for i, _ in predictions])
            for label in np.unique(lang):
                module.log(f"val/f1_{label}", float(precision_recall_f1(counts[lang == label].sum(0))[2]))


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
        log_model=True,
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
        gradient_clip_val=cfg.optim.grad_clip,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=data)

    model, _ = load_model(checkpoint.best_model_path)
    word = {split: predict_probs(model, data.examples[split], cfg.data.eval_batch_size) for split in SPLITS}
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
    """`--epochs 10`, `--model.hidden_size 512`: one flag per field."""
    parser = argparse.ArgumentParser()
    for key, default in dotted_fields(Config()):
        kind = (lambda v: v.lower() in {"1", "true", "yes"}) if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{key}", type=kind, default=default)
    cfg = Config()
    for key, value in vars(parser.parse_args()).items():
        *group, name = key.split(".")
        setattr(getattr(cfg, group[0]) if group else cfg, name, value)
    return cfg


if __name__ == "__main__":
    run(parse_args())
