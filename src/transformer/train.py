"""A training run: fine-tune an encoder with HF Trainer, scoring val and sample by the competition F1 every epoch,
then the best epoch -> probs.npz of every split and the submission.

    python -m src.transformer.train --model.name xlm-roberta-base --data.max_length 256   # one flag per Config field
"""
import json
import os
import shutil
from dataclasses import asdict

import numpy as np
import torch
import wandb
from transformers import Trainer, TrainingArguments, set_seed

from src.cli_config import changed_fields, config_from_args
from src.config import MODELS_DIR, WANDB_ENTITY, WANDB_TRANSFORMER_PROJECT
from src.dataset import EVAL_SETS, SPLITS, load_eval, load_test_ids, load_train
from src.ensemble import save_probs
from src.metric import counts_per_post, scores
from src.submission import gold_and_predicted, write_submissions
from src.transformer.collator import Collator
from src.transformer.config import Config
from src.transformer.dataset import WindowDataset, encode, load_tokenizer
from src.transformer.models.token_classifier import load_model
from src.transformer.predict import CONFIG, predict_probs
from src.transformer.word_probs import post_probs

NAME_SKIP = {"name", "wandb_mode"}


def run_name(cfg: Config):
    return cfg.name or "-".join([cfg.model.name.split("/")[-1], *changed_fields(cfg, NAME_SKIP)])


class Scorer:
    """compute_metrics of HF Trainer: logits of the val windows, then the sample ones -> answers -> P/R/F1 of each.
    One eval set, cut by window counts: compute_metrics is not told which set it scores."""

    def __init__(self, posts, windows):
        self.posts, self.windows = posts, windows

    def __call__(self, prediction):
        logits = prediction.predictions   # [windows, longest, 3]; rows padded with -100, cut off by post_probs
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        out, first = {}, 0
        for split in EVAL_SETS:
            last = first + len(self.windows[split])
            words = post_probs(self.posts[split], self.windows[split], probs[first:last])
            counts = counts_per_post(*gold_and_predicted(self.posts[split], words))
            out.update({f"{split}_{k}": v for k, v in scores(counts).items()})
            first = last
        return out


def fit(cfg: Config, run_dir, posts, tokenizer):
    """Fine-tune -> the model of the best epoch by val F1."""
    data = cfg.data
    assert data.max_length <= tokenizer.model_max_length, f"{cfg.model.name} reads {tokenizer.model_max_length} at most"
    train = encode(load_train(data.train_size, cfg.seed), tokenizer, data.max_length, data.stride, keep_offsets=False)
    windows = {split: encode(posts[split], tokenizer, data.max_length, data.stride) for split in EVAL_SETS}
    args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        report_to="wandb",
        seed=cfg.seed,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=data.batch_size,
        per_device_eval_batch_size=data.eval_batch_size,
        group_by_length=True,                 # batches of windows of similar length: less padding
        optim=cfg.optim.optimizer,
        lr_scheduler_type=cfg.optim.scheduler,
        learning_rate=cfg.optim.lr,
        warmup_ratio=cfg.optim.warmup_ratio,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.grad_clip,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="val_f1",
        eval_accumulation_steps=50,           # logits to the CPU every 50 batches, not all val on the GPU
        logging_steps=50,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=load_model(cfg.model.name),
        args=args,
        train_dataset=WindowDataset(train),
        eval_dataset=WindowDataset(windows["val"] + windows["sample"]),
        data_collator=Collator(tokenizer.pad_token_id),
        compute_metrics=Scorer(posts, windows),
    )
    trainer.train()
    wandb.run.summary["best/val_f1"] = trainer.state.best_metric
    return trainer.model


def export(name, model, tokenizer, cfg: Config, run_dir, posts):
    """Best model -> its weights, tokenizer and Config in run_dir, probs.npz of every split and the submission pair,
    the last three logged as a W&B artifact; -> the path of the submission."""
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    (run_dir / CONFIG).write_text(json.dumps(asdict(cfg), indent=2))
    for checkpoint in run_dir.glob("checkpoint-*"):   # Trainer's; the best is saved above
        shutil.rmtree(checkpoint)

    probs = {split: predict_probs(model, tokenizer, posts[split], cfg) for split in SPLITS}
    save_probs(run_dir / "probs.npz", probs)
    paths = write_submissions(name, load_test_ids(), posts["test"], probs["test"])

    artifact = wandb.Artifact(f"submission-{wandb.run.id}", type="submission")
    for file in (*paths, run_dir / "probs.npz"):
        artifact.add_file(str(file))
    wandb.run.log_artifact(artifact)
    return paths[0]


def run(cfg: Config):
    set_seed(cfg.seed)
    name = run_name(cfg)
    os.environ["WANDB_WATCH"] = "gradients"   # gradient histograms every logging_steps: the gradient flow
    wandb.init(entity=WANDB_ENTITY, project=WANDB_TRANSFORMER_PROJECT, name=name, config=asdict(cfg), mode=cfg.wandb_mode)
    run_dir = MODELS_DIR / f"{name}-{wandb.run.id}"

    posts = load_eval()
    if cfg.data.val_size:
        posts["val"] = posts["val"][:cfg.data.val_size]
    tokenizer = load_tokenizer(cfg.model.name)
    model = fit(cfg, run_dir, posts, tokenizer)
    path = export(name, model, tokenizer, cfg, run_dir, posts)
    wandb.finish()
    print("submission:", path)
    print("weights:", run_dir)
    return path


if __name__ == "__main__":
    run(config_from_args(Config))
