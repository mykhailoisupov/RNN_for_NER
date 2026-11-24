import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from src.config import ROOT, WANDB_ENTITY, WANDB_PROJECT

FIGURES = ROOT / "docs" / "figures"
MEMBERS = {
    "b1yzs79t": "fastText, v0",
    "lr5hudf6": "Flair, v0",
    "e4ob4l0j": "fastText, v1",
    "fi9xjw14": "Flair, v1",
}
LAYERS = {
    "weight_ih_l0": "LSTM layer 1, input",
    "weight_hh_l0": "LSTM layer 1, recurrent",
    "weight_ih_l1": "LSTM layer 2, input",
    "weight_hh_l1": "LSTM layer 2, recurrent",
    "head.weight": "linear head",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def history(run_id):
    run = wandb.Api().run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    return list(run.scan_history(page_size=2000))


def per_epoch(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("gradients/")} for r in rows])
    train = df.dropna(subset=["train/loss"]).groupby("epoch")["train/loss"].mean()
    val = df.dropna(subset=["val/f1"]).groupby("epoch")[["val/loss", "val/f1", "sample/f1"]].last()
    return val.join(train).reset_index()


def mean_abs(hist):
    packed = hist["packedBins"]
    centers = packed["min"] + packed["size"] * (np.arange(packed["count"]) + 0.5)
    counts = np.asarray(hist["values"])
    return float((counts * np.abs(centers)).sum() / counts.sum())


def gradient_flow(rows):
    out = {}
    for name in LAYERS:
        key = f"gradients/{name}" if "." in name else f"gradients/encoder.rnn.{name}"
        points = [(r["trainer/global_step"], mean_abs(r[key])) for r in rows if isinstance(r.get(key), dict)]
        out[name] = np.array(points)
    return out


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = {run_id: history(run_id) for run_id in MEMBERS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for (run_id, label), color in zip(MEMBERS.items(), COLORS):
        curve = per_epoch(rows[run_id])
        epoch = curve["epoch"] + 1
        dash = "--" if "v0" in label else "-"
        axes[0].plot(epoch, curve["train/loss"], dash, color=color, lw=2, marker="o", ms=4, label=f"{label} ({run_id})")
        axes[1].plot(epoch, curve["val/loss"], dash, color=color, lw=2, marker="o", ms=4)
        axes[2].plot(epoch, curve["val/f1"], dash, color=color, lw=2, marker="o", ms=4)
    style(axes[0], "Train loss (mean per epoch)", "cross-entropy")
    style(axes[1], "Val loss", "cross-entropy")
    style(axes[2], "Val F1 (competition metric)", "F1")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.legend(loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "curves.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, run_id in zip(axes, ("e4ob4l0j", "fi9xjw14")):
        for (name, label), color in zip(LAYERS.items(), COLORS):
            points = gradient_flow(rows[run_id])[name]
            smooth = pd.Series(points[:, 1]).rolling(5, center=True, min_periods=1).mean()
            ax.plot(points[:, 0], smooth, color=color, lw=2, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        style(ax, f"{MEMBERS[run_id]} ({run_id})", "mean |gradient|")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "gradient_flow.png", dpi=130)

    table = pd.concat({MEMBERS[r]: per_epoch(rows[r]) for r in MEMBERS})
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
