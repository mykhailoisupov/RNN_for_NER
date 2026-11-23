"""Tables and figures of the v3 runs for the report: W&B history -> docs/results/*.csv and *.png.

    uv run --group eda python -m scripts.report_figures            # needs WANDB_API_KEY (.env)

LB scores of the ensembles are not in W&B: docs/results/leaderboard.csv is written by hand from Kaggle.
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from src.colab import load_env
from src.config import ROOT, WANDB_ENTITY, WANDB_PROJECT

OUT = ROOT / "docs" / "results"
STEPS_PER_EPOCH = 3437   # 439,827 train posts of normalized:v3 / batch 128

RUNS = {   # label -> W&B run id; the order fixes the colours
    "A fastText+Flair BiLSTM": "wzoh8guc",
    "B Flair BiLSTM": "nfece7iu",
    "C fastText BiLSTM": "3yd46rm3",
    "D fastText BiGRU": "pbue6w0u",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, validated (light surface)
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

EPOCH_KEYS = ["epoch", "val/f1", "val/precision", "val/recall", "val/loss", "val/f1_seen", "val/f1_unseen",
              "val/f1_uk", "val/f1_ru", "val/f1_tg", "sample/f1", "lr-AdamW"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


# --- data -------------------------------------------------------------------------------------------------------
def fetch():
    api = wandb.Api()
    runs, epochs, losses, grads = [], [], [], []
    for label, run_id in RUNS.items():
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        cfg, s = run.config, run.summary
        ep = run.history(keys=EPOCH_KEYS[:-1], pandas=True).drop(columns="_step")
        lr = run.history(keys=["trainer/global_step", "lr-AdamW"], pandas=True)   # logged as each epoch starts
        lr = lr.set_index(lr["trainer/global_step"] // STEPS_PER_EPOCH)["lr-AdamW"]
        ep = ep.assign(**{"lr-AdamW": ep["epoch"].map(lr).to_numpy(), "model": label})
        best = ep.loc[ep["val/f1"].idxmax()]
        runs.append({"model": label, "run_id": run_id, "embedding": cfg["vocab"]["embedding"],
                     "rnn": cfg["model"]["rnn"], "gpu": (run.metadata or {}).get("gpu"),
                     "epochs_run": int(ep["epoch"].max()) + 1, "best_epoch": int(best["epoch"]),
                     **{k: best[k] for k in EPOCH_KEYS[1:-1]}, "lb/f1": s.get("lb/f1"),
                     "runtime_min": round(s.get("_runtime", 0) / 60)})
        epochs.append(ep)
        tl = run.history(keys=["trainer/global_step", "train/loss"], samples=10_000, pandas=True)
        losses.append(tl.drop(columns="_step").assign(model=label))
        grads.append(gradient_norms(run).assign(model=label))
    return pd.DataFrame(runs), pd.concat(epochs), pd.concat(losses), pd.concat(grads)


def gradient_norms(run):
    """L2 norm of every weight's gradient at each logged step, from the W&B histograms (bin centres x counts)."""
    keys = [k for k in run.summary.keys()
            if k.startswith("gradients/") and ("weight" in k or k.endswith("crf.transitions"))]
    rows = []
    for h in run.scan_history(keys=["trainer/global_step", *keys]):
        for k in keys:
            v = h.get(k)
            if not isinstance(v, dict):
                continue
            if packed := v.get("packedBins"):   # bins stored as min + size * i
                bins = packed["min"] + packed["size"] * np.arange(packed["count"] + 1)
            elif v.get("bins") is not None:
                bins = np.array(v["bins"])
            else:
                continue
            centres = (bins[:-1] + bins[1:]) / 2
            rows.append({"step": h["trainer/global_step"], "param": k.removeprefix("gradients/"),
                         "norm": float(np.sqrt((np.array(v["values"]) * centres ** 2).sum()))})
    return pd.DataFrame(rows)


def layer_name(param, char_part):
    """`embedder.parts.2.conv.weight` -> `CharCNN conv`, `encoder.rnn.weight_hh_l0_reverse` -> `RNN L0 ← hh`."""
    if m := re.search(r"embedder\.parts\.(\d+)\.(\w+)\.weight", param):
        part, module = int(m[1]), m[2]
        if part == char_part:
            return "CharCNN chars" if module == "embedding" else "CharCNN conv"
        return "word shape"
    if m := re.search(r"encoder\.rnn\.weight_(ih|hh)_l(\d)(_reverse)?", param):
        return f"RNN L{m[2]} {'←' if m[3] else '→'} {m[1]}"
    return "linear head" if param.endswith("head.weight") else "CRF transitions" if param.endswith("crf.transitions") else param


# --- figures ----------------------------------------------------------------------------------------------------
def label_ends(ax, series, fmt="{:.4f}"):
    """Direct labels at the right end of each line, nudged apart so they do not collide."""
    ends = sorted(((y[-1], x[-1], label, color) for label, x, y, color in series), key=lambda e: e[0])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    placed = []
    for y, x, label, color in ends:
        ty = max(y, placed[-1] + gap) if placed else y
        placed.append(ty)
        ax.annotate(f"{label.split()[0]}  {fmt.format(y)}", (x, y), xytext=(8, (ty - y) / (hi - lo) * ax.bbox.height),
                    textcoords="offset points", va="center", fontsize=9, color=INK)


def per_epoch(epochs, key, title, ylabel, path, legend="lower right"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = []
    for (label, color) in zip(RUNS, COLORS):
        ep = epochs[epochs.model == label].sort_values("epoch")
        x, y = ep["epoch"].to_numpy(), ep[key].to_numpy()
        ax.plot(x, y, color=color, lw=2, marker="o", ms=4, label=label)
        i = ep["val/f1"].to_numpy().argmax()
        ax.plot(x[i], y[i], marker="o", ms=10, mfc="none", mec=color, mew=2)
        series.append((label, x, y, color))
    ax.set(title=title, xlabel="epoch", ylabel=ylabel, xticks=range(int(epochs.epoch.max()) + 1))
    ax.set_xlim(right=epochs.epoch.max() + 1.6)
    label_ends(ax, series)
    ax.legend(loc=legend, fontsize=9)
    ax.text(0, -0.2, "Ring: the best epoch, whose checkpoint made the submission.", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_loss(losses, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, color in zip(RUNS, COLORS):
        tl = losses[losses.model == label].sort_values("trainer/global_step")
        smooth = tl["train/loss"].rolling(20, min_periods=1).mean()
        ax.plot(tl["trainer/global_step"] / STEPS_PER_EPOCH, smooth, color=color, lw=2, label=label)
    ax.set(title="Training loss (CRF negative log-likelihood)", xlabel="epoch", ylabel="train loss, rolling mean",
           yscale="log")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def gradient_flow(grads, path):
    """One heatmap per model: rows are the weights from input (top) to output (bottom), columns the logged steps,
    colour the log10 gradient norm. A row that fades to white would be a vanishing gradient."""
    fig, axes = plt.subplots(len(RUNS), 1, figsize=(9, 3.1 * len(RUNS)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    vmin, vmax = np.log10(grads.norm[grads.norm > 0].quantile(0.01)), np.log10(grads.norm.max())
    for ax, label in zip(axes, RUNS):
        g = grads[grads.model == label].copy()
        char_part = int(next(re.search(r"embedder\.parts\.(\d+)\.conv", p)[1] for p in g.param if ".conv." in p))
        g["layer"] = [layer_name(p, char_part) for p in g.param]
        table = g.pivot_table(index="layer", columns="step", values="norm")
        order = [n for n in ["CharCNN chars", "CharCNN conv", "word shape"] if n in table.index]
        order += sorted(n for n in table.index if n.startswith("RNN"))
        order += [n for n in ["linear head", "CRF transitions"] if n in table.index]
        table = table.loc[order]
        im = ax.imshow(np.log10(table.clip(lower=1e-9)), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", extent=(0, table.columns.max() / STEPS_PER_EPOCH, len(order), 0))
        ax.set_yticks(np.arange(len(order)) + 0.5, order, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(False)
        ax.set_xlabel("epoch", fontsize=8)
    fig.suptitle("Gradient flow: L2 norm of each weight's gradient (log10), every 200 steps", x=0.02, ha="left",
                 fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.5, label="log10 ‖grad‖ (clip at 1.0 = 0)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def cv_vs_lb(path):
    lb = pd.read_csv(OUT / "leaderboard.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for kind, color, marker in (("model", COLORS[0], "o"), ("ensemble", COLORS[1], "s")):
        part = lb[lb.kind == kind]
        ax.scatter(part.val_f1, part.lb_f1, s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=2,
                   label={"model": "single model", "ensemble": "ensemble (mean of probabilities)"}[kind], zorder=3)
        for row in part.itertuples():
            ax.annotate(row.name, (row.val_f1, row.lb_f1), xytext=(7, 4), textcoords="offset points", fontsize=8.5)
    ax.set(title="CV and LB disagree: val F1 (local) vs public leaderboard F1", xlabel="val F1 (normalized:v3)",
           ylabel="public LB F1")
    ax.margins(x=0.25, y=0.15)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, epochs, losses, grads = fetch()
    runs.round(4).to_csv(OUT / "runs.csv", index=False)
    epochs.round(5).to_csv(OUT / "epochs.csv", index=False)
    per_epoch(epochs, "val/f1", "Validation F1 per epoch", "val F1", OUT / "val_f1.png")
    per_epoch(epochs, "val/loss", "Validation loss per epoch", "val loss", OUT / "val_loss.png", legend="upper left")
    train_loss(losses, OUT / "train_loss.png")
    gradient_flow(grads, OUT / "gradient_flow.png")
    cv_vs_lb(OUT / "cv_vs_lb.png")
    print(runs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
