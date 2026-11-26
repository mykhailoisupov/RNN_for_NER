"""Setup of a Colab session for notebooks/rnn_train.ipynb and transformer_train.ipynb: keys, packages, the W&B login, the competition data and the
train/val of scripts.build_dataset. load_env reads the same .env locally."""
import importlib.util
import os
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from getpass import getpass
from pathlib import Path

from src.config import DATA_DIR, KAGGLE_COMPETITION, NORMALIZED_DIR, ROOT, WANDB_ENTITY, WANDB_PROJECT

PACKAGES = {"wandb": "wandb", "kaggle": "kaggle",
            "pymorphy3": "pymorphy3", "pymorphy3_dicts_uk": "pymorphy3-dicts-uk", "pymorphy3_dicts_ru": "pymorphy3-dicts-ru"}
RNN_PACKAGES = {"lightning": "lightning", "flair": "flair>=0.15"}
# installed every time: Colab has transformers 5, which renamed the TrainingArguments src.transformer.train uses
TRANSFORMER_PACKAGES = ["transformers>=4.57.6,<5", "accelerate", "sentencepiece", "protobuf"]
DATA_FILES = ["test.csv", "labeling_sample.csv"]
NORMALIZED = ["train.parquet", "val.parquet"]
ENV_FILES = [Path("/content/.env"), ROOT / ".env"]


@contextmanager
def step(title):
    print(f"\n▶ {title}", flush=True)
    start = time.time()
    yield
    print(f"✔ {title} ({time.time() - start:.0f}s)", flush=True)


def sh(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in process.stdout:
        print("  " + line, end="", flush=True)
        lines.append(line)
    if process.wait():
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{''.join(lines[-30:])}")


def load_env():
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        print("env:", path)
        return
    print("env: .env not found, keys will be prompted")


def colab_secret(name):
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return None


def secret(name):
    return os.environ.get(name) or colab_secret(name) or getpass(f"{name}: ")


def install(transformer=False):
    packages = PACKAGES if transformer else PACKAGES | RNN_PACKAGES
    missing = [spec for module, spec in packages.items() if importlib.util.find_spec(module) is None]
    missing += TRANSFORMER_PACKAGES if transformer else []
    if not missing:
        print("packages: already installed")
        return
    print("installing:", ", ".join(missing))
    sh([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", *missing])


def download_data():
    missing = [f for f in DATA_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        print("data: ok")
        return
    token = os.environ.get("KAGGLE_API_TOKEN") or colab_secret("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    else:
        os.environ["KAGGLE_USERNAME"] = secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secret("KAGGLE_KEY")
    DATA_DIR.mkdir(exist_ok=True)
    sh(["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(DATA_DIR)])
    for archive in DATA_DIR.glob("*.zip"):
        print("unzipping:", archive.name)
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA_DIR)
        archive.unlink()
    missing = [f for f in DATA_FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"missing after download: {missing}; found: {sorted(p.name for p in DATA_DIR.iterdir())}")
    print("data: downloaded")


def download_normalized():
    """train/val built by scripts/build_dataset.py, uploaded there with --upload."""
    if all((NORMALIZED_DIR / f).exists() for f in NORMALIZED):
        print("normalized: ok")
        return
    import wandb

    artifact = wandb.Api().artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/normalized:latest")
    artifact.download(str(NORMALIZED_DIR))
    print("normalized:", artifact.version)


def login_wandb():
    import wandb

    wandb.login(key=secret("WANDB_API_KEY"))


def setup(transformer=False):
    with step("1/4 keys"):
        load_env()
    with step("2/4 packages"):
        install(transformer)
    with step("3/4 W&B login"):
        login_wandb()
    with step("4/4 data"):
        download_data()
        download_normalized()
    import torch

    print("\ndevice:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
