import importlib.util
import os
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from getpass import getpass
from pathlib import Path

from src.config import DATA_DIR, KAGGLE_COMPETITION, ROOT, WANDB_ENTITY, WANDB_PROJECT

PACKAGES = {"lightning": "lightning", "wandb": "wandb", "bpemb": "bpemb", "kaggle": "kaggle", "flair": "flair>=0.15"}
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


def install():
    missing = [spec for module, spec in PACKAGES.items() if importlib.util.find_spec(module) is None]
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
    if not KAGGLE_COMPETITION:
        raise RuntimeError("set KAGGLE_COMPETITION in src/config.py or put the files into data/ manually")
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
    out = DATA_DIR / "normalized"
    if all((out / f).exists() for f in NORMALIZED):
        print("normalized: ok")
        return
    import wandb

    artifact = wandb.Api().artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/normalized:latest")
    artifact.download(str(out))
    print("normalized:", artifact.version)


def login_wandb():
    import wandb

    wandb.login(key=secret("WANDB_API_KEY"))


def setup():
    with step("1/4 keys"):
        load_env()
    with step("2/4 packages"):
        install()
    with step("3/4 W&B login"):
        login_wandb()
    with step("4/4 data"):
        download_data()
        download_normalized()
    import torch

    print("\ndevice:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
