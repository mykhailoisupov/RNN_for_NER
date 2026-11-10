import os
import subprocess
import sys
import zipfile
from getpass import getpass
from pathlib import Path

from src.config import DATA_DIR, KAGGLE_COMPETITION, ROOT

PACKAGES = ["lightning", "wandb", "bpemb", "kaggle", "flair>=0.15"]
DATA_FILES = ["uk_geo_dataset.csv", "test.csv", "labeling_sample.csv"]
ENV_FILES = [Path("/content/.env"), ROOT / ".env"]


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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)


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
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(DATA_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"kaggle download failed:\n{result.stdout}\n{result.stderr}")
    for archive in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA_DIR)
        archive.unlink()
    missing = [f for f in DATA_FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"missing after download: {missing}; found: {sorted(p.name for p in DATA_DIR.iterdir())}")
    print("data: downloaded")


def login_wandb():
    import wandb

    wandb.login(key=secret("WANDB_API_KEY"))


def setup():
    load_env()
    install()
    download_data()
    login_wandb()
    import torch

    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
