"""Підготовка Colab: залежності, дані з Kaggle, логін у W&B.

Викликається з першої клітинки notebooks/train.ipynb після того, як код гілки вже склоновано.
Ключі беруться з Colab Secrets (WANDB_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY),
а якщо їх немає — зі змінних оточення або питаються вручну.
"""

import os
import subprocess
import sys
import zipfile
from getpass import getpass

from src.config import DATA_DIR, KAGGLE_COMPETITION

PACKAGES = ["lightning", "wandb", "bpemb", "kaggle", "flair>=0.15"]
DATA_FILES = ["uk_geo_dataset.csv", "test.csv", "labeling_sample.csv"]


def secret(name):
    try:
        from google.colab import userdata

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name) or getpass(f"{name}: ")


def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)


def download_data():
    missing = [f for f in DATA_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        print("data: ok")
        return
    if not KAGGLE_COMPETITION:
        raise RuntimeError("заповни KAGGLE_COMPETITION у src/config.py або поклади файли в data/ вручну")
    os.environ["KAGGLE_USERNAME"] = secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secret("KAGGLE_KEY")
    DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(DATA_DIR)], check=True)
    for archive in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA_DIR)
        archive.unlink()
    missing = [f for f in DATA_FILES if not (DATA_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"після завантаження бракує: {missing}; є: {sorted(p.name for p in DATA_DIR.iterdir())}")
    print("data: downloaded")


def login_wandb():
    import wandb

    wandb.login(key=secret("WANDB_API_KEY"))


def setup():
    install()
    download_data()
    login_wandb()
    import torch

    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
