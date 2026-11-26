"""Where the project's things live, on disk and online. Shared by src, src/rnn and scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized"     # train/val of scripts.build_dataset
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submission"

WANDB_ENTITY = "whoresandbaddies"
WANDB_PROJECT = "rnn-for-ner"                  # RNN runs and the normalized dataset
WANDB_TRANSFORMER_PROJECT = "transformers-for-ner"
KAGGLE_COMPETITION = "kse-ua-location-extraction-2026"
