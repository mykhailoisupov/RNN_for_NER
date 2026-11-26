"""Any HF encoder + a linear O/B/I head on every subword: AutoModelForTokenClassification, the same for XLM-R,
mDeBERTa, mBERT."""
from transformers import AutoModelForTokenClassification

from src.tags import TAGS


def load_model(name):
    """Pretrained encoder from the hub with a new head, or a fine-tuned one from its run directory."""
    return AutoModelForTokenClassification.from_pretrained(
        name, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id={t: i for i, t in enumerate(TAGS)})
