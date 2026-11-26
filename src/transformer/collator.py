"""Windows of src.transformer.dataset -> a batch the model reads: input_ids, attention_mask, labels, padded to the
longest window. Only these keys: HF Trainer calls model(**batch).

    python -m src.transformer.collator    # check: shapes, mask and padding of a batch
"""
import numpy as np
import torch

from src.tags import IGNORE


def pad(rows, value, length):
    """Rows of different lengths -> a [rows, length] LongTensor, the tail of every row filled with `value`."""
    out = np.full((len(rows), length), value, np.int64)
    for k, row in enumerate(rows):
        out[k, :len(row)] = row
    return torch.from_numpy(out)


class Collator:
    def __init__(self, pad_token_id, multiple_of=8):
        self.pad_token_id = pad_token_id
        self.multiple_of = multiple_of   # tensor cores of fp16 work on multiples of 8

    def __call__(self, windows):
        lengths = [len(w["input_ids"]) for w in windows]
        length = -(-max(lengths) // self.multiple_of) * self.multiple_of   # rounded up
        return {
            "input_ids": pad([w["input_ids"] for w in windows], self.pad_token_id, length),
            "attention_mask": pad([np.ones(n) for n in lengths], 0, length),
            "labels": pad([w["labels"] for w in windows], IGNORE, length),
        }


# --- check ------------------------------------------------------------------------------------------------------
def _check(name="xlm-roberta-base"):
    from src.dataset import load_eval
    from src.transformer.dataset import encode, load_tokenizer

    tokenizer = load_tokenizer(name)
    windows = encode(load_eval()["sample"][:8], tokenizer, max_length=512, stride=128)
    batch = Collator(tokenizer.pad_token_id)(windows)
    lengths = [len(w["input_ids"]) for w in windows]
    print({k: tuple(v.shape) for k, v in batch.items()}, "window lengths", lengths)

    assert all(v.dtype == torch.long for v in batch.values())
    assert batch["attention_mask"].sum(1).tolist() == lengths
    pads = batch["attention_mask"] == 0
    assert (batch["input_ids"][pads] == tokenizer.pad_token_id).all() and (batch["labels"][pads] == IGNORE).all()
    for k, w in enumerate(windows):
        assert batch["input_ids"][k, :lengths[k]].tolist() == w["input_ids"].tolist()
    print("ok")


if __name__ == "__main__":
    _check()
