"""Model tags -> submission strings in the form of the ORIGINAL text.

The model reads `prepare(text)`, which has the same length as `text`. Token offsets
found on the prepared text are therefore valid in the original, and the answer is cut
from the original: its apostrophes, spaces and case reach the submission untouched.
"""
import re
import string

from src.text import tags_to_spans


def hard_processing(text):
    """Verbatim copy of the competition metric's normalisation (the missing `f` is theirs)."""
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\b\w\b\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def bio_to_strings(original_text, offsets, tags):
    return [original_text[a:b] for a, b in tags_to_spans(offsets, tags)]


def to_submission(original_text, offsets, tags):
    """`offsets` come from tokenize(prepare(original_text)); the strings come from the original."""
    out = {}
    for label in bio_to_strings(original_text, offsets, tags):
        label = re.sub(r"^[\W_]+|[\W_]+$", "", label)   # the metric scores `Київ,` as a miss
        if hard_processing(label):                      # an empty string counts as a false positive
            out[label] = None                           # the metric compares sets
    return list(out)
