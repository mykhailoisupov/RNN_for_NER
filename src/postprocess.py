"""Model tags -> submission strings in the form of the ORIGINAL text (steps 44-47).

The model reads `prepare(text)`, which has the same length as `text`. Token offsets
found on the prepared text are therefore valid in the original, and the answer is cut
from the original: its apostrophes, spaces and case reach the submission untouched.

    python -m src.postprocess     # self-check
"""
import re
import string

from src.data import tags_to_spans


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
        label = re.sub(r"^[\W_]+|[\W_]+$", "", label)   # step 45: `Київ,` scores 0, `Київ` scores 1
        if hard_processing(label):                      # step 46: an empty string counts as a false positive
            out[label] = None                           # step 47: one string per name
    return list(out)


def check():
    import ast

    import pandas as pd

    from src.data import DATA, normalize, parse_spans, spans_to_bio, tokenize

    original = "🫶Київ, вул.\xa0Хрещатик, 28"
    offsets = tokenize(normalize(original))
    tags, _ = spans_to_bio(offsets, [(1, 5), (7, 23)])
    assert to_submission(original, offsets, tags) == ["Київ", "вул.\xa0Хрещатик, 28"]   # original form, not normalised

    offsets = tokenize("Київ, 2024 і Київ,")
    assert to_submission("Київ, 2024 і Київ,", offsets, ["B", "I", "B", "O", "B", "I"]) == ["Київ"]

    # Reference tags of the labelled sample must come back as the reference strings.
    sample = pd.read_csv(DATA / "labeling_sample.csv")
    mismatches = []
    for text, markers, locations in zip(sample.text, sample.loc_markers, sample.locations):
        offsets = tokenize(normalize(text))
        tags, _ = spans_to_bio(offsets, parse_spans(markers))
        got = {hard_processing(s) for s in to_submission(text, offsets, tags)}
        expected = {hard_processing(s) for s in ast.literal_eval(locations)}
        if got != expected:
            mismatches.append((sorted(expected - got), sorted(got - expected)))
    assert not mismatches, mismatches
    print(f"postprocess.py: all checks passed, {len(sample)} sample posts reproduced exactly")


if __name__ == "__main__":
    check()
