"""Checks of the span repairs and the capitals augmentation.  python tests/test_submission.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # an installed `tests` package shadows `python -m`

from scripts.build_dataset import add_caps
from src.submission import drop_hashtags, split_at_newlines, trim_edges
from src.tags import aligned
from src.text import tokenize


def test_split_at_newlines():
    text = "на Межовій\n\nКиївщина: тихо"
    assert trim_edges(text, split_at_newlines(text, [(3, 20)])) == [(3, 10), (12, 20)]
    assert split_at_newlines(text, [(3, 10)]) == [(3, 10)]


def test_drop_hashtags():
    text = "у Протасів Яр #ПротасівЯр #Роман_Ратушний та #Kyivpatrol"
    spans = [(2, 13), (15, 25), (33, 41), (46, 56)]          # the third starts inside a hashtag
    assert [text[a:b] for a, b in drop_hashtags(text, spans)] == ["Протасів Яр"]


def test_add_caps():
    text = "Сьогодні у Києві дуже тепло.\nПогода гарна\nЗавтра в Одесі буде дощ, кажуть синоптики."
    texts = [text] * 300
    spans = [[(text.index(name), text.index(name) + len(name)) for name in ("Києві", "Одесі")]] * 300
    out = add_caps(texts, spans, 0.5, 0.5, 0.5, 0)
    assert all(len(a) == len(b) and a.lower() == b.lower() for a, b in zip(out, texts))
    assert all(aligned(tokenize(t), s) for t, s in zip(out, spans))
    assert any("у КИЄВІ дуже" in t for t in out)                     # a name alone
    assert any("ПОГОДА ГАРНА" in t for t in out)                     # a sentence
    changed = [[a for a, b in zip(text.split(), t.split()) if a != b] for t in out]
    assert ["дуже"] in changed or ["тепло."] in changed                # one plain word alone
    assert 50 < sum(a != b for a, b in zip(out, texts)) < 300


if __name__ == "__main__":
    for check in (test_split_at_newlines, test_drop_hashtags, test_add_caps):
        check()
    print("ok")
