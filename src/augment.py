import random
import re
from collections import Counter


def _attach(text, pairs, block, block_spans, rng):
    """Put a block before or after the post, a blank line between them."""
    if rng.random() < 0.5:
        shift = len(text) + 2
        return text + "\n\n" + block, pairs + [(a + shift, b + shift) for a, b in block_spans]
    shift = len(block) + 2
    return block + "\n\n" + text, block_spans + [(a + shift, b + shift) for a, b in pairs]


#  Lines a model should leave alone: `Україні` in a slogan is not a place mention.
SLOGANS = ["Слава Україні!", "Героям слава!", "Слава ЗСУ!", "Слава нації!"]
NAME = re.compile(r"[А-ЯІЇЄҐA-Z][^\d\n]*")
FIRST_WORD = re.compile(r"[А-ЯІЇЄҐ][а-яіїєґ']{2,}")
#  An oblique-case ending: test lists names in the nominative (`Фастів - 12`), so `Фастові`, `Києва` stay out.
#  ponytail: genitive `Одеси` passes (same ending as plural `Черкаси`); pymorphy3 if list recall does not grow.
OBLIQUE = re.compile(r"(?:і|ї|у|ю|ом|ем|єм|ою|ею|ах|ях|ова|ева|єва|ого|ої|ій|ому|ім|ни|ви|пи)$")   # `України`, `Росії`


def name_block(rng, names, weights, negatives):
    """As in test: 30% a list `Чорнобиль - 19` of 3-8 lines (10% of them no name), 70% one heading `Київщина:`
    (half of them no name: `Сьогодні:`, `Слава Україні!`). No emoji: prepare() turns them into spaces anyway."""
    is_list = rng.random() < 0.3
    sep = rng.choice([" - ", " – ", " — ", ": "])
    block, spans = "", []
    for _ in range(rng.randint(3, 8) if is_list else 1):
        block += "\n" if block else ""
        number = f"{sep}{rng.randint(1, 99)}"
        if rng.random() < (0.1 if is_list else 0.5):
            word = rng.choice(negatives)
            block += (word + number if is_list else rng.choice(SLOGANS) if rng.random() < 0.3 else word + ":")
            continue
        name = rng.choices(names, weights)[0]
        lead, tail = ("", number) if is_list else rng.choice([("", ":"), ("- ", ""), ("", "")])
        spans.append((len(block) + len(lead), len(block) + len(lead) + len(name)))
        block += lead + name + tail
    return block, spans


def add_name_lines(texts, spans, share, seed):
    """Add a block of bare names to `share` of the posts that have names. Names (nominative-looking, drawn by how
    often they are labelled) and negatives (capitalised first words outside labels: `Сьогодні`) come from these
    posts, so val gets no train names."""
    counts = Counter(text[a:b] for text, pairs in zip(texts, spans) for a, b in pairs
                     if NAME.fullmatch(text, a, b) and len(text[a:b].split()) <= 3 and not OBLIQUE.search(text[a:b]))
    names = sorted(n for n, k in counts.items() if k >= 2)
    weights = [counts[n] for n in names]
    negatives = sorted({m.group() for text, pairs in zip(texts, spans) if (m := FIRST_WORD.match(text))
                        and not any(a < m.end() for a, _ in pairs)} - set(names))
    rng = random.Random(seed)
    out_texts, out_spans = [], []
    for text, pairs in zip(texts, spans):
        pairs = [(int(a), int(b)) for a, b in pairs]
        if pairs and names and rng.random() < share:
            text, pairs = _attach(text, pairs, *name_block(rng, names, weights, negatives), rng)
        out_texts.append(text)
        out_spans.append(pairs)
    return out_texts, out_spans
