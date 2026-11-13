import random
import re
from collections import Counter

STREET_TYPES = ["вул.", "вулиця", "просп.", "проспект", "бульв.", "бульвар", "пров.", "провулок"]
STREET = re.compile(r"^(?:вул\.|вулиця|просп\.|проспект|бульв\.|бульвар|пров\.|провулок)\s+([А-ЯІЇЄҐ][^,\d]{1,39}?[^\W\d_])\s*(?:,?\s*\d.*)?$")
CITIES = ["Київ", "Бровари", "Ірпінь", "Буча", "Вишгород", "Бориспіль", "Біла Церква", "Фастів", "Обухів",
          "Васильків", "Вінниця", "Львів", "Одеса", "Дніпро", "Харків"]
LEADS = ["📍", "📍 ", "Адреса: ", "▪️", "👉 ", "• ", "", ""]
TAILS = ["", "", ".", ";"]


def street_names(texts, spans):
    names = set()
    for text, pairs in zip(texts, spans):
        for a, b in pairs:
            match = STREET.match(text[int(a):int(b)])
            if match:
                names.add(match.group(1).strip())
    return sorted(names)


def house_number(rng):
    n = rng.randint(1, 200)
    return rng.choice([f"{n}", f"{n}", f"{n}-{rng.choice('АБВ')}", f"{n}{rng.choice('абв')}",
                       f"{n}/{rng.randint(1, 9)}", f"{n}-{n + rng.randint(1, 8)}"])


def address_line(rng, names):
    line, spans = rng.choice(LEADS), []
    if rng.random() < 0.3:
        city = rng.choice(CITIES)
        spans.append((len(line) + 3, len(line) + 3 + len(city)))
        line += f"м. {city}, "
    street = f"{rng.choice(STREET_TYPES)} {rng.choice(names)}"
    if rng.random() < 0.8:
        street += rng.choice([", ", ",", " "]) + house_number(rng)
    spans.append((len(line), len(line) + len(street)))
    return line + street + rng.choice(TAILS), spans


def add_addresses(texts, spans, share, seed):
    names = street_names(texts, spans)
    rng = random.Random(seed)
    out_texts, out_spans = [], []
    for text, pairs in zip(texts, spans):
        pairs = [(int(a), int(b)) for a, b in pairs]
        if rng.random() < share:
            block, block_spans = "", []
            for _ in range(rng.randint(1, 3)):
                line, line_spans = address_line(rng, names)
                block += "\n" if block else ""
                block_spans += [(len(block) + a, len(block) + b) for a, b in line_spans]
                block += line
            text, pairs = _attach(text, pairs, block, block_spans, rng)
        out_texts.append(text)
        out_spans.append(pairs)
    return out_texts, out_spans


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


def name_block(rng, names, negatives):
    """A list `Чорнобиль - 19` of 3-10 lines or one heading `Київщина:`, as in test; 20% of lines carry no name.
    No emoji: prepare() turns them into spaces anyway."""
    sep = rng.choice([" - ", " – ", " — ", ": "])
    block, spans = "", []
    for _ in range(rng.randint(3, 10) if rng.random() < 0.6 else 1):
        block += "\n" if block else ""
        if rng.random() < 0.2:
            block += rng.choice(SLOGANS) if rng.random() < 0.3 else f"{rng.choice(negatives)}{sep}{rng.randint(1, 99)}"
            continue
        name = rng.choice(names)
        lead, tail = rng.choice([("", f"{sep}{rng.randint(1, 99)}"), ("", ":"), ("- ", ""), ("", "")])
        spans.append((len(block) + len(lead), len(block) + len(lead) + len(name)))
        block += lead + name + tail
    return block, spans


def add_name_lines(texts, spans, share, seed):
    """Add a block of bare names to `share` of the posts that have names. Names and negatives (capitalised first
    words outside labels: `Сьогодні`) come from these posts, so val gets no train names.
    ponytail: names keep their case form (`Києві - 12`); lemmatise if list recall does not grow."""
    names = Counter(text[a:b] for text, pairs in zip(texts, spans) for a, b in pairs
                    if NAME.fullmatch(text, a, b) and len(text[a:b].split()) <= 3)
    names = sorted(n for n, k in names.items() if k >= 2)
    negatives = sorted({m.group() for text, pairs in zip(texts, spans) if (m := FIRST_WORD.match(text))
                        and not any(a < m.end() for a, _ in pairs)} - set(names))
    rng = random.Random(seed)
    out_texts, out_spans = [], []
    for text, pairs in zip(texts, spans):
        pairs = [(int(a), int(b)) for a, b in pairs]
        if pairs and names and rng.random() < share:
            text, pairs = _attach(text, pairs, *name_block(rng, names, negatives), rng)
        out_texts.append(text)
        out_spans.append(pairs)
    return out_texts, out_spans
