import random
import re

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
            if rng.random() < 0.5:
                shift = len(text) + 2
                text, pairs = text + "\n\n" + block, pairs + [(a + shift, b + shift) for a, b in block_spans]
            else:
                shift = len(block) + 2
                text, pairs = block + "\n\n" + text, block_spans + [(a + shift, b + shift) for a, b in pairs]
        out_texts.append(text)
        out_spans.append(pairs)
    return out_texts, out_spans
