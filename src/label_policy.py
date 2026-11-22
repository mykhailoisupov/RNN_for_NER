"""How test is labelled: POLICY, steps `(text, spans) -> spans` over the original text.

relabel() is the one implementation: build_dataset runs it over the train labels (TRAIN_POLICY there), submission
over the model output. Sections follow the order of POLICY; each holds the patterns its step reads.
"""
import re
from functools import cache, lru_cache

import pymorphy3

# street type words; also the `street` feature of build_dataset and EDA
STREET = (r"(?:вул\.|ул\.|вулиц(?:я|і|ю|е|ею)|улиц(?:а|ы|е|у|ей)|проспект(?:і|у|ом|а|е)?|просп\.|пр\.|"
          r"провул(?:ок|ку|ком)|переул(?:ок|ке|ка)|бульвар(?:і|у|ом|а|е)?|бульв\.|бул\.|узв(?:із|озі|озу|озом)|"
          r"площ(?:а|і|у|ею|адь|ади)|пл\.|набережн\w*)")


# --- cut_at_bracket ---------------------------------------------------------------------------------------------
def cut_at_bracket(text, spans):
    """`России (СКР)`: a bracket is never inside a name."""
    return [(a, text.index("(", a) if "(" in text[a:b] else b) for a, b in spans]


# --- trim_edges -------------------------------------------------------------------------------------------------
QUOTES = {"«": "»", "“": "”", "„": "“"}
EDGES = re.compile(r"[\W_]*(.*?)[\W_]*", re.S)   # the punctuation around a span


def trim(text, a, b):
    """Span without edge punctuation (the metric scores `Київ,` as a miss), but with the closing quote of a quote
    it opens (`метро «Лісова»`). Cuts submission strings and train labels alike."""
    m = EDGES.fullmatch(text, a, b)
    a, b = m.start(1), m.end(1)
    for open_, close in QUOTES.items():
        if text.count(open_, a, b) > text.count(close, a, b) and text[b:b + 1] == close:
            b += 1
    if text.count('"', a, b) % 2 and text[b:b + 1] == '"':   # a straight quote opens and closes alike: `ТЦ "Ocean Plaza"`
        b += 1
    return a, b


def trim_edges(text, spans):
    """`Донецьку,` -> `Донецьку`."""
    return [(a, b) for a, b in (trim(text, a, b) for a, b in spans) if a < b]


# --- join_street_type -------------------------------------------------------------------------------------------
# street type right before a name: `вул. Хрещатик`. No `вулиці`, `площа`: `вулиці Києва` is not a street
TYPE_BEFORE = re.compile(r"(?<!\w)(?:вул\.|ул\.|вулиц(?:я|ю|ею)|улиц(?:а|е|у|ей)|просп\.|проспект(?:і|у|ом|а|е)?"
                         r"|провул(?:ок|ку|ка)|переул(?:ок|ке|ка)|бульвар(?:і|у|а|е|ом)?|бул\.|пл\.|площади"
                         r"|набережной)\s*$", re.I)


def join_street_type(text, spans):
    """`вул. Хрещатик`, where the label leaves `вул.` outside."""
    return [(m.start() if (m := TYPE_BEFORE.search(text, max(0, a - 35), a)) else a, b) for a, b in spans]


# --- join_house_number ------------------------------------------------------------------------------------------
# a span with one of these is an address: `вул. Шевченка`, `площа Ринок`
STREET_ANY = re.compile(rf"(?:^|\s)(?:{STREET}|майд\.|шосе|алеї|алея|алею)(?:\s|$)", re.I)
# house number after a street: `, 3`, `, 23а`, `, 69/1`. Not `, 5 людей`
HOUSE = re.compile(r",\s*\d{1,3}(?:[а-яіїєґa-z]|/\d+[а-яіїєґa-z]?)?(?=[\s,.;:)!?]|$)(?!\s*(?:року|рік|р\.|рр|людин|осіб))",
                   re.I)


def join_house_number(text, spans):
    """`вул. С. Бандери` -> `вул. С. Бандери, 23`; only after a street type: `Шевченка, 50 дронів` is not an address."""
    out = []
    for a, b in spans:
        while b < len(text) and text[b - 1].isdigit() and text[b].isdigit():   # `Галицька, 9|3`
            b += 1
        house = HOUSE.match(text, b)
        if house and STREET_ANY.search(text[a:b]) and not re.search(r"\d", text[a:b]):
            b = house.end()
        out.append((a, b))
    return out


# --- join_settlement_type ---------------------------------------------------------------------------------------
# settlement or station type right before a name: `селі Забуччя`, `метро Почайна`. Not `міста`: LB -0.0047 with it
SETTLEMENT_LEFT = re.compile(r"(?<!\w)(?:сел(?:о|а|і|ом|е)|селищ(?:е|а|і|ем)|смт|пгт|пос[её]л(?:ок|ке|ка|ком)|метро"
                             r"|станці(?:я|ї|ю|єю)|станци(?:я|и|ю|ей))\s+[«\"]?$", re.I)


def join_settlement_type(text, spans):
    """`Забуччя` -> `селі Забуччя`, `метро Почайна`: the type word within 20 characters to the left; not over
    another name."""
    out = []
    for a, b in spans:
        m = SETTLEMENT_LEFT.search(text, max(0, a - 20), a)
        start = m.start() if m else a
        out.append((start if not any(c < a and start < d for c, d in spans) else a, b))
    return out


# --- join_area_type ---------------------------------------------------------------------------------------------
# area type right after a name: `Дарницькому районі`. Singular only: `районах` ends a list of names
AREA_AFTER = re.compile(r"\s+(?:район(?:і|у|ом|а|е)?|област(?:ь|і|и|ю|ью)|обл\.|р-н|кра(?:й|я|е|ю|ем)"
                        r"|республ[іи]к(?:а|и|е|у|ою|ой)|республіці)(?!\w)", re.I)
# `Донецької`: an adjective, so an area type may follow. Not `Росією`
ADJECTIVE_END = re.compile(r"(?:ськ|цьк|зьк|ск|цк)\w{1,3}$")


def join_area_type(text, spans):
    """`Донецької` -> `Донецької області`."""
    return [(a, m.end() if ADJECTIVE_END.search(text, a, b) and (m := AREA_AFTER.match(text, b)) else b)
            for a, b in spans]


# --- drop_org_names ---------------------------------------------------------------------------------------------
# first word of an organisation or a post: `поліція Києва`, `мер Ірпеня`, `полиция Москвы`
ORG = (r"поліці\w*|нацполіці\w*|поліцейськ\w*|прокуратур\w*|прокурор\w*|рад(?:а|и|і|у|ою)|міськрад\w*|облрад\w*|"
       r"суд(?:у|ом|і|и|ів)?|райсуд\w*|метрополітен\w*|міністр\w*|міністерств\w*|мер(?:а|ом|ії|ія|ію)?|"
       r"уряд\w*|президент\w*|влад(?:а|и|і|у|ою)|адміністраці\w*|департамент\w*|управлінн\w*|"
       r"служб(?:а|и|і|у|ою)|фонд\w*|посольств\w*|парламент\w*|СБУ|СБ|ДСНС|ГУНП|ЗСУ|"
       r"полици\w*|мэр(?:а|у|ом|ии|ия|ию)?|министр\w*|министерств\w*|правительств\w*|власт(?:ь|и|ей)|"
       r"администраци\w*|управлени\w*|служб(?:ы|е|ой)|суд(?:а|е|ы|ов)|госдум\w*|ФСБ|МВД|СК")
PREP = r"(?:у|в|во|на|по|з|із|зі|с|со|из|до|від|от|за|під|под|над|біля|та|і|й|и)"
# that word, up to 2 lowercase words, then the name: `міністра оборони України`. Not `поліція у Києві`
ORG_BEFORE = re.compile(rf"\b(?:{ORG})(?:[ \t\xa0]+(?!{PREP}[ \t\xa0])(?-i:[а-яіїєґё][а-яіїєґё'ʼ’.-]*)){{0,2}}[ \t\xa0]+$",
                        re.I)


@cache
def _analyzer(lang):
    return pymorphy3.MorphAnalyzer(lang=lang)


@lru_cache(maxsize=None)
def _tags(word):
    """pymorphy3 tags of every reading of a word: Ukrainian, Russian when only the Russian dictionary knows it."""
    uk, ru = _analyzer("uk"), _analyzer("ru")
    morph = ru if not uk.word_is_known(word) and ru.word_is_known(word) else uk
    return [p.tag for p in morph.parse(word)]


def can_be(word, *grammemes):
    """Has the word a reading with all these grammemes: can_be(`оборони`, "NOUN", "gent")."""
    return any(set(grammemes) <= tag.grammemes for tag in _tags(word))


def in_org_name(text, a, b):
    """Is the mention text[a:b] part of an organisation or post name: a genitive after an ORG word, with only
    genitive nouns or adjectives between (`міністра оборони України`; not `СБУ затримала мешканця Харкова`)."""
    words = text[a:b].split()
    org = ORG_BEFORE.search(text, max(0, a - 80), a)
    return bool(words and org and can_be(words[-1], "gent")
                and all(can_be(w, "NOUN", "gent") or can_be(w, "ADJF", "gent")
                        for w in (w.strip(".-'’ʼ") for w in org.group().split()[1:])))


def drop_org_names(text, spans):
    return [(a, b) for a, b in spans if not in_org_name(text, a, b)]


# --- drop_common_nouns ------------------------------------------------------------------------------------------
# «столиця», Kyiv in test: kept in lowercase. Not labelled by the policy any more (LB +0.0049 without, 2026-11-22)
CAPITAL = re.compile(r"(?i)столиц(?:я|і|ю|ею)")
# `брянської`: an adjective made from a place name, a name even in lowercase
REGION_ADJ = re.compile(r"\w+(?:ськ|цьк|зьк)(?:ої|ій|а|ий|ого|ому|ою|ие|ой|ая)")
# names that media also write in lowercase: `москви`, `рф`
LOWERABLE = re.compile(r"Москв(?:а|и|і|у|ою)|Росі(?:я|ї|ю|єю)|РФ")


def is_common_noun(label):
    """One word without a capital letter is not a name (`пляжі`, `вулиць`); `аль-Букамаль` is. Except what test
    labels in lowercase: `столиці`, `обл.`, `брянської`, `москви`, `рф`."""
    return (len(label.split()) == 1 and not any(c.isupper() for c in label)
            and not (CAPITAL.fullmatch(label) or "." in label or REGION_ADJ.fullmatch(label)
                     or LOWERABLE.fullmatch(label[:1].upper() + label[1:]) or label == "рф"))


def drop_common_nouns(text, spans):
    return [(a, b) for a, b in spans if not is_common_noun(text[a:b])]


# --- the policy -------------------------------------------------------------------------------------------------
def clean_spans(text, spans):
    """Tidy a span list: sorted, overlapping or touching spans merged, edge whitespace trimmed, spans without
    letters or digits dropped. Not rampart's merge of neighbours across a connector."""
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    cleaned = []
    for a, b in merged:
        chunk = text[a:b]
        a, b = a + len(chunk) - len(chunk.lstrip()), b - (len(chunk) - len(chunk.rstrip()))
        if a < b and any(c.isalnum() for c in text[a:b]):
            cleaned.append((a, b))
    return cleaned


# trim_edges twice: the joins need a clean name, and a join can leave a quote open (`метро "Вокзальна`)
# LB 2026-11-22, best ensemble: without join_settlement_type -0.0029, without drop_org_names -0.0029; label_capital
# and label_facilities cost +0.0049 and +0.0019 and are gone; join_house_number moved 0 and runs on train only.
JOINS = [cut_at_bracket, trim_edges, join_street_type, join_settlement_type, join_area_type, trim_edges]
REST = [drop_org_names, drop_common_nouns]
POLICY = JOINS + REST


def relabel(text, spans, steps=POLICY, stats=None):
    """Spans as test labels them; `stats` counts the texts each step changes."""
    for step in steps:
        new = step(text, spans)
        if stats is not None:
            stats[step.__name__] += new != spans
        spans = new
    return clean_spans(text, spans)
