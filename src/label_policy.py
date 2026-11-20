"""How test is labelled: POLICY, steps `(text, spans) -> spans` over the original text.

relabel() is the one implementation: build_dataset runs it over the train labels, submission over the model output.
"""
import re

QUOTES = {"«": "»", "“": "”", "„": "“"}
# the punctuation around a span
EDGES = re.compile(r"[\W_]*(.*?)[\W_]*", re.S)
# «столиця» for Kyiv. Not `столиці Польщі`: a name follows
CAPITAL = re.compile(r"(?i)\bстолиц(?:я|і|ю|ею)\b(?![ \t]+(?-i:[А-ЯІЇЄҐA-Z]))")
# `французькій столиці`: the capital of another country
FOREIGN_BEFORE = re.compile(r"\b(?!україн)\w+(?:ськ|цьк|зьк)\w{1,3}[ \t]+$", re.I)
KYIV = re.compile(r"\b(?:Київ|Києва|Києві|Києву|Києвом|Киев|Киева|Киеве|Киеву|Киевом)\b", re.I)
# names that media also write in lowercase: `москви`, `рф`
LOWERABLE = re.compile(r"Москв(?:а|и|і|у|ою)|Росі(?:я|ї|ю|єю)|РФ")
# `брянської`: an adjective made from a place name, a name even in lowercase
REGION_ADJ = re.compile(r"\w+(?:ськ|цьк|зьк)(?:ої|ій|а|ий|ого|ому|ою|ие|ой|ая)")
# street type words
STREET = (r"вул\.|ул\.|вулиц(?:я|і|ю|ею)|улиц(?:а|ы|е|у|ей)|проспект(?:і|у|ом|а|е)?|просп\.|пр\.|"
          r"провул(?:ок|ку|ком)|переул(?:ок|ке|ка)|бульвар(?:і|у|ом|а|е)?|бульв\.|бул\.|узв(?:із|озі|озу|озом)")
# a span with one of these is an address: `вул. Шевченка`, `площа Ринок`
STREET_ANY = re.compile(r"(?:^|\s)(?:" + STREET + r"|площ(?:а|і|у|ею)|пл\.|майд\.|шосе|алеї|алея|алею)(?:\s|$)", re.I)
# house number after a street: `, 3`, `, 23а`, `, 69/1`. Not `, 5 людей`
HOUSE = re.compile(r",\s*\d{1,3}(?:[а-яіїєґa-z]|/\d+[а-яіїєґa-z]?)?(?=[\s,.;:)!?]|$)(?!\s*(?:року|рік|р\.|рр|людин|осіб))",
                   re.I)
# street type right before a name: `вул. Хрещатик`. No `вулиці`, `площа`: `вулиці Києва` is not a street
TYPE_BEFORE = re.compile(r"(?<!\w)(?:вул\.|ул\.|вулиц(?:я|ю|ею)|улиц(?:а|е|у|ей)|просп\.|проспект(?:і|у|ом|а|е)?"
                         r"|провул(?:ок|ку|ка)|переул(?:ок|ке|ка)|бульвар(?:і|у|а|е|ом)?|бул\.|пл\.|площади"
                         r"|набережной)\s*$", re.I)
# area type right after a name: `Дарницькому районі`. Singular only: `районах` ends a list of names
AREA_AFTER = re.compile(r"\s+(?:район(?:і|у|ом|а|е)?|област(?:ь|і|и|ю|ью)|обл\.|р-н|кра(?:й|я|е|ю|ем)"
                        r"|республ[іи]к(?:а|и|е|у|ою|ой)|республіці)(?!\w)", re.I)
# `Донецької`: an adjective, so an area type may follow. Not `Росією`
ADJECTIVE_END = re.compile(r"(?:ськ|цьк|зьк|ск|цк)\w{1,3}$")
# facility type words
FACILITY = (r"(?:собор(?:у|і|ом|а|е)?|лавр(?:а|и|і|у|ою|е)|парк(?:у|і|ом|а|е)?|сквер(?:у|і|ом|а|е)?|мост(?:у|і|ом|а|е)?"
            r"|шляхопров(?:ід|оду|оді|одом)|аеропорт(?:у|і|ом|а|е)?|аэропорт(?:у|ом|а|е)?|ТРЦ|ТЦ|ЖК)(?!\w)")
# one capitalised word
NAME_WORD = r"[А-ЯІЇЄҐA-Z][\w'’ʼ]*(?:-[А-ЯІЇЄҐA-Z]?[\w'’ʼ]+)*"
# `Михайлівського собору`: an adjective, then the type
FACILITY_AFTER = re.compile(rf"{NAME_WORD}(?:ого|ому|ий|ій|ої|ою|им|ім|ый|ая|ой|ую|(?:ськ|цьк|зьк|ов|ев|єв|ин|н)[ау])"
                            rf"\s+(?:{FACILITY}|міст(?!\w))")   # `міст` (bridge) only after an adjective
# `парку Шевченка`, `ТЦ "Аркадія"`: the type, then the name
FACILITY_BEFORE = re.compile(rf"(?<!\w)(?P<type>(?i:{FACILITY}))\s+(?:(?P<quoted>[«\"„“][^»\"”“\n]{{1,40}}[»\"”“])"
                             rf"|{NAME_WORD}(?:\s+{NAME_WORD})?)")
# first word of an organisation or a post: `поліція Києва`, `мер Ірпеня`, `полиция Москвы`
ORG = (r"поліці\w*|нацполіці\w*|поліцейськ\w*|прокуратур\w*|прокурор\w*|рад(?:а|и|і|у|ою)|міськрад\w*|облрад\w*|"
       r"суд(?:у|ом|і|и|ів)?|райсуд\w*|метрополітен\w*|міністр\w*|міністерств\w*|мер(?:а|ом|ії|ія|ію)?|"
       r"уряд\w*|президент\w*|влад(?:а|и|і|у|ою)|адміністраці\w*|департамент\w*|управлінн\w*|"
       r"служб(?:а|и|і|у|ою)|фонд\w*|посольств\w*|парламент\w*|СБУ|СБ|ДСНС|ГУНП|ЗСУ|"
       r"полици\w*|мэр(?:а|у|ом|ии|ия|ию)?|министр\w*|министерств\w*|правительств\w*|власт(?:ь|и|ей)|"
       r"администраци\w*|управлени\w*|служб(?:ы|е|ой)|суд(?:а|е|ы|ов)|госдум\w*|ФСБ|МВД|СК")
PREP = r"(?:у|в|во|на|по|з|із|зі|с|со|из|до|від|от|за|під|под|над|біля|та|і|й|и)"
# that word, up to 2 lowercase words, then the name: `міністра оборони України`. Not `поліція у Києві`
# ponytail: a verb between passes (`СБУ затримала мешканця Харкова`); pymorphy3 if it matters
ORG_BEFORE = re.compile(rf"\b(?:{ORG})(?:[ \t\xa0]+(?!{PREP}[ \t\xa0])(?-i:[а-яіїєґё][а-яіїєґё'ʼ’.-]*)){{0,2}}[ \t\xa0]+$",
                        re.I)
# `Києва`, `України`: the case of a name inside an organisation name
GENITIVE = re.compile(r"\w*[аяиіїы]")
# settlement or station type right before a name: `селі Забуччя`, `метро Почайна`. Not `міста`
SETTLEMENT_LEFT = re.compile(r"(?<!\w)(?:сел(?:о|а|і|ом|е)|селищ(?:е|а|і|ем)|смт|пгт|пос[её]л(?:ок|ке|ка|ком)|метро"
                             r"|станці(?:я|ї|ю|єю)|станци(?:я|и|ю|ей))\s+[«\"]?$", re.I)


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


def is_common_noun(label):
    """One word without a capital letter is not a name (`пляжі`, `вулиць`); `аль-Букамаль` is."""
    return (len(label.split()) == 1 and not any(c.isupper() for c in label)
            and not (CAPITAL.fullmatch(label) or "." in label or REGION_ADJ.fullmatch(label)
                     or LOWERABLE.fullmatch(label[:1].upper() + label[1:]) or label == "рф"))


def in_org_name(text, a, b):
    """Is the mention text[a:b] part of an organisation or post name?"""
    words = text[a:b].split()
    return bool(words) and bool(GENITIVE.fullmatch(words[-1])) and bool(ORG_BEFORE.search(text, max(0, a - 80), a))


def settlement_start(text, a):
    """Start of the mention at `a` with its village/station type word, if there is one."""
    m = SETTLEMENT_LEFT.search(text, max(0, a - 20), a)
    return m.start() if m else a


def clean_spans(text, spans):
    """Merge touching spans, trim edge whitespace, drop spans without letters or digits."""
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


# --- the policy: (text, spans) -> spans -----------------------------------------------------------------------
def cut_at_bracket(text, spans):
    """`России (СКР)`: a bracket is never inside a name."""
    return [(a, text.index("(", a) if "(" in text[a:b] else b) for a, b in spans]


def trim_edges(text, spans):
    """`Донецьку,` -> `Донецьку`."""
    return [(a, b) for a, b in (trim(text, a, b) for a, b in spans) if a < b]


def join_street_type(text, spans):
    """`вул. Хрещатик`, where the label leaves `вул.` outside."""
    return [(m.start() if (m := TYPE_BEFORE.search(text, max(0, a - 35), a)) else a, b) for a, b in spans]


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


def join_settlement_type(text, spans):
    """`Забуччя` -> `селі Забуччя`, `метро Почайна`; not over another name."""
    out = []
    for a, b in spans:
        start = settlement_start(text, a)
        out.append((start if not any(c < a and start < d for c, d in spans) else a, b))
    return out


def join_area_type(text, spans):
    """`Донецької` -> `Донецької області`."""
    return [(a, m.end() if ADJECTIVE_END.search(text, a, b) and (m := AREA_AFTER.match(text, b)) else b)
            for a, b in spans]


def label_capital(text, spans):
    """«столиця» is a name of its own in every mention: `Дарницькому районі столиці` is two names."""
    out = list(spans)
    for m in CAPITAL.finditer(text):
        a, b = m.span()
        if FOREIGN_BEFORE.search(text, max(0, a - 25), a):
            continue
        over = [(c, d) for c, d in out if c < b and a < d]
        rest = [p for c, d in over for p in ((c, len(text[:a].rstrip())), (d - len(text[b:d].lstrip()), d)) if p[0] < p[1]]
        out = [s for s in out if s not in over] + [(a, b)] + rest
    return out


def label_kyiv(text, spans):
    """Test labels every mention of Kyiv."""
    return spans + [m.span() for m in KYIV.finditer(text) if not any(c < m.end() and m.start() < d for c, d in spans)]


def label_facilities(text, spans):
    out = list(spans)
    for m in [*FACILITY_AFTER.finditer(text), *FACILITY_BEFORE.finditer(text)]:
        a, b = m.span()
        inside = [(c, d) for c, d in out if c < b and a < d]
        if (a, b) in inside or any(c < a or b < d for c, d in inside):
            continue                    # labelled whole already, or a span runs past the facility
        if m.re is FACILITY_AFTER and not inside:
            continue                    # `Об'єднавчого собору`, `Міжнародний аеропорт`: only extend a labelled name
        if m.re is FACILITY_BEFORE and not m["quoted"] and (inside or m["type"].lower().startswith("собор")
                                                            or not re.search(r"[аяиіїу]$", m.group())):
            continue    # `вокзалі Каїра`: a labelled name is a city; `собору Вселенського`: a council;
                        # `директор парку Олег Деркач`: a person, the name of a facility is genitive (`парку Шевченка`)
        out = [s for s in out if s not in inside] + [(a, b)]
    return out


def drop_org_names(text, spans):
    return [(a, b) for a, b in spans if not in_org_name(text, a, b)]


def drop_common_nouns(text, spans):
    return [(a, b) for a, b in spans if not is_common_noun(text[a:b])]


# trim_edges twice: the joins need a clean name, and a join can leave a quote open (`метро "Вокзальна`)
_JOINS = [cut_at_bracket, trim_edges, join_street_type, join_house_number, join_settlement_type, join_area_type,
          trim_edges]
_REST = [label_capital, label_facilities, drop_org_names, drop_common_nouns]
POLICY = _JOINS + _REST
# label_kyiv needs the ORG spans of the source to skip `«Вечірній Київ»`, so it runs on train only
TRAIN_POLICY = _JOINS + [label_kyiv] + _REST


def relabel(text, spans, steps=POLICY, stats=None):
    """Spans as test labels them; `stats` counts the texts each step changes."""
    for step in steps:
        new = step(text, spans)
        if stats is not None:
            stats[step.__name__] += new != spans
        spans = new
    return clean_spans(text, spans)
