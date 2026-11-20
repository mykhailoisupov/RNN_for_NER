"""How test is labelled: regexes and checks over a span of the original text.

build_dataset relabels train with them, submission applies them to the model output.
"""
import re

QUOTES = {"«": "»", "“": "”", "„": "“"}
EDGES = re.compile(r"[\W_]*(.*?)[\W_]*", re.S)
CAPITAL = re.compile(r"(?i)\bстолиц(?:я|і|ю|ею)\b")
#  One lowercase word is a common noun, not a name (`общепитах`, `десанте`): LB +0.004 as a submission rule.
#  Kept: «столиця», glued abbreviations (`просп.Правди`), adjectival regions (`брянської ... областей`).
#  Written in lowercase in Ukrainian media (`москви`, `рф`): train has 20% of them lowercased, so not common nouns.
LOWERABLE = re.compile(r"Москв(?:а|и|і|у|ою)|Росі(?:я|ї|ю|єю)|РФ")
REGION_ADJ = re.compile(r"\w+(?:ськ|цьк|зьк)(?:ої|ій|а|ий|ого|ому|ою|ие|ой|ая)")
#  Street types that mark an address (`площа Ринок, 1` too) / that are added to the left of a street name.
#  Not added: `площа`, `шосе` (`площа Київської області` is an area), plural `вулиць`.
STREET = (r"вул\.|ул\.|вулиц(?:я|і|ю|ею)|улиц(?:а|ы|е|у|ей)|проспект(?:і|у|ом|а|е)?|просп\.|пр\.|"
          r"провул(?:ок|ку|ком)|переул(?:ок|ке|ка)|бульвар(?:і|у|ом|а|е)?|бульв\.|бул\.|узв(?:із|озі|озу|озом)")
STREET_ANY = re.compile(r"(?:^|\s)(?:" + STREET + r"|площ(?:а|і|у|ею)|пл\.|шосе|алеї|алея|алею)(?:\s|$)", re.I)
STREET_LEFT = re.compile(r"(?:" + STREET + r")\s*$", re.I)
#  `, 3` / `, 23а` / `, 69/1` after a street: 1-3 digits, not `50-річний`, not a year, not `5 людей`
HOUSE = re.compile(r",\s*\d{1,3}(?:[а-яіїєґa-z]|/\d+[а-яіїєґa-z]?)?(?=[\s,.;:)!?]|$)(?!\s*(?:року|рік|р\.|рр|людин|осіб))",
                   re.I)
HOUSE_REST = re.compile(r"(?:/\d+)?[а-яіїєґa-z]?(?=[\s,.;:)!?]|$)", re.I)
#  street-shaped names: after a person (Лукʼяненка) or adjectival (Бориспільська)
STREET_FORM = re.compile(r"\w+(?:енка|ська|цька|зька|ського|цького|зького)$")
#  Test does not label a city/country inside an organisation or post name (`поліція Києва`, `міністра оборони
#  України`, `мер Ірпеня`): LB +0.0073. An organisation word, 0-2 words on the same line that are not prepositions,
#  then a genitive name. `поліція у Києві` and `суд повернув Києву` stay locations.
ORG = (r"поліці\w*|нацполіці\w*|поліцейськ\w*|прокуратур\w*|прокурор\w*|рад(?:а|и|і|у|ою)|міськрад\w*|облрад\w*|"
       r"суд(?:у|ом|і|и|ів)?|райсуд\w*|метрополітен\w*|міністр\w*|міністерств\w*|мер(?:а|ом|ії|ія|ію)?|"
       r"уряд\w*|президент\w*|влад(?:а|и|і|у|ою)|адміністраці\w*|департамент\w*|управлінн\w*|"
       r"служб(?:а|и|і|у|ою)|фонд\w*|посольств\w*|парламент\w*|СБУ|СБ|ДСНС|ГУНП|ЗСУ|"
       #  ru: `полиция Москвы`, `мэр Москвы`, `правительства России`
       r"полици\w*|мэр(?:а|у|ом|ии|ия|ию)?|министр\w*|министерств\w*|правительств\w*|власт(?:ь|и|ей)|"
       r"администраци\w*|управлени\w*|служб(?:ы|е|ой)|суд(?:а|е|ы|ов)|госдум\w*|ФСБ|МВД|СК")
PREP = r"(?:у|в|во|на|по|з|із|зі|с|со|из|до|від|от|за|під|под|над|біля|та|і|й|и)"
ORG_BEFORE = re.compile(rf"\b(?:{ORG})(?:[ \t\xa0]+(?!{PREP}[ \t\xa0])[\w'ʼ’.-]+){{0,2}}[ \t\xa0]+$", re.I)
GENITIVE = re.compile(r"\w*[аяиіїы]")
#  Test labels a village/station with its type word (`селі Забуччя`, `метро Почайна`): LB +0.0058.
#  Not `міста Києва`: adding `міст*` cost −0.0047. `с.`/`м.` do not matter, the metric drops one-letter words.
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
