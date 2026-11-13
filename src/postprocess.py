"""Model tags -> submission strings in the form of the ORIGINAL text.

The model reads `prepare(text)`, which has the same length as `text`. Token offsets
found on the prepared text are therefore valid in the original, and the answer is cut
from the original: its apostrophes, spaces and case reach the submission untouched.
"""
import re
import string

from src.text import prepare, tags_to_spans

#  Compound names split on a hyphen are glued (Києво-Печерська, Івано-Франківськ, Нью-Делі). A plain hyphen between
#  two names is a route (`Київ-Одеса`, `Брест-Ковель`): two answers, so only these left parts glue.
COMPOUND_LEFT = re.compile(r"\w*(?:ськ|цьк|зьк|ев|єв|ов|ан)о|Нью|Сан|Санкт|Сен|Усть|Ріо|Лос|Буенос")
QUOTES = {"«": "»", "“": "”", "„": "“"}
EDGES = re.compile(r"[\W_]*(.*?)[\W_]*", re.S)
CAPITAL = re.compile(r"(?i)\bстолиц(?:я|і|ю|ею)\b")
#  One lowercase word is a common noun, not a name (`общепитах`, `десанте`): LB +0.004 when it was a submission rule.
#  Kept: «столиця», glued abbreviations (`просп.Правди`), adjectival regions (`брянської ... областей`).
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


def hard_processing(text):
    """Verbatim copy of the competition metric's normalisation (the missing `f` is theirs)."""
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\b\w\b\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def glue_hyphens(text, offsets, tags):
    tags = list(tags)
    for k in range(1, len(offsets) - 1):
        (pa, pb), (a, b), (na, _) = offsets[k - 1], offsets[k], offsets[k + 1]
        if (text[a:b] in ("-", "–") and pb == a and b == na and tags[k - 1] != "O" and tags[k + 1] != "O"
                and COMPOUND_LEFT.fullmatch(text[pa:pb])):
            tags[k] = tags[k + 1] = "I"
    return tags


def trim(text, a, b):
    """Span without edge punctuation (the metric scores `Київ,` as a miss), but with the closing quote of a quote
    it opens (`метро «Лісова»`). Cuts submission strings and train labels alike."""
    m = EDGES.fullmatch(text, a, b)
    a, b = m.start(1), m.end(1)
    for open_, close in QUOTES.items():
        if text.count(open_, a, b) > text.count(close, a, b) and text[b:b + 1] == close:
            b += 1
    return a, b


def to_submission(original_text, offsets, tags):
    """`offsets` come from tokenize(prepare(original_text)); the strings come from the original."""
    out = {}
    for a, b in tags_to_spans(offsets, glue_hyphens(prepare(original_text), offsets, tags)):
        label = original_text[slice(*trim(original_text, a, b))]
        if hard_processing(label):          # an empty string counts as a false positive
            out[label] = None               # the metric compares sets
    return list(out)


def common_noun(label):
    """One word without a capital letter is not a name (`пляжі`, `вулиць`); `аль-Букамаль` is. Relabels train."""
    return (len(label.split()) == 1 and not any(c.isupper() for c in label)
            and not (CAPITAL.fullmatch(label) or "." in label or REGION_ADJ.fullmatch(label)))


def _occurrences(text, s):
    return [m.start() for m in re.finditer(re.escape(s), text)]


def full_addresses(text, locations):
    """Test labels addresses with the street type and the house number (`вул. Солом'янська, 28`); the metric drops
    digits but keeps the comma, so `Лукʼяненка` != `Лукʼяненка, 3`. LB +0.004."""
    out = []
    for loc in locations:
        new = loc
        for k in _occurrences(text, loc):
            if any(o != loc and loc in o and any(j <= k < j + len(o) for j in _occurrences(text, o)) for o in locations):
                continue                                         # this mention is inside a longer answer
            a, b = k, k + len(loc)
            left = STREET_LEFT.search(text[max(0, a - 15):a])
            if left and not STREET_ANY.search(loc):
                a -= len(left.group())                           # `Мілютенка` -> `вул. Мілютенка`
            house = HOUSE.match(text, b)
            street = (left or STREET_ANY.search(loc) or STREET_FORM.search(loc)
                      or re.search(r"(?:" + STREET + r")\s*" + re.escape(loc), text, re.I)
                      or (house and re.search(r"[^\d,\s]", house.group()[1:])))
            if street and house and not re.search(r"\d", loc):
                b = house.end()                                  # `Лукʼяненка` -> `Лукʼяненка, 3`
            elif re.search(r",\s*\d+$", loc) and (rest := HOUSE_REST.match(text, b)):
                b = rest.end()                                   # `Відрадний, 69` -> `Відрадний, 69/1`
            new = text[a:b]
            break
        out.append(new)
    # `Бориспільська` next to `Бориспільська, 9` and never alone in the text: a cut copy of the address
    cut = {o for o in out if _occurrences(text, o) and all(
        any(p.startswith(o + ",") and text.startswith(p, k) for p in out) for k in _occurrences(text, o))}
    return list(dict.fromkeys(o for o in out if o not in cut))


def in_org_name(text, a, b):
    """Is the mention text[a:b] part of an organisation or post name? Also relabels train in build_dataset."""
    words = text[a:b].split()
    return bool(words) and bool(GENITIVE.fullmatch(words[-1])) and bool(ORG_BEFORE.search(text, max(0, a - 80), a))


def settlement_start(text, a):
    """Start of the mention at `a` with its village/station type word, if there is one. Also used in build_dataset."""
    m = SETTLEMENT_LEFT.search(text, max(0, a - 20), a)
    return m.start() if m else a


def test_policy(original_text, locations):
    """How test is labelled where the training sources differ: full addresses, «столиця» is a location (LB +0.006),
    a city/country in an organisation name is not, when every its mention is one (LB +0.0073), a village/station
    keeps its type word (LB +0.0058).
    Only for the submission, so val/f1 keeps measuring the model against its own labels."""
    text = original_text
    locations = full_addresses(text, locations)
    found = [text[m.start():m.end()] for m in CAPITAL.finditer(prepare(text))]
    locations = [loc for loc in dict.fromkeys(locations + found)
                 if not (k := _occurrences(text, loc)) or not all(in_org_name(text, i, i + len(loc)) for i in k)]
    return list(dict.fromkeys(m for loc in locations for m in _with_settlement_type(text, loc)))


def _with_settlement_type(text, loc):
    """Every mention of `loc`, with its type word where it has one; not inside a longer word (`Киев` in `Киев-...`)."""
    whole = [k for k in _occurrences(text, loc) if not re.match(r"[\w-]", text[k + len(loc):k + len(loc) + 1])]
    return [text[settlement_start(text, k):k + len(loc)] for k in whole] or [loc]


if __name__ == "__main__":
    def org(text, label):
        k = text.index(label)
        return in_org_name(text, k, k + len(label))

    assert org("Патрульна поліція Києва повідомила", "Києва")
    assert org("заява міністра оборони України", "України")
    assert org("Рада оборони Києва", "Києва")
    assert not org("поліція у Києві", "Києві")                    # preposition: a place
    assert not org("Нацполіція\n\nКиївщина", "Київщина")         # new line: a heading
    assert not org("Верховний суд повернув Києву", "Києву")      # not genitive
    assert not org("Україна переможе", "Україна")
    assert org("сообщила полиция Москвы", "Москвы")
    assert not org("полиция в Москве", "Москве")
    assert not common_noun("аль-Букамаль") and common_noun("пляжі") and not common_noun("столиці")
    assert trim("у Київ, а", 2, 7) == (2, 6)
    assert trim("метро «Лісова» є", 0, 14) == (0, 14)       # closing quote of an opened one is kept
    assert trim("метро «Лісова» є", 6, 14) == (7, 13)       # edge quotes go
    assert test_policy("суд Києва, а потім Києва", ["Києва"]) == ["Києва"]   # one mention is outside an org name
    assert test_policy("Поліція Києва", ["Києва"]) == []
    assert test_policy("у селі Забуччя", ["Забуччя"]) == ["селі Забуччя"]
    assert test_policy("на метро «Лісова»", ["Лісова»"]) == ["метро «Лісова»"]
    assert test_policy("гостям міста Києва", ["Києва"]) == ["Києва"]           # city type stays out
    assert test_policy("Почайна. Метро Почайна", ["Почайна"]) == ["Почайна", "Метро Почайна"]
    assert test_policy("Киев. станции Киев-Пассажирский", ["Киев"]) == ["Киев"]   # not inside a longer name
    print("ok")


