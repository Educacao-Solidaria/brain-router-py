"""Separação de texto em parágrafos e sentenças (pt-BR), por posição no texto original.

As funções devolvem `Span = (início, fim)` em vez de strings: quem chama recorta
`text[início:fim]` e sabe exatamente de onde cada pedaço saiu. As pontas nunca têm
espaço em branco, e nenhum span é vazio.

Uma sentença termina em `.`, `!`, `?` ou `…` (aspas e parênteses de fechamento juntos)
quando o que vem depois do espaço começa com maiúscula, dígito ou abertura de aspas,
parêntese ou travessão. O ponto não fecha a sentença depois de uma abreviação comum
("Sr.", "Dra.", "art.", "fls."), de uma inicial ("J. Silva"), de uma sigla com pontos
("S.A.") ou de um número de lista ("1."). Na dúvida, não corta: juntar duas frases
custa menos que partir uma ao meio.
"""

import re

Span = tuple[int, int]

# Sem "etc.": ele também fecha frase, e aí a regra da maiúscula decide. Agrupadas por tipo
# numa string: como lista literal, o formatador poria uma por linha (~70 linhas).
ABBREVIATIONS = frozenset(
    (  # noqa: SIM905
        # tratamento e títulos
        "sr sra srs sras srta dr dra drs dras prof profa profs profas eng enga arq adv "
        "exmo exma ilmo ilma sto sta jr sa exa sas exas "
        # texto legal e referências
        "art arts inc incs al alín par parág cap caps tít dec res port nº núm num "
        "pág págs pag pp fl fls vol vols ed op cit cf ib ibid obs ex "
        # medidas, endereços e empresas
        "aprox máx mín séc tel cel av ltda cia"
    ).split()
)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# Pontuação final + fechamentos, desde que seguida de espaço (o ponto de "3.5" não conta).
_TERMINATOR = re.compile(r"[.!?…]+[\"'”’»)\]]*(?=\s)")
_OPENERS = frozenset("\"'“‘«([—–-")
_LEADING_PUNCTUATION = "\"'“‘«(["


def _strip(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_paragraphs(text: str) -> list[Span]:
    """Parágrafos separados por linha em branco (uma ou mais)."""
    spans: list[Span] = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        spans.append(_strip(text, start, match.start()))
        start = match.end()
    spans.append(_strip(text, start, len(text)))
    return [span for span in spans if span[0] < span[1]]


def _is_abbreviation(text: str, dot: int, floor: int) -> bool:
    """O ponto em `text[dot]` fecha uma abreviação, e não a sentença?"""
    start = dot
    while start > floor and not text[start - 1].isspace():
        start -= 1
    token = text[start:dot].lstrip(_LEADING_PUNCTUATION).lower()
    if not token:
        return False
    # "Profª." e "Drª." são a mesma abreviação; "1º." não (o número ordinal fecha frase).
    if token in ABBREVIATIONS or token.rstrip("ªº") in ABBREVIATIONS or token.isdigit():
        return True
    # Inicial ("J.") ou sigla com pontos ("S.A", "e.g"): todas as partes com uma letra.
    return all(len(part) == 1 and part.isalpha() for part in token.split("."))


def split_sentences(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Sentenças de `text[start:end]`, com spans relativos a `text` inteiro."""
    end = len(text) if end is None else end
    spans: list[Span] = []
    begin = start
    for match in _TERMINATOR.finditer(text, start, end):
        after = match.end()
        while after < end and text[after].isspace():
            after += 1
        if after == end:
            break
        if text[after] not in _OPENERS and not (text[after].isupper() or text[after].isdigit()):
            continue
        punctuation = match.group().rstrip("\"'”’»)]")
        if punctuation == "." and _is_abbreviation(text, match.start(), begin):
            continue
        spans.append(_strip(text, begin, match.end()))
        begin = after
    spans.append(_strip(text, begin, end))
    return [span for span in spans if span[0] < span[1]]
