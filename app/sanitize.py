"""Sanitização de texto antes de gravar no Postgres e de mandar ao modelo de embedding.

`sanitize_text` devolve um texto que o Postgres aceita em `text`, que codifica em UTF-8
sem erro, que a busca textual separa em palavras certas e que não esconde nada de quem lê:

- Surrogate solto (de JSON com "\\ud800" ou de bytes lidos com `surrogateescape`) vira
  U+FFFD, e um par de surrogates separado vira o caractere que ele codifica. Sem isso o
  encode UTF-8 levanta `UnicodeEncodeError`: no httpx, antes de o pedido de embedding sair.
- Quebra de linha vira `\\n`: `\\r\\n`, `\\r` e as demais fronteiras do `str.splitlines`
  (VT, FF, separadores de arquivo, grupo e registro, NEL, U+2028, U+2029). Trocar em vez
  de apagar: "fim da página\\fInício" não pode virar "fim da páginaInício". O separador
  de unidade (0x1F) vira espaço, pelo mesmo motivo.
- Os demais controles C0 e C1 saem; `\\t` e `\\n` ficam. O NUL é o caso grave: o Postgres
  recusa 0x00 em `text`, e um NUL derruba o COPY do lote inteiro.
- Invisíveis saem: zero-width (ZWSP, ZWNJ, ZWJ, word joiner), BOM, soft hyphen, marcas e
  overrides bidi (o "Trojan Source": o texto exibido não é o texto lido), tags Unicode e
  seletores de variação suplementares, que carregam texto invisível que o modelo lê
  ("ASCII smuggling"). Um "ma\\u200btrícula" também não casaria com "matrícula" na busca
  textual. O ZWJ sai junto: em pt-BR ele só junta emoji (a família vira três pessoas).
- NFC no fim, para que o "é" composto e o "e" + acento combinante sejam a mesma sequência
  (busca textual e chave do cache de embeddings dependem disso). NFKC não: ele troca o
  caractere por um "equivalente" e muda o texto. "nº 3" viraria "no 3" (e "no" é stopword
  em português), "Art. 5º, § 2º" viraria "Art. 5o, § 2o", "m²" viraria "m2".

Idempotente: `sanitize_text(sanitize_text(x)) == sanitize_text(x)`.

Onde aplicar: na fronteira de entrada, antes de qualquer outra etapa. `content` e `title`
do `index_document` (PR 34) passam por aqui antes do `chunk_text`, para que as posições de
cada chunk apontem para o texto que foi gravado; o `query` do `hybrid_search` (PRs 30 e
32), antes do embedding e da busca textual.
"""

import unicodedata

# Fronteiras de linha do str.splitlines, fora o próprio \n (o \r\n é trocado antes).
_LINE_BREAKS = (0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)
_CONTROLS = (*range(0x00, 0x09), *range(0x0E, 0x1C), 0x7F, *range(0x80, 0x85), *range(0x86, 0xA0))
_INVISIBLE = (
    0x00AD,  # soft hyphen
    0x061C,  # arabic letter mark (bidi)
    0x180E,  # mongolian vowel separator
    *range(0x200B, 0x2010),  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    *range(0x202A, 0x202F),  # LRE, RLE, PDF, LRO, RLO
    *range(0x2060, 0x2065),  # word joiner e operadores invisíveis
    *range(0x2066, 0x2070),  # isolates bidi (LRI, RLI, FSI, PDI) e controles obsoletos
    0xFEFF,  # BOM / zero width no-break space
    *range(0xFFF9, 0xFFFC),  # anotação interlinear
    0xE0001,  # language tag
    *range(0xE0020, 0xE0080),  # tags
    *range(0xE0100, 0xE01F0),  # seletores de variação suplementares
)
_TABLE: dict[int, str | None] = {
    **dict.fromkeys(_CONTROLS + _INVISIBLE),
    **dict.fromkeys(_LINE_BREAKS, "\n"),
    0x1F: " ",
}


def sanitize_text(text: str) -> str:
    """Texto seguro para o `text` do Postgres, a busca textual e o modelo de embedding."""
    # UTF-16 com surrogatepass reúne o par separado; "replace" troca o solto por U+FFFD.
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    text = text.replace("\r\n", "\n").translate(_TABLE)
    return unicodedata.normalize("NFC", text)
