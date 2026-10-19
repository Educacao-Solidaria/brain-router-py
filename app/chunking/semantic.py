"""Chunking semântico de texto corrido: parágrafos → sentenças → chunks com sobreposição.

`chunk_text` só corta entre sentenças, e prefere cortar entre parágrafos: um parágrafo
que cabe inteiro num chunk não é partido entre dois, desde que o chunk atual já esteja
pela metade. Uma sentença maior que o máximo se parte entre palavras, nunca no meio de
uma. A sobreposição repete as últimas sentenças inteiras do chunk anterior.

O `content` de cada chunk é a fatia exata do texto original (quebras de linha
preservadas), e `metadata["position"]` diz onde ela está (um `ChunkPosition`).
"""

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass

from app.chunking.sentences import Span, split_paragraphs, split_sentences
from app.schemas import Chunk, ChunkPosition

TokenCounter = Callable[[str], int]

_WORD = re.compile(r"\S+")


def count_words(text: str) -> int:
    """Contagem padrão de tokens: palavras separadas por espaço.

    ponytail: palavra ≈ token. O contador do tiktoken (PR 39) entra pelo parâmetro
    `count_tokens` de `chunk_text`, sem mudar mais nada aqui.
    """
    return len(text.split())


@dataclass(frozen=True, slots=True)
class _Piece:
    start: int
    end: int
    tokens: int


def _pieces(text: str, sentence: Span, max_tokens: int, count: TokenCounter) -> list[_Piece]:
    """A sentença inteira, ou partida entre palavras quando passa de `max_tokens`."""
    start, end = sentence
    tokens = count(text[start:end])
    if tokens <= max_tokens:
        return [_Piece(start, end, tokens)]
    pieces: list[_Piece] = []
    piece_start = piece_end = start
    used = 0
    for word in _WORD.finditer(text, start, end):
        word_tokens = count(word.group())
        if used and used + word_tokens > max_tokens:
            pieces.append(_Piece(piece_start, piece_end, used))
            piece_start, used = word.start(), 0
        piece_end = word.end()
        used += word_tokens
    pieces.append(_Piece(piece_start, piece_end, used))
    return pieces


def _pack(paragraphs: list[list[_Piece]], max_tokens: int, overlap: int) -> list[list[_Piece]]:
    windows: list[list[_Piece]] = []
    current: list[_Piece] = []
    used = fresh = 0  # tokens em `current`; peças dele que ainda não saíram em chunk algum

    def cut(incoming: int) -> None:
        nonlocal current, used, fresh
        windows.append(current)
        # As últimas peças inteiras do chunk que saiu, sem estourar o próximo. A cauda
        # nunca é o chunk inteiro (ele só saiu porque não cabia com `incoming`), então
        # cada chunk começa depois do anterior e o laço sempre anda.
        budget = min(overlap, max_tokens - incoming)
        tail: list[_Piece] = []
        total = 0
        for piece in reversed(current):
            if total + piece.tokens > budget:
                break
            tail.append(piece)
            total += piece.tokens
        current, used, fresh = tail[::-1], total, 0

    for paragraph in paragraphs:
        size = sum(piece.tokens for piece in paragraph)
        if fresh and used + size > max_tokens >= size and 2 * used >= max_tokens:
            cut(size)
        for piece in paragraph:
            if fresh and used + piece.tokens > max_tokens:
                cut(piece.tokens)
            current.append(piece)
            used += piece.tokens
            fresh += 1
    if fresh:  # um resto feito só de sobreposição já saiu no chunk anterior
        windows.append(current)
    return windows


def chunk_text(
    text: str,
    document_id: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    count_tokens: TokenCounter = count_words,
) -> list[Chunk]:
    """Divide `text` em chunks de até `max_tokens`, repetindo até `overlap_tokens` do anterior.

    Padrões iguais aos de `IndexDocumentInput` (`chunk_size=500`, `chunk_overlap=50`). O id
    é `"{document_id}:{n}"`: o mesmo texto gera os mesmos ids. Texto vazio ou só espaço
    não gera chunk. Uma palavra sozinha maior que `max_tokens` sai num chunk acima do
    teto: cortá-la ao meio seria pior.

    `overlap_tokens >= max_tokens` é aceito e vale como "o máximo que couber": o
    `IndexDocumentInput` permite `chunk_size=50` com o `chunk_overlap=50` padrão.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens deve ser >= 1, veio {max_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens deve ser >= 0, veio {overlap_tokens}")

    paragraphs = [
        [
            piece
            for sentence in split_sentences(text, start, end)
            for piece in _pieces(text, sentence, max_tokens, count_tokens)
        ]
        for start, end in split_paragraphs(text)
    ]
    newlines = [match.start() for match in re.finditer("\n", text)]
    chunks: list[Chunk] = []
    for index, window in enumerate(_pack(paragraphs, max_tokens, overlap_tokens)):
        start, end = window[0].start, window[-1].end
        content = text[start:end]
        line_start = bisect.bisect_left(newlines, start) + 1
        position = ChunkPosition(
            start_char=start,
            end_char=end,
            line_start=line_start,
            line_end=line_start + content.count("\n"),
        )
        chunks.append(
            Chunk(
                id=f"{document_id}:{index}",
                document_id=document_id,
                content=content,
                token_count=count_tokens(content),
                metadata={"position": position.model_dump()},
            )
        )
    return chunks
