"""Chunking semântico: sentenças pt-BR, parágrafos, teto de tokens e sobreposição."""

import pytest

from app.chunking import chunk_text, split_paragraphs, split_sentences
from app.schemas import Chunk, ChunkPosition

# Seis palavras cada: com max_tokens=12 cabem exatamente duas por chunk.
SENTENCES = [f"Frase número {n} fala de matrícula." for n in range(1, 13)]


def _sentences(text: str) -> list[str]:
    return [text[start:end] for start, end in split_sentences(text)]


def _contents(chunks: list[Chunk]) -> list[str]:
    return [chunk.content for chunk in chunks]


def _position(chunk: Chunk) -> ChunkPosition:
    return ChunkPosition.model_validate(chunk.metadata["position"])


def test_splits_on_final_punctuation_followed_by_a_capital() -> None:
    text = "Olá. Tudo bem? Sim! Até logo… Então tchau."
    assert _sentences(text) == ["Olá.", "Tudo bem?", "Sim!", "Até logo…", "Então tchau."]


def test_common_abbreviations_do_not_end_a_sentence() -> None:
    text = "O Sr. Silva e a Dra. Ana citaram o art. 5º e as fls. 12 do processo. Depois saíram."
    assert _sentences(text) == [
        "O Sr. Silva e a Dra. Ana citaram o art. 5º e as fls. 12 do processo.",
        "Depois saíram.",
    ]


def test_initials_dotted_acronyms_and_list_numbers_do_not_end_a_sentence() -> None:
    # "S.A. Em" não corta: na dúvida, juntar duas frases custa menos que partir uma.
    text = "J. R. Silva fundou a Acme S.A. Em 1990 abriu filiais. 1. Primeiro item."
    assert _sentences(text) == [
        "J. R. Silva fundou a Acme S.A. Em 1990 abriu filiais.",
        "1. Primeiro item.",
    ]


def test_etc_ends_a_sentence_only_before_a_capital() -> None:
    text = "Levou lápis, cadernos etc. e voltou. Comprou livros etc. Depois saiu."
    assert _sentences(text) == [
        "Levou lápis, cadernos etc. e voltou.",
        "Comprou livros etc.",
        "Depois saiu.",
    ]


def test_quotes_decimals_and_lowercase_continuations() -> None:
    text = 'Ele disse: "Vou sair." Ela ficou. O índice subiu 3.5 pontos. depois caiu. Fim .  Ok.'
    assert _sentences(text) == [
        'Ele disse: "Vou sair."',
        "Ela ficou.",
        "O índice subiu 3.5 pontos. depois caiu.",
        "Fim .",
        "Ok.",
    ]


def test_ordinal_indicator_keeps_the_abbreviation_but_not_the_ordinal_number() -> None:
    text = "A Profª. Ana e a Drª. Lia chegaram em 1º. Depois saíram."
    assert _sentences(text) == ["A Profª. Ana e a Drª. Lia chegaram em 1º.", "Depois saíram."]


def test_sentence_spans_point_into_the_original_text_without_surrounding_space() -> None:
    assert split_sentences("  Primeira frase.   Segunda frase.  ") == [(2, 17), (20, 34)]


def test_split_sentences_stays_inside_the_requested_window() -> None:
    text = "Fora. Dentro um. Dentro dois. Fora de novo."
    start, end = text.index("Dentro"), text.index(" Fora de")
    assert [text[s:e] for s, e in split_sentences(text, start, end)] == [
        "Dentro um.",
        "Dentro dois.",
    ]


def test_paragraphs_split_on_one_or_more_blank_lines() -> None:
    text = "Primeiro parágrafo\ncom duas linhas.\n\n\n  Segundo.\r\n\r\nTerceiro."
    assert [text[s:e] for s, e in split_paragraphs(text)] == [
        "Primeiro parágrafo\ncom duas linhas.",
        "Segundo.",
        "Terceiro.",
    ]


def test_blank_text_has_no_paragraphs_sentences_or_chunks() -> None:
    for blank in ("", "   ", " \n\n \t "):
        assert split_paragraphs(blank) == []
        assert split_sentences(blank) == []
        assert chunk_text(blank, "doc") == []


def test_short_text_becomes_one_chunk_that_knows_where_it_came_from() -> None:
    text = "Linha um.\nLinha dois."
    (chunk,) = chunk_text(text, "doc-1")
    assert (chunk.id, chunk.document_id, chunk.content, chunk.token_count) == (
        "doc-1:0",
        "doc-1",
        text,
        4,
    )
    assert _position(chunk) == ChunkPosition(
        start_char=0, end_char=len(text), line_start=1, line_end=2
    )


def test_chunks_respect_the_maximum_and_cut_only_between_sentences() -> None:
    text = " ".join(SENTENCES)
    chunks = chunk_text(text, "doc", max_tokens=12, overlap_tokens=0)

    assert [chunk.id for chunk in chunks] == [f"doc:{n}" for n in range(6)]
    assert _contents(chunks) == [" ".join(SENTENCES[n : n + 2]) for n in range(0, 12, 2)]
    assert all(chunk.token_count == 12 for chunk in chunks)


def test_overlap_repeats_whole_sentences_from_the_previous_chunk() -> None:
    text = " ".join(SENTENCES[:6])
    with_overlap = chunk_text(text, "doc", max_tokens=18, overlap_tokens=6)
    assert _contents(with_overlap) == [
        " ".join(SENTENCES[0:3]),
        " ".join(SENTENCES[2:5]),
        " ".join(SENTENCES[4:6]),
    ]
    # Uma sentença (6) não cabe numa sobreposição de 5: nada se repete.
    too_small = chunk_text(text, "doc", max_tokens=18, overlap_tokens=5)
    assert _contents(too_small) == [" ".join(SENTENCES[0:3]), " ".join(SENTENCES[3:6])]


def test_a_sentence_longer_than_the_maximum_splits_between_words() -> None:
    text = " ".join(f"palavra{n}" for n in range(1, 26)) + "."
    chunks = chunk_text(text, "doc", max_tokens=10, overlap_tokens=0)

    assert [chunk.token_count for chunk in chunks] == [10, 10, 5]
    assert " ".join(_contents(chunks)) == text  # nenhuma palavra cortada ou perdida


def test_a_single_word_above_the_maximum_is_never_cut() -> None:
    chunks = chunk_text("Supercalifragilístico é grande.", "doc", max_tokens=10, count_tokens=len)
    assert _contents(chunks) == ["Supercalifragilístico", "é grande."]
    assert chunks[0].token_count == 21  # acima do teto, mas inteira


def test_a_paragraph_that_fits_whole_goes_to_the_next_chunk() -> None:
    first = "Um dois três quatro cinco seis."
    second = "Sete oito. Nove dez. Onze doze."
    chunks = chunk_text(f"{first}\n\n{second}", "doc", max_tokens=10, overlap_tokens=0)
    assert _contents(chunks) == [first, second]


def test_a_short_heading_is_not_left_alone_in_a_chunk() -> None:
    text = "Capítulo um.\n\nSete oito nove. Dez onze doze. Treze catorze quinze."
    chunks = chunk_text(text, "doc", max_tokens=10, overlap_tokens=0)
    assert _contents(chunks) == [
        "Capítulo um.\n\nSete oito nove. Dez onze doze.",
        "Treze catorze quinze.",
    ]


def test_positions_and_lines_locate_each_chunk_in_the_original() -> None:
    text = "Linha um.\nLinha dois.\n\nLinha quatro. Linha quatro de novo."
    chunks = chunk_text(text, "doc", max_tokens=4, overlap_tokens=0)

    assert _contents(chunks) == ["Linha um.\nLinha dois.", "Linha quatro.", "Linha quatro de novo."]
    assert [(_position(c).line_start, _position(c).line_end) for c in chunks] == [
        (1, 2),
        (4, 4),
        (4, 4),
    ]


def test_an_overlap_bigger_than_the_chunk_keeps_only_what_fits() -> None:
    # Como chunk_size=50 com o chunk_overlap=50 padrão do IndexDocumentInput.
    chunks = chunk_text(" ".join(SENTENCES), "doc", max_tokens=12, overlap_tokens=50)
    assert _contents(chunks) == [" ".join(SENTENCES[n : n + 2]) for n in range(11)]


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (-1, 0), (10, -1)])
def test_invalid_sizes_are_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("Texto.", "doc", max_tokens=max_tokens, overlap_tokens=overlap)


def test_any_valid_size_covers_every_word_and_always_moves_forward() -> None:
    text = "Art. 1º O Sr. Silva leu o texto. " * 7 + "\n\n" + " ".join(f"p{n}" for n in range(40))
    for max_tokens in (1, 2, 3, 5, 8, 13):
        for overlap in range(max_tokens + 2):
            chunks = chunk_text(text, "doc", max_tokens=max_tokens, overlap_tokens=overlap)
            covered: set[int] = set()
            for chunk in chunks:
                position = _position(chunk)
                assert text[position.start_char : position.end_char] == chunk.content
                assert chunk.token_count <= max_tokens
                covered.update(range(position.start_char, position.end_char))
            assert all(i in covered for i, char in enumerate(text) if not char.isspace())
            starts = [_position(chunk).start_char for chunk in chunks]
            assert starts == sorted(set(starts)), (max_tokens, overlap)
