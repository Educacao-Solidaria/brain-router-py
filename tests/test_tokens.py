"""Contador de tokens (`app.tokens`) com as codificações reais do tiktoken.

As contagens usam o arquivo de cada codificação: o tiktoken o baixa no primeiro uso, confere o
SHA-256 e o guarda em `TIKTOKEN_CACHE_DIR` (o CI guarda essa pasta com `actions/cache`). Sem
rede e sem o arquivo, os testes de contagem falham em vez de pular. O mapa de modelos não
precisa de arquivo nenhum.
"""

import io
import json
from collections.abc import Iterator

import pytest
import tiktoken

from app import tokens
from app.chunking import chunk_text, count_words
from app.log import configure_logging, shutdown_logging
from app.tokens import FALLBACK_ENCODING, count_tokens, encoding_name_for, token_counter

PT_BR = (
    "A matrícula dos alunos da rede municipal começa em março, "
    "com atendimento presencial nas escolas."
)


@pytest.fixture
def fresh_caches() -> Iterator[None]:
    encoding_name_for.cache_clear()
    tokens._encoding.cache_clear()
    yield
    encoding_name_for.cache_clear()
    tokens._encoding.cache_clear()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/text-embedding-3-small", "cl100k_base"),  # o padrão do cliente de embeddings
        ("text-embedding-3-large", "cl100k_base"),
        ("openai/text-embedding-ada-002", "cl100k_base"),
        ("openai/gpt-4o", "o200k_base"),
        ("openai/gpt-4o-mini", "o200k_base"),
        ("openai/gpt-4o:free", "o200k_base"),  # variante da OpenRouter
        ("openai/gpt-4.1-nano", "o200k_base"),
        ("openai/gpt-5", "o200k_base"),
        ("openai/o3-mini", "o200k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("ft:gpt-4o-mini:org::abc123", "o200k_base"),  # fine-tune: o ":" não é variante
        ("  openai/gpt-4  ", "cl100k_base"),
    ],
)
def test_model_names_map_to_the_encoding_of_the_model(model: str, expected: str) -> None:
    assert encoding_name_for(model) == expected


def test_unknown_model_falls_back_to_cl100k_with_a_single_warning(fresh_caches: None) -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    try:
        for _ in range(3):
            assert encoding_name_for("anthropic/claude-sonnet-4.5") == FALLBACK_ENCODING
    finally:
        shutdown_logging()  # esvazia a fila antes de ler

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    warnings = [event for event in events if event["event"] == "tokens.unknown_model"]
    assert FALLBACK_ENCODING == "cl100k_base"
    assert [warning["model"] for warning in warnings] == ["anthropic/claude-sonnet-4.5"]


@pytest.mark.parametrize("model", ["", "   ", "openai/"])
def test_empty_model_name_is_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="vazio"):
        encoding_name_for(model)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("hello world", 2), ("tiktoken is great!", 6), (PT_BR, 25), ("", 0)],
)
def test_counts_match_the_cl100k_tokenizer_of_the_default_model(text: str, expected: int) -> None:
    assert count_tokens(text) == expected


def test_the_model_decides_the_count() -> None:
    # Mesmo texto, outra codificação: o o200k do gpt-4o junta mais pedaços do português.
    assert count_tokens(PT_BR, "openai/gpt-4o") == 17
    assert count_tokens(PT_BR, "openai/text-embedding-3-small") == 25


def test_special_token_text_from_a_document_counts_as_plain_text() -> None:
    assert count_tokens("<|endoftext|>") == 7
    with pytest.raises(ValueError):  # o encode padrão recusaria o texto do documento
        tiktoken.get_encoding("cl100k_base").encode("<|endoftext|>")


def test_plugs_into_the_chunker_as_count_tokens() -> None:
    count = token_counter()
    text = "\n\n".join([PT_BR] * 6)

    chunks = chunk_text(text, "doc", max_tokens=60, overlap_tokens=10, count_tokens=count)

    assert len(chunks) > 1
    assert all(chunk.token_count == count(chunk.content) for chunk in chunks)
    # Palavra subestima token em pt-BR: o padrão do chunker cortaria chunks quase 2x maiores.
    assert count(text) > 1.5 * count_words(text)


def test_encoding_failure_shows_up_when_the_counter_is_built(
    fresh_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(name: str) -> tiktoken.Encoding:
        raise ConnectionError(f"sem rede para baixar {name}")

    monkeypatch.setattr(tiktoken, "get_encoding", offline)

    with pytest.raises(ConnectionError, match="cl100k_base"):
        token_counter()
