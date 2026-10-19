"""Chunking de texto para indexação: parágrafos, sentenças pt-BR e sobreposição."""

from app.chunking.semantic import TokenCounter, chunk_text, count_words
from app.chunking.sentences import ABBREVIATIONS, Span, split_paragraphs, split_sentences

__all__ = [
    "ABBREVIATIONS",
    "Span",
    "TokenCounter",
    "chunk_text",
    "count_words",
    "split_paragraphs",
    "split_sentences",
]
