from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import math

from app.errors.exceptions import CollectionNotFoundError
from app.schemas.domain import Chunk, SearchResult


class BaseVectorStore(ABC):
    """Interface abstrata base para provedores de Vector Store."""

    @abstractmethod
    async def add_chunks(self, collection_name: str, chunks: List[Chunk]) -> int:
        """Armazena uma lista de chunks associados a uma colecao."""
        pass

    @abstractmethod
    async def search_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Realiza busca por similaridade vetorial."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Remove todos os dados de uma colecao."""
        pass

    @abstractmethod
    async def count_chunks(self, collection_name: str) -> int:
        """Retorna o numero de chunks armazenados em uma colecao."""
        pass


class BaseTextSearch(ABC):
    """Interface abstrata base para motores de busca textual (BM25/FTS)."""

    @abstractmethod
    async def search_text(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Executa busca textual full-text/lexical por palavras-chave."""
        pass


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(BaseVectorStore, BaseTextSearch):
    """Implementacao em memoria para desenvolvimento local e testes isolados."""

    def __init__(self) -> None:
        self._collections: Dict[str, List[Chunk]] = {}

    async def add_chunks(self, collection_name: str, chunks: List[Chunk]) -> int:
        if collection_name not in self._collections:
            self._collections[collection_name] = []
        self._collections[collection_name].extend(chunks)
        return len(chunks)

    async def search_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        if collection_name not in self._collections:
            raise CollectionNotFoundError(collection_name)

        scored: List[tuple[Chunk, float]] = []
        for chunk in self._collections[collection_name]:
            if not chunk.embedding:
                continue
            sim = _cosine_similarity(query_vector, chunk.embedding)
            if sim >= min_score:
                scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[SearchResult] = []
        for rank, (chunk, score) in enumerate(scored[:limit], start=1):
            results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=score,
                    sparse_score=None,
                    rrf_score=score,
                    rank=rank,
                )
            )
        return results

    async def search_text(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        if collection_name not in self._collections:
            raise CollectionNotFoundError(collection_name)

        terms = [t.lower() for t in query.split() if t]
        scored: List[tuple[Chunk, float]] = []

        for chunk in self._collections[collection_name]:
            text_lower = chunk.content.lower()
            matches = sum(1 for term in terms if term in text_lower)
            if matches > 0:
                score = matches / max(len(terms), 1)
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[SearchResult] = []
        for rank, (chunk, score) in enumerate(scored[:limit], start=1):
            results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=None,
                    sparse_score=score,
                    rrf_score=score,
                    rank=rank,
                )
            )
        return results

    async def delete_collection(self, collection_name: str) -> bool:
        if collection_name in self._collections:
            del self._collections[collection_name]
            return True
        return False

    async def count_chunks(self, collection_name: str) -> int:
        return len(self._collections.get(collection_name, []))
