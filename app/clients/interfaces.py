import hashlib
from abc import ABC, abstractmethod

from app.errors.exceptions import EmbeddingServiceError, RerankServiceError
from app.schemas.domain import Chunk, SearchResult


class BaseEmbeddingClient(ABC):
    """Interface abstrata para geradores de embeddings densos."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensao dos vetores gerados pelo modelo."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Gera o vetor de embedding para uma consulta isolada."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Gera vetores de embeddings em lote para multiplos textos."""
        pass


class BaseRerankerClient(ABC):
    """Interface abstrata para modelos de reordenacao de contexto (cross-encoders)."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """Reordena chunks candidatos calculando relevance score com base na query."""
        pass


class MockEmbeddingClient(BaseEmbeddingClient):
    """Cliente simulado de embeddings deterministicos para testes sem rede."""

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _generate_vector(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingServiceError("Texto vazio para geracao de embedding")
        # Gera vetor deterministico via SHA-256
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(b) / 255.0 for b in h]
        # Repete ou trunca ate atingir a dimensao desejada
        repeated = (raw * (self._dim // len(raw) + 1))[: self._dim]
        # Normalizacao L2
        norm = sum(x * x for x in repeated) ** 0.5
        return [x / norm for x in repeated]

    async def embed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]


class MockRerankerClient(BaseRerankerClient):
    """Cliente simulado de reranking baseado em sobreposicao de termos."""

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_n: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            raise RerankServiceError("Query vazia para operacao de reranking")

        terms = [t.lower() for t in query.split() if t]
        scored: list[tuple[Chunk, float]] = []

        for chunk in chunks:
            content_lower = chunk.content.lower()
            matches = sum(1 for t in terms if t in content_lower)
            score = matches / max(len(terms), 1)
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[SearchResult] = []
        for rank, (chunk, score) in enumerate(scored[:top_n], start=1):
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
