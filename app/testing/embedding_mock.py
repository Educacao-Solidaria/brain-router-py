"""Módulo de geradores simulados e utilitários de teste para o ecossistema RAG."""

import asyncio
import hashlib
import math
import random
from typing import Any

from app.clients.interfaces import BaseEmbeddingClient
from app.errors.exceptions import EmbeddingServiceError


class DeterministicEmbeddingGenerator(BaseEmbeddingClient):
    """Gerador determinístico de vetores sintéticos parametrizado por seed para testes locais.

    Permite testar pipelines de busca vetorial sem depender de chamadas externas de API,
    com suporte a simulação de latência, injeção de falhas e controle de dimensionalidade.
    """

    def __init__(
        self,
        dimension: int = 384,
        seed: int = 42,
        simulated_delay_ms: float = 0.0,
        fail_rate: float = 0.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError("A dimensão deve ser um inteiro estritamente positivo.")
        if not (0.0 <= fail_rate <= 1.0):
            raise ValueError("fail_rate deve estar no intervalo [0.0, 1.0].")

        self._dimension = dimension
        self._seed = seed
        self._simulated_delay_ms = simulated_delay_ms
        self._fail_rate = fail_rate
        self.call_count: int = 0
        self.total_tokens_estimated: int = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Gera um vetor unitário determinístico baseado no hash do texto e na seed."""
        if not text.strip():
            raise EmbeddingServiceError("Texto vazio não pode ser convertido em embedding.")

        self.call_count += 1
        self.total_tokens_estimated += max(1, len(text.split()))

        # Injeção determinística de falha
        if self._fail_rate > 0.0:
            pseudo_rnd = (
                int(hashlib.md5(f"{text}_{self._seed}_{self.call_count}".encode()).hexdigest(), 16)
                % 10000
            ) / 10000.0
            if pseudo_rnd < self._fail_rate:
                raise EmbeddingServiceError("Falha simulada no serviço de embeddings.")

        # Seed local combinando texto e seed de inicialização
        hasher = hashlib.sha512(f"{self._seed}:{text}".encode())
        digest = hasher.digest()

        # Constrói floats a partir dos bytes
        raw_values: list[float] = []
        rng = random.Random(int.from_bytes(digest[:8], byteorder="big"))
        for _ in range(self._dimension):
            raw_values.append(rng.gauss(0.0, 1.0))

        # Normalização L2 para garantir norma unitária (essencial para Cosseno e Dot Product)
        squared_sum = sum(x * x for x in raw_values)
        norm = math.sqrt(squared_sum)
        if norm == 0.0:
            return [1.0 / math.sqrt(self._dimension)] * self._dimension

        return [x / norm for x in raw_values]

    async def _simulate_latency(self) -> None:
        if self._simulated_delay_ms > 0:
            await asyncio.sleep(self._simulated_delay_ms / 1000.0)

    async def embed_query(self, text: str) -> list[float]:
        await self._simulate_latency()
        return self._generate_deterministic_vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        await self._simulate_latency()
        return [self._generate_deterministic_vector(t) for t in texts]

    def reset_stats(self) -> None:
        """Reinicia contadores de métricas de telemetria."""
        self.call_count = 0
        self.total_tokens_estimated = 0

    def get_stats(self) -> dict[str, Any]:
        """Retorna resumo estatístico de execuções."""
        return {
            "dimension": self._dimension,
            "call_count": self.call_count,
            "total_tokens_estimated": self.total_tokens_estimated,
            "simulated_delay_ms": self._simulated_delay_ms,
        }
