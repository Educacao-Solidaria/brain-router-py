"""Cache em memória (LRU + TTL) para o embedding de consultas que se repetem.

`CachedEmbeddingClient` envolve qualquer `BaseEmbeddingClient` pelo mesmo contrato
(decorator): quem chama `embed_query` não muda nada e deixa de pagar, em latência e em
custo de API, por vetorizar de novo a mesma pergunta.

- Só `embed_query` passa pelo cache. `embed_documents` vai direto ao cliente: na ingestão
  os textos quase nunca se repetem (encheriam o cache e expulsariam as perguntas
  frequentes), e há modelos em que o vetor de consulta e o de documento do mesmo texto
  diferem (prefixos "query:"/"passage:").
- A chave é (modelo, dimensão, SHA-256 do texto normalizado). O resumo de 32 bytes deixa a
  memória por entrada fixa (~12 KB com 1536 dimensões; ~12 MB no padrão de 1024 entradas):
  o `query` do `SearchInput` não tem teto de tamanho.
- Normalização mínima, que só junta grafias da mesma pergunta: NFC ("é" composto e "e" +
  acento combinante viram o mesmo texto) e espaço nas pontas. Caixa e espaço interno ficam
  como vieram: "Rio" e "rio" são perguntas diferentes, e indentação conta em código. O
  cliente interno recebe o texto já normalizado, então o vetor é o mesmo com ou sem cache.
- Falha não entra no cache: a chamada seguinte tenta de novo.
- O vetor fica num `array('d')` (8 bytes por dimensão) e cada chamada recebe uma lista
  nova: o chamador pode alterá-la à vontade sem corromper o que está guardado.
- Misses simultâneos da mesma chave fazem uma chamada só: os outros esperam a mesma tarefa,
  por `asyncio.shield`, então cancelar um chamador não cancela a chamada dos demais (e o
  vetor ainda entra no cache quando ela termina).

Taxa de acerto: `1 - get_stats("embeddings.cache.miss").calls /
get_stats("embeddings.cache.query").calls` (`app.perf`); cada miss é uma chamada paga ao
cliente interno, inclusive com o cache desligado.
ponytail: por processo e sem lock (o asyncio roda numa thread só). Cache compartilhado
entre réplicas (Redis) só se a taxa de acerto por réplica ficar baixa.
"""

import asyncio
import hashlib
import math
import time
import unicodedata
from array import array
from collections import OrderedDict
from collections.abc import Callable

from app.clients.interfaces import BaseEmbeddingClient
from app.perf import timed

DEFAULT_MAX_ITEMS = 1024
DEFAULT_TTL_SECONDS = 3600.0

_Key = tuple[str, int, bytes]


def normalize_query(text: str) -> str:
    """NFC e sem espaço nas pontas: o texto que vira chave e que vai ao cliente interno."""
    return unicodedata.normalize("NFC", text).strip()


class CachedEmbeddingClient(BaseEmbeddingClient):
    """`embed_query` com cache; `embed_documents` e `dimension` são os do cliente interno.

    `max_items=0` desliga o cache (tudo vai ao cliente). `ttl_seconds=None`: sem expiração,
    só o LRU. Fechar o cliente interno continua sendo trabalho de quem o criou.
    """

    def __init__(
        self,
        inner: BaseEmbeddingClient,
        *,
        model: str,
        max_items: int = DEFAULT_MAX_ITEMS,
        ttl_seconds: float | None = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not model:
            raise ValueError("model não pode ser vazio: ele separa os vetores de cada modelo")
        if max_items < 0:
            raise ValueError(f"max_items deve ser >= 0, veio {max_items}")
        if ttl_seconds is not None and not (math.isfinite(ttl_seconds) and ttl_seconds > 0):
            raise ValueError(f"ttl_seconds deve ser finito e > 0 (ou None), veio {ttl_seconds}")
        self._inner = inner
        self._model = model
        self._max_items = max_items
        self._ttl = ttl_seconds
        self._clock = clock
        # Ordem de uso: a primeira entrada é a menos recente, a primeira a sair.
        self._entries: OrderedDict[_Key, tuple[float, array[float]]] = OrderedDict()
        self._pending: dict[_Key, asyncio.Task[array[float]]] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._inner!r}, model={self._model!r}, "
            f"max_items={self._max_items}, ttl_seconds={self._ttl})"
        )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @timed("embeddings.cache.query")
    async def embed_query(self, text: str) -> list[float]:
        query = normalize_query(text)
        if self._max_items == 0:
            return await self._miss(query)
        # surrogatepass: um surrogate solto não derruba o cache (quem recusa é o cliente).
        digest = hashlib.sha256(query.encode("utf-8", "surrogatepass")).digest()
        key = (self._model, self._inner.dimension, digest)
        vector = self._lookup(key)
        if vector is None:
            task = self._pending.get(key)
            # done(): tarefa cancelada antes de começar (fim do event loop) não serve mais.
            if task is None or task.done():
                task = asyncio.create_task(self._fetch(key, query))
                self._pending[key] = task
            vector = await asyncio.shield(task)
        return vector.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_documents(texts)

    def _lookup(self, key: _Key) -> array[float] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, vector = entry
        if expires_at <= self._clock():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return vector

    @timed("embeddings.cache.miss")
    async def _miss(self, query: str) -> list[float]:
        return await self._inner.embed_query(query)

    async def _fetch(self, key: _Key, query: str) -> array[float]:
        try:
            vector = array("d", await self._miss(query))
        finally:
            # Antes de a tarefa terminar: quem chegar depois de uma falha chama de novo.
            self._pending.pop(key, None)
        expires_at = math.inf if self._ttl is None else self._clock() + self._ttl
        self._entries[key] = (expires_at, vector)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_items:
            self._entries.popitem(last=False)
        return vector
