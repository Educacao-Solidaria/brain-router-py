"""O que o `loadtests/locustfile.py` usa e se testa sem o locust (que só vem no extra `load`)."""

import random
from dataclasses import dataclass

from pydantic import ValidationError

from app.schemas.mcp import IndexDocumentOutput

VOCABULARY = (  # texto de secretaria escolar, em frases e parágrafos que o chunker parte
    *("a", "matrícula", "da", "escola", "abre", "em", "fevereiro", "e", "os", "alunos", "levam"),
    *("documentos", "à", "secretaria", "para", "conferência", "do", "histórico", "com", "vagas"),
)


class Rejected(Exception):
    """O upload não foi aceito; a mensagem é o motivo da tabela de falhas do locust."""


def document(n: int, words: int) -> str:
    """O texto do upload `n` (o mesmo para o mesmo `n`): frases de 12 palavras, 5 por parágrafo."""
    if words < 1:
        raise ValueError(f"words deve ser >= 1, veio {words}")
    rng = random.Random(n)
    count = -(-words // 12)  # frases inteiras: arredonda para cima
    sentences = [" ".join(rng.choices(VOCABULARY, k=12)).capitalize() + "." for _ in range(count)]
    return "\n\n".join(" ".join(sentences[i : i + 5]) for i in range(0, count, 5))


def payload(run: str, n: int, words: int) -> dict[str, str]:
    uri = f"loadtest://{run}/{n}"  # único na execução: todo upload é documento novo
    return {"title": f"carga {n}", "content": document(n, words), "source_uri": uri}


def accepted(status: int, body: str) -> int:
    """Os chunks criados por um upload aceito (201 com documento novo); senão `Rejected`."""
    if status != 201:
        raise Rejected(f"HTTP {status}")  # o FastHttpUser dá 0 quando não conecta
    try:
        output = IndexDocumentOutput.model_validate_json(body)
    except ValidationError:
        raise Rejected("201 com corpo fora do contrato") from None
    if not output.success or output.chunks_created < 1:
        raise Rejected("201 sem chunk novo")
    return output.chunks_created


@dataclass
class Totals:
    """Os uploads aceitos, na linha que tem de bater com a do `ingest_server` ao desligar."""

    documents: int = 0
    chunks: int = 0

    def __str__(self) -> str:
        return f"locust: documentos={self.documents} chunks={self.chunks}"
