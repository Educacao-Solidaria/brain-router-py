"""Dataset sintético em português para validar o pipeline RAG sem chamar LLM nem API.

`generate_dataset(n, seed=s)` devolve `n` amostras `EvalInput` (o contrato de `app.evals`):
pergunta, contextos (o trecho que responde + distratores de outros temas, embaralhados) e a
resposta de referência em `answer` e `ground_truth`. Mesmo `seed` → mesmo dataset, byte a
byte: só `random.Random(seed)` e templates fixos, sem relógio, rede ou estado global. (O
Python só garante a sequência de `choice`/`sample`/`shuffle` dentro da mesma versão; o
projeto fixa 3.12 em `.python-version`.)

`metadata["gold_index"]` aponta qual contexto responde à pergunta — é o gabarito para
medir recuperação (o trecho certo veio?) separado de geração (a resposta está ancorada?).
"""

import random
from dataclasses import dataclass

from app.evals.interfaces import EvalInput


@dataclass(frozen=True)
class _Template:
    topic: str
    context: str
    question: str
    answer: str


_TEMPLATES = (
    _Template(
        "matrícula",
        "A matrícula do curso de {curso} na unidade {unidade} vai de {inicio} até {fim}, "
        "pelo portal do aluno ou na secretaria.",
        "Até quando posso fazer a matrícula de {curso} na unidade {unidade}?",
        "A matrícula de {curso} na unidade {unidade} vai até {fim}.",
    ),
    _Template(
        "frequência",
        "Para aprovação em {curso} na unidade {unidade}, o aluno precisa de frequência "
        "mínima de {frequencia}% das aulas.",
        "Qual a frequência mínima para aprovação em {curso} na unidade {unidade}?",
        "A frequência mínima para aprovação em {curso} é de {frequencia}% das aulas.",
    ),
    _Template(
        "avaliação",
        "A prova final de {curso} na unidade {unidade} acontece em {fim} e vale {peso} "
        "pontos da nota do semestre.",
        "Quando acontece a prova final de {curso} e quanto ela vale?",
        "A prova final de {curso} acontece em {fim} e vale {peso} pontos.",
    ),
    _Template(
        "bolsa",
        "Alunos de {curso} com renda familiar de até {salarios} salários mínimos podem "
        "pedir bolsa de {bolsa}% na secretaria da unidade {unidade}.",
        "Qual o percentual da bolsa de estudos para o curso de {curso}?",
        "A bolsa de {curso} é de {bolsa}% com renda familiar de até {salarios} salários mínimos.",
    ),
    _Template(
        "certificado",
        "O certificado digital de conclusão de {curso} é emitido em até {dias} dias úteis "
        "depois da última aula, sem custo para o aluno.",
        "Em quanto tempo sai o certificado de conclusão de {curso}?",
        "O certificado de conclusão de {curso} é emitido em até {dias} dias úteis.",
    ),
)

_CURSOS = ("Alfabetização de Adultos", "Informática Básica", "Matemática Financeira")
_UNIDADES = ("Centro", "Vila Nova", "Jardim América", "Bom Retiro")
_MESES = ("fevereiro", "março", "abril", "agosto", "setembro", "outubro")


def _slots(rng: random.Random) -> dict[str, str]:
    mes = rng.choice(_MESES)
    inicio, fim = sorted(rng.sample(range(1, 29), 2))
    return {
        "curso": rng.choice(_CURSOS),
        "unidade": rng.choice(_UNIDADES),
        "inicio": f"{inicio} de {mes}",
        "fim": f"{fim} de {mes}",
        "frequencia": rng.choice(("70", "75", "80")),
        "peso": rng.choice(("40", "50", "60")),
        "salarios": rng.choice(("1,5", "2", "3")),
        "bolsa": rng.choice(("50", "70", "100")),
        "dias": rng.choice(("5", "10", "15")),
    }


def generate_dataset(size: int, *, seed: int = 0, distractors: int = 2) -> list[EvalInput]:
    """`size` amostras determinísticas; `distractors` contextos de outros temas por amostra."""
    if size < 0:
        raise ValueError("size não pode ser negativo")
    if not 0 <= distractors < len(_TEMPLATES):
        raise ValueError(f"distractors deve estar entre 0 e {len(_TEMPLATES) - 1}")

    rng = random.Random(seed)
    samples = []
    for index in range(size):
        template = rng.choice(_TEMPLATES)
        values = _slots(rng)
        gold = template.context.format(**values)
        others = [t for t in _TEMPLATES if t is not template]
        noise = [t.context.format(**_slots(rng)) for t in rng.sample(others, distractors)]
        contexts = [gold, *noise]
        rng.shuffle(contexts)
        answer = template.answer.format(**values)
        samples.append(
            EvalInput(
                query=template.question.format(**values),
                contexts=contexts,
                answer=answer,
                ground_truth=answer,
                metadata={
                    "id": f"sintetico-{seed}-{index}",
                    "topic": template.topic,
                    "gold_index": contexts.index(gold),
                    "seed": seed,
                },
            )
        )
    return samples
