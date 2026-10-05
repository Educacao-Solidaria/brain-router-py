"""Testes unitários para o módulo de avaliação e métricas de RAG."""

import pytest

from app.evals import (
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    EvalInput,
    FaithfulnessEvaluator,
)


@pytest.mark.asyncio
async def test_faithfulness_evaluator_high_score() -> None:
    evaluator = FaithfulnessEvaluator()
    data = EvalInput(
        query="Qual a capital do Brasil?",
        contexts=["A capital do Brasil é Brasília, inaugurada em 1960 pelo presidente JK."],
        answer="A capital do Brasil é Brasília.",
    )

    result = await evaluator.evaluate(data)
    assert result.metric_name == "faithfulness"
    assert result.score > 0.8
    assert result.passed is True
    assert "ancorados no contexto" in result.reason


@pytest.mark.asyncio
async def test_faithfulness_evaluator_hallucination() -> None:
    evaluator = FaithfulnessEvaluator()
    data = EvalInput(
        query="Qual a capital do Brasil?",
        contexts=["A capital do Brasil é Brasília."],
        answer="A capital do Brasil é Buenos Aires, na Argentina.",
    )

    result = await evaluator.evaluate(data, threshold=0.7)
    assert result.metric_name == "faithfulness"
    assert result.score < 0.7
    assert result.passed is False


@pytest.mark.asyncio
async def test_faithfulness_evaluator_empty_contexts() -> None:
    evaluator = FaithfulnessEvaluator()
    data = EvalInput(
        query="O que é RRF?",
        contexts=[],
        answer="RRF é Reciprocal Rank Fusion.",
    )

    result = await evaluator.evaluate(data)
    assert result.score == 0.0
    assert result.passed is False
    assert "Nenhum contexto" in result.reason


@pytest.mark.asyncio
async def test_context_relevance_evaluator() -> None:
    evaluator = ContextRelevanceEvaluator()
    data = EvalInput(
        query="algoritmo reciprocal rank fusion busca vetorial",
        contexts=[
            "O algoritmo reciprocal rank fusion combina scores de busca densa com bm25 sparse.",
            "Postgres pgvector armazena os vetores em colunas indexadas por hnsw.",
        ],
        answer="Qualquer resposta.",
    )

    result = await evaluator.evaluate(data, threshold=0.6)
    assert result.metric_name == "context_relevance"
    assert result.score >= 0.6
    assert result.passed is True


@pytest.mark.asyncio
async def test_answer_relevance_evaluator() -> None:
    evaluator = AnswerRelevanceEvaluator()
    data = EvalInput(
        query="como configurar pgvector postgres",
        contexts=[],
        answer="Para configurar o pgvector no postgres execute create extension vector.",
    )

    result = await evaluator.evaluate(data)
    assert result.metric_name == "answer_relevance"
    assert result.score > 0.5
    assert result.passed is True


@pytest.mark.asyncio
async def test_eval_input_validation() -> None:
    with pytest.raises(ValueError):
        EvalInput(query="", answer="Alguma resposta")
