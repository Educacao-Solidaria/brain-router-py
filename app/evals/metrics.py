"""Implementações canônicas de métricas de avaliação de RAG."""

import re

from app.evals.interfaces import BaseEvaluator, EvalInput, EvalOutput


def _tokenize(text: str) -> set[str]:
    """Tokenizador léxico determinístico simples para cálculo de sobreposição."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = {token for token in clean.split() if len(token) > 2}
    return tokens


class FaithfulnessEvaluator(BaseEvaluator):
    """Mede a fidelidade da resposta aos contextos recuperados (ausência de alucinação)."""

    @property
    def name(self) -> str:
        return "faithfulness"

    async def evaluate(self, input_data: EvalInput, threshold: float | None = None) -> EvalOutput:
        limit = threshold if threshold is not None else self.default_threshold

        if not input_data.answer.strip():
            return EvalOutput(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=limit,
                reason="Resposta gerada vazia.",
            )

        if not input_data.contexts:
            return EvalOutput(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=limit,
                reason="Nenhum contexto foi fornecido para validar a resposta.",
            )

        answer_tokens = _tokenize(input_data.answer)
        if not answer_tokens:
            return EvalOutput(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=limit,
                reason="Resposta sem termos substantivos avaliáveis.",
            )

        context_tokens: set[str] = set()
        for ctx in input_data.contexts:
            context_tokens.update(_tokenize(ctx))

        supported_tokens = answer_tokens.intersection(context_tokens)
        score = round(len(supported_tokens) / len(answer_tokens), 4)
        passed = score >= limit

        reason = (
            f"Fidelidade léxica: {len(supported_tokens)}/{len(answer_tokens)} tokens da resposta "
            f"estão ancorados no contexto recuperado."
        )

        return EvalOutput(
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=limit,
            reason=reason,
            metadata={
                "supported_tokens_count": len(supported_tokens),
                "total_answer_tokens": len(answer_tokens),
            },
        )


class ContextRelevanceEvaluator(BaseEvaluator):
    """Mede se os contextos recuperados contêm os termos e temas da consulta."""

    @property
    def name(self) -> str:
        return "context_relevance"

    async def evaluate(self, input_data: EvalInput, threshold: float | None = None) -> EvalOutput:
        limit = threshold if threshold is not None else self.default_threshold

        query_tokens = _tokenize(input_data.query)
        if not query_tokens:
            return EvalOutput(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=limit,
                reason="Consulta sem tokens substantivos identificados.",
            )

        if not input_data.contexts:
            return EvalOutput(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=limit,
                reason="Nenhum contexto recuperado.",
            )

        context_tokens: set[str] = set()
        for ctx in input_data.contexts:
            context_tokens.update(_tokenize(ctx))

        matched_tokens = query_tokens.intersection(context_tokens)
        score = round(len(matched_tokens) / len(query_tokens), 4)
        passed = score >= limit

        reason = (
            f"Relevância de contexto: {len(matched_tokens)}/{len(query_tokens)} tokens da consulta "
            f"foram encontrados nos contextos recuperados."
        )

        return EvalOutput(
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=limit,
            reason=reason,
            metadata={
                "matched_query_tokens": len(matched_tokens),
                "total_query_tokens": len(query_tokens),
            },
        )


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Mede a adequação e cobertura da resposta em relação à pergunta original."""

    @property
    def name(self) -> str:
        return "answer_relevance"

    async def evaluate(self, input_data: EvalInput, threshold: float | None = None) -> EvalOutput:
        limit = threshold if threshold is not None else self.default_threshold

        query_tokens = _tokenize(input_data.query)
        answer_tokens = _tokenize(input_data.answer)

        if not answer_tokens:
            return EvalOutput(
                metric_name=self.name,
                score=0.0,
                passed=False,
                threshold=limit,
                reason="Resposta vazia não é relevante à consulta.",
            )

        if not query_tokens:
            return EvalOutput(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=limit,
                reason="Consulta vazia considerada trivial.",
            )

        overlap = query_tokens.intersection(answer_tokens)
        score = round(len(overlap) / len(query_tokens), 4)
        passed = score >= limit

        reason = (
            f"Relevância da resposta: {len(overlap)}/{len(query_tokens)} termos-chave da pergunta "
            f"foram abordados diretamente na resposta."
        )

        return EvalOutput(
            metric_name=self.name,
            score=score,
            passed=passed,
            threshold=limit,
            reason=reason,
            metadata={"overlap_count": len(overlap)},
        )
