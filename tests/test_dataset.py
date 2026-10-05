"""Gerador de dataset sintético: determinismo, forma das amostras e gabarito utilizável."""

import pytest

from app.evals.dataset import generate_dataset
from app.evals.interfaces import EvalInput
from app.evals.metrics import FaithfulnessEvaluator


def test_same_seed_gives_identical_dataset() -> None:
    first = [s.model_dump_json() for s in generate_dataset(30, seed=7)]
    again = [s.model_dump_json() for s in generate_dataset(30, seed=7)]
    other = [s.model_dump_json() for s in generate_dataset(30, seed=8)]

    assert first == again
    assert first != other


def test_prefix_is_stable_when_size_grows() -> None:
    assert generate_dataset(5, seed=3) == generate_dataset(10, seed=3)[:5]


def test_samples_follow_the_eval_contract() -> None:
    samples = generate_dataset(25, seed=1, distractors=3)

    assert len(samples) == 25
    for index, sample in enumerate(samples):
        assert isinstance(sample, EvalInput)
        assert sample.query.endswith("?")
        assert len(sample.contexts) == 4
        assert len(set(sample.contexts)) == 4  # distratores vêm de outros temas
        assert sample.answer == sample.ground_truth
        assert sample.metadata["id"] == f"sintetico-1-{index}"
        assert sample.metadata["seed"] == 1


async def test_gold_context_alone_grounds_the_answer_and_distractors_do_not() -> None:
    evaluator = FaithfulnessEvaluator()
    for sample in generate_dataset(40, seed=5):
        gold_index = sample.metadata["gold_index"]
        gold = [sample.contexts[gold_index]]
        noise = [c for i, c in enumerate(sample.contexts) if i != gold_index]

        with_gold = await evaluator.evaluate(sample.model_copy(update={"contexts": gold}))
        with_noise = await evaluator.evaluate(sample.model_copy(update={"contexts": noise}))

        assert with_gold.score == 1.0, (sample.metadata["id"], with_gold.reason)
        assert with_noise.score < with_gold.score, sample.metadata["id"]


def test_covers_all_topics_and_text_is_portuguese() -> None:
    samples = generate_dataset(60, seed=2)

    topics = {s.metadata["topic"] for s in samples}
    assert topics == {"matrícula", "frequência", "avaliação", "bolsa", "certificado"}
    text = " ".join(s.query + " ".join(s.contexts) for s in samples)
    assert all(char in text for char in "ãçéú")


def test_zero_size_and_zero_distractors() -> None:
    assert generate_dataset(0) == []
    (sample,) = generate_dataset(1, distractors=0)
    assert sample.metadata["gold_index"] == 0
    assert len(sample.contexts) == 1


@pytest.mark.parametrize(("size", "distractors"), [(-1, 2), (1, -1), (1, 5)])
def test_invalid_arguments_are_rejected(size: int, distractors: int) -> None:
    with pytest.raises(ValueError):
        generate_dataset(size, distractors=distractors)
