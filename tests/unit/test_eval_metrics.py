import json

import pytest

from evals import metrics
from tests.fakes import make_response

A, B, C = "data/a.txt", "data/b.txt", "data/c.txt"


def test_retrieval_metrics():
    assert metrics.hit_at_k([A], [B, A]) == 1.0
    assert metrics.hit_at_k([A], [B, C]) == 0.0
    assert metrics.recall_at_k([A, B], [A, C]) == 0.5
    assert metrics.reciprocal_rank([A], [B, C, A]) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank([A], [B]) == 0.0


@pytest.mark.parametrize("facts, answer, expected", [
    (["15000"], "Доплата — 15 000 рублей в неделю", 1.0),
    (["99.9"], "гарантируется 99,9% доступности", 1.0),
    (["3 мин", "retry-after"], "Откат занимает до 3 минут", 0.5),
    (["3 мин"], "не более **трёх минут**", 1.0),
    (["3 мин"], "внутри минуты", 0.0),  # "три" внутри слова — не число
    (["15000"], "15\u202f000 ₽", 1.0),
    (["go"], "Шлюз написан на Go.", 1.0),
    (["go"], "Используем Google Cloud", 0.0),  # короткий факт — только целым словом
    ([], "что угодно", None),
])
def test_key_fact_recall(facts, answer, expected):
    assert metrics.key_fact_recall(facts, answer) == expected


def test_judge_parses_json_and_survives_garbage(fake_llm):
    fake_llm(lambda kw: make_response(json.dumps({"verdict": "pass", "reason": "всё из контекста"})))
    assert metrics.judge_faithfulness("ctx", "ans") == {"verdict": "PASS", "reason": "всё из контекста"}

    fake_llm(lambda kw: make_response("Вердикт: PASS"))
    assert metrics.judge_faithfulness("ctx", "ans")["verdict"] == "ERROR"


def test_judge_uses_separate_model(fake_llm):
    from rag_agent import config
    fake = fake_llm(lambda kw: make_response('{"verdict": "CORRECT", "reason": ""}'))
    metrics.judge_correctness("q", "ref", "ans")
    assert fake.calls[0]["model"] == config.JUDGE_MODEL != config.LLM_MODEL


def test_cohens_kappa():
    from evals.judge_calibration import cohens_kappa
    assert cohens_kappa(["P", "F", "P", "F"], ["P", "F", "P", "F"]) == 1.0
    # согласие ровно на уровне случайного — kappa 0, хотя accuracy 50%
    assert cohens_kappa(["P", "P", "F", "F"], ["P", "F", "P", "F"]) == 0.0
