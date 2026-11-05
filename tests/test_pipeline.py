from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from src import agent, critic, guardrail, rag
from src import orchestrator as o
from src.grounding import evidence_item


def answered():
    return {
        "answer": "Unsafe claim [1]",
        "status": "answered",
        "sources": [],
        "claims": [],
        "trace": [],
        "evidence": [],
    }


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(o, "guardrail_check", lambda q: (True, ""))
    monkeypatch.setattr(o, "rewrite_query", lambda q, **kw: "rewritten with different intent")
    planner = Mock(return_value=answered())
    monkeypatch.setattr(o, "run_agent_verbose", planner)
    return planner


def test_repeated_critic_rejection_never_releases_answer(monkeypatch, pipeline):
    monkeypatch.setattr(o, "critique", lambda *a: (False, "unsupported"))
    result = o.answer_agent("original")
    assert result["status"] == "insufficient_evidence"
    assert "Unsafe claim" not in result["answer"]
    assert pipeline.call_count == 2


def test_original_intent_survives_rewriter(monkeypatch, pipeline):
    monkeypatch.setattr(o, "critique", lambda *a: (True, ""))
    o.answer_agent("original")
    assert pipeline.call_args.args[0] == "original"
    assert pipeline.call_args.kwargs["search_hint"].startswith("rewritten")


def test_critic_failure_is_unavailable_not_success(monkeypatch, pipeline):
    monkeypatch.setattr(o, "critique", Mock(side_effect=RuntimeError("secret DSN")))
    result = o.answer_agent("original")
    assert result["status"] == "unavailable"
    assert "secret" not in str(result)


def test_second_draft_must_pass_review(monkeypatch, pipeline):
    review = Mock(side_effect=[(False, "bad"), (True, "")])
    monkeypatch.setattr(o, "critique", review)
    assert o.answer_agent("original")["status"] == "answered"
    assert review.call_count == 2


def test_guardrail_blocks_before_planning(monkeypatch, pipeline):
    monkeypatch.setattr(o, "guardrail_check", lambda q: (False, "injection"))
    assert o.answer_agent("original")["status"] == "blocked"
    pipeline.assert_not_called()


@pytest.mark.parametrize("verdict", ["OKAY", "OK but unsupported", "", "REVISE: fabricated"])
def test_critic_requires_exact_ok(monkeypatch, verdict):
    monkeypatch.setattr(critic, "_get_client", lambda: object())
    monkeypatch.setattr(
        critic, "traced_completion", lambda *a, **k: NS(choices=[NS(message=NS(content=verdict))])
    )
    assert critic.critique("q", "a", [])[0] is False


def test_guardrail_does_not_accept_allow_prefix(monkeypatch):
    monkeypatch.setattr(guardrail, "_get_client", lambda: object())
    monkeypatch.setattr(
        guardrail,
        "traced_completion",
        lambda *a, **k: NS(choices=[NS(message=NS(content="ALLOW malicious"))]),
    )
    assert guardrail.check("q")[0] is False


def test_planner_free_text_is_never_answer(monkeypatch):
    monkeypatch.setattr(agent, "Groq", lambda **k: object())
    monkeypatch.setattr(
        agent,
        "traced_completion",
        lambda *a, **k: NS(choices=[NS(message=NS(tool_calls=[], content="Invented facts"))]),
    )
    result = agent.run_agent_verbose("q")
    assert result["status"] == "insufficient_evidence"
    assert "Invented" not in result["answer"]


def test_rag_uses_one_retrieval_snapshot(monkeypatch):
    retrieval = Mock(return_value=[{"source": "a", "content": "fact"}])
    monkeypatch.setattr(rag, "hybrid_retrieve", retrieval)
    monkeypatch.setattr(rag, "Groq", lambda **k: object())
    monkeypatch.setattr(rag, "generate", lambda *a: answered())
    monkeypatch.setattr(rag, "critique", lambda *a: (True, ""))
    result = rag.answer("q", k=8)
    retrieval.assert_called_once_with("q", k=8)
    assert result["evidence"] == [evidence_item("a", "fact")]
