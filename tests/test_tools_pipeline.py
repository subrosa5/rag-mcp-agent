import importlib
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from src import agent, config, embeddings


def response(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls or []))])


def call(name, arguments):
    return NS(id="call1", function=NS(name=name, arguments=arguments))


@pytest.mark.parametrize(
    "name,args",
    [
        ("search_documents", '{"query":7}'),
        ("unknown", "{}"),
        ("search_documents", "[]"),
        ("calculate", "{broken"),
    ],
)
def test_bad_tool_arguments_cannot_become_evidence(monkeypatch, name, args):
    monkeypatch.setattr(agent, "Groq", lambda **k: object())
    completion = Mock(
        side_effect=[response(calls=[call(name, args)]), response(content="invented")]
    )
    monkeypatch.setattr(agent, "traced_completion", completion)
    result = agent.run_agent_verbose("q")
    assert result["status"] == "insufficient_evidence"
    assert result["evidence"] == []


def test_successful_tool_supplies_exact_snapshot_to_generator(monkeypatch):
    monkeypatch.setattr(agent, "Groq", lambda **k: object())
    monkeypatch.setattr(
        agent,
        "traced_completion",
        Mock(
            side_effect=[
                response(calls=[call("calculate", '{"expression":"12*7"}')]),
                response(content="84"),
            ]
        ),
    )
    generate = Mock(
        return_value={"status": "answered", "answer": "84 [1]", "sources": [], "claims": []}
    )
    monkeypatch.setattr(agent, "generate", generate)
    result = agent.run_agent_verbose("12*7")
    assert result["evidence"][0]["content"] == "12*7 = 84"
    assert generate.call_args.args[2] == result["evidence"]


def test_provider_error_not_leaked(monkeypatch):
    monkeypatch.setattr(agent, "Groq", lambda **k: object())
    monkeypatch.setattr(
        agent,
        "traced_completion",
        Mock(
            side_effect=[
                response(calls=[call("search_documents", '{"query":"q"}')]),
                response(content="invented"),
            ]
        ),
    )
    monkeypatch.setitem(
        agent.TOOL_FUNCTIONS,
        "search_documents",
        Mock(side_effect=RuntimeError("postgres://secret")),
    )
    result = agent.run_agent_verbose("q")
    assert "secret" not in str(result)
    assert result["status"] == "unavailable"


def test_cohere_embedding_batching_and_shape(monkeypatch):
    monkeypatch.setattr(config, "USE_COHERE", True)
    monkeypatch.setattr(config, "EMBEDDING_DIM", 3)
    importlib.reload(embeddings)
    fake = Mock()
    fake.embed.side_effect = lambda **kw: NS(
        embeddings=NS(float_=[[1.0, 0.0, 0.0]] * len(kw["texts"]))
    )
    monkeypatch.setattr(embeddings, "_client", lambda: fake)
    try:
        assert embeddings.embed(["doc"] * 193).shape == (193, 3)
        assert [len(c.kwargs["texts"]) for c in fake.embed.call_args_list] == [96, 96, 1]
        fake.embed.side_effect = lambda **kw: NS(embeddings=NS(float_=[[float("nan")] * 3]))
        with pytest.raises(ValueError):
            embeddings.embed(["doc"])
    finally:
        monkeypatch.undo()
        importlib.reload(embeddings)
