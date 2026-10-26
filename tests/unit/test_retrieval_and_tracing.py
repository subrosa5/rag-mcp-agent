from types import SimpleNamespace

from rag_agent.observability import tracing
from rag_agent.retrieval import hybrid


def test_hybrid_dedupes_by_id_before_rerank(monkeypatch):
    monkeypatch.setattr(hybrid, "vector_search", lambda q: [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}])
    monkeypatch.setattr(hybrid, "bm25_search", lambda q: [{"id": 2, "content": "b"}, {"id": 3, "content": "c"}])
    seen = {}

    def fake_rerank(q, candidates):
        seen["ids"] = sorted(c["id"] for c in candidates)
        return candidates[:2]
    monkeypatch.setattr(hybrid, "_rerank", fake_rerank)

    assert len(hybrid.hybrid_retrieve("q")) == 2
    assert seen["ids"] == [1, 2, 3]


def test_hybrid_empty_index(monkeypatch):
    monkeypatch.setattr(hybrid, "vector_search", lambda q: [])
    monkeypatch.setattr(hybrid, "bm25_search", lambda q: [])
    assert hybrid.hybrid_retrieve("q") == []


def test_tracing_cost_and_reset():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))

    tracing.traced_completion(client, "main", model="m")
    stats = tracing.get_stats()
    assert stats["total_llm_calls"] == 1
    assert stats["total_cost_usd"] == tracing.PRICE_PER_1M_INPUT_USD + tracing.PRICE_PER_1M_OUTPUT_USD

    tracing.reset()
    assert tracing.get_stats()["total_llm_calls"] == 0


def test_tracing_survives_missing_usage():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace())))
    tracing.traced_completion(client, "x", model="m")
    assert tracing.get_stats()["total_tokens"] == 0
