"""tracing.py — cost/latency трекинг поверх Groq-вызовов.

Зачем отдельный модуль, а не time.time() в каждом файле: четыре агента
(guardrail, query_rewriter, agent, critic) дёргают Groq независимо друг от
друга, и без единой точки сбора статистики нет способа ответить на вопрос
"сколько стоил этот запрос и где ушло время" — а это ровно то, что требуют
от продакшн-LLM-систем: контроль cost/latency, а не только "работает ли
ответ" (см. eval.py — тот отвечает на другой вопрос, "правда ли ответ").

Механика:
  - traced_completion() — тонкая обёртка вокруг client.chat.completions.create,
    вызывается вместо него в каждом из пяти мест (guardrail, query_rewriter,
    agent, critic, rag).
  - Записи копятся в contextvar, а не в обычной module-level переменной —
    так параллельные запросы в одном процессе (несколько uvicorn-воркеров/
    asyncio-тасков) не путают чужую статистику между собой.
  - reset() вызывается в начале запроса (orchestrator.py / rag.py / mcp_server.py),
    get_stats() — в конце, чтобы приложить статистику к ответу.

Известное ограничение: цены ниже захардкожены на дату написания (Groq,
openai/gpt-oss-20b) — для реального прода их надо тянуть из конфига или
billng API, а не хардкодить. Это демонстрация механизма трекинга, не
биллинговая система.
"""

import os
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass

from . import config

PRICE_PER_1M_INPUT_USD = float(os.environ.get("LLM_INPUT_USD_PER_MILLION", "0.075"))
PRICE_PER_1M_OUTPUT_USD = float(os.environ.get("LLM_OUTPUT_USD_PER_MILLION", "0.30"))


@dataclass
class CallRecord:
    agent: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


_calls: ContextVar[list[CallRecord] | None] = ContextVar("_calls", default=None)


def reset() -> None:
    """Начать новый запрос — сбросить накопленные записи текущего контекста."""
    _calls.set([])


def traced_completion(client, agent: str, **kwargs):
    """Замена client.chat.completions.create(**kwargs) — тот же вызов,
    плюс запись latency/токенов/стоимости под именем `agent` (guardrail,
    query_rewriter, main[step N], critic, rag)."""
    calls = _calls.get()
    if calls is None:
        # reset() не вызвали заранее (например, вызов вне orchestrator.py) —
        # не роняем запрос из-за отсутствующей телеметрии, просто заводим
        # список на лету.
        calls = []
        _calls.set(calls)

    t0 = time.perf_counter()
    kwargs.setdefault("max_completion_tokens", config.MAX_COMPLETION_TOKENS)
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost = (
        prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_USD
        + completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_USD
    )

    calls.append(
        CallRecord(
            agent=agent,
            model=kwargs.get("model", ""),
            latency_ms=round(latency_ms, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6),
        )
    )
    return resp


def get_stats() -> dict:
    """Итоговая статистика по всем LLM-вызовам текущего запроса — сколько
    агентов дёрнули Groq, сколько это заняло по времени и в деньгах."""
    calls = _calls.get() or []
    return {
        "cost_scope": "estimated LLM only; excludes embeddings/reranking; configure prices when changing models",
        "calls": [asdict(c) for c in calls],
        "total_llm_calls": len(calls),
        "total_latency_ms": round(sum(c.latency_ms for c in calls), 1),
        "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls),
        "total_cost_usd": round(sum(c.cost_usd for c in calls), 6),
    }
