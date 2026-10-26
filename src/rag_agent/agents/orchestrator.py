"""orchestrator.py — собирает четырёх независимых агентов в один пайплайн.

    вопрос
      │
      ▼
   Guardrail Agent ──BLOCK──▶ вежливый отказ, дальше не идём (экономим вызовы)
      │ ALLOW
      ▼
   Query Rewriter Agent  (сырой вопрос → чистая формулировка)
      │
      ▼
   Main Agent (main_agent.py) — решает сам: search_documents / calculate / оба
      │
      ▼
   Critic Agent ──REVISE──▶ Main Agent пробует снова (максимум 1 раз)
      │ OK
      ▼
   ответ пользователю

Четыре роли, у каждой свой периметр ответственности — это то, что делает
это НАСТОЯЩИМ multi-agent, а не косметикой:
  Guardrail       — безопасность на входе (защита от prompt injection)
  Query Rewriter  — качество поискового запроса (реальный RAG-приём)
  Main Agent      — выполнение задачи через инструменты
  Critic          — обоснованность ответа на выходе

Честная цена: до 5 вызовов LLM на один вопрос в худшем случае (guardrail +
rewriter + agent + critic + 1 повтор agent'а). Для игрушечного демо это
избыточно; для системы, которая не должна врать и не должна вестись на
инъекции, — осознанный и оправданный trade-off.
"""
from ..observability.tracing import reset as reset_tracing
from .critic import critique
from .guardrail import check as guardrail_check
from .main_agent import run_agent_verbose
from .memory import trim_history
from .query_rewriter import rewrite as rewrite_query

BLOCKED_ANSWER = "Извините, не могу обработать этот запрос."


def run_pipeline(
    question: str,
    history: list[dict] | None = None,
    max_revisions: int = 1,
    verbose: bool = True,
) -> dict:
    """Полный пайплайн с подробностями: ответ, trace инструментов, решение
    guardrail'а и сколько было ревизий. Нужен evals/run.py — чтобы проверить
    faithfulness, судье нужен не только ответ, но и то, на чём он основан."""
    reset_tracing()  # новый запрос — новая статистика cost/latency (см. tracing.py)
    history = trim_history(history or [])  # см. memory.py — иначе контекст растёт бесконечно

    allowed, reason = guardrail_check(question)
    if verbose:
        print(f"  [guardrail] {'ALLOW' if allowed else 'BLOCK: ' + reason}")
    if not allowed:
        return {"answer": BLOCKED_ANSWER, "trace": [], "blocked": True, "revisions": 0}

    clean_question = rewrite_query(question, history=history)
    if verbose and clean_question != question:
        print(f"  [rewriter] '{question}' → '{clean_question}'")

    result = run_agent_verbose(clean_question, history=history, verbose=verbose)

    # Раньше здесь был баг: после REVISE агент перезапускался, но новый,
    # исправленный ответ уходил пользователю БЕЗ повторной проверки —
    # критик проверял только первый черновик, а не тот, что реально
    # отправляется. Теперь критик проверяет КАЖДУЮ версию ответа, включая
    # финальную после ревизии — цикл продолжается, пока либо критик не
    # скажет OK, либо не кончатся попытки.
    revisions_left = max_revisions
    while True:
        ok, crit_reason = critique(question, result["answer"], result["trace"])
        if verbose:
            print(f"  [критик] {'OK' if ok else 'REVISE (' + crit_reason + ')'}")
        if ok or revisions_left <= 0:
            break
        # передаём замечание критика обратно агенту как инструкцию на доработку
        result = run_agent_verbose(
            clean_question, history=history, extra_instruction=crit_reason, verbose=verbose
        )
        revisions_left -= 1

    # после max_revisions критик не блокирует ответ навсегда — лучше вернуть
    # неидеальный, но честно перепроверенный ответ, чем зависнуть в бесконечных
    # проверках. Но теперь это "неидеальный и мы это знаем", а не "не проверяли вовсе".
    return {**result, "blocked": False, "revisions": max_revisions - revisions_left}


def run_agent_with_review(
    question: str,
    history: list[dict] | None = None,
    max_revisions: int = 1,
    verbose: bool = True,
) -> str:
    """Только текст ответа — для api.py, MCP-сервера и CLI."""
    return run_pipeline(question, history=history, max_revisions=max_revisions, verbose=verbose)["answer"]


if __name__ == "__main__":
    q = input("Вопрос: ")
    print("\n" + run_agent_with_review(q))
