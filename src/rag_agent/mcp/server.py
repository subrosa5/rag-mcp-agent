"""mcp/server.py — MCP-обвязка поверх существующего RAG-агента.

Зачем отдельный файл, а не ещё один HTTP-роут в api.py: MCP и HTTP — два
разных транспорта для одного и того же движка. api.py остаётся тонким
слоем для фронта (Next.js), этот файл — тонкий слой для MCP-клиентов
(Claude Desktop, Claude Code, Cursor и т.д.). Сама логика (agent.py,
orchestrator.py, rag.py, retrieval.py) используется как есть, без
дублирования — ровно тот же код, что отвечает на проде.

Запуск (stdio, для локального MCP-клиента):
    python -m rag_agent.mcp.server

Регистрация в Claude Code:
    claude mcp add --transport stdio rag-agent -- python -m rag_agent.mcp.server
"""
from mcp.server.mcpserver import MCPServer

from ..agents.main_agent import run_agent
from ..agents.orchestrator import run_agent_with_review
from ..generation.rag import answer as compute_rag_answer
from ..observability.tracing import get_stats
from ..retrieval.hybrid import hybrid_retrieve
from ..tools.finance import get_stock_quote as fetch_stock_quote

mcp = MCPServer("rag-mcp-agent")


@mcp.tool()
def ask_agent(question: str) -> dict:
    """Задать вопрос базе знаний через полный multi-agent пайплайн
    (guardrail → query rewriter → main agent с tool-calling → critic).
    Медленнее и дороже, чем ask_agent_fast, зато critic проверяет
    обоснованность ответа перед тем, как его вернуть. Используй по
    умолчанию — это тот же путь, что и продакшн-эндпоинт /agent/ask.
    Возвращает ответ вместе со stats (латентность/токены/стоимость каждого
    из четырёх агентов, см. tracing.py) — для контроля cost/latency."""
    answer = run_agent_with_review(question, verbose=False)
    return {"answer": answer, "stats": get_stats()}


@mcp.tool()
def ask_agent_fast(question: str) -> dict:
    """То же самое, но без critic-агента — один проход main-агента вместо
    двух-трёх вызовов LLM. Используй, когда важна скорость/стоимость, а не
    максимальная обоснованность ответа. Тоже возвращает stats."""
    answer = run_agent(question, verbose=False)
    return {"answer": answer, "stats": get_stats()}


@mcp.tool()
def search_documents(question: str, k: int = 4) -> list[dict]:
    """Чистый retrieval без обращения к LLM: hybrid search (векторный +
    BM25) с реранкингом. Возвращает top-k релевантных чанков документов
    как есть, без генерации ответа — полезно, когда нужны сырые источники,
    а не сформулированный текст."""
    return hybrid_retrieve(question)[:k]


@mcp.tool()
def rag_answer(question: str) -> dict:
    """Фиксированный RAG-пайплайн без агентного принятия решений (всегда
    ищет по документам и отвечает) — для сравнения с ask_agent, который
    сам решает, нужен ли вообще поиск. Уже включает stats (rag.py сам
    прикладывает их к ответу)."""
    return compute_rag_answer(question)


@mcp.tool()
def get_stock_quote(ticker: str) -> dict:
    """Текущая цена акции по биржевому тикеру (например AAPL, TSLA) —
    live-данные с Yahoo Finance, без API-ключа. Тот же инструмент, что
    main-агент может вызвать сам внутри ask_agent (см. tools.py); здесь
    доступен и напрямую, в обход агентного цикла."""
    return fetch_stock_quote(ticker)


if __name__ == "__main__":
    mcp.run()
