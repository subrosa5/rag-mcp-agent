"""mcp_server.py — MCP-обвязка поверх существующего RAG-агента.

Зачем отдельный файл, а не ещё один HTTP-роут в api.py: MCP и HTTP — два
разных транспорта для одного и того же движка. api.py остаётся тонким
слоем для фронта (Next.js), этот файл — тонкий слой для MCP-клиентов
(Claude Desktop, Claude Code, Cursor и т.д.). Сама логика (agent.py,
orchestrator.py, rag.py, retrieval.py) используется как есть, без
дублирования — ровно тот же код, что отвечает на проде.

Запуск (stdio, для локального MCP-клиента):
    python -m src.mcp_server

Регистрация в Claude Code:
    claude mcp add --transport stdio rag-agent -- python -m src.mcp_server
"""
from mcp.server.mcpserver import MCPServer

from .agent import run_agent
from .orchestrator import run_agent_with_review
from .rag import answer as compute_rag_answer
from .retrieval import hybrid_retrieve

mcp = MCPServer("rag-agent")


@mcp.tool()
def ask_agent(question: str) -> str:
    """Задать вопрос базе знаний через полный multi-agent пайплайн
    (guardrail → query rewriter → main agent с tool-calling → critic).
    Медленнее и дороже, чем ask_agent_fast, зато critic проверяет
    обоснованность ответа перед тем, как его вернуть. Используй по
    умолчанию — это тот же путь, что и продакшн-эндпоинт /agent/ask."""
    return run_agent_with_review(question, verbose=False)


@mcp.tool()
def ask_agent_fast(question: str) -> str:
    """То же самое, но без critic-агента — один проход main-агента вместо
    двух-трёх вызовов LLM. Используй, когда важна скорость/стоимость, а не
    максимальная обоснованность ответа."""
    return run_agent(question, verbose=False)


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
    сам решает, нужен ли вообще поиск."""
    return compute_rag_answer(question)


if __name__ == "__main__":
    mcp.run()
