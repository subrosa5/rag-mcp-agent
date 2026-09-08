"""api.py — тонкий HTTP-слой поверх агента.

Запуск: uvicorn src.api:app --reload
Зачем: "production-ready" в резюме проекта значит, что систему можно
дёрнуть не только из терминала. Сам агент (agent.py) от FastAPI не зависит —
это намеренно: логика и транспорт разделены, api.py можно заменить на
CLI/Slack-бота/что угодно, не трогая agent.py.
"""
import os
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import run_agent
from .orchestrator import run_agent_with_review
from .rag import answer as rag_answer

app = FastAPI(title="RAG + Agent lab")

# Фронт (Next.js на Vercel) и бэкенд (Render) — разные домены, значит браузер
# будет делать cross-origin запрос, и без CORS-заголовков он его заблокирует.
# ALLOWED_ORIGIN — env-переменная, а не хардкод домена, чтобы не редактировать
# код при каждом передеплое фронта на новый preview-URL.
_allowed_origin = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin] if _allowed_origin != "*" else ["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# --- защита: auth, rate limit, конкурентность ---
# Без этого блока /agent/ask мог дёргать кто угодно, сколько угодно раз в
# секунду, и каждый вызов — это до 5 запросов к Groq (см. orchestrator.py) —
# то есть чужой скрипт может незаметно съесть весь платный лимит за минуты.

# API_KEY не задан -> эндпоинт открыт (удобно для локальной разработки).
# Задан -> каждый запрос обязан прислать точно такой же заголовок X-API-Key.
_API_KEY = os.environ.get("API_KEY")

# rate limit: простое скользящее окно в памяти процесса, без Redis — этого
# достаточно для одного инстанса лабы. На нескольких инстансах за балансировщиком
# каждый инстанс считал бы отдельно — для прод-масштаба нужен внешний стор
# (Redis), это осознанное упрощение, не забытый пункт.
_RATE_LIMIT = 20          # запросов
_RATE_WINDOW_SEC = 60     # за столько секунд, на один ключ (IP или API-ключ)
_request_log: dict[str, deque] = defaultdict(deque)

# ограничение одновременных дорогих запросов (агент делает до 5 LLM-вызовов
# подряд) — без этого 50 параллельных запросов запустят 50 параллельных
# цепочек вызовов к Groq одновременно.
# threading.Semaphore, а не asyncio.Semaphore: эндпоинты ниже — обычные
# (синхронные) def-функции, FastAPI сам крутит их в пуле потоков, поэтому
# примитив синхронизации тоже должен быть "потоковым", не asyncio-шным —
# иначе он молча не будет ничего ограничивать.
import threading

_MAX_CONCURRENT = 5
_concurrent_semaphore = threading.Semaphore(_MAX_CONCURRENT)


def _client_key(request: Request) -> str:
    """Ключ для rate limit: API-ключ, если он есть в заголовке, иначе IP.
    Так честные клиенты с ключом не делят лимит с анонимными по той же сети."""
    return request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")


def enforce_limits(request: Request) -> None:
    """FastAPI-зависимость: проверяет API-ключ (если настроен) и rate limit
    ДО того, как эндпоинт начнёт что-либо делать — отклонённый запрос не
    стоит ни одного вызова Groq."""
    if _API_KEY is not None:
        if request.headers.get("X-API-Key") != _API_KEY:
            raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")

    key = _client_key(request)
    now = time.monotonic()
    log = _request_log[key]
    while log and now - log[0] > _RATE_WINDOW_SEC:
        log.popleft()
    if len(log) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Слишком много запросов — лимит {_RATE_LIMIT} в {_RATE_WINDOW_SEC} сек.",
        )
    log.append(now)


class HistoryMessage(BaseModel):
    # Literal, а не str: клиент не может передать role="system" и незаметно
    # подсунуть модели поддельную системную инструкцию через историю диалога
    # (prompt injection через историю) — Pydantic отклонит такой запрос
    # с 422 ещё до того, как он дойдёт до agent.py.
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8_000)


class AskRequest(BaseModel):
    # min_length=1 отсекает пустую строку прямо на уровне валидации запроса —
    # без этого пустой вопрос доходил бы до retrieval/Groq и тратил впустую
    # вызов внешнего API. FastAPI сам вернёт 422 с понятной ошибкой, наш код
    # даже не запустится. max_length — защита от гигантского вопроса, который
    # раздует промпт и стоимость одного вызова.
    question: str = Field(min_length=1, max_length=4_000)
    # history приходит с фронта целиком (см. web/app/Chat.tsx) — фронт не
    # обрезает её сам, обрезка под контекстное окно — забота бэкенда
    # (src/memory.py), а не каждого клиента по отдельности. max_length здесь —
    # защита от клиента, который в обход фронта пришлёт историю на 10 000 сообщений.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)


class AskResponse(BaseModel):
    answer: str


@app.post("/agent/ask", response_model=AskResponse, dependencies=[Depends(enforce_limits)])
def ask_agent(req: AskRequest):
    """Агент + критик (multi-agent, reflection-паттерн): второй независимый
    вызов LLM проверяет ответ первого перед тем, как отдать пользователю.
    Дороже и медленнее одного агента (см. orchestrator.py), но это то, что
    реально отвечает на фронте — сознательный выбор в пользу надёжности."""
    history = [m.model_dump() for m in req.history]
    with _concurrent_semaphore:
        return {"answer": run_agent_with_review(req.question, history=history, verbose=False)}


@app.post("/agent/ask-raw", response_model=AskResponse, dependencies=[Depends(enforce_limits)])
def ask_agent_raw(req: AskRequest):
    """Тот же агент, но БЕЗ критика — один вызов вместо двух-трёх. Для
    сравнения задержки/поведения и как честный пример, что multi-agent —
    не бесплатная надстройка, а конкретный trade-off надёжность/стоимость."""
    history = [m.model_dump() for m in req.history]
    with _concurrent_semaphore:
        return {"answer": run_agent(req.question, history=history, verbose=False)}


@app.post("/rag/ask", dependencies=[Depends(enforce_limits)])
def ask_rag(req: AskRequest):
    """Чистый RAG без агентного цикла — для сравнения и для случаев,
    когда нужен именно фиксированный pipeline, а не решение модели."""
    with _concurrent_semaphore:
        return rag_answer(req.question)


@app.get("/health")
def health():
    return {"status": "ok"}
