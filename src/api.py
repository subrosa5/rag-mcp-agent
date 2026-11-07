"""Bounded HTTP transport. Public demo mode must be explicitly enabled."""

import hmac
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .orchestrator import answer_agent
from .rag import answer as rag_answer

app = FastAPI(title="Evidence-backed RAG agent", version="0.2.0")
_allowed = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)
_API_KEY = os.environ.get("API_KEY") or None
_DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
_RATE_LIMIT, _RATE_WINDOW_SEC = 20, 60
_request_log: dict[str, deque] = {}
_lock = threading.Lock()
_concurrent_semaphore = threading.BoundedSemaphore(5)


def _client_key(request: Request) -> str:
    # An arbitrary unvalidated header must never create a fresh rate-limit bucket.
    return "authenticated" if _API_KEY else (request.client.host if request.client else "unknown")


def enforce_limits(request: Request) -> None:
    if not _API_KEY and not _DEMO_MODE:
        raise HTTPException(503, "Configure API_KEY or explicitly enable DEMO_MODE")
    if _API_KEY and not hmac.compare_digest(
        request.headers.get("X-API-Key", "").encode(), _API_KEY.encode()
    ):
        raise HTTPException(401, "Invalid API key")
    key, now = _client_key(request), time.monotonic()
    with _lock:
        for old in list(_request_log):
            if not _request_log[old] or now - _request_log[old][-1] >= _RATE_WINDOW_SEC:
                del _request_log[old]
        if key not in _request_log and len(_request_log) >= 10000:
            raise HTTPException(429, "Rate limit capacity reached", headers={"Retry-After": "60"})
        entries = _request_log.setdefault(key, deque())
        while entries and now - entries[0] >= _RATE_WINDOW_SEC:
            entries.popleft()
        if len(entries) >= _RATE_LIMIT:
            raise HTTPException(429, "Too many requests", headers={"Retry-After": "60"})
        entries.append(now)


@contextmanager
def capacity():
    if not _concurrent_semaphore.acquire(blocking=False):
        raise HTTPException(503, "Service busy", headers={"Retry-After": "5"})
    try:
        yield
    finally:
        _concurrent_semaphore.release()


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "insufficient_evidence", "blocked", "unavailable"]
    reason: str | None = None
    sources: list[dict] = Field(default_factory=list)
    claims: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)


def public_result(result: dict) -> dict:
    if result["status"] == "unavailable":
        raise HTTPException(
            503,
            "Answer dependencies unavailable",
            headers={"Retry-After": str(result.get("retry_after", 10))},
        )
    # Do not expose raw tool traces or unused retrieved documents over HTTP.
    return AskResponse.model_validate(result).model_dump()


@app.post("/agent/ask", response_model=AskResponse, dependencies=[Depends(enforce_limits)])
def ask_agent(req: AskRequest):
    with capacity():
        return public_result(answer_agent(req.question, [m.model_dump() for m in req.history]))


@app.post(
    "/agent/ask-raw",
    response_model=AskResponse,
    dependencies=[Depends(enforce_limits)],
    deprecated=True,
)
def ask_agent_raw(req: AskRequest):
    # Compatibility alias; skipping validation is no longer a public capability.
    return ask_agent(req)


@app.post("/rag/ask", response_model=AskResponse, dependencies=[Depends(enforce_limits)])
def ask_rag(req: AskRequest):
    with capacity():
        return public_result(rag_answer(req.question))


@app.get("/health")
def health():
    # Liveness only, deliberately no paid inference.
    return {"status": "ok"}
