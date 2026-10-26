"""conftest.py — общие фикстуры.

Главное правило юнит-тестов здесь: НИ ОДНОГО сетевого вызова. Groq, Cohere и
Yahoo подменяются фейками, иначе тесты становятся медленными, платными и
недетерминированными (LLM на один и тот же вопрос отвечает по-разному).

Переменные окружения выставляются ДО импорта rag_agent: config.py читает их
на этапе импорта. USE_COHERE=true — чтобы embeddings.py не тянул torch и
sentence-transformers (полминуты на импорт и ~1GB RAM в CI ради тестов,
которые эмбеддинги всё равно подменяют).
"""
import os

os.environ["USE_COHERE"] = "true"
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")

import pytest  # noqa: E402

from rag_agent import llm  # noqa: E402
from rag_agent.observability import tracing  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402


@pytest.fixture
def fake_llm(monkeypatch):
    """Использование: fake = fake_llm(lambda kwargs: make_response("ALLOW"))."""
    def install(handler) -> FakeLLM:
        fake = FakeLLM(handler)
        monkeypatch.setattr(llm, "get_client", lambda: fake)
        return fake
    return install


@pytest.fixture(autouse=True)
def _fresh_tracing():
    tracing.reset()
    yield
