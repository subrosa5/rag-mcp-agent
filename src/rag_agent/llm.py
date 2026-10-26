"""llm.py — единственная точка, где создаётся клиент LLM-провайдера.

Раньше каждый агент (guardrail, query_rewriter, critic, rag, main agent)
держал свою копию `_get_client()` — пять одинаковых кусков кода, и main agent
вообще создавал новый Groq-клиент на каждый вопрос. Вынесено сюда, чтобы:
  - был один клиент на процесс (переиспользуется HTTP connection pool);
  - тесты подменяли клиента в ОДНОМ месте (tests/conftest.py), а не в пяти;
  - каскад моделей (дешёвая для router/guardrail, сильная для ответа)
    добавлялся здесь, а не размазывался по агентам.
"""
from functools import lru_cache

from groq import Groq

from . import config


@lru_cache(maxsize=1)
def get_client() -> Groq:
    # max_retries: SDK сам повторяет 429/5xx с экспоненциальной паузой. Дефолт
    # (2) не переживает прогон evals на бесплатном тарифе Groq — там лимит
    # запросов в минуту, а один вопрос через полный пайплайн это 4-6 вызовов.
    return Groq(api_key=config.GROQ_API_KEY, max_retries=6)
