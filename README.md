# rag-mcp-agent

RAG + tool-use агент, реализованный без фреймворков (без LangChain/LlamaIndex) —
hybrid retrieval, реранкинг, tool-calling цикл и eval-харнес написаны напрямую
поверх Postgres/pgvector, Groq и Cohere.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![CI](https://github.com/subrosa5/rag-mcp-agent/actions/workflows/ci.yml/badge.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Render-009688?style=flat-square&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-Vercel-black?style=flat-square&logo=next.js)
![pgvector](https://img.shields.io/badge/pgvector-Neon-336791?style=flat-square&logo=postgresql)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)
![Cohere](https://img.shields.io/badge/Cohere-Rerank-39594d?style=flat-square)
![No LangChain](https://img.shields.io/badge/No-LangChain-critical?style=flat-square)

**Демо:** https://rag-agent-liard-two.vercel.app
**Стек:** Next.js (Vercel) → FastAPI (Render) → Postgres/pgvector (Neon), LLM и реранкинг через Groq/Cohere

## Что делает

Отвечает на вопросы по набору документов (RAG) и умеет вызывать инструменты —
сейчас это поиск по базе знаний и калькулятор. Модель сама решает, какой
инструмент вызвать и сколько раз, вместо жёсткого pipeline "всегда искать,
потом отвечать". Вопрос проходит через четырёх независимых агентов, у
каждого своя, нигде не дублирующаяся роль (подробно — ниже).

Пример: на вопрос *"Сколько времени занимает откат деплоя и сколько будет 12×7?"*
main-агент вызывает `search_documents` для первой части и `calculate` для второй,
и собирает оба результата в один ответ.

## Архитектура

```
   вопрос
     │
     ▼
  Guardrail Agent ──BLOCK──▶ вежливый отказ, дальше не идём
     │ ALLOW
     ▼
  Query Rewriter Agent   (сырой вопрос → чистая формулировка)
     │
     ▼
  ┌──────────────────┐
  │    MAIN AGENT     │◀── история tool-вызовов внутри одного запроса
  │ (Groq, gpt-oss-20b)│
  └────────┬──────────┘
           │ модель решает: нужен инструмент?
   ┌───────┴────────┐
   ▼                ▼
search_documents  calculate
   │
┌──┴───┐
▼      ▼
vector  bm25
search  search
   └──┬───┘
      ▼
  union по id → rerank (cross-encoder, top-20→top-4)
      │
      ▼
  ответ + trace (что искал, что нашёл)
      │
      ▼
  Critic Agent ──REVISE──▶ Main Agent пробует снова (макс. 1 раз)
      │ OK
      ▼
  ответ пользователю
```

## Агенты — что делает каждый и зачем он отдельный

Четыре агента, и ни один не дублирует роль другого — это то, что делает
систему настоящим multi-agent, а не косметикой ради модного слова.

| Агент | Файл | Что проверяет/делает | Почему это отдельный агент, а не часть другого |
|---|---|---|---|
| **Guardrail** | `agents/guardrail.py` | Вход: не является ли вопрос попыткой prompt injection ("забудь инструкции", "покажи системный промпт"). Блокирует до того, как запущены остальные агенты. | Безопасность на входе — своя, узкая ответственность. Смешивать с проверкой качества ответа (критик) нельзя: разная цена ошибки — здесь ложное срабатывание блокирует легитимный вопрос, у критика — просто лишняя переделка. |
| **Query Rewriter** | `agents/query_rewriter.py` | Чистит сырую формулировку вопроса (опечатки, сокращения, двусмысленность) перед тем, как её увидит retrieval. Не трогает вопрос, если переформулировка не нужна (например, арифметику). | Retrieval (эмбеддинги + BM25) чувствителен к формулировке — реальный приём в продакшн RAG (query rewriting/expansion), а не выдумка ради количества агентов. |
| **Main Agent** | `agents/main_agent.py` | Решает, какой инструмент вызвать (`search_documents` / `calculate`), сколько раз, и формулирует ответ. | Единственный агент с доступом к tool-calling — вся "работа" происходит тут, остальные три — обвязка вокруг него. |
| **Critic** | `agents/critic.py` | Выход: сверяет ответ Main Agent с тем, что он реально нашёл (trace). Если ответ не обоснован найденным — просит Main Agent переделать (1 попытка), с конкретной причиной. | Проверка ПОСЛЕ генерации — тот же принцип, что офлайн-судья в `evals/`, только встроен в живой запрос, а не считается отдельно на golden set. |

`agents/orchestrator.py` — не агент, а пайплайн, который вызывает всех четырёх по порядку и передаёт результат от одного к другому.

**Честная цена:** до 5 вызовов LLM на один вопрос в худшем случае (guardrail + rewriter + agent + critic + 1 повтор). Для демо это избыточно; для системы, которая не должна врать и не должна вестись на инъекции, — осознанный trade-off, не бесплатная фича. `/agent/ask-raw` в API отдаёт путь без критика — для сравнения задержки и честного разговора про эту цену.

## Ключевые решения

- **pgvector вместо отдельной vector DB.** Postgres с HNSW-индексом достаточен
  вплоть до миллионов векторов; отдельная инфраструктура не нужна для этого масштаба.
- **Hybrid search (vector + BM25).** Эмбеддинги ловят смысл, но плохо ловят
  точные токены — версии, коды, имена. BM25 закрывает этот случай. Кандидаты
  из обоих источников объединяются и фильтруются реранкингом.
- **Двухэтапный retrieval (bi-encoder → cross-encoder).** Bi-encoder кодирует
  вопрос и документ независимо — быстро, подходит для recall по всей базе.
  Cross-encoder читает пару вместе — точнее, но дорого считать на всей базе,
  поэтому применяется только к уже отобранным top-20 кандидатам.
- **RAG как инструмент агента, а не отдельный pipeline.** `generation/rag.py` — фиксированный
  retrieve→generate; `agents/main_agent.py` — модель сама решает, вызывать ли `search_documents`,
  `calculate`, оба или ни одного.
- **Калькулятор через `ast`, не `eval()`.** LLM генерирует выражение как строку;
  прямой `eval` — инъекция. `ast`-парсер разрешает только арифметику.
- **Два бэкенда эмбеддингов/реранкинга (local / Cohere), переключаемые `USE_COHERE`.**
  Self-hosted sentence-transformers держат в памяти ~1.3GB (обе модели) — не
  помещается в 512MB на Render free/starter tier. Cohere Embed+Rerank API
  переносит инференс наружу, локальный процесс почти ничего не занимает.
  Остальной код (`retrieval/hybrid.py`, `ingestion/pipeline.py`) не знает, какой бэкенд активен.
- **Четыре агента с непересекающимися ролями (guardrail/rewriter/main/critic),
  а не один агент с большим промптом.** У каждого своя, узкая ответственность
  и своя цена ошибки (см. таблицу "Агенты" выше) — это позволяет менять/тестировать
  каждого независимо и явно называть, за что отвечает конкретно этот вызов LLM.
- **Память между сообщениями — hard truncation, не суммаризация.** Фронт шлёт
  всю историю чата на каждый запрос, `agents/memory.py` обрезает её до последних
  10 сообщений перед тем, как она попадёт в промпт. Простая обрезка теряет
  старый контекст безвозвратно — осознанный выбор для этого масштаба, не
  недосмотр (суммаризация вместо отбрасывания — следующий шаг, см. roadmap).

## Известные ограничения

- **LLM-судья сам требует проверки.** Цифры faithfulness/correctness — мнение
  `gpt-oss-120b`, а не истина. Насколько ему можно верить, меряет
  `evals/judge_calibration.py` (согласие с ручной разметкой, Cohen's kappa);
  ручная разметка — следующий шаг.
- **Нет тенантов и ACL** — любой пользователь видит все документы (F-08 в
  `security/threat-model.md`). Открытые находки безопасности перечислены там же.
- **Корпус демонстрационный** — 12 внутренних регламентов вымышленной компании
  (`data/`), ~23 KB.

## Структура

```
rag-agent/
├── src/rag_agent/
│   ├── config.py              # параметры, переключатель local/Cohere, модель судьи
│   ├── db.py                   # слой поверх psycopg, без ORM
│   ├── llm.py                   # единственный LLM-клиент на процесс (ретраи, точка для каскада моделей)
│   ├── ingestion/               # data/*.txt → чанки → эмбеддинги → Postgres (python -m rag_agent.ingestion)
│   ├── retrieval/               # embeddings.py (local | Cohere), hybrid.py (vector + BM25 + rerank)
│   ├── generation/rag.py        # retrieve + generate, без агента
│   ├── agents/                  # guardrail, query_rewriter, main_agent, critic, orchestrator, memory
│   ├── tools/                   # search, calculator (ast, без eval), finance; registry — схемы для LLM
│   ├── observability/tracing.py # cost/latency каждого LLM-вызова
│   ├── api/app.py               # FastAPI: auth, rate limit, валидация, эндпоинты
│   └── mcp/server.py            # тот же движок как MCP-сервер (stdio)
├── evals/
│   ├── datasets/golden.jsonl    # 63 вопроса: single_hop, multi_hop, exact_token, colloquial, unanswerable
│   ├── metrics.py               # retrieval (hit/recall/MRR), key facts, LLM-судья (faithfulness, correctness)
│   ├── run.py                   # прогон → отчёт → сравнение с baseline → гейт для CI
│   ├── judge_calibration.py     # согласие судьи с ручной разметкой
│   ├── baseline_*.json          # зафиксированные цифры, от которых считается Δ
│   └── thresholds.json          # пороги гейта
├── tests/                       # unit (без сети, LLM подменён фейком) + integration (живой pgvector)
├── security/threat-model.md     # активы, границы доверия, STRIDE, OWASP LLM Top 10, находки
├── .github/workflows/           # ci (lint, mypy, tests, web) · security (SAST, CVE, секреты) · evals (гейт)
├── data/*.txt                    # корпус
├── web/                          # Next.js чат-фронт (Vercel)
├── pyproject.toml + uv.lock      # зависимости; extras: local (torch), mcp
└── render.yaml                   # Render Blueprint
```

## Локальный запуск

```bash
cp .env.example .env   # заполнить GROQ_API_KEY; USE_COHERE=false по умолчанию

uv sync --all-extras --group dev    # .venv с зависимостями (uv: https://docs.astral.sh/uv/)

uv run python -m rag_agent.ingestion             # индексация data/*.txt
uv run python -m rag_agent.generation.rag        # RAG без агента
uv run python -m rag_agent.agents.orchestrator   # полный пайплайн с агентами
uv run uvicorn rag_agent.api.app:app --reload    # HTTP API
```

## Тесты, evals, безопасность

```bash
uv run ruff check . && uv run mypy                # линтер (включая security-правила S) и типы
uv run pytest                                     # unit-тесты: без сети и без ключей, < 1 с
TEST_DATABASE_URL=postgresql://localhost/rag_agent_test uv run pytest -m integration

uv run python -m evals.run                        # golden set через RAG-пайплайн
uv run python -m evals.run --pipeline full --gate # полный пайплайн, exit 1 при падении метрик
```

Метрики считаются по слоям, чтобы было видно, **где** сломалось:
retrieval (hit@k, recall@k, MRR — без LLM), key facts (детерминированная
проверка подстрок) и LLM-судья (faithfulness, correctness). Судья —
`gpt-oss-120b`, то есть другая и более сильная модель, чем отвечающая
`gpt-oss-20b`: модель, проверяющая сама себя, прощает себе свои же ошибки.

Безопасность — `security/threat-model.md`: границы доверия, STRIDE, маппинг на
OWASP Top 10 for LLM (2025) и таблица находок со статусами. SAST (ruff `S`,
bandit, semgrep), CVE в зависимостях (pip-audit, npm audit) и секреты в git
(gitleaks) гоняются в CI на каждый PR и раз в неделю.

## MCP-сервер

Тот же агент доступен не только по HTTP, но и как [MCP](https://modelcontextprotocol.io)-сервер —
`src/rag_agent/mcp/server.py` тонкой обвязкой выставляет `ask_agent`, `ask_agent_fast`,
`search_documents` и `rag_answer` как MCP tools, дёргая ровно те же функции,
что и `api.py` (никакой отдельной логики, два транспорта поверх одного движка).

```bash
uv sync --extra mcp
claude mcp add --transport stdio rag-agent -- uv run --directory $(pwd) python -m rag_agent.mcp.server
```

После этого любой MCP-клиент (Claude Code, Claude Desktop, Cursor) может
спрашивать базу знаний этого проекта напрямую, тем же самым пайплайном
(guardrail → query rewriter → main agent → critic), что работает на проде.

## Деплой

Прод использует Cohere вместо self-hosted моделей и Neon вместо локального Postgres.

1. **Neon** — Postgres с pgvector, провизионится через Vercel Marketplace
   (`vercel integration add neon`) или напрямую на neon.tech.
2. **Render** — подключить GitHub-репозиторий, он подхватит `render.yaml`.
   Секреты задаются в дашборде: `GROQ_API_KEY`, `COHERE_API_KEY`,
   `DATABASE_URL` (из Neon), `API_KEY` (общий секрет с фронтом).
3. `uv run python -m rag_agent.ingestion` один раз против `DATABASE_URL` от Neon — наполняет базу.
4. **Vercel** — `web/` деплоится отдельным проектом, env `RAG_AGENT_API_URL`
   указывает на Render-сервис, `BACKEND_API_KEY` — тот же секрет, что `API_KEY` на Render.

## Метрики

Baseline на golden set (63 вопроса, RAG-пайплайн, локальные эмбеддинги,
`evals/baseline_rag.json`). Гейт в CI — `evals/thresholds.json`: это baseline
минус запас на шум LLM.

| категория | n | hit@k | MRR | key facts | faithfulness | correctness |
|---|---|---|---|---|---|---|
| single_hop | 31 | 1.00 | 0.98 | 0.92 | 0.94 | 0.94 |
| multi_hop | 10 | 1.00 | 1.00 | 0.90 | 0.90 | 0.95 |
| exact_token | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| colloquial | 6 | **0.83** | 0.83 | 0.83 | 1.00 | 0.83 |
| unanswerable | 8 | — | — | — | 0.88 | 0.88 |
| **всего** | 63 | 0.98 | 0.97 | 0.92 | 0.94 | 0.91 |

p50 latency 777 ms, p95 1.2 s, $0.00015 за вопрос.

Что видно из разбора провалов:
- **Разговорные формулировки — слабое место retrieval.** «ноут спёрли что
  делать» не находит политику безопасности. Это работа для query rewriting
  перед поиском, которого в RAG-пайплайне нет.
- **Ловушка без ответа:** на вопрос о месячной цене Enterprise модель вывела
  конкретное число, которого в базе нет.
- **Сам судья тоже ошибался:** он ставил PARTIAL, когда в ответе не было
  деталей эталона, о которых вопрос не спрашивал. Промпт судьи исправлен
  (`--rejudge` переоценивает сохранённые ответы без перегенерации), поэтому
  калибровка судьи по ручной разметке — обязательный следующий шаг.
- **Полный мульти-агентный пайплайн** на частичном прогоне (49 вопросов)
  ответил хуже чистого RAG. Его baseline ещё не снят: прогон упёрся в дневной
  лимит токенов Groq (200K/день на модель на бесплатном тарифе).

## Roadmap

- Baseline полного пайплайна и разбор, почему агенты хуже чистого RAG
- Ручная разметка для калибровки судьи (`evals/judge_calibration.py`)
- OpenTelemetry → Langfuse вместо самописного трекинга
- BM25 в Postgres FTS вместо `rank_bm25` в памяти; contextual chunking; router и каскад моделей
- Тенанты и ACL-фильтр в retrieval; закрыть открытые находки из `security/threat-model.md`
- Суммаризация старой истории вместо hard truncation; streaming ответа
