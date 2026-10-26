"""metrics.py — метрики eval-харнеса, по слоям.

Почему по слоям, а не одна цифра "качество": когда ответ неверный, нужно
понять, ГДЕ сломалось. Retrieval не нашёл нужный документ → чинить поиск
(чанкинг, эмбеддинги, реранк). Нашёл, но модель ответила не то → чинить
генерацию (промпт, модель). Одна общая цифра этого не говорит.

  retrieval:   hit@k, recall@k, MRR — чистые функции, без LLM, детерминированы
  generation:  key_fact_recall — детерминированная проверка подстрок (дёшево, грубо)
               faithfulness, correctness — LLM-судья (дорого, тоньше, но сам
               требует проверки — см. judge_calibration.py)
"""
import json
import re

from rag_agent import config, llm

# --- retrieval ---

def hit_at_k(expected: list[str], retrieved: list[str]) -> float:
    """1.0, если хотя бы один ожидаемый документ есть среди найденных."""
    return float(any(src in retrieved for src in expected))


def recall_at_k(expected: list[str], retrieved: list[str]) -> float:
    """Доля ожидаемых документов, которые нашлись. Для multi-hop важнее hit:
    найти один из двух нужных документов — это ещё не ответ."""
    return sum(src in retrieved for src in expected) / len(expected)


def reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
    """1/позиция первого релевантного документа. Показывает не только "нашёл
    ли", но и "насколько высоко" — реранкер должен поднимать нужное наверх."""
    for rank, src in enumerate(retrieved, 1):
        if src in expected:
            return 1.0 / rank
    return 0.0


# --- генерация: детерминированно ---

# "не более трёх минут" — правильный ответ, но подстрока "3 мин" его не
# находит. Малые числительные словами приводим к цифрам (с падежами).
_NUMBER_WORDS = [
    (r"один|одн(?:ого|ой|ому|им|а|у|о)", "1"), (r"дв(?:а|е|ух|ум|умя)", "2"), (r"тр(?:и|ех|ем|емя)", "3"),
    (r"четыр(?:е|ех|ем|ьмя)", "4"), (r"пят(?:ь|и|ью)", "5"), (r"шест(?:ь|и|ью)", "6"), (r"сем(?:ь|и|ью)", "7"),
    (r"восемь|восьм(?:и|ью)", "8"), (r"девят(?:ь|и|ью)", "9"), (r"десят(?:ь|и|ью)", "10"),
]


def normalize(text: str) -> str:
    """"15 000" == "15000", "99,5" == "99.5", "трёх" == "3", регистр и
    markdown не важны — иначе правильный ответ проваливается из-за формы записи."""
    text = text.lower().replace("\u00a0", " ").replace("\u202f", " ").replace("ё", "е").replace("*", "")
    for pattern, digit in _NUMBER_WORDS:
        text = re.sub(rf"\b(?:{pattern})\b", digit, text)
    text = re.sub(r"(\d)\s+(?=\d)", r"\1", text)
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    return text


def key_fact_recall(key_facts: list[str], answer: str) -> float | None:
    """Доля ключевых фактов, встретившихся в ответе подстрокой. Короткие
    латинские факты ("go") ищутся как отдельное слово, чтобы не засчитать
    "google". None — у вопроса нет фактов (ловушки без ответа)."""
    if not key_facts:
        return None
    norm = normalize(answer)
    found = 0
    for fact in key_facts:
        f = normalize(fact)
        if re.fullmatch(r"[a-z0-9]{1,3}", f):
            found += bool(re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", norm))
        else:
            found += f in norm
    return found / len(key_facts)


# --- генерация: LLM-судья ---

FAITHFULNESS_PROMPT = """Ты — строгий судья RAG-системы. Проверь, следует ли ОТВЕТ из КОНТЕКСТА.
FAIL — если в ответе есть хоть одно фактическое утверждение, которого нет в контексте.
PASS — если все утверждения ответа подтверждаются контекстом, ИЛИ если ответ честно
говорит, что информации нет (это не выдумка, а правильное поведение).

КОНТЕКСТ:
{context}

ОТВЕТ:
{answer}

Верни JSON: {{"verdict": "PASS" | "FAIL", "reason": "<одно предложение>"}}"""

CORRECTNESS_PROMPT = """Ты — строгий судья. Сравни ОТВЕТ системы с ЭТАЛОНОМ по смыслу, не по словам.
CORRECT — ответ отвечает на то, что СПРОШЕНО, и не противоречит эталону. Детали эталона,
о которых вопрос не спрашивал, не обязательны.
PARTIAL — ответ верный, но упускает часть того, о чём вопрос спрашивал прямо
(например, в вопросе две части, а отвечено на одну).
WRONG — ответ противоречит эталону или не отвечает на вопрос.
Если эталон говорит, что ответа в базе нет: CORRECT — только если система тоже честно
отказалась отвечать (или ответила ровно в границах того, что указано в эталоне);
любая выдуманная конкретика — WRONG.

ВОПРОС: {question}
ЭТАЛОН: {reference}
ОТВЕТ: {answer}

Верни JSON: {{"verdict": "CORRECT" | "PARTIAL" | "WRONG", "reason": "<одно предложение>"}}"""

CORRECTNESS_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}


def _judge(prompt: str) -> dict:
    """Структурированный вывод (JSON mode) вместо "ответь одним словом":
    раньше судья иногда писал "Вердикт: PASS" или пояснение перед словом, и
    startswith("PASS") считал это FAIL. reason нужен для разбора провалов
    глазами — без него непонятно, ошибся ли судья или система."""
    resp = llm.get_client().chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # судье нужна повторяемость, а не креативность
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {"verdict": str(data.get("verdict", "")).upper(), "reason": str(data.get("reason", ""))}
    except (json.JSONDecodeError, AttributeError):
        return {"verdict": "ERROR", "reason": "судья вернул невалидный JSON"}


def judge_faithfulness(context: str, answer: str) -> dict:
    return _judge(FAITHFULNESS_PROMPT.format(context=context or "(пусто)", answer=answer))


def judge_correctness(question: str, reference: str, answer: str) -> dict:
    return _judge(CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer))
