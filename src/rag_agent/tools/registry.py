"""registry.py — инструменты, которые агент может вызывать.

Три инструмента нарочно очень разной природы — это делает "агентность"
видимой: модель должна САМА понять, какой инструмент нужен под конкретный
вопрос, а не просто всегда дёргать один и тот же.

  search_documents — RAG как инструмент: агент лезет в базу знаний,
                      когда вопрос требует фактов из документов.
  calculate        — обычная функция: агент считает, когда вопрос про число,
                      а не про знание из документов.
  get_stock_quote  — внешний API: агент дёргает live-данные, которых нет
                      ни в базе документов, ни в его собственных знаниях
                      (цена акции меняется каждую секунду).

Реализации лежат рядом (search.py, calculator.py, finance.py), а здесь —
реестр: каждый инструмент описан в двух местах:
  1) TOOLS_SCHEMA — JSON-описание для модели (что это, какие параметры) —
     ничего "не выполняет", это просто текстовая инструкция для LLM.
  2) TOOL_FUNCTIONS — реальная python-функция, которую МЫ вызываем, когда
     модель попросила именно этот tool_call. Модель никогда не выполняет
     код сама — она только просит, выполнение — всегда на нашей стороне.
"""
from collections.abc import Callable

from .calculator import calculate
from .finance import get_stock_quote
from .search import search_documents

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Найти релевантную информацию в базе документов пользователя. Используй, когда вопрос требует конкретных фактов из загруженных документов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос — переформулированный вопрос пользователя"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Посчитать арифметическое выражение (+, -, *, /, **). Используй для любой математики.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Например: '12 * (3 + 4)'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Получить текущую цену акции по биржевому тикеру (например AAPL, TSLA). Используй для вопросов про live-котировки — этого нет ни в базе документов, ни в знаниях модели.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Биржевой тикер, например AAPL"}
                },
                "required": ["ticker"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., object]] = {
    "search_documents": search_documents,
    "calculate": calculate,
    "get_stock_quote": get_stock_quote,
}
