"""tools.py — инструменты, которые агент может вызывать.

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

Каждый инструмент описан в двух местах:
  1) TOOLS_SCHEMA — JSON-описание для модели (что это, какие параметры) —
     ничего "не выполняет", это просто текстовая инструкция для LLM.
  2) TOOL_FUNCTIONS — реальная python-функция, которую МЫ вызываем, когда
     модель попросила именно этот tool_call. Модель никогда не выполняет
     код сама — она только просит, выполнение — всегда на нашей стороне.
"""

import ast
import math
import operator

from .finance import get_stock_quote
from .retrieval import hybrid_retrieve


def search_documents(query: str) -> list[dict]:
    """Return a structured evidence snapshot; source IDs are assigned by the caller."""
    return hybrid_retrieve(query)


# --- безопасный калькулятор ---
# Мы намеренно НЕ используем eval(expression) — LLM генерирует строку,
# которую мы выполняем как код, а это классическая дыра (модель могла
# "нафантазировать" что угодно, включая __import__('os').system(...)).
# Вместо этого разбираем выражение через ast и разрешаем только арифметику.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


# Защита от resource exhaustion: _safe_eval уже не даёт выполнить чужой код
# (это была единственная защита раньше), но арифметика сама по себе может
# быть дорогой — Python считает степени с числами произвольной точности, и
# что-то вроде "9**9**9**9" не выполнит код, но займёт CPU/RAM надолго и
# вернёт число из миллионов цифр. Ограничиваем длину выражения, показатель
# степени и размер итогового результата — сама арифметика остаётся такой же.
_MAX_EXPRESSION_LEN = 200
_MAX_EXPONENT = 1000
_MAX_RESULT_DIGITS = 1000


def _safe_eval(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if not math.isfinite(node.value):
            raise ValueError("Non-finite number")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Показатель степени слишком большой (>{_MAX_EXPONENT})")
            # Ограничения только на показатель НЕ ловят цепочку вложенных
            # возведений в степень с маленьким показателем на каждом шаге
            # ("повторное возведение в квадрат") — числа всё равно растут
            # чудовищно быстро (удвоение количества цифр на каждом уровне).
            # Поэтому проверяем ещё и само основание перед операцией.
            if abs(left) > 10**50:
                raise ValueError("Основание степени слишком большое")
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Недопустимое выражение: {ast.dump(node)}")


def calculate(expression: str) -> str:
    if len(expression) > _MAX_EXPRESSION_LEN:
        return f"Ошибка: выражение длиннее {_MAX_EXPRESSION_LEN} символов"
    try:
        tree = ast.parse(expression, mode="eval")
        # ast.walk считает ВСЕ узлы дерева, не только допустимые операции —
        # грубый, но дешёвый способ отсечь патологически огромные выражения
        # ещё до вычисления (тысячи вложенных скобок и т.п.).
        if sum(1 for _ in ast.walk(tree)) > 100:
            return "Ошибка: выражение слишком сложное"
        result = _safe_eval(tree.body)
        if not isinstance(result, (int, float)) or not math.isfinite(result):
            raise ValueError("Non-real or non-finite result")
        if len(str(result)) > _MAX_RESULT_DIGITS:
            return f"Ошибка: результат слишком большой (>{_MAX_RESULT_DIGITS} цифр)"
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Найти релевантную информацию в базе документов пользователя. Используй, когда вопрос требует конкретных фактов из загруженных документов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос — переформулированный вопрос пользователя",
                    }
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

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "calculate": calculate,
    "get_stock_quote": get_stock_quote,
}
