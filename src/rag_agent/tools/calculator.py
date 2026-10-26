"""calculator.py — инструмент calculate: арифметика без eval()."""
import ast
import operator
from collections.abc import Callable
from typing import Any

# --- безопасный калькулятор ---
# Мы намеренно НЕ используем eval(expression) — LLM генерирует строку,
# которую мы выполняем как код, а это классическая дыра (модель могла
# "нафантазировать" что угодно, включая __import__('os').system(...)).
# Вместо этого разбираем выражение через ast и разрешаем только арифметику.
_ALLOWED_OPS: dict[type[ast.AST], Callable[..., Any]] = {
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
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
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
        if len(str(result)) > _MAX_RESULT_DIGITS:
            return f"Ошибка: результат слишком большой (>{_MAX_RESULT_DIGITS} цифр)"
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"
