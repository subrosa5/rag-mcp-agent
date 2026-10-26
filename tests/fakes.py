"""fakes.py — фейковые ответы LLM в форме groq ChatCompletion."""
from types import SimpleNamespace


def make_response(content: str | None = None, tool_calls: list | None = None,
                  prompt_tokens: int = 100, completion_tokens: int = 20):
    """Ответ в форме groq ChatCompletion — ровно те поля, которые читает наш код."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def make_tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeLLM:
    """Подменяет Groq-клиента. handler(kwargs) -> ответ; все вызовы пишутся в
    self.calls, чтобы тест мог проверить, КТО и СКОЛЬКО раз дёрнул модель."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(kwargs)


def system_prompt(kwargs: dict) -> str:
    return kwargs["messages"][0]["content"] if kwargs["messages"][0]["role"] == "system" else ""
