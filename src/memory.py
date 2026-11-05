"""History is untrusted context, bounded equally for CLI, HTTP and MCP callers."""

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARS = 16000


def trim_history(history: list[dict]) -> list[dict]:
    if not isinstance(history, list) or len(history) > 50:
        raise ValueError("Invalid history")
    result, remaining = [], MAX_HISTORY_CHARS
    for item in reversed(history):
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            raise ValueError("History role must be user or assistant")
        content = item.get("content")
        if not isinstance(content, str) or len(content) > 8000:
            raise ValueError("Invalid history content")
        if len(result) < MAX_HISTORY_MESSAGES and len(content) <= remaining:
            result.append({"role": item["role"], "content": content})
            remaining -= len(content)
        else:
            # Still validate all roles, but don't skip into older context.
            remaining = -1
    return list(reversed(result))
