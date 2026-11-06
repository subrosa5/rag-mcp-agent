"""One public answer pipeline; rejected or failed reviews never release a draft."""

import logging

from groq import RateLimitError

from .agent import run_agent_verbose
from .critic import critique
from .grounding import abstain
from .guardrail import check as guardrail_check
from .memory import trim_history
from .query_rewriter import rewrite as rewrite_query
from .tracing import get_stats, reset

log = logging.getLogger(__name__)


def answer_agent(
    question: str, history: list[dict] | None = None, max_revisions: int = 1, verbose: bool = False
) -> dict:
    reset()
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("Question must contain 1–4000 non-blank characters")
    if not 0 <= max_revisions <= 1:
        raise ValueError("max_revisions must be 0 or 1")
    history = trim_history(history or [])
    result = abstain()
    try:
        allowed, _ = guardrail_check(question)
        if not allowed:
            return {**abstain("blocked"), "status": "blocked", "stats": get_stats()}
        # Rewriting is a retrieval hint, never a replacement for the user's intent.
        hint = rewrite_query(question, history=history)
        feedback = None
        for _ in range(max_revisions + 1):
            result = run_agent_verbose(
                question,
                history=history,
                search_hint=hint,
                extra_instruction=feedback,
                verbose=verbose,
            )
            if result["status"] != "answered":
                break
            ok, feedback = critique(question, result["answer"], result["trace"])
            if ok:
                break
            result = {**abstain("review_rejected"), "evidence": result.get("evidence", [])}
    except RateLimitError as exc:
        try:
            retry_after = max(
                1, min(86400, int(float(exc.response.headers.get("retry-after", "60"))))
            )
        except (ValueError, AttributeError):
            retry_after = 60
        result = {**abstain("rate_limited"), "status": "unavailable", "retry_after": retry_after}
    except Exception as exc:
        log.warning("Answer pipeline unavailable (%s)", type(exc).__name__)
        result = {**abstain("dependency_unavailable"), "status": "unavailable"}
    return {**result, "stats": get_stats()}


def run_agent_with_review(
    question: str, history: list[dict] | None = None, max_revisions: int = 1, verbose: bool = True
) -> str:
    return answer_agent(question, history, max_revisions, verbose)["answer"]


if __name__ == "__main__":
    print(run_agent_with_review(input("Вопрос: ")))
