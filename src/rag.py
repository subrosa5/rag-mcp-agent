"""Fixed retrieval → structured answer → review, using the exact retrieved snapshot."""

from groq import Groq

from . import config
from .critic import critique
from .grounding import abstain, evidence_item, generate
from .retrieval import hybrid_retrieve
from .tracing import get_stats, reset


def answer(question: str, k: int | None = None) -> dict:
    reset()
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("Invalid question")
    if k is not None and (type(k) is not int or not 1 <= k <= 20):
        raise ValueError("k must be between 1 and 20")
    evidence = []
    try:
        chunks = hybrid_retrieve(question, k=k)
        evidence = [evidence_item(c["source"], c["content"]) for c in chunks]
        client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_TIMEOUT, max_retries=1)
        result = generate(client, question, evidence)
        if result["status"] == "answered":
            ok, _ = critique(
                question,
                result["answer"],
                [{"tool": "search_documents", "args": {"query": question}, "result": evidence}],
            )
            if not ok:
                result = abstain("review_rejected")
    except Exception:
        result = {**abstain("dependency_unavailable"), "status": "unavailable"}
    return {**result, "evidence": evidence, "stats": get_stats()}


if __name__ == "__main__":
    print(answer(input("Вопрос: "))["answer"])
