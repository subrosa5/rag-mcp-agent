"""Postgres vector + indexed multilingual lexical retrieval, RRF, then rerank.

Lexical ranking is Postgres ts_rank_cd, not BM25. RRF combines ranks, not
incompatible scores. The rejection threshold must be calibrated per reranker.
"""

import math

from . import config, db
from .embeddings import embed
from .embeddings import rerank as _rerank


def vector_search(question: str, k: int | None = None) -> list[dict]:
    q_vec = embed([question], input_type="search_query")[0]
    return db.vector_search(q_vec, config.VECTOR_TOP_K if k is None else k)


def lexical_search(question: str, k: int | None = None) -> list[dict]:
    return db.lexical_search(question, config.BM25_TOP_K if k is None else k)


def reciprocal_rank_fusion(*rankings: list[dict]) -> list[dict]:
    merged = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            if hit["id"] in seen:
                continue
            seen.add(hit["id"])
            item = merged.setdefault(hit["id"], {**hit, "rrf_score": 0.0})
            item["rrf_score"] += 1 / (60 + rank)
    return sorted(merged.values(), key=lambda c: (-c["rrf_score"], str(c["id"])))


def hybrid_retrieve(question: str, k: int | None = None) -> list[dict]:
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("Invalid query")
    k = config.RERANK_TOP_K if k is None else k
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")
    candidates = reciprocal_rank_fusion(vector_search(question), lexical_search(question))
    if not candidates:
        return []
    hits = _rerank(question, candidates[: config.RERANK_CANDIDATES], top_k=k)
    return [
        c
        for c in hits
        if math.isfinite(c["rerank_score"]) and c["rerank_score"] >= config.MIN_RERANK_SCORE
    ]
