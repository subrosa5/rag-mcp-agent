"""Интеграционные тесты против живого Postgres + pgvector.

Запуск: TEST_DATABASE_URL=postgresql://localhost/rag_agent_test uv run pytest -m integration

Отдельная переменная, а не DATABASE_URL, — нарочно: тесты делают TRUNCATE,
и случайно направить их на рабочую базу (или Neon с демо) не должно быть
возможно одной опечаткой в .env.
"""
import os

import numpy as np
import pytest

from rag_agent import config, db

pytestmark = pytest.mark.integration

TEST_DB = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    if not TEST_DB:
        pytest.skip("TEST_DATABASE_URL не задан")
    monkeypatch.setattr(config, "DATABASE_URL", TEST_DB)
    with db._raw_conn() as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
    db.init_schema()
    yield


def _unit(*xs) -> np.ndarray:
    v = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def test_vector_search_orders_by_cosine():
    db.reindex([
        ("a.txt", 0, "про откат", _unit(1, 0)),
        ("b.txt", 0, "про онбординг", _unit(0, 1)),
        ("c.txt", 0, "между", _unit(1, 1)),
    ])
    hits = db.vector_search(_unit(1, 0.1), k=3)
    assert [h["source"] for h in hits] == ["a.txt", "c.txt", "b.txt"]
    assert hits[0]["score"] > hits[1]["score"] > hits[2]["score"]


def test_reindex_is_atomic():
    db.reindex([("old.txt", 0, "старое", _unit(1))])
    bad_rows = [("new.txt", 0, "новое", _unit(1)), ("broken.txt", 0, None, _unit(1))]  # NOT NULL упадёт
    with pytest.raises(Exception):  # noqa: B017 — конкретный тип ошибки psycopg тут не важен
        db.reindex(bad_rows)
    # TRUNCATE откатился вместе с упавшей вставкой — старые данные на месте
    assert [c["source"] for c in db.fetch_all_chunks()] == ["old.txt"]
