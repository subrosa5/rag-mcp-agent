"""Opt-in tests require a DISPOSABLE database: RAG_TEST_DATABASE_URL only."""

import os

import numpy as np
import pytest

from src import config, db

pytestmark = pytest.mark.skipif(
    not os.environ.get("RAG_TEST_DATABASE_URL"), reason="disposable Postgres not configured"
)


@pytest.fixture
def index(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", os.environ["RAG_TEST_DATABASE_URL"])
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "test-fixture")
    monkeypatch.setattr(config, "EMBEDDING_DIM", 384)
    db.init_schema()
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    db.reindex(
        [
            ("deployment", 0, "Откат деплоя занимает 3 минуты.", vector),
            ("onboarding", 0, "Новый инженер получает доступ на второй неделе.", -vector),
        ]
    )
    return vector


def test_lexical_russian_stemming_and_no_match(index):
    assert db.lexical_search("отката", 4)[0]["source"] == "deployment"
    assert db.lexical_search("несуществующее_слово", 4) == []


def test_vector_and_schema_idempotency(index):
    db.init_schema()
    assert db.vector_search(index, 1)[0]["source"] == "deployment"


def test_failed_reindex_rolls_back_snapshot_and_metadata(index):
    with pytest.raises(Exception):
        db.reindex([("bad", 0, "bad dimensions", np.zeros(2))])
    assert db.vector_search(index, 1)[0]["source"] == "deployment"


def test_empty_reindex_preserves_documents(index):
    with pytest.raises(ValueError):
        db.reindex([])
    assert len(db.fetch_all_chunks()) == 2


def test_model_change_rejected_even_at_same_dimension(index, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "different")
    with pytest.raises(ValueError):
        db.vector_search(index, 1)


def test_natural_language_query_does_not_require_every_term(index):
    assert (
        db.lexical_search("Сколько времени требуется для отката деплоя?", 4)[0]["source"]
        == "deployment"
    )
