"""db.py — тонкий слой поверх psycopg. Никакого ORM.

Почему без ORM (SQLAlchemy и т.п.): для 3 запросов ORM — это лишний слой
абстракции, который на интервью сложнее объяснить, чем сам SQL. Здесь ты
видишь ровно тот запрос, который уходит в базу.
"""

import re
from contextlib import contextmanager

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import config


@contextmanager
def _raw_conn():
    """Соединение БЕЗ register_vector — нужно ровно один раз, для init_schema,
    потому что register_vector сам читает тип 'vector' из pg_type, а на
    свежей базе (например, только что созданный Neon-проект) этого типа ещё
    нет, пока не выполнен CREATE EXTENSION. Курица и яйцо: нельзя
    зарегистрировать тип раньше, чем расширение, которое его создаёт."""
    conn = psycopg.connect(
        config.DATABASE_URL,
        autocommit=True,
        connect_timeout=5,
        options="-c statement_timeout=10000",
    )
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_conn():
    """Контекстный менеджер: открыли соединение, зарегистрировали тип vector, отдали, закрыли.
    Предполагает, что init_schema() уже отработал (расширение существует) —
    иначе см. _raw_conn."""
    conn = psycopg.connect(
        config.DATABASE_URL,
        autocommit=True,
        connect_timeout=5,
        options="-c statement_timeout=10000",
    )
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    """Создаёт расширение, таблицу и HNSW-индекс, если их ещё нет.

    Размерность вектора берём из config.EMBEDDING_DIM, а не хардкодим —
    у local-бэкенда 384, у Cohere 1024 (см. config.py). Если ты руками
    поменял USE_COHERE на базе, где таблица уже создана под другую
    размерность, CREATE TABLE IF NOT EXISTS ничего не подскажет — нужно
    пересоздать базу вручную (DROP TABLE chunks). Это осознанно не
    автоматизировано: молчаливая ALTER-миграция типа колонки — плохая идея.
    """
    with _raw_conn() as conn:
        if conn.execute("SELECT lower('АБВ')").fetchone()[0] != "абв":
            raise ValueError(
                "Database LC_CTYPE must support Unicode case folding (e.g. en_US.UTF-8)"
            )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                    id          BIGSERIAL PRIMARY KEY,
                    source      TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector({config.EMBEDDING_DIM}) NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
               ON chunks USING hnsw (embedding vector_cosine_ops)"""
        )

        actual = conn.execute("""SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute WHERE attrelid = 'chunks'::regclass
            AND attname = 'embedding'""").fetchone()[0]
        if actual != f"vector({config.EMBEDDING_DIM})":
            raise ValueError("Embedding dimension changed: use a new database and re-ingest")
        conn.execute("""CREATE TABLE IF NOT EXISTS index_metadata (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            embedding_model TEXT NOT NULL, embedding_dim INTEGER NOT NULL)""")
        conn.execute("""CREATE INDEX IF NOT EXISTS chunks_lexical_idx ON chunks USING gin
            ((to_tsvector('russian', content) || to_tsvector('simple', content)))""")


def insert_chunks(rows: list[tuple[str, int, str, "np.ndarray"]]) -> None:
    """rows: список (source, chunk_index, content, embedding)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (source, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s)""",
                rows,
            )


def clear_chunks() -> None:
    """Полная переиндексация проще, чем инкрементальные апдейты — для лабы этого достаточно."""
    with get_conn() as conn:
        conn.execute("TRUNCATE chunks RESTART IDENTITY")


@contextmanager
def transaction():
    """Одно соединение, одна транзакция: либо всё внутри блока применяется
    разом (commit), либо ничего (rollback при любом исключении). autocommit
    здесь намеренно False — в отличие от get_conn()/_raw_conn(), где каждый
    вызов сам по себе атомарен, но independent от соседних вызовов."""
    conn = psycopg.connect(
        config.DATABASE_URL,
        autocommit=False,
        connect_timeout=5,
        options="-c statement_timeout=10000",
    )
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reindex(all_rows: list[tuple[str, int, str, "np.ndarray"]]) -> None:
    """Полная переиндексация АТОМАРНО: TRUNCATE и все вставки — в одной
    транзакции. Раньше clear_chunks() и insert_chunks() каждый открывали
    свою автокоммитную транзакцию по отдельности (по одной на документ) —
    сбой на середине переиндексации (или чужой запрос ровно в этот момент)
    мог увидеть пустую или наполовину заполненную таблицу chunks."""
    with transaction() as conn:
        if not all_rows:
            raise ValueError("Refusing empty reindex")
        conn.execute("TRUNCATE chunks RESTART IDENTITY")
        conn.execute(
            """INSERT INTO index_metadata VALUES (TRUE, %s, %s)
            ON CONFLICT (singleton) DO UPDATE SET embedding_model = EXCLUDED.embedding_model,
            embedding_dim = EXCLUDED.embedding_dim""",
            (config.EMBEDDING_MODEL, config.EMBEDDING_DIM),
        )
        if all_rows:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (source, chunk_index, content, embedding)
                       VALUES (%s, %s, %s, %s)""",
                    all_rows,
                )


def vector_search(query_embedding, k: int) -> list[dict]:
    """Поиск k ближайших чанков по косинусному расстоянию.

    Оператор <=> — косинусное расстояние (1 - cosine_similarity), встроен pgvector.
    ORDER BY <=> ASC значит "сначала самые близкие" (расстояние маленькое = похоже).
    """
    with get_conn() as conn:
        validate_index(conn)
        rows = conn.execute(
            """SELECT id, source, content, 1 - (embedding <=> %s) AS score
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (query_embedding, query_embedding, k),
        ).fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2], "score": r[3]} for r in rows]


def fetch_all_chunks() -> list[dict]:
    """Administrative snapshot for offline checks; never used in request retrieval."""
    with get_conn() as conn:
        validate_index(conn)
        rows = conn.execute("SELECT id, source, content FROM chunks").fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2]} for r in rows]


def validate_index(conn) -> None:
    row = conn.execute(
        "SELECT embedding_model, embedding_dim FROM index_metadata WHERE singleton"
    ).fetchone()
    if row != (config.EMBEDDING_MODEL, config.EMBEDDING_DIM):
        raise ValueError("Index model mismatch: run ingestion with the configured embedding model")


def lexical_search(question: str, k: int) -> list[dict]:
    # Recall stage: natural-language questions need not match every word.
    # Quote individual terms so user-supplied query operators remain data.
    terms = re.findall(r"[\w]+(?:[.-][\w]+)*", question)[:64]
    if not terms:
        return []
    query = " OR ".join(f'"{term}"' for term in terms)
    with get_conn() as conn:
        validate_index(conn)
        rows = conn.execute(
            """WITH query AS (
            SELECT websearch_to_tsquery('russian', %s) || websearch_to_tsquery('simple', %s) AS q
        ) SELECT id, source, content,
            ts_rank_cd(to_tsvector('russian', content) || to_tsvector('simple', content), q) AS score
        FROM chunks, query
        WHERE (to_tsvector('russian', content) || to_tsvector('simple', content)) @@ q
        ORDER BY score DESC, id LIMIT %s""",
            (query, query, k),
        ).fetchall()
    return [{"id": r[0], "source": r[1], "content": r[2], "score": r[3]} for r in rows]
