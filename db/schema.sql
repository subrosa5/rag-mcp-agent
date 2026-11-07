-- Reference only. src/db.py::init_schema() is the executable migration.
-- Requires Unicode-aware LC_CTYPE; local embeddings use 384, Cohere v3 uses 1024.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_lexical_idx ON chunks USING gin
    ((to_tsvector('russian', content) || to_tsvector('simple', content)));
CREATE TABLE IF NOT EXISTS index_metadata (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL
);
-- Metadata is committed together with the actual indexed vectors, never guessed.
