"""ingest.py — строит индекс: файлы из data/ -> чанки -> эмбеддинги -> Postgres.

Запуск: python -m src.ingest
"""

import glob
import os

from . import db
from .chunking import chunk_text
from .embeddings import embed


def load_documents(data_dir: str = "data") -> list[tuple[str, str]]:
    """Возвращает [(путь_к_файлу, содержимое), ...]."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append((path, f.read()))
    return docs


def run():
    docs = load_documents()
    if not docs:
        print("Нет файлов в data/*.txt — добавь документы и запусти снова.")
        return

    print(f"Найдено документов: {len(docs)}")
    db.init_schema()  # идемпотентно: создаст таблицу/индекс, если их ещё нет (новая база — Neon и т.п.)

    # Сначала считаем чанки и эмбеддинги для ВСЕХ документов в памяти, и
    # только потом одним махом пишем в базу через db.reindex() — TRUNCATE и
    # все вставки атомарно, одной транзакцией. Раньше TRUNCATE и вставки по
    # каждому документу были отдельными автокоммитами: сбой посреди процесса
    # (или параллельный запрос ровно в эту секунду) мог застать базу пустой
    # или наполовину заполненной.
    all_rows = []
    per_doc_counts = []
    for source, text in docs:
        pieces = chunk_text(text)
        if not pieces:
            continue
        vectors = embed(pieces)  # одним батчем — быстрее, чем по одному чанку
        rows = [
            (source, i, chunk, vec)
            for i, (chunk, vec) in enumerate(zip(pieces, vectors, strict=True))
        ]
        all_rows.extend(rows)
        per_doc_counts.append((source, len(rows)))

    if not all_rows:
        raise ValueError("Refusing to replace index with an empty corpus")
    db.reindex(all_rows)

    for source, count in per_doc_counts:
        print(f"  {source}: {count} чанков")
    print(f"Готово: {len(all_rows)} чанков проиндексировано в Postgres.")


if __name__ == "__main__":
    run()
