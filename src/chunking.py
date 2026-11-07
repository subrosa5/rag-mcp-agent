"""chunking.py — режем текст на куски с перекрытием.

Вынесено в отдельный модуль, потому что и ingest.py (индексация), и любой
будущий eval-скрипт должны резать текст ОДИНАКОВО — иначе результаты
эксперимента "поменял chunk_size" будет неверно сравнивать разные вещи.
"""

from . import config


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Режем строку на куски по `size` символов, сдвигаясь с шагом (size - overlap).

    Пример при size=500, overlap=100: чанк 2 начинается на 400 символов
    раньше конца чанка 1 — та же мысль, разорванная на границе, попадёт
    целиком хотя бы в один из двух чанков.
    """
    size = config.CHUNK_SIZE if size is None else size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    # Без этой проверки overlap >= size даёт шаг size - overlap <= 0 —
    # start никогда не растёт (или уменьшается), и `while start < len(text)`
    # зависает навсегда на непустом тексте. Раньше это можно было словить
    # только руками, поменяв параметры по умолчанию в config.py.
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) должен быть меньше size ({size}) — иначе бесконечный цикл"
        )
    if size <= 0:
        raise ValueError(f"size должен быть положительным, получено {size}")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks
