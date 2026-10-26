import pytest

from rag_agent.agents.memory import MAX_HISTORY_MESSAGES, trim_history
from rag_agent.ingestion.chunking import chunk_text


class TestChunking:
    def test_empty_text(self):
        assert chunk_text("   ") == []

    def test_overlap_must_be_smaller_than_size(self):
        # раньше overlap >= size давал бесконечный цикл
        with pytest.raises(ValueError):
            chunk_text("abc", size=10, overlap=10)

    def test_chunks_respect_size_and_cover_text(self):
        text = "".join(chr(ord("a") + i % 26) for i in range(1234))
        chunks = chunk_text(text, size=100, overlap=20)
        assert all(len(c) <= 100 for c in chunks)
        assert chunks[0] == text[:100]
        assert chunks[1].startswith(text[80:90])  # шаг = size - overlap
        assert text.endswith(chunks[-1])


class TestMemory:
    def test_short_history_untouched(self):
        h = [{"role": "user", "content": "1"}]
        assert trim_history(h) == h

    def test_keeps_only_last_messages(self):
        h = [{"role": "user", "content": str(i)} for i in range(25)]
        trimmed = trim_history(h)
        assert len(trimmed) == MAX_HISTORY_MESSAGES
        assert trimmed[-1]["content"] == "24"
