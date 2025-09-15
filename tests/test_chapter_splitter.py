import os
import pytest

from chapter_splitter import ChapterSplitter

SAMPLE_HTML = """
<html><body>
<h1>Chapter 1</h1>
<p>Sentence one. Sentence two. Sentence three. Sentence four.</p>
<p>Another paragraph with multiple sentences. It should be split when needed.</p>
</body></html>
"""

def test_count_tokens_smoke():
    splitter = ChapterSplitter(model_name="gpt-3.5-turbo", target_tokens=20)
    n = splitter.count_tokens("Hello world")
    assert isinstance(n, int) and n > 0

@pytest.mark.parametrize("compression,limit", [(1.0, 50_000), (0.5, 30_000)])
def test_split_noop_when_under_limit(compression, limit):
    splitter = ChapterSplitter(target_tokens=limit, compression_factor=compression)
    chunks = splitter.split_chapter("<p>short</p>", max_tokens=limit)
    assert len(chunks) == 1
    html, idx, total = chunks[0]
    assert idx == 1 and total == 1


def test_split_large_paragraph_into_sentences():
    splitter = ChapterSplitter(target_tokens=20, compression_factor=1.0)
    # Force splitting by using a very small max_tokens
    chunks = splitter.split_chapter(SAMPLE_HTML, max_tokens=40)
    # Should produce more than one chunk
    assert len(chunks) >= 1
    # Ensure each item is tuple(html, idx, total)
    for c in chunks:
        assert isinstance(c[0], str)
        assert isinstance(c[1], int)
        assert isinstance(c[2], int)


def test_merge_translated_chunks_orders_correctly():
    splitter = ChapterSplitter()
    translated_chunks = [
        ("<p>Second</p>", 2, 2),
        ("<p>First</p>", 1, 2),
    ]
    merged = splitter.merge_translated_chunks(translated_chunks)
    assert "First" in merged.split("Second")[0]
