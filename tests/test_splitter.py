from app.ingestion.chunking.splitter import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []


def test_small_paragraphs_are_packed_together():
    chunks = chunk_text("alpha\n\nbeta\n\ngamma")
    assert chunks == ["alpha\n\nbeta\n\ngamma"]


def test_text_without_blank_lines_is_split_not_one_giant_chunk():
    # This is the real-world HTML/PDF case: thousands of chars, no "\n\n"
    text = "\n".join(f"line {i} about kubernetes autoscaling" for i in range(1000))
    chunks = chunk_text(text, chunk_size=1500)
    assert len(chunks) > 5
    assert all(len(c) <= 1500 for c in chunks)


def test_no_content_is_lost_when_splitting_long_text():
    words = [f"w{i}" for i in range(3000)]
    chunks = chunk_text(" ".join(words), chunk_size=500, overlap=50)
    joined = " ".join(chunks)
    # every word must appear in at least one chunk (overlap may repeat some)
    assert all(w in joined.split() for w in words)


def test_oversized_paragraph_does_not_swallow_neighbours():
    text = "intro\n\n" + ("x " * 2000) + "\n\noutro"
    chunks = chunk_text(text, chunk_size=1000)
    assert chunks[0] == "intro"
    assert chunks[-1] == "outro"


def test_windows_line_endings_are_handled():
    chunks = chunk_text("one\r\n\r\ntwo", chunk_size=5)
    assert chunks == ["one", "two"]
