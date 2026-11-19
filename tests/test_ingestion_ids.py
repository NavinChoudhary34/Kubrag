from app.ingestion.processor import make_point_id


def test_same_chunk_gets_same_id_so_reingest_is_idempotent():
    a = make_point_id("true", "doc.txt", 0, "hello world")
    b = make_point_id("true", "doc.txt", 0, "hello world")
    assert a == b


def test_different_chunk_text_index_or_file_changes_id():
    base = make_point_id("true", "doc.txt", 0, "hello")
    assert make_point_id("true", "doc.txt", 0, "hello!") != base
    assert make_point_id("true", "doc.txt", 1, "hello") != base
    assert make_point_id("true", "other.txt", 0, "hello") != base
    assert make_point_id("noisy", "doc.txt", 0, "hello") != base
