from ingest import load_documents, split_into_chunks


def test_load_documents_skips_empty_and_unsupported_files(tmp_path):
    (tmp_path / "policy.md").write_text("Refunds within 30 days.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Shipping is free.", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    sources = {doc.metadata["source"] for doc in load_documents(tmp_path)}

    assert sources == {"policy.md", "notes.txt"}


def test_load_documents_uses_paths_relative_to_docs_dir(tmp_path):
    (tmp_path / "team").mkdir()
    (tmp_path / "team" / "guide.md").write_text("Hello", encoding="utf-8")

    [document] = load_documents(tmp_path)

    assert document.metadata["source"] == "team/guide.md"


def test_chunks_are_numbered_per_source(tmp_path):
    (tmp_path / "a.md").write_text("word " * 200, encoding="utf-8")
    (tmp_path / "b.md").write_text("short", encoding="utf-8")

    chunks = split_into_chunks(load_documents(tmp_path), chunk_size=100, chunk_overlap=10)
    numbers_by_source = {}
    for chunk in chunks:
        numbers_by_source.setdefault(chunk.metadata["source"], []).append(chunk.metadata["chunk"])

    assert numbers_by_source["b.md"] == [0]
    assert numbers_by_source["a.md"] == list(range(len(numbers_by_source["a.md"])))
    assert len(numbers_by_source["a.md"]) > 1
