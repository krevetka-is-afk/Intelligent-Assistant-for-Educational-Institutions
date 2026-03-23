from __future__ import annotations

import zipfile
from pathlib import Path

from langchain_chroma import Chroma

from src.server.app.document_ingestion import (
    build_chunk_records,
    chunk_text,
    index_directory,
    load_document,
    normalize_text,
)


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        length = float(len(text))
        checksum = float(sum(ord(char) for char in text) % 997)
        return [length, checksum, 1.0]


def _write_docx(path: Path, *, title: str, paragraphs: list[str]) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        {paragraphs}
      </w:body>
    </w:document>
    """.format(
        paragraphs="".join(
            f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
        )
    )
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>{title}</dc:title>
    </cp:coreProperties>
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("docProps/core.xml", core_xml)


def test_chunk_text_uses_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=5, overlap=2)
    assert chunks == [(0, 5, "abcde"), (3, 8, "defgh"), (6, 10, "ghij")]


def test_load_document_reads_txt_with_cp1251_fallback(tmp_path):
    txt_path = tmp_path / "notice.txt"
    txt_path.write_bytes("Привет студент".encode("cp1251"))

    parsed = load_document(txt_path, root_dir=tmp_path)

    assert parsed.title == "notice"
    assert parsed.source == "notice.txt"
    assert parsed.sections[0].text == "Привет студент"


def test_load_document_reads_html_and_docx(tmp_path):
    html_path = tmp_path / "page.html"
    html_path.write_text(
        "<html><head><title>FAQ</title><style>.x{}</style></head>"
        "<body><h1>Заголовок</h1><script>alert(1)</script><p>Ответ</p></body></html>",
        encoding="utf-8",
    )
    docx_path = tmp_path / "rules.docx"
    _write_docx(docx_path, title="Правила", paragraphs=["Первый абзац", "Второй абзац"])

    html_doc = load_document(html_path, root_dir=tmp_path)
    docx_doc = load_document(docx_path, root_dir=tmp_path)

    assert html_doc.title == "FAQ"
    assert "Заголовок" in html_doc.sections[0].text
    assert "alert" not in html_doc.sections[0].text
    assert docx_doc.title == "Правила"
    assert docx_doc.sections[0].text == "Первый абзац\n\nВторой абзац"


def test_build_chunk_records_preserves_metadata(tmp_path):
    txt_path = tmp_path / "handbook.txt"
    txt_path.write_text("line one\nline two\nline three", encoding="utf-8")
    parsed = load_document(txt_path, root_dir=tmp_path)

    chunks = build_chunk_records(
        parsed, chunk_size=12, overlap=2, indexed_at="2026-03-22T00:00:00Z"
    )

    assert len(chunks) >= 2
    assert chunks[0].metadata["document_id"] == parsed.document_id
    assert chunks[0].metadata["chunk_id"] == chunks[0].id
    assert chunks[0].metadata["source"] == "handbook.txt"
    assert chunks[0].metadata["indexed_at"] == "2026-03-22T00:00:00Z"


def test_index_directory_does_not_duplicate_chunks_and_rebuilds(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text(normalize_text("A" * 620), encoding="utf-8")
    (input_dir / "b.txt").write_text("B" * 200, encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    first = index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=True)
    store = Chroma(
        collection_name="test_docs",
        persist_directory=str(persist_dir),
        embedding_function=_FakeEmbeddings(),
    )
    first_count = store._collection.count()

    second = index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=False)
    second_count = store._collection.count()

    third = index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=True)
    rebuilt_store = Chroma(
        collection_name="test_docs",
        persist_directory=str(persist_dir),
        embedding_function=_FakeEmbeddings(),
    )
    rebuilt_count = rebuilt_store._collection.count()

    assert first.indexed_files == 2
    assert second.indexed_files == 2
    assert third.indexed_files == 2
    assert first_count == second_count == rebuilt_count
