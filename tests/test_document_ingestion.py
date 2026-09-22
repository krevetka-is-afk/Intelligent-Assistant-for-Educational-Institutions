from __future__ import annotations

import zipfile
from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma

from src.server.app.document_ingestion import (
    DocumentParsingError,
    _ocr_pdf_pages,
    build_chunk_records,
    chunk_text,
    index_directory,
    load_document,
    normalize_text,
)
from src.server.app.lexical import get_indexed_document_ids, search_lexical


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


def _write_docx(
    path: Path,
    *,
    title: str,
    paragraphs: list[str],
    hyperlink_count: int = 0,
) -> None:
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
        if hyperlink_count:
            relationships = "\n".join(
                (
                    f'<Relationship Id="rId{index}" '
                    'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                    f'relationships/hyperlink" Target="https://example.test/{index}" '
                    'TargetMode="External"/>'
                )
                for index in range(1, hyperlink_count + 1)
            )
            archive.writestr(
                "word/_rels/document.xml.rels",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/'
                    f'package/2006/relationships">{relationships}</Relationships>'
                ),
            )


def _empty_pdf_page_class(*, image_payload: bytes | None = None):
    class _ImageFile:
        data = image_payload or b"image-bytes"

    class _Page:
        images = [_ImageFile()] if image_payload is not None else []

        @staticmethod
        def extract_text() -> str:
            return ""

    return _Page


class _PdfPage:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


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


def test_load_document_runs_ocr_when_pdf_has_no_extractable_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _ImageFile:
        data = b"image-bytes"

    class _Page:
        images = [_ImageFile()]

        @staticmethod
        def extract_text() -> str:
            return ""

    class _Reader:
        pages = [_Page()]

        def __init__(self, path: str) -> None:
            assert path == str(pdf_path)

    rendered_pages: list[int] = []

    def render_page(path: Path, *, page_index: int, timeout_seconds: float):
        assert path == pdf_path
        assert timeout_seconds > 0
        rendered_pages.append(page_index)
        return f"rendered-page-{page_index}"

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        render_page,
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        lambda image, remaining_seconds: "Правила пересдачи из OCR",
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert rendered_pages == [1]
    assert parsed.sections[0].text == "Правила пересдачи из OCR"
    assert parsed.sections[0].page == 1
    assert parsed.metadata["ocr_status"] == "succeeded"
    assert parsed.metadata["ocr_pages_attempted"] == 1
    assert parsed.metadata["ocr_pages_succeeded"] == 1


def test_load_document_reports_unavailable_ocr_for_pdf_without_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Page:
        images = []

        @staticmethod
        def extract_text() -> str:
            return ""

    class _Reader:
        pages = [_Page()]

        def __init__(self, path: str) -> None:
            del path

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr(
        "src.server.app.document_ingestion._ocr_available",
        lambda: (False, "tesseract is not installed"),
    )

    try:
        load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)
    except DocumentParsingError as exc:
        error = exc
    else:  # pragma: no cover - assertion message is clearer than pytest.raises attrs here
        raise AssertionError("PDF without extractable text should fail when OCR is unavailable")

    assert error.reason == "no_extractable_text"
    assert error.details["ocr_status"] == "unavailable"
    assert error.details["ocr_attempted"] is True


def test_load_document_limits_pdf_ocr_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _ImageFile:
        data = b"image-bytes"

    class _Page:
        images = [_ImageFile()]

        @staticmethod
        def extract_text() -> str:
            return ""

    class _Reader:
        pages = [_Page(), _Page()]

        def __init__(self, path: str) -> None:
            del path

    rendered_pages: list[int] = []

    def render_page(path: Path, *, page_index: int, timeout_seconds: float):
        assert path == pdf_path
        assert timeout_seconds > 0
        rendered_pages.append(page_index)
        return f"rendered-page-{page_index}"

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        render_page,
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        lambda image, remaining_seconds: "OCR text",
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion.config.DOCUMENT_OCR_MAX_PAGES",
        1,
        raising=False,
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert len(parsed.sections) == 1
    assert rendered_pages == [1]
    assert parsed.metadata["pdf_pages"] == 2
    assert parsed.metadata["ocr_pages_attempted"] == 1
    assert parsed.metadata["ocr_pages_skipped"] == 1


def test_load_document_adds_ocr_for_scanned_pages_after_existing_pdf_text(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Reader:
        pages = [
            _PdfPage("Текстовая страница"),
            _PdfPage(""),
        ]

        def __init__(self, path: str) -> None:
            assert path == str(pdf_path)

    rendered_pages: list[int] = []

    def render_page(path: Path, *, page_index: int, timeout_seconds: float):
        assert path == pdf_path
        assert timeout_seconds > 0
        rendered_pages.append(page_index)
        return f"rendered-page-{page_index}"

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        render_page,
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        lambda image, remaining_seconds: (
            "OCR скан страницы" if image == "rendered-page-2" else "DUPLICATE"
        ),
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert [(section.page, section.text) for section in parsed.sections] == [
        (1, "Текстовая страница"),
        (2, "OCR скан страницы"),
    ]
    assert rendered_pages == [2]
    assert parsed.metadata["pdf_pages"] == 2
    assert parsed.metadata["pdf_raw_no_extractable_text"] is False
    assert parsed.metadata["ocr_attempted"] is True
    assert parsed.metadata["ocr_pages_attempted"] == 1
    assert parsed.metadata["ocr_pages_succeeded"] == 1
    assert parsed.metadata["ocr_pages_failed"] == 0
    assert parsed.metadata["ocr_chars"] == len("OCR скан страницы")


def test_load_document_keeps_existing_pdf_text_when_partial_ocr_page_fails(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "partial-fail.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Reader:
        pages = [
            _PdfPage("Извлеченный текст первой страницы"),
            _PdfPage(""),
        ]

        def __init__(self, path: str) -> None:
            del path

    def fail_ocr(image, remaining_seconds):
        del image, remaining_seconds
        raise RuntimeError("ocr page failed")

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        lambda path, *, page_index, timeout_seconds: f"rendered-page-{page_index}",
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        fail_ocr,
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert [(section.page, section.text) for section in parsed.sections] == [
        (1, "Извлеченный текст первой страницы")
    ]
    assert parsed.metadata["ocr_attempted"] is True
    assert parsed.metadata["ocr_pages_attempted"] == 1
    assert parsed.metadata["ocr_pages_failed"] == 1
    assert parsed.metadata["ocr_error"] == "ocr page failed"


def test_load_document_does_not_fallback_to_page_images_when_rendered_ocr_fails(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "rendered-ocr-fails.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _ImageFile:
        data = b"embedded-image-should-not-be-used"

    class _BlankPageWithImage(_PdfPage):
        images = [_ImageFile()]

    class _Reader:
        pages = [
            _PdfPage("Извлеченный текст первой страницы"),
            _BlankPageWithImage(""),
        ]

        def __init__(self, path: str) -> None:
            assert path == str(pdf_path)

    rendered_pages: list[int] = []
    image_fallback_calls: list[bytes] = []

    def render_page(path: Path, *, page_index: int, timeout_seconds: float):
        assert path == pdf_path
        assert timeout_seconds > 0
        rendered_pages.append(page_index)
        return f"rendered-page-{page_index}"

    def fail_tesseract(image, remaining_seconds):
        assert image == "rendered-page-2"
        assert remaining_seconds > 0
        raise RuntimeError("tesseract failed after render")

    def record_forbidden_image_fallback(payload: bytes):
        image_fallback_calls.append(payload)
        raise AssertionError("page.images fallback must not run after rendered OCR fails")

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        render_page,
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        fail_tesseract,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._to_ocr_image",
        record_forbidden_image_fallback,
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert [(section.page, section.text) for section in parsed.sections] == [
        (1, "Извлеченный текст первой страницы")
    ]
    assert rendered_pages == [2]
    assert image_fallback_calls == []
    assert parsed.metadata["ocr_attempted"] is True
    assert parsed.metadata["ocr_pages_attempted"] == 1
    assert parsed.metadata["ocr_pages_failed"] == 1
    assert parsed.metadata["ocr_status"] == "failed"
    assert parsed.metadata["ocr_error"] == "tesseract failed after render"


def test_load_document_keeps_existing_pdf_text_when_partial_ocr_unavailable(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "partial-unavailable.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Reader:
        pages = [
            _PdfPage("Извлеченный текст первой страницы"),
            _PdfPage(""),
        ]

        def __init__(self, path: str) -> None:
            del path

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr(
        "src.server.app.document_ingestion._ocr_available",
        lambda: (False, "tesseract is not installed"),
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert [(section.page, section.text) for section in parsed.sections] == [
        (1, "Извлеченный текст первой страницы")
    ]
    assert parsed.metadata["ocr_attempted"] is True
    assert parsed.metadata["ocr_status"] == "unavailable"
    assert parsed.metadata["ocr_pages_attempted"] == 0
    assert parsed.metadata["ocr_error"] == "tesseract is not installed"


def test_load_document_runs_partial_ocr_for_sparse_pdf_pages_in_page_order(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sparse.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Reader:
        pages = [
            _PdfPage("Первая текстовая страница"),
            _PdfPage(""),
            _PdfPage("Третья текстовая страница"),
            _PdfPage(""),
        ]

        def __init__(self, path: str) -> None:
            assert path == str(pdf_path)

    rendered_pages: list[int] = []

    def render_page(path: Path, *, page_index: int, timeout_seconds: float):
        assert path == pdf_path
        assert timeout_seconds > 0
        rendered_pages.append(page_index)
        return f"rendered-page-{page_index}"

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        render_page,
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        lambda image, remaining_seconds: f"OCR {image}",
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert [(section.page, section.text) for section in parsed.sections] == [
        (1, "Первая текстовая страница"),
        (2, "OCR rendered-page-2"),
        (3, "Третья текстовая страница"),
        (4, "OCR rendered-page-4"),
    ]
    assert rendered_pages == [2, 4]
    assert parsed.metadata["ocr_pages_attempted"] == 2
    assert parsed.metadata["ocr_pages_succeeded"] == 2


def test_load_document_limits_partial_pdf_ocr_pages_and_reports_skipped_pages(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "partial-limit.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Reader:
        pages = [
            _PdfPage("Извлеченный текст первой страницы"),
            _PdfPage(""),
            _PdfPage(""),
        ]

        def __init__(self, path: str) -> None:
            del path

    rendered_pages: list[int] = []

    def render_page(path: Path, *, page_index: int, timeout_seconds: float):
        assert path == pdf_path
        assert timeout_seconds > 0
        rendered_pages.append(page_index)
        return f"rendered-page-{page_index}"

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr(
        "src.server.app.document_ingestion._render_pdf_page_for_ocr",
        render_page,
        raising=False,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        lambda image, remaining_seconds: (
            "OCR второй страницы" if image == "rendered-page-2" else "OCR третьей страницы"
        ),
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion.config.DOCUMENT_OCR_MAX_PAGES",
        1,
        raising=False,
    )

    parsed = load_document(pdf_path, root_dir=tmp_path, enable_ocr=True)

    assert [(section.page, section.text) for section in parsed.sections] == [
        (1, "Извлеченный текст первой страницы"),
        (2, "OCR второй страницы"),
    ]
    assert rendered_pages == [2]
    assert parsed.metadata["ocr_status"] == "page_limit"
    assert parsed.metadata["ocr_pages_attempted"] == 1
    assert parsed.metadata["ocr_pages_succeeded"] == 1
    assert parsed.metadata["ocr_pages_skipped"] == 1


def test_pdf_ocr_timeout_does_not_double_count_pages_beyond_limit(monkeypatch):
    class _ImageFile:
        data = b"image-bytes"

    class _Page:
        images = [_ImageFile()]

    class _Reader:
        pages = [_Page(), _Page(), _Page(), _Page()]

    times = iter([0.0, 0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr("src.server.app.document_ingestion.perf_counter", lambda: next(times, 2.0))
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr("src.server.app.document_ingestion._to_ocr_image", lambda payload: object())
    monkeypatch.setattr(
        "src.server.app.document_ingestion._image_to_string_with_timeout",
        lambda image, remaining_seconds: "OCR text",
    )

    texts, metrics = _ocr_pdf_pages(_Reader(), max_pages=2, timeout_seconds=1.0)

    assert texts == [("OCR text", 1)]
    assert metrics["ocr_status"] == "timeout"
    assert metrics["ocr_pages_attempted"] == 1
    assert metrics["ocr_pages_skipped"] == 3


def test_load_document_marks_noisy_docx_without_rejecting_it(tmp_path):
    docx_path = tmp_path / "navigation.docx"
    _write_docx(
        docx_path,
        title="Навигация",
        paragraphs=[
            "1 2 3 4 5 6 7 8 9 10",
            "11 12 13 14 15 16 17 18 19 20",
            "Перейти к разделу Содержание назад далее",
        ],
    )

    parsed = load_document(docx_path, root_dir=tmp_path)

    assert parsed.metadata["quality_status"] == "review"
    assert "navigation_lines" in parsed.metadata["quality_reasons"]
    assert "many_numeric_lines" in parsed.metadata["quality_reasons"]
    assert "Перейти к разделу" in parsed.sections[0].text


def test_load_document_does_not_mark_normal_numbered_docx_as_noisy(tmp_path):
    docx_path = tmp_path / "order.docx"
    _write_docx(
        docx_path,
        title="Приказ",
        paragraphs=[
            "Приказ № 12/34 от 05.09.2026 устанавливает две даты пересдачи.",
            "Студент подает заявление через LMS до 18:00.",
        ],
    )

    parsed = load_document(docx_path, root_dir=tmp_path)

    assert parsed.metadata["quality_status"] == "clean"
    assert parsed.metadata["quality_reasons"] == ""


def test_load_document_does_not_mark_legitimate_grade_mapping_docx_as_noisy(tmp_path):
    docx_path = tmp_path / "grade_mapping.docx"
    _write_docx(
        docx_path,
        title="Шкала оценивания",
        paragraphs=[
            "Соответствие баллов и оценок используется для учебной дисциплины.",
            "0 1 2 3 4 5 6 7 8 9 10",
            "0-3 неудовлетворительно, 4-5 удовлетворительно.",
            "6-7 хорошо, 8-10 отлично.",
            "Итоговая оценка рассчитывается по утвержденной формуле.",
        ],
    )

    parsed = load_document(docx_path, root_dir=tmp_path)

    assert parsed.metadata["quality_status"] == "clean"
    assert parsed.metadata["quality_reasons"] == ""


def test_load_document_marks_scraped_navigation_docx_with_many_links_as_noisy(tmp_path):
    docx_path = tmp_path / "scraped_navigation.docx"
    _write_docx(
        docx_path,
        title="Навигационная выгрузка",
        paragraphs=[
            "Содержание раздела",
            "Перейти к странице программы",
            "Назад Далее",
            "Оглавление",
            "Основной текст почти отсутствует",
        ],
        hyperlink_count=84,
    )

    parsed = load_document(docx_path, root_dir=tmp_path)

    assert parsed.metadata["docx_hyperlinks"] == 84
    assert parsed.metadata["quality_status"] == "review"
    assert "many_hyperlinks" in parsed.metadata["quality_reasons"]
    assert "navigation_lines" in parsed.metadata["quality_reasons"]


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
    assert chunks[0].metadata["source_type"] == "txt"
    assert chunks[0].metadata["source_size"] == txt_path.stat().st_size
    assert chunks[0].metadata["source_sha256"] == sha256(txt_path.read_bytes()).hexdigest()
    assert chunks[0].metadata["indexed_at"] == "2026-03-22T00:00:00Z"


def test_build_chunk_records_preserves_document_quality_flags(tmp_path):
    docx_path = tmp_path / "navigation.docx"
    _write_docx(
        docx_path,
        title="Навигация",
        paragraphs=["1 2 3 4 5 6 7 8 9 10", "назад далее содержание"],
    )
    parsed = load_document(docx_path, root_dir=tmp_path)

    chunks = build_chunk_records(
        parsed, chunk_size=80, overlap=5, indexed_at="2026-03-22T00:00:00Z"
    )

    assert chunks[0].metadata["quality_status"] == "review"
    assert "many_numeric_lines" in chunks[0].metadata["quality_reasons"]


def test_index_directory_reports_extension_reason_counts_and_no_text_rate(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    input_dir.mkdir()
    (input_dir / "ok.txt").write_text("Полезный текст про пересдачу", encoding="utf-8")
    (input_dir / "empty.txt").write_text("", encoding="utf-8")
    (input_dir / "broken.docx").write_text("not a zip archive", encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    summary = index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=True)
    report = summary.to_report()

    assert report["files_seen"] == 3
    assert report["indexed_files"] == 1
    assert report["failed_files"] == 2
    assert report["counts_by_extension"][".txt"] == 2
    assert report["counts_by_extension"][".docx"] == 1
    assert report["counts_by_extension_status"][".txt"] == {"failed": 1, "indexed": 1}
    assert report["counts_by_extension_status"][".docx"] == {"failed": 1}
    assert report["counts_by_extension_reason"][".txt"] == {"no_extractable_text": 1}
    docx_reasons = report["counts_by_extension_reason"][".docx"]
    assert sum(docx_reasons.values()) == 1
    assert set(docx_reasons) <= {"docx_parser_error", "parser_error"}
    assert report["counts_by_reason"]["no_extractable_text"] == 1
    assert any(
        report["counts_by_reason"].get(reason) == 1
        for reason in ("docx_parser_error", "parser_error")
    )
    assert report["no_extractable_text_rate"] == 0.3333
    results_by_source = {entry["source"]: entry for entry in report["results"]}
    assert results_by_source["ok.txt"]["status"] == "indexed"
    assert results_by_source["ok.txt"]["reason"] in {None, "indexed"}
    assert results_by_source["empty.txt"]["status"] == "failed"
    assert results_by_source["empty.txt"]["reason"] == "no_extractable_text"
    assert results_by_source["broken.docx"]["status"] == "failed"
    assert results_by_source["broken.docx"]["reason"] in {"docx_parser_error", "parser_error"}


def test_index_directory_writes_audit_report_with_ocr_success_metrics(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    report_path = tmp_path / "reports" / "indexing.json"
    input_dir.mkdir()
    pdf_path = input_dir / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _ImageFile:
        data = b"image-bytes"

    class _Page:
        images = [_ImageFile()]

        @staticmethod
        def extract_text() -> str:
            return ""

    class _Reader:
        pages = [_Page()]

        def __init__(self, path: str) -> None:
            del path

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr("src.server.app.document_ingestion._to_ocr_image", lambda payload: object())
    monkeypatch.setattr(
        "src.server.app.document_ingestion.pytesseract.image_to_string",
        lambda image, **kwargs: "OCR text",
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion.config.DOCUMENT_OCR_LANG",
        "rus+eng",
        raising=False,
    )

    summary = index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        audit_only=True,
        report_path=report_path,
        enable_ocr=True,
    )

    assert summary.ocr == {
        "documents_attempted": 1,
        "documents_succeeded": 1,
        "success_rate": 1.0,
        "statuses": {"succeeded": 1},
        "pages_attempted": 1,
        "pages_succeeded": 1,
        "pages_failed": 0,
        "pages_skipped": 0,
        "chars": 8,
    }
    assert report_path.exists()
    assert '"success_rate": 1.0' in report_path.read_text(encoding="utf-8")


def test_index_directory_counts_empty_docx_and_no_text_pdfs_in_no_text_rate(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    input_dir.mkdir()
    (input_dir / "ok.txt").write_text("Полезный текст", encoding="utf-8")
    _write_docx(input_dir / "empty.docx", title="Пустой", paragraphs=[])
    (input_dir / "not_recovered.pdf").write_bytes(b"%PDF-1.4\n")
    (input_dir / "recovered.pdf").write_bytes(b"%PDF-1.4\n")

    class _Reader:
        def __init__(self, path: str) -> None:
            if Path(path).name == "recovered.pdf":
                page_class = _empty_pdf_page_class(image_payload=b"recovered-image")
            else:
                page_class = _empty_pdf_page_class()
            self.pages = [page_class()]

    monkeypatch.setattr("src.server.app.document_ingestion.PdfReader", _Reader)
    monkeypatch.setattr("src.server.app.document_ingestion._ocr_available", lambda: (True, None))
    monkeypatch.setattr("src.server.app.document_ingestion._to_ocr_image", lambda payload: payload)
    monkeypatch.setattr(
        "src.server.app.document_ingestion.pytesseract.image_to_string",
        lambda image, **kwargs: "OCR text" if image == b"recovered-image" else "",
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    summary = index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=True,
        enable_ocr=True,
    )
    report = summary.to_report()

    assert report["files_seen"] == 4
    assert report["failed_files"] == 2
    assert report["indexed_files"] == 2
    assert report["counts_by_reason"]["docx_empty"] == 1
    assert any(
        report["counts_by_reason"].get(reason) == 1
        for reason in ("no_extractable_text", "pdf_no_extractable_text")
    )
    assert report["pdf_no_extractable_text_count"] == 2
    assert report["ocr"]["documents_attempted"] == 2
    assert report["ocr"]["documents_succeeded"] == 1
    assert report["no_extractable_text_rate"] == 0.75


def test_index_directory_audit_only_does_not_create_indexes(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    lexical_index_path = tmp_path / "lexical.sqlite3"
    input_dir.mkdir()
    (input_dir / "ok.txt").write_text("Полезный текст про пересдачу", encoding="utf-8")

    def fail_create_vector_store(*args, **kwargs):
        raise AssertionError("audit-only mode must not open or mutate the vector store")

    def fail_initialize_lexical_index(*args, **kwargs):
        raise AssertionError("audit-only mode must not open or mutate the lexical index")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.create_vector_store",
        fail_create_vector_store,
    )
    monkeypatch.setattr(
        "src.server.app.document_ingestion.lexical.initialize_lexical_index",
        fail_initialize_lexical_index,
    )

    summary = index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=True,
        lexical_index_path=lexical_index_path,
        audit_only=True,
    )

    assert summary.indexed_files == 1
    assert summary.chunks_written > 0
    assert not persist_dir.exists()
    assert not lexical_index_path.exists()


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


def test_index_directory_ignores_temporary_word_lock_files(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    input_dir.mkdir()
    (input_dir / "guide.txt").write_text("Полезный текст для индексации", encoding="utf-8")
    (input_dir / "~$guide.docx").write_text("not a real docx", encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    summary = index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=True)

    assert summary.files_seen == 1
    assert summary.indexed_files == 1
    assert summary.failed_files == 0


def test_index_directory_deletes_stale_documents_without_rebuild(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("A" * 400, encoding="utf-8")
    stale_path = input_dir / "b.txt"
    stale_path.write_text("B" * 400, encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=True)
    store = Chroma(
        collection_name="test_docs",
        persist_directory=str(persist_dir),
        embedding_function=_FakeEmbeddings(),
    )
    initial = store._collection.get(include=["metadatas"])
    initial_document_ids = {
        metadata["document_id"] for metadata in initial["metadatas"] if metadata is not None
    }
    assert len(initial_document_ids) == 2

    stale_path.unlink()
    index_directory(input_dir, persist_dir, collection_name="test_docs", rebuild=False)

    after_cleanup = store._collection.get(include=["metadatas"])
    remaining_document_ids = {
        metadata["document_id"] for metadata in after_cleanup["metadatas"] if metadata is not None
    }

    assert len(remaining_document_ids) == 1


def test_index_directory_writes_and_cleans_lexical_index(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    lexical_index_path = tmp_path / "custom" / "lexical.sqlite3"
    input_dir.mkdir()
    (input_dir / "retake.txt").write_text("Правила первой пересдачи", encoding="utf-8")
    stale_path = input_dir / "discipline.txt"
    stale_path.write_text("Дисциплинарное взыскание", encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=True,
        lexical_index_path=lexical_index_path,
    )
    assert len(get_indexed_document_ids(lexical_index_path)) == 2
    assert search_lexical("пересдача", limit=5, index_path=lexical_index_path)

    stale_path.unlink()
    index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=False,
        lexical_index_path=lexical_index_path,
    )

    assert len(get_indexed_document_ids(lexical_index_path)) == 1
    assert search_lexical("взыскание", limit=5, index_path=lexical_index_path) == []


def test_index_directory_parse_failure_preserves_previous_lexical_rows(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    lexical_index_path = tmp_path / "lexical.sqlite3"
    input_dir.mkdir()
    source_path = input_dir / "retake.txt"
    source_path.write_text("Старый текст про пересдачу", encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=True,
        lexical_index_path=lexical_index_path,
    )
    source_path.write_text("", encoding="utf-8")

    summary = index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=False,
        lexical_index_path=lexical_index_path,
    )

    assert summary.failed_files == 1
    assert search_lexical("пересдача", limit=5, index_path=lexical_index_path)


def test_index_directory_rolls_back_dense_rows_when_lexical_write_fails(tmp_path, monkeypatch):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    lexical_index_path = tmp_path / "lexical.sqlite3"
    input_dir.mkdir()
    (input_dir / "retake.txt").write_text("Правила пересдачи" * 20, encoding="utf-8")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.get_embedding_function", lambda: _FakeEmbeddings()
    )

    def fail_replace_document_chunks(*args, **kwargs):
        raise RuntimeError("lexical write failed")

    monkeypatch.setattr(
        "src.server.app.document_ingestion.lexical.replace_document_chunks",
        fail_replace_document_chunks,
    )

    summary = index_directory(
        input_dir,
        persist_dir,
        collection_name="test_docs",
        rebuild=True,
        lexical_index_path=lexical_index_path,
    )
    store = Chroma(
        collection_name="test_docs",
        persist_directory=str(persist_dir),
        embedding_function=_FakeEmbeddings(),
    )

    assert summary.failed_files == 1
    assert summary.indexed_files == 0
    assert store._collection.count() == 0
    assert get_indexed_document_ids(lexical_index_path) == set()
