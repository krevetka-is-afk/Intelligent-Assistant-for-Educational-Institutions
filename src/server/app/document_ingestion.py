from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from time import perf_counter
from typing import Any
from xml.etree import ElementTree as ET

import pytesseract
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from PIL import Image, ImageOps
from pypdf import PdfReader

from . import config, lexical
from .vector import get_embedding_function

logger = logging.getLogger("server.indexing")

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".html", ".htm", ".txt", ".docx"})
OCR_RENDER_DPI = 200
OCR_RENDER_SCALE_TO = 1600
OCR_MAX_IMAGE_SIZE = (2500, 3500)
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
CORE_NS = {"dc": "http://purl.org/dc/elements/1.1/"}


class DocumentParsingError(RuntimeError):
    """Raised when a supported document cannot be parsed."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "parser_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details or {}


@dataclass(slots=True)
class TextSection:
    text: str
    page: int | None
    base_offset: int


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    source: str
    title: str
    mime_type: str
    source_type: str
    source_size: int
    source_sha256: str
    sections: list[TextSection]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChunkRecord:
    id: str
    page_content: str
    metadata: dict[str, Any]

    def to_document(self) -> Document:
        metadata = dict(self.metadata)
        flags = metadata.get("quality_flags")
        if isinstance(flags, list):
            metadata["quality_flags"] = ",".join(str(flag) for flag in flags)
        return Document(id=self.id, page_content=self.page_content, metadata=metadata)


@dataclass(slots=True)
class IndexingSummary:
    generated_at: str | None = None
    input_dir: str | None = None
    files_seen: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    chunks_written: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    counts_by_extension: dict[str, int] | None = None
    counts_by_extension_total: dict[str, int] | None = None
    counts_by_status: dict[str, int] | None = None
    counts_by_reason: dict[str, int] | None = None
    counts_by_extension_status: dict[str, dict[str, int]] | None = None
    counts_by_extension_reason: dict[str, dict[str, int]] | None = None
    ocr: dict[str, Any] | None = None
    pdf_no_extractable_text_count: int = 0
    no_extractable_text_rate: float = 0.0

    def to_report(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_dir": self.input_dir,
            "files_seen": self.files_seen,
            "indexed_files": self.indexed_files,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
            "chunks_written": self.chunks_written,
            "counts_by_extension": self.counts_by_extension or {},
            "counts_by_extension_total": self.counts_by_extension_total or {},
            "counts_by_status": self.counts_by_status or {},
            "counts_by_reason": self.counts_by_reason or {},
            "counts_by_extension_status": self.counts_by_extension_status or {},
            "counts_by_extension_reason": self.counts_by_extension_reason or {},
            "ocr": self.ocr or {},
            "pdf_no_extractable_text_count": self.pdf_no_extractable_text_count,
            "no_extractable_text_rate": self.no_extractable_text_rate,
            "results": self.results,
        }


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._current_tag: str | None = None
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._current_tag = tag.lower()
        if self._current_tag in {"script", "style"}:
            self._skip_depth += 1
        elif self._current_tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif lowered in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._text_parts.append("\n")
        self._current_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._current_tag == "title":
            self._title_parts.append(data)
        self._text_parts.append(data)

    @property
    def title(self) -> str | None:
        title = normalize_text(" ".join(self._title_parts))
        return title or None

    @property
    def text(self) -> str:
        return normalize_text("".join(self._text_parts))


def normalize_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", " ").replace("\x0c", "\n")
    lines = [" ".join(line.split()) for line in normalized.split("\n")]

    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if compact_lines and not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(line)
        previous_blank = False

    return "\n".join(compact_lines).strip()


def build_document_id(relative_path: Path) -> str:
    return sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:24]


def _is_temporary_office_file(path: Path) -> bool:
    return path.suffix.lower() == ".docx" and path.name.startswith("~$")


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[tuple[int, int, str]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= text_length:
            break
        start += step

    return chunks


def _build_sections(texts: Iterable[tuple[str, int | None]]) -> list[TextSection]:
    sections: list[TextSection] = []
    offset = 0
    for text, page in texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        sections.append(TextSection(text=normalized, page=page, base_offset=offset))
        offset += len(normalized) + 2
    return sections


def _empty_parse_metrics() -> dict[str, Any]:
    return {
        "ocr_attempted": False,
        "ocr_status": "not_applicable",
        "ocr_pages_attempted": 0,
        "ocr_pages_succeeded": 0,
        "ocr_pages_failed": 0,
        "ocr_pages_unavailable": 0,
        "ocr_pages_skipped": 0,
        "ocr_renderer_fallback_pages": 0,
        "ocr_chars": 0,
        "ocr_page_results_json": "[]",
        "pdf_pages": 0,
    }


def _ocr_available(*, require_renderer: bool = False) -> tuple[bool, str | None]:
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
    if require_renderer and shutil.which("pdftoppm") is None:
        return False, "pdftoppm is not installed"
    return True, None


def _image_to_string_with_timeout(image: Image.Image, *, remaining_seconds: float) -> str:
    return pytesseract.image_to_string(
        image,
        lang=config.DOCUMENT_OCR_LANG,
        timeout=remaining_seconds,
    )


def _to_ocr_image(image_payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_payload))
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "P"}:
        image = image.convert("RGB")
    image.thumbnail(OCR_MAX_IMAGE_SIZE)
    return image


def _render_pdf_page_image(
    pdf_path: Path,
    page_number: int,
    *,
    timeout_seconds: float | None = None,
) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="pdf-ocr-") as tmp_dir:
        output_prefix = Path(tmp_dir) / "page"
        completed = subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-r",
                str(OCR_RENDER_DPI),
                "-scale-to",
                str(OCR_RENDER_SCALE_TO),
                "-png",
                str(pdf_path),
                str(output_prefix),
            ],
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(stderr or f"pdftoppm failed with code {completed.returncode}")
        rendered_pages = sorted(Path(tmp_dir).glob("page-*.png"))
        if not rendered_pages:
            raise RuntimeError("pdftoppm did not produce a page image")
        image = Image.open(rendered_pages[0])
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "P"}:
            image = image.convert("RGB")
        image.thumbnail(OCR_MAX_IMAGE_SIZE)
        return image


def _render_pdf_page_for_ocr(
    pdf_path: Path,
    *,
    page_index: int,
    timeout_seconds: float,
) -> Image.Image:
    return _render_pdf_page_image(
        pdf_path,
        page_index,
        timeout_seconds=timeout_seconds,
    )


def _ocr_pdf_pages(
    reader: PdfReader,
    *,
    max_pages: int,
    timeout_seconds: float,
    page_indices: Iterable[int] | None = None,
    pdf_path: Path | None = None,
) -> tuple[list[tuple[str, int | None]], dict[str, Any]]:
    metrics = _empty_parse_metrics()
    metrics["ocr_attempted"] = True
    metrics["pdf_pages"] = len(reader.pages)
    candidate_page_indices = (
        list(range(1, len(reader.pages) + 1)) if page_indices is None else list(page_indices)
    )
    page_results: list[dict[str, Any]] = []
    try:
        available, unavailable_reason = _ocr_available(require_renderer=pdf_path is not None)
    except TypeError:
        available, unavailable_reason = _ocr_available()
    if not available:
        metrics["ocr_status"] = "unavailable"
        metrics["ocr_error"] = unavailable_reason
        metrics["ocr_pages_unavailable"] = len(candidate_page_indices)
        page_results.extend(
            {"page": page_index, "status": "unavailable"} for page_index in candidate_page_indices
        )
        metrics["ocr_page_results_json"] = json.dumps(
            page_results,
            ensure_ascii=False,
            sort_keys=True,
        )
        return [], metrics

    page_texts: list[tuple[str, int | None]] = []
    started = perf_counter()
    page_limit_reached = len(candidate_page_indices) > max_pages
    process_page_indices = candidate_page_indices[:max_pages]
    skipped_by_limit = candidate_page_indices[max_pages:]
    if skipped_by_limit:
        metrics["ocr_pages_skipped"] = len(skipped_by_limit)
        page_results.extend(
            {"page": page_index, "status": "skipped_page_limit"} for page_index in skipped_by_limit
        )
    for ocr_index, page_index in enumerate(process_page_indices):
        elapsed = perf_counter() - started
        if elapsed >= timeout_seconds:
            metrics["ocr_status"] = "timeout"
            timed_out_pages = process_page_indices[ocr_index:]
            metrics["ocr_pages_skipped"] += len(timed_out_pages)
            page_results.extend(
                {"page": skipped_page_index, "status": "skipped_timeout"}
                for skipped_page_index in timed_out_pages
            )
            break
        page = reader.pages[page_index - 1]
        metrics["ocr_pages_attempted"] += 1
        image_texts: list[str] = []
        try:
            if pdf_path is not None:
                remaining = timeout_seconds - (perf_counter() - started)
                if remaining <= 0:
                    metrics["ocr_status"] = "timeout"
                    metrics["ocr_pages_failed"] += 1
                    page_results.append({"page": page_index, "status": "failed_timeout"})
                    timed_out_pages = process_page_indices[ocr_index + 1 :]
                    metrics["ocr_pages_skipped"] += len(timed_out_pages)
                    page_results.extend(
                        {"page": skipped_page_index, "status": "skipped_timeout"}
                        for skipped_page_index in timed_out_pages
                    )
                    break
                try:
                    image = _render_pdf_page_for_ocr(
                        pdf_path,
                        page_index=page_index,
                        timeout_seconds=remaining,
                    )
                except RuntimeError:
                    page_images = getattr(page, "images", [])
                    if not page_images:
                        raise
                    metrics["ocr_renderer_fallback_pages"] += 1
                    for image_file in page_images:
                        remaining = timeout_seconds - (perf_counter() - started)
                        if remaining <= 0:
                            metrics["ocr_status"] = "timeout"
                            metrics["ocr_pages_failed"] += 1
                            page_results.append({"page": page_index, "status": "failed_timeout"})
                            timed_out_pages = process_page_indices[ocr_index + 1 :]
                            metrics["ocr_pages_skipped"] += len(timed_out_pages)
                            page_results.extend(
                                {"page": skipped_page_index, "status": "skipped_timeout"}
                                for skipped_page_index in timed_out_pages
                            )
                            break
                        image = _to_ocr_image(image_file.data)
                        remaining = timeout_seconds - (perf_counter() - started)
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired("pdf-ocr", timeout_seconds)
                        image_texts.append(
                            _image_to_string_with_timeout(
                                image,
                                remaining_seconds=remaining,
                            )
                        )
                else:
                    remaining = timeout_seconds - (perf_counter() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired("pdf-ocr", timeout_seconds)
                    image_texts.append(
                        _image_to_string_with_timeout(image, remaining_seconds=remaining)
                    )
            else:
                page_images = getattr(page, "images", [])
                if not page_images:
                    metrics["ocr_pages_failed"] += 1
                    page_results.append({"page": page_index, "status": "failed_no_images"})
                    continue
                for image_file in page_images:
                    remaining = timeout_seconds - (perf_counter() - started)
                    if remaining <= 0:
                        metrics["ocr_status"] = "timeout"
                        metrics["ocr_pages_failed"] += 1
                        page_results.append({"page": page_index, "status": "failed_timeout"})
                        timed_out_pages = process_page_indices[ocr_index + 1 :]
                        metrics["ocr_pages_skipped"] += len(timed_out_pages)
                        page_results.extend(
                            {"page": skipped_page_index, "status": "skipped_timeout"}
                            for skipped_page_index in timed_out_pages
                        )
                        break
                    image = _to_ocr_image(image_file.data)
                    remaining = timeout_seconds - (perf_counter() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired("pdf-ocr", timeout_seconds)
                    image_texts.append(
                        _image_to_string_with_timeout(image, remaining_seconds=remaining)
                    )
        except (
            RuntimeError,
            subprocess.TimeoutExpired,
            pytesseract.TesseractError,
            pytesseract.TesseractNotFoundError,
            OSError,
        ) as exc:
            metrics["ocr_pages_failed"] += 1
            metrics["ocr_error"] = str(exc)
            if perf_counter() - started >= timeout_seconds:
                metrics["ocr_status"] = "timeout"
                page_results.append({"page": page_index, "status": "failed_timeout"})
                timed_out_pages = process_page_indices[ocr_index + 1 :]
                metrics["ocr_pages_skipped"] += len(timed_out_pages)
                page_results.extend(
                    {"page": skipped_page_index, "status": "skipped_timeout"}
                    for skipped_page_index in timed_out_pages
                )
                break
            page_results.append({"page": page_index, "status": "failed_error"})
            continue

        if metrics["ocr_status"] == "timeout":
            break

        normalized = normalize_text("\n".join(image_texts))
        if normalized:
            metrics["ocr_pages_succeeded"] += 1
            metrics["ocr_chars"] += len(normalized)
            page_texts.append((normalized, page_index))
            page_results.append(
                {"page": page_index, "status": "succeeded", "chars": len(normalized)}
            )
        else:
            metrics["ocr_pages_failed"] += 1
            page_results.append({"page": page_index, "status": "failed_empty_text"})

    if metrics["ocr_status"] != "timeout":
        if page_texts and (page_limit_reached or metrics["ocr_pages_failed"]):
            metrics["ocr_status"] = "page_limit" if page_limit_reached else "partial"
        elif page_texts:
            metrics["ocr_status"] = "succeeded"
        elif page_limit_reached:
            metrics["ocr_status"] = "page_limit"
        else:
            metrics["ocr_status"] = "failed"
    metrics["ocr_page_results_json"] = json.dumps(
        sorted(page_results, key=lambda item: (int(item["page"]), str(item["status"]))),
        ensure_ascii=False,
        sort_keys=True,
    )
    return page_texts, metrics


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_pdf(
    path: Path,
    *,
    enable_ocr: bool = False,
) -> tuple[str, list[TextSection], dict[str, Any]]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentParsingError(
            f"Could not read PDF {path}",
            reason="parser_error",
            details={"detail_reason": "pdf_parser_error"},
        ) from exc

    page_texts = [(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
    normalized_by_page = {
        page: normalized for text, page in page_texts if (normalized := normalize_text(text))
    }
    blank_page_indices = [
        page for text, page in page_texts if page is not None and not normalize_text(text)
    ]
    metrics = _empty_parse_metrics()
    metrics["pdf_pages"] = len(reader.pages)
    metrics["pdf_extractable_chars"] = sum(len(text) for text in normalized_by_page.values())
    metrics["pdf_extractable_pages"] = len(normalized_by_page)
    metrics["pdf_no_text_pages"] = len(blank_page_indices)
    metrics["pdf_raw_no_extractable_text"] = metrics["pdf_extractable_chars"] == 0
    ocr_by_page: dict[int, str] = {}
    if blank_page_indices:
        if enable_ocr:
            ocr_texts, ocr_metrics = _ocr_pdf_pages(
                reader,
                max_pages=config.DOCUMENT_OCR_MAX_PAGES,
                timeout_seconds=config.DOCUMENT_OCR_TIMEOUT_SECONDS,
                page_indices=blank_page_indices,
                pdf_path=path,
            )
            metrics.update(ocr_metrics)
            ocr_by_page = {
                page: normalized
                for text, page in ocr_texts
                if page is not None and (normalized := normalize_text(text))
            }
        else:
            metrics["ocr_status"] = "disabled"
            metrics["ocr_pages_skipped"] = len(blank_page_indices)
            metrics["ocr_page_results_json"] = json.dumps(
                [
                    {"page": page_index, "status": "skipped_disabled"}
                    for page_index in blank_page_indices
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
    combined_texts = [
        (
            (normalized_by_page[page_index], page_index)
            if page_index in normalized_by_page
            else (ocr_by_page.get(page_index, ""), page_index)
        )
        for page_index in range(1, len(reader.pages) + 1)
    ]
    sections = _build_sections(combined_texts)
    if not sections:
        raise DocumentParsingError(
            f"PDF {path} did not contain extractable text",
            reason="no_extractable_text",
            details={**metrics, "detail_reason": "pdf_no_extractable_text"},
        )

    title = normalize_text(path.stem)
    return title or path.stem, sections, metrics


def _load_html(path: Path) -> tuple[str, list[TextSection], dict[str, Any]]:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(_read_text_file(path))
        parser.close()
    except Exception as exc:
        raise DocumentParsingError(
            f"Could not parse HTML {path}",
            reason="parser_error",
            details={"detail_reason": "html_parser_error"},
        ) from exc

    if not parser.text:
        raise DocumentParsingError(
            f"HTML {path} did not contain extractable text",
            reason="no_extractable_text",
        )

    title = parser.title or normalize_text(path.stem) or path.stem
    return title, _build_sections([(parser.text, None)]), {}


def _load_txt(path: Path) -> tuple[str, list[TextSection], dict[str, Any]]:
    text = normalize_text(_read_text_file(path))
    if not text:
        raise DocumentParsingError(f"TXT {path} is empty", reason="no_extractable_text")
    return normalize_text(path.stem) or path.stem, _build_sections([(text, None)]), {}


def _extract_docx_title(archive: zipfile.ZipFile) -> str | None:
    try:
        raw_core = archive.read("docProps/core.xml")
    except KeyError:
        return None

    try:
        root = ET.fromstring(raw_core)
    except ET.ParseError:
        return None

    title_node = root.find(".//dc:title", CORE_NS)
    if title_node is None or title_node.text is None:
        return None

    title = normalize_text(title_node.text)
    return title or None


def _extract_docx_text(archive: zipfile.ZipFile) -> tuple[str, dict[str, Any]]:
    try:
        raw_document = archive.read("word/document.xml")
    except KeyError as exc:
        raise DocumentParsingError(
            "DOCX is missing word/document.xml",
            reason="parser_error",
            details={"detail_reason": "docx_parser_error"},
        ) from exc

    try:
        root = ET.fromstring(raw_document)
    except ET.ParseError as exc:
        raise DocumentParsingError(
            "DOCX XML is malformed",
            reason="parser_error",
            details={"detail_reason": "docx_parser_error"},
        ) from exc

    paragraphs: list[str] = []
    navigation_lines = 0
    navigation_hits = 0
    numeric_lines = 0
    navigation_markers = (
        "оглавление",
        "назад",
        "далее",
        "перейти",
        "меню",
        "карта сайта",
        "личный кабинет",
    )
    for paragraph in root.findall(".//w:p", WORD_NS):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)]
        joined = normalize_text("".join(texts))
        if joined:
            paragraphs.append(joined)
            lowered = joined.lower()
            line_navigation_hits = sum(1 for marker in navigation_markers if marker in lowered)
            if line_navigation_hits:
                navigation_lines += 1
                navigation_hits += line_navigation_hits
            if sum(char.isdigit() for char in joined) >= max(3, int(len(joined) * 0.45)):
                numeric_lines += 1

    text = "\n\n".join(paragraphs)
    plain_length = len(text)
    digit_count = sum(char.isdigit() for char in text)
    relationship_files = [name for name in archive.namelist() if name.endswith(".rels")]
    hyperlink_count = 0
    for name in relationship_files:
        try:
            rels_root = ET.fromstring(archive.read(name))
        except ET.ParseError:
            continue
        hyperlink_count += sum(
            1 for node in rels_root if "hyperlink" in str(node.attrib.get("Type", "")).lower()
        )
    media_count = sum(1 for name in archive.namelist() if name.startswith("word/media/"))
    metrics = {
        "docx_paragraphs": len(paragraphs),
        "docx_chars": plain_length,
        "docx_digit_ratio": round(digit_count / plain_length, 4) if plain_length else 0.0,
        "docx_navigation_line_ratio": (
            round(navigation_lines / len(paragraphs), 4) if paragraphs else 0.0
        ),
        "docx_navigation_hits": navigation_hits,
        "docx_numeric_line_ratio": round(numeric_lines / len(paragraphs), 4) if paragraphs else 0.0,
        "docx_hyperlinks": hyperlink_count,
        "docx_hyperlink_density": round(hyperlink_count / max(plain_length / 1000, 1), 4),
        "docx_media": media_count,
    }
    return text, metrics


def _classify_docx_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    text_length = int(metrics.get("docx_chars") or 0)
    digit_ratio = float(metrics.get("docx_digit_ratio") or 0.0)
    numeric_line_ratio = float(metrics.get("docx_numeric_line_ratio") or 0.0)
    navigation_line_ratio = float(metrics.get("docx_navigation_line_ratio") or 0.0)
    navigation_hits = int(metrics.get("docx_navigation_hits") or 0)
    hyperlink_count = int(metrics.get("docx_hyperlinks") or 0)
    hyperlink_density = float(metrics.get("docx_hyperlink_density") or 0.0)
    media_count = int(metrics.get("docx_media") or 0)

    numeric_evidence = digit_ratio > 0.2 or numeric_line_ratio > 0.3
    strong_numeric_garble = digit_ratio > 0.55 and numeric_line_ratio > 0.75 and text_length > 500
    hyperlink_evidence = hyperlink_count >= 20 and hyperlink_density >= 3.0
    concentrated_navigation_evidence = navigation_line_ratio > 0.08
    web_navigation_evidence = hyperlink_evidence and navigation_hits >= 1
    navigation_evidence = concentrated_navigation_evidence or web_navigation_evidence

    if text_length < 200 and media_count > 0:
        reasons.append("media_heavy_short_text")
    if numeric_evidence:
        reasons.append("high_digit_ratio")
    if navigation_evidence:
        reasons.append("navigation_lines")
    if hyperlink_evidence:
        reasons.append("many_hyperlinks")
    if numeric_line_ratio > 0.45 and numeric_evidence:
        reasons.append("many_numeric_lines")
    if strong_numeric_garble:
        reasons.append("numeric_garble")

    quality_flags: list[str] = []
    if "media_heavy_short_text" in reasons:
        quality_flags.append("media_heavy_short_text")
    if strong_numeric_garble:
        quality_flags.append("numeric_garble")
    if navigation_evidence and hyperlink_evidence:
        quality_flags.append("navigation_noise")
    if concentrated_navigation_evidence and numeric_evidence:
        quality_flags.append("numeric_navigation_noise")

    noisy = bool(quality_flags)
    return {
        **metrics,
        "quality_status": "review" if noisy else "clean",
        "quality_score": len(reasons),
        "quality_reasons": ",".join(reasons),
        "quality_flags": quality_flags,
    }


def _load_docx(path: Path) -> tuple[str, list[TextSection], dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as archive:
            title = _extract_docx_title(archive)
            text, metrics = _extract_docx_text(archive)
    except zipfile.BadZipFile as exc:
        raise DocumentParsingError(
            f"DOCX {path} is not a valid archive",
            reason="parser_error",
            details={"detail_reason": "docx_parser_error"},
        ) from exc

    normalized_text = normalize_text(text)
    quality_metadata = _classify_docx_quality(metrics)
    if not normalized_text:
        raise DocumentParsingError(
            f"DOCX {path} is empty",
            reason="no_extractable_text",
            details={**quality_metadata, "detail_reason": "docx_empty"},
        )

    resolved_title = title or normalize_text(path.stem) or path.stem
    return resolved_title, _build_sections([(normalized_text, None)]), quality_metadata


def load_document(path: Path, *, root_dir: Path, enable_ocr: bool | None = None) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(f"Unsupported file type: {path}")

    loaders = {
        ".pdf": lambda item: _load_pdf(
            item,
            enable_ocr=config.DOCUMENT_OCR_ENABLED if enable_ocr is None else enable_ocr,
        ),
        ".html": _load_html,
        ".htm": _load_html,
        ".txt": _load_txt,
        ".docx": _load_docx,
    }

    title, sections, extra_metadata = loaders[suffix](path)
    relative_path = path.resolve().relative_to(root_dir.resolve())
    raw_content = path.read_bytes()
    return ParsedDocument(
        document_id=build_document_id(relative_path),
        source=relative_path.as_posix(),
        title=title,
        mime_type=MIME_TYPES[suffix],
        source_type=suffix.lstrip("."),
        source_size=len(raw_content),
        source_sha256=sha256(raw_content).hexdigest(),
        sections=sections,
        metadata=extra_metadata,
    )


def build_chunk_records(
    parsed_document: ParsedDocument,
    *,
    chunk_size: int,
    overlap: int,
    indexed_at: str | None = None,
) -> list[ChunkRecord]:
    timestamp = indexed_at or datetime.now(UTC).isoformat()
    records: list[ChunkRecord] = []
    chunk_index = 0

    for section in parsed_document.sections:
        for start, end, chunk_text_value in chunk_text(
            section.text,
            chunk_size=chunk_size,
            overlap=overlap,
        ):
            chunk_id = f"{parsed_document.document_id}:{chunk_index:05d}"
            metadata: dict[str, Any] = {
                "document_id": parsed_document.document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "source": parsed_document.source,
                "title": parsed_document.title,
                "mime_type": parsed_document.mime_type,
                "source_type": parsed_document.source_type,
                "source_size": parsed_document.source_size,
                "source_sha256": parsed_document.source_sha256,
                "char_start": section.base_offset + start,
                "char_end": section.base_offset + end,
                "indexed_at": timestamp,
            }
            metadata.update(parsed_document.metadata)
            if section.page is not None:
                metadata["page"] = section.page
            records.append(
                ChunkRecord(id=chunk_id, page_content=chunk_text_value, metadata=metadata)
            )
            chunk_index += 1

    return records


def _iter_supported_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_temporary_office_file(path)
    )


def _new_file_result(path: Path, *, input_dir: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(input_dir.resolve()).as_posix()
    return {
        "source": relative,
        "extension": path.suffix.lower(),
        "status": "pending",
        "reason": None,
        "chunks": 0,
        "metadata": {},
    }


def _finish_report(summary: IndexingSummary) -> None:
    results = summary.results or []
    summary.counts_by_extension_total = dict(Counter(item["extension"] for item in results))
    summary.counts_by_extension = dict(summary.counts_by_extension_total)
    summary.counts_by_status = dict(Counter(item["status"] for item in results))
    summary.counts_by_reason = dict(
        Counter(
            item["reason"]
            for item in results
            if item.get("reason") and item.get("status") != "indexed"
        )
    )
    extension_status: dict[str, Counter[str]] = {}
    extension_reason: dict[str, Counter[str]] = {}
    for item in results:
        extension = str(item.get("extension") or "")
        extension_status.setdefault(extension, Counter()).update([str(item.get("status"))])
        metadata = item.get("metadata") or {}
        reason = metadata.get("detail_reason") or item.get("reason")
        if reason and item.get("status") != "indexed":
            extension_reason.setdefault(extension, Counter()).update([str(reason)])
    summary.counts_by_extension_status = {
        extension: dict(counter) for extension, counter in extension_status.items()
    }
    summary.counts_by_extension_reason = {
        extension: dict(counter) for extension, counter in extension_reason.items()
    }
    summary.pdf_no_extractable_text_count = sum(
        1
        for item in results
        if item.get("extension") == ".pdf"
        and (item.get("metadata") or {}).get("pdf_extractable_chars") == 0
    )
    public_no_text_count = sum(
        1
        for item in results
        if item.get("reason") in {"no_extractable_text", "pdf_no_extractable_text", "docx_empty"}
    )
    recovered_pdf_no_text_count = sum(
        1
        for item in results
        if item.get("extension") == ".pdf"
        and item.get("status") != "failed"
        and (item.get("metadata") or {}).get("pdf_extractable_chars") == 0
    )
    no_text_count = public_no_text_count + recovered_pdf_no_text_count
    summary.no_extractable_text_rate = (
        round(no_text_count / summary.files_seen, 4) if summary.files_seen else 0.0
    )
    ocr_results = [
        item.get("metadata") or {}
        for item in results
        if (item.get("metadata") or {}).get("ocr_attempted") is True
    ]
    ocr_statuses = Counter(str(item.get("ocr_status")) for item in ocr_results)
    attempted = len(ocr_results)
    succeeded = sum(1 for item in ocr_results if int(item.get("ocr_pages_succeeded") or 0) > 0)
    summary.ocr = {
        "documents_attempted": attempted,
        "documents_succeeded": succeeded,
        "success_rate": round(succeeded / attempted, 4) if attempted else 0.0,
        "statuses": dict(ocr_statuses),
        "pages_attempted": sum(int(item.get("ocr_pages_attempted") or 0) for item in ocr_results),
        "pages_succeeded": sum(int(item.get("ocr_pages_succeeded") or 0) for item in ocr_results),
        "pages_failed": sum(int(item.get("ocr_pages_failed") or 0) for item in ocr_results),
        "pages_skipped": sum(int(item.get("ocr_pages_skipped") or 0) for item in ocr_results),
        "chars": sum(int(item.get("ocr_chars") or 0) for item in ocr_results),
    }


def write_indexing_report(summary: IndexingSummary, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary.to_report(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _delete_document_chunks(vector_store: Chroma, document_id: str) -> None:
    collection = vector_store._collection
    existing = collection.get(where={"document_id": document_id}, include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)


def _get_indexed_document_ids(vector_store: Chroma) -> set[str]:
    collection = vector_store._collection
    existing = collection.get(include=["metadatas"])
    document_ids: set[str] = set()

    for metadata in existing.get("metadatas") or []:
        if not isinstance(metadata, dict):
            continue
        document_id = metadata.get("document_id")
        if isinstance(document_id, str) and document_id:
            document_ids.add(document_id)

    return document_ids


def _delete_stale_document_chunks(
    vector_store: Chroma, *, active_document_ids: set[str]
) -> list[str]:
    stale_document_ids = sorted(_get_indexed_document_ids(vector_store) - active_document_ids)
    for document_id in stale_document_ids:
        _delete_document_chunks(vector_store, document_id)
    return stale_document_ids


def create_vector_store(persist_directory: Path, collection_name: str, *, rebuild: bool) -> Chroma:
    embedding_function = get_embedding_function()
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )

    if rebuild:
        client = vector_store._client
        try:
            client.delete_collection(collection_name)
        except NotFoundError:
            pass
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )

    return vector_store


def index_directory(
    input_dir: Path,
    persist_directory: Path,
    *,
    collection_name: str | None = None,
    rebuild: bool = False,
    chunk_size: int | None = None,
    overlap: int | None = None,
    lexical_index_path: Path | None = None,
    audit_only: bool = False,
    report_path: Path | None = None,
    enable_ocr: bool | None = None,
) -> IndexingSummary:
    config.validate_chunk_settings()
    collection = collection_name or config.CHROMA_COLLECTION_NAME
    resolved_chunk_size = chunk_size or config.CHUNK_SIZE
    resolved_overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    resolved_enable_ocr = config.DOCUMENT_OCR_ENABLED if enable_ocr is None else enable_ocr

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not audit_only:
        persist_directory.mkdir(parents=True, exist_ok=True)
        vector_store = create_vector_store(persist_directory, collection, rebuild=rebuild)
        resolved_lexical_index_path = lexical.resolve_lexical_index_path(
            persist_directory=persist_directory,
            lexical_index_path=lexical_index_path,
        )
        lexical.initialize_lexical_index(resolved_lexical_index_path, rebuild=rebuild)
    else:
        vector_store = None
        resolved_lexical_index_path = None

    indexed_at = datetime.now(UTC).isoformat()
    summary = IndexingSummary(
        generated_at=indexed_at,
        input_dir=str(input_dir.resolve()),
        results=[],
    )
    indexed_paths = _iter_supported_files(input_dir)
    root_dir = input_dir.resolve()
    active_document_ids = {
        build_document_id(path.resolve().relative_to(root_dir)) for path in indexed_paths
    }

    for path in indexed_paths:
        summary.files_seen += 1
        file_result = _new_file_result(path, input_dir=input_dir)
        summary.results.append(file_result)
        try:
            parsed_document = load_document(
                path,
                root_dir=input_dir,
                enable_ocr=resolved_enable_ocr,
            )
            chunk_records = build_chunk_records(
                parsed_document,
                chunk_size=resolved_chunk_size,
                overlap=resolved_overlap,
                indexed_at=indexed_at,
            )
            file_result["metadata"] = parsed_document.metadata
            file_result["chunks"] = len(chunk_records)
            if not chunk_records:
                summary.skipped_files += 1
                file_result["status"] = "skipped"
                file_result["reason"] = "no_chunks"
                logger.warning("Skipping %s because it produced no chunks", path)
                continue

            if audit_only:
                summary.indexed_files += 1
                summary.chunks_written += len(chunk_records)
                file_result["status"] = "parsed"
                continue

            try:
                assert vector_store is not None
                assert resolved_lexical_index_path is not None
                _delete_document_chunks(vector_store, parsed_document.document_id)
                vector_store.add_documents(
                    documents=[record.to_document() for record in chunk_records],
                    ids=[record.id for record in chunk_records],
                )
                lexical.replace_document_chunks(
                    resolved_lexical_index_path,
                    parsed_document.document_id,
                    chunk_records,
                )
            except Exception:
                assert vector_store is not None
                assert resolved_lexical_index_path is not None
                _delete_document_chunks(vector_store, parsed_document.document_id)
                lexical.delete_document_chunks(
                    resolved_lexical_index_path,
                    parsed_document.document_id,
                )
                raise
            summary.indexed_files += 1
            summary.chunks_written += len(chunk_records)
            file_result["status"] = "indexed"
        except DocumentParsingError as exc:
            summary.failed_files += 1
            file_result["status"] = "failed"
            file_result["reason"] = exc.details.get("detail_reason") or exc.reason
            file_result["error"] = str(exc)
            if exc.details:
                file_result["metadata"] = exc.details
            if audit_only:
                logger.debug(
                    "Failed to parse document %s: reason=%s error=%s",
                    path,
                    exc.reason,
                    exc,
                )
            else:
                logger.exception("Failed to parse document %s", path)
        except Exception as exc:
            summary.failed_files += 1
            file_result["status"] = "failed"
            file_result["reason"] = "unexpected_error"
            file_result["error"] = str(exc)
            logger.exception("Unexpected indexing failure for %s", path)

    if not audit_only:
        assert vector_store is not None
        assert resolved_lexical_index_path is not None
        stale_document_ids = _delete_stale_document_chunks(
            vector_store, active_document_ids=active_document_ids
        )
        stale_lexical_document_ids = lexical.delete_stale_documents(
            resolved_lexical_index_path,
            active_document_ids=active_document_ids,
        )
        if stale_document_ids:
            logger.info(
                "Removed stale indexed documents: count=%s",
                len(stale_document_ids),
            )
        if stale_lexical_document_ids:
            logger.info(
                "Removed stale lexical documents: count=%s",
                len(stale_lexical_document_ids),
            )

    _finish_report(summary)
    if report_path is not None:
        write_indexing_report(summary, report_path)
    return summary
