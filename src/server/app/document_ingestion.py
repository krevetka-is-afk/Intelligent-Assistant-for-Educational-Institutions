from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

from . import config
from .lexical import delete_document_chunks as delete_lexical_document_chunks
from .lexical import (
    initialize_lexical_index,
    upsert_document_chunks,
)
from .vector import get_embedding_function

logger = logging.getLogger("server.indexing")

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".html", ".htm", ".txt", ".docx"})
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
    sections: list[TextSection]


@dataclass(slots=True)
class ChunkRecord:
    id: str
    page_content: str
    metadata: dict[str, Any]

    def to_document(self) -> Document:
        return Document(id=self.id, page_content=self.page_content, metadata=self.metadata)


@dataclass(slots=True)
class IndexingSummary:
    files_seen: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    chunks_written: int = 0


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


def _build_sections(texts: list[tuple[str, int | None]]) -> list[TextSection]:
    sections: list[TextSection] = []
    offset = 0
    for text, page in texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        sections.append(TextSection(text=normalized, page=page, base_offset=offset))
        offset += len(normalized) + 2
    return sections


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_pdf(path: Path) -> tuple[str, list[TextSection]]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentParsingError(f"Could not read PDF {path}") from exc

    page_texts = [(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
    sections = _build_sections(page_texts)
    if not sections:
        raise DocumentParsingError(f"PDF {path} did not contain extractable text")

    title = normalize_text(path.stem)
    return title or path.stem, sections


def _load_html(path: Path) -> tuple[str, list[TextSection]]:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(_read_text_file(path))
        parser.close()
    except Exception as exc:
        raise DocumentParsingError(f"Could not parse HTML {path}") from exc

    if not parser.text:
        raise DocumentParsingError(f"HTML {path} did not contain extractable text")

    title = parser.title or normalize_text(path.stem) or path.stem
    return title, _build_sections([(parser.text, None)])


def _load_txt(path: Path) -> tuple[str, list[TextSection]]:
    text = normalize_text(_read_text_file(path))
    if not text:
        raise DocumentParsingError(f"TXT {path} is empty")
    return normalize_text(path.stem) or path.stem, _build_sections([(text, None)])


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


def _extract_docx_text(archive: zipfile.ZipFile) -> str:
    try:
        raw_document = archive.read("word/document.xml")
    except KeyError as exc:
        raise DocumentParsingError("DOCX is missing word/document.xml") from exc

    try:
        root = ET.fromstring(raw_document)
    except ET.ParseError as exc:
        raise DocumentParsingError("DOCX XML is malformed") from exc

    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", WORD_NS):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", WORD_NS)]
        joined = normalize_text("".join(texts))
        if joined:
            paragraphs.append(joined)

    return "\n\n".join(paragraphs)


def _load_docx(path: Path) -> tuple[str, list[TextSection]]:
    try:
        with zipfile.ZipFile(path) as archive:
            title = _extract_docx_title(archive)
            text = _extract_docx_text(archive)
    except zipfile.BadZipFile as exc:
        raise DocumentParsingError(f"DOCX {path} is not a valid archive") from exc

    normalized_text = normalize_text(text)
    if not normalized_text:
        raise DocumentParsingError(f"DOCX {path} is empty")

    resolved_title = title or normalize_text(path.stem) or path.stem
    return resolved_title, _build_sections([(normalized_text, None)])


def load_document(path: Path, *, root_dir: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(f"Unsupported file type: {path}")

    loaders = {
        ".pdf": _load_pdf,
        ".html": _load_html,
        ".htm": _load_html,
        ".txt": _load_txt,
        ".docx": _load_docx,
    }

    title, sections = loaders[suffix](path)
    relative_path = path.resolve().relative_to(root_dir.resolve())
    return ParsedDocument(
        document_id=build_document_id(relative_path),
        source=relative_path.as_posix(),
        title=title,
        mime_type=MIME_TYPES[suffix],
        sections=sections,
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
                "char_start": section.base_offset + start,
                "char_end": section.base_offset + end,
                "indexed_at": timestamp,
            }
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

    for metadata in existing.get("metadatas", []):
        if not isinstance(metadata, dict):
            continue
        document_id = metadata.get("document_id")
        if isinstance(document_id, str) and document_id:
            document_ids.add(document_id)

    return document_ids


def _delete_stale_document_chunks(
    vector_store: Chroma, *, active_document_ids: set[str], lexical_enabled: bool
) -> list[str]:
    stale_document_ids = sorted(_get_indexed_document_ids(vector_store) - active_document_ids)
    for document_id in stale_document_ids:
        _delete_document_chunks(vector_store, document_id)
        if lexical_enabled:
            try:
                delete_lexical_document_chunks(document_id)
            except Exception:
                logger.exception(
                    "Failed to delete stale lexical chunks for document %s",
                    document_id,
                )
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
) -> IndexingSummary:
    config.validate_chunk_settings()
    collection = collection_name or config.CHROMA_COLLECTION_NAME
    resolved_chunk_size = chunk_size or config.CHUNK_SIZE
    resolved_overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    persist_directory.mkdir(parents=True, exist_ok=True)
    vector_store = create_vector_store(persist_directory, collection, rebuild=rebuild)
    lexical_enabled = True
    try:
        initialize_lexical_index(rebuild=rebuild)
    except Exception:
        lexical_enabled = False
        logger.exception("Failed to initialize lexical index; continuing with dense indexing only")
    summary = IndexingSummary()
    indexed_at = datetime.now(UTC).isoformat()
    indexed_paths = _iter_supported_files(input_dir)
    root_dir = input_dir.resolve()
    active_document_ids = {
        build_document_id(path.resolve().relative_to(root_dir)) for path in indexed_paths
    }

    for path in indexed_paths:
        summary.files_seen += 1
        try:
            parsed_document = load_document(path, root_dir=input_dir)
            chunk_records = build_chunk_records(
                parsed_document,
                chunk_size=resolved_chunk_size,
                overlap=resolved_overlap,
                indexed_at=indexed_at,
            )
            if not chunk_records:
                summary.skipped_files += 1
                logger.warning("Skipping %s because it produced no chunks", path)
                continue

            _delete_document_chunks(vector_store, parsed_document.document_id)
            vector_store.add_documents(
                documents=[record.to_document() for record in chunk_records],
                ids=[record.id for record in chunk_records],
            )
            if lexical_enabled:
                try:
                    upsert_document_chunks(parsed_document.document_id, chunk_records)
                except Exception:
                    logger.exception(
                        "Failed to update lexical index for %s",
                        path,
                    )
            summary.indexed_files += 1
            summary.chunks_written += len(chunk_records)
        except DocumentParsingError:
            summary.failed_files += 1
            logger.exception("Failed to parse document %s", path)
        except Exception:
            summary.failed_files += 1
            logger.exception("Unexpected indexing failure for %s", path)

    stale_document_ids = _delete_stale_document_chunks(
        vector_store,
        active_document_ids=active_document_ids,
        lexical_enabled=lexical_enabled,
    )
    if stale_document_ids:
        logger.info(
            "Removed stale indexed documents: count=%s",
            len(stale_document_ids),
        )

    return summary
