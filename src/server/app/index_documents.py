from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app_runtime import log_extra, setup_logging

from . import config
from .document_ingestion import index_directory
from .vector import clear_vector_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index educational documents into Chroma.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.DOCUMENTS_DIR,
        help="Directory with source documents.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=config.VECTOR_DB_DIR,
        help="Directory where Chroma persists the collection.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the collection before indexing.",
    )
    parser.add_argument(
        "--lexical-index-path",
        type=Path,
        default=None,
        help="SQLite FTS5 index path. Defaults to <persist-dir>/lexical_index.sqlite3.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Parse documents and build a JSON-friendly report without touching Chroma or FTS.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write structured per-file ingestion results to this JSON path.",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable bounded OCR for PDF pages with no extractable text.",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=["fixed", "structure_v1"],
        default=config.CHUNK_STRATEGY,
        help="Chunking strategy to use while indexing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override RAG_CHUNK_SIZE for this indexing run.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override RAG_CHUNK_OVERLAP for this indexing run.",
    )
    return parser


def main() -> int:
    setup_logging("server")
    parser = build_parser()
    args = parser.parse_args()

    try:
        clear_vector_cache()
        summary = index_directory(
            args.input_dir.resolve(),
            args.persist_dir.resolve(),
            rebuild=args.rebuild,
            lexical_index_path=(
                args.lexical_index_path.resolve() if args.lexical_index_path is not None else None
            ),
            audit_only=args.audit_only,
            report_path=args.report_json.resolve() if args.report_json is not None else None,
            enable_ocr=args.enable_ocr if args.enable_ocr else None,
            chunk_strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
    except Exception as exc:
        logging.getLogger("server.indexing").exception(
            "Fatal indexing error: %s",
            exc,
            extra=log_extra(stage="indexing", error_type=type(exc).__name__),
        )
        return 1

    logging.getLogger("server.indexing").info(
        (
            "Indexing finished: files_seen=%s indexed_files=%s "
            "skipped_files=%s failed_files=%s chunks_written=%s "
            "counts_by_reason=%s no_extractable_text_rate=%s"
        ),
        summary.files_seen,
        summary.indexed_files,
        summary.skipped_files,
        summary.failed_files,
        summary.chunks_written,
        summary.counts_by_reason or {},
        summary.no_extractable_text_rate,
        extra=log_extra(stage="indexing"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
