from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    try:
        clear_vector_cache()
        summary = index_directory(
            args.input_dir.resolve(),
            args.persist_dir.resolve(),
            rebuild=args.rebuild,
        )
    except Exception as exc:
        logging.getLogger("server.indexing").exception("Fatal indexing error: %s", exc)
        return 1

    logging.getLogger("server.indexing").info(
        (
            "Indexing finished: files_seen=%s indexed_files=%s "
            "skipped_files=%s failed_files=%s chunks_written=%s"
        ),
        summary.files_seen,
        summary.indexed_files,
        summary.skipped_files,
        summary.failed_files,
        summary.chunks_written,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
