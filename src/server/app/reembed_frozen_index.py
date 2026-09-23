from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


class EmbeddingFunction(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class ReembedFrozenIndexResult:
    source_count: int
    target_count: int
    model: str
    normalized: bool
    source: Path
    target: Path
    collection_name: str
    manifest_path: Path
    lexical_source_path: Path
    lexical_target_path: Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _new_embedding_function(*, model: str, normalize_embeddings: bool) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model=model,
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
        query_encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def _get_or_fail_source_collection(source: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(source))
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:  # pragma: no cover - chromadb exception type varies by version
        raise RuntimeError(
            f"Source Chroma collection {collection_name!r} does not exist in {source}"
        ) from exc


def _prepare_target_collection(
    target: Path,
    collection_name: str,
    *,
    rebuild_target: bool,
    collection_metadata: dict[str, Any] | None = None,
):
    if rebuild_target and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target))
    collection = client.get_or_create_collection(name=collection_name, metadata=collection_metadata)
    count = int(collection.count())
    if count:
        raise RuntimeError(
            f"Target Chroma collection {collection_name!r} in {target} is not empty "
            f"({count} rows); choose a fresh target or pass --rebuild-target."
        )
    return collection


def _validate_batch_documents(
    raw_documents: list[Any] | None, *, count: int, offset: int
) -> list[str]:
    if raw_documents is None:
        raise RuntimeError(f"Source batch at offset {offset} did not include documents")
    if len(raw_documents) != count:
        raise RuntimeError(
            f"Source batch documents length mismatch at offset {offset}: "
            f"got {len(raw_documents)}, expected {count}"
        )
    return [str(item or "") for item in raw_documents]


def _validate_batch_metadatas(
    raw_metadatas: list[Any] | None,
    *,
    count: int,
    offset: int,
) -> list[dict[str, Any]]:
    if raw_metadatas is None:
        raise RuntimeError(f"Source batch at offset {offset} did not include metadatas")
    if len(raw_metadatas) != count:
        raise RuntimeError(
            f"Source batch metadatas length mismatch at offset {offset}: "
            f"got {len(raw_metadatas)}, expected {count}"
        )
    return [dict(item or {}) for item in raw_metadatas]


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _copy_lexical_index(*, source: Path, target: Path) -> tuple[Path, Path, int]:
    source_path = source / "lexical_index.sqlite3"
    if not source_path.is_file():
        raise RuntimeError(f"Source lexical index does not exist: {source_path}")
    target_path = target / "lexical_index.sqlite3"
    if target_path.exists():
        raise RuntimeError(f"Target lexical index already exists: {target_path}")
    shutil.copy2(source_path, target_path)
    return source_path, target_path, target_path.stat().st_size


def _write_manifest(
    *,
    manifest_path: Path,
    source: Path,
    target: Path,
    collection_name: str,
    model: str,
    normalized: bool,
    batch_size: int,
    source_count: int,
    target_count: int,
    lexical_source_path: Path,
    lexical_target_path: Path,
    lexical_size_bytes: int,
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "target": str(target),
        "collection_name": collection_name,
        "embedding_model": model,
        "normalize_embeddings": normalized,
        "batch_size": batch_size,
        "source_count": source_count,
        "target_count": target_count,
        "lexical_index_source": str(lexical_source_path),
        "lexical_index_target": str(lexical_target_path),
        "lexical_index_size_bytes": lexical_size_bytes,
        "lexical_index_copied": True,
        "ids_documents_metadatas_preserved": True,
        "source_files_rechunked": False,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reembed_frozen_index(
    *,
    source: Path,
    target: Path,
    model: str,
    collection_name: str,
    batch_size: int,
    normalize_embeddings: bool = True,
    rebuild_target: bool = False,
    expected_count: int | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> ReembedFrozenIndexResult:
    source = _resolve_path(source)
    target = _resolve_path(target)
    if source == target or _is_relative_to(target, source) or _is_relative_to(source, target):
        raise RuntimeError("Source and target Chroma directories must not overlap")

    source_collection = _get_or_fail_source_collection(source, collection_name)
    source_count = int(source_collection.count())
    if expected_count is not None and source_count != expected_count:
        raise RuntimeError(
            f"Source count mismatch: expected {expected_count}, got {source_count} in {source}"
        )
    if source_count == 0:
        raise RuntimeError(f"Source Chroma collection {collection_name!r} in {source} is empty")

    target_collection = _prepare_target_collection(
        target,
        collection_name,
        rebuild_target=rebuild_target,
        collection_metadata=getattr(source_collection, "metadata", None),
    )
    embedder = embedding_function or _new_embedding_function(
        model=model,
        normalize_embeddings=normalize_embeddings,
    )

    seen_ids: set[str] = set()
    for offset in range(0, source_count, batch_size):
        batch = source_collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = [str(item) for item in batch.get("ids", [])]
        if not ids:
            raise RuntimeError(f"Source returned an empty batch at offset {offset}")
        duplicate_ids = seen_ids.intersection(ids)
        if duplicate_ids:
            sample = sorted(duplicate_ids)[:5]
            raise RuntimeError(f"Duplicate source ids encountered: {sample}")
        seen_ids.update(ids)

        documents = _validate_batch_documents(batch.get("documents"), count=len(ids), offset=offset)
        metadatas = _validate_batch_metadatas(batch.get("metadatas"), count=len(ids), offset=offset)
        embeddings = embedder.embed_documents(documents)
        if len(embeddings) != len(ids):
            raise RuntimeError(
                f"Embedding count mismatch at offset {offset}: "
                f"got {len(embeddings)}, expected {len(ids)}"
            )

        target_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    if len(seen_ids) != source_count:
        raise RuntimeError(
            f"Copied id count mismatch: got {len(seen_ids)}, expected {source_count}"
        )

    target_count = int(target_collection.count())
    if target_count != source_count:
        raise RuntimeError(
            f"Target count mismatch: source has {source_count}, target has {target_count}"
        )

    lexical_source_path, lexical_target_path, lexical_size_bytes = _copy_lexical_index(
        source=source,
        target=target,
    )

    manifest_path = target / "reembed_manifest.json"
    _write_manifest(
        manifest_path=manifest_path,
        source=source,
        target=target,
        collection_name=collection_name,
        model=model,
        normalized=normalize_embeddings,
        batch_size=batch_size,
        source_count=source_count,
        target_count=target_count,
        lexical_source_path=lexical_source_path,
        lexical_target_path=lexical_target_path,
        lexical_size_bytes=lexical_size_bytes,
    )
    return ReembedFrozenIndexResult(
        source_count=source_count,
        target_count=target_count,
        model=model,
        normalized=normalize_embeddings,
        source=source,
        target=target,
        collection_name=collection_name,
        manifest_path=manifest_path,
        lexical_source_path=lexical_source_path,
        lexical_target_path=lexical_target_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-embed an existing frozen Chroma collection into a separate target index "
            "without re-reading or re-chunking source files."
        )
    )
    parser.add_argument(
        "--source", required=True, type=_resolve_path, help="Source Chroma directory"
    )
    parser.add_argument(
        "--target", required=True, type=_resolve_path, help="Target Chroma directory"
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Hugging Face embedding model used for the target index",
    )
    parser.add_argument(
        "--collection-name",
        default=config.CHROMA_COLLECTION_NAME,
        help="Chroma collection name to copy",
    )
    parser.add_argument("--batch-size", default=64, type=_positive_int, help="Embedding batch size")
    parser.add_argument(
        "--expected-count",
        type=_positive_int,
        default=None,
        help="Optional source row count assertion, for example 21127 for the frozen Stage 7 index",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help="Disable embedding normalization for the target index",
    )
    parser.add_argument(
        "--rebuild-target",
        action="store_true",
        help="Delete the target directory before writing the new Chroma index",
    )
    parser.set_defaults(normalize_embeddings=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = reembed_frozen_index(
        source=args.source,
        target=args.target,
        model=args.model,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        rebuild_target=args.rebuild_target,
        expected_count=args.expected_count,
    )
    print(
        json.dumps(
            {
                "source_count": result.source_count,
                "target_count": result.target_count,
                "model": result.model,
                "normalize_embeddings": result.normalized,
                "collection_name": result.collection_name,
                "source": str(result.source),
                "target": str(result.target),
                "manifest_path": str(result.manifest_path),
                "lexical_index_source": str(result.lexical_source_path),
                "lexical_index_target": str(result.lexical_target_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
