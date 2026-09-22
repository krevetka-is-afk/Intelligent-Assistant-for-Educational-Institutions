from __future__ import annotations

from pathlib import Path

from src.server.app import index_documents
from src.server.app.document_ingestion import IndexingSummary


def test_index_documents_cli_passes_audit_only_to_index_directory(monkeypatch, tmp_path):
    input_dir = tmp_path / "docs"
    persist_dir = tmp_path / "db"
    lexical_index_path = tmp_path / "lexical.sqlite3"
    report_path = tmp_path / "indexing-report.json"
    input_dir.mkdir()
    calls: list[dict[str, object]] = []

    def fake_index_directory(
        input_dir_arg: Path,
        persist_dir_arg: Path,
        *,
        rebuild: bool,
        lexical_index_path: Path | None,
        audit_only: bool,
        report_path: Path | None,
        enable_ocr: bool | None,
    ) -> IndexingSummary:
        calls.append(
            {
                "input_dir": input_dir_arg,
                "persist_dir": persist_dir_arg,
                "rebuild": rebuild,
                "lexical_index_path": lexical_index_path,
                "audit_only": audit_only,
                "report_path": report_path,
                "enable_ocr": enable_ocr,
            }
        )
        return IndexingSummary(files_seen=1, indexed_files=1, chunks_written=1)

    monkeypatch.setattr(index_documents, "index_directory", fake_index_directory)
    monkeypatch.setattr(index_documents, "clear_vector_cache", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "index_documents",
            "--input-dir",
            str(input_dir),
            "--persist-dir",
            str(persist_dir),
            "--lexical-index-path",
            str(lexical_index_path),
            "--audit-only",
            "--report-json",
            str(report_path),
            "--enable-ocr",
        ],
    )

    exit_code = index_documents.main()

    assert exit_code == 0
    assert calls == [
        {
            "input_dir": input_dir.resolve(),
            "persist_dir": persist_dir.resolve(),
            "rebuild": False,
            "lexical_index_path": lexical_index_path.resolve(),
            "audit_only": True,
            "report_path": report_path.resolve(),
            "enable_ocr": True,
        }
    ]
