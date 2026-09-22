import json

from src.server.app.rag_evaluation import (
    RAGEvaluationCase,
    capture_rag_evaluation_case,
    load_evaluation_cases,
    write_capture_report,
)


def test_rag_evaluation_case_keeps_legacy_constructor_and_fixture_compatibility(tmp_path):
    case = RAGEvaluationCase(
        id="legacy-case",
        question="Какой тестовый вопрос?",
        conversation_history=[],
        expected_documents=["doc"],
        forbidden_clusters=["other"],
        minimum_answer_points=["ключевой пункт"],
        allow_no_calendar_dates_statement=False,
    )

    assert case.reference_answer is None
    assert case.reference_source_urls == []
    assert case.reference_checked_at is None
    assert case.reference_note is None
    assert case.source_pdf_page is None

    fixture_path = tmp_path / "cases.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "legacy-fixture-case",
                        "question": "Какой тестовый вопрос из фикстуры?",
                        "conversation_history": [],
                        "expected_documents": ["doc"],
                        "forbidden_clusters": ["other"],
                        "minimum_answer_points": ["ключевой пункт"],
                        "allow_no_calendar_dates_statement": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded_case = load_evaluation_cases(fixture_path)[0]

    assert loaded_case.reference_answer is None
    assert loaded_case.reference_source_urls == []
    assert loaded_case.source_pdf_page is None


def test_capture_report_includes_reference_metadata_and_manual_answer_points(tmp_path):
    case = RAGEvaluationCase(
        id="reference-case",
        question="Какие документы нужны для тестирования?",
        conversation_history=[],
        expected_documents=["pmi"],
        forbidden_clusters=["unrelated"],
        minimum_answer_points=["называет входные данные", "называет ожидаемый результат"],
        allow_no_calendar_dates_statement=False,
        reference_answer="Эталонный ответ для ручного сравнения.",
        reference_source_urls=["https://example.test/reference"],
        reference_checked_at="2026-09-22",
        reference_note="Ответ проверен вручную.",
        source_pdf_page=11,
    )

    capture = capture_rag_evaluation_case(case, search_fn=lambda query, *, k: [])
    report_path = tmp_path / "report.json"

    write_capture_report(report_path, [capture])

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    captured_case = payload["captures"][0]

    assert payload["schema_version"] == 3
    assert capture.minimum_answer_points == [
        "называет входные данные",
        "называет ожидаемый результат",
    ]
    assert captured_case["minimum_answer_points"] == capture.minimum_answer_points
    assert captured_case["reference_answer"] == "Эталонный ответ для ручного сравнения."
    assert captured_case["reference_source_urls"] == ["https://example.test/reference"]
    assert captured_case["reference_checked_at"] == "2026-09-22"
    assert captured_case["reference_note"] == "Ответ проверен вручную."
    assert captured_case["source_pdf_page"] == 11
