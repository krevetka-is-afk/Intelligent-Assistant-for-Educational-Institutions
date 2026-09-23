from pathlib import Path
from urllib.parse import urlparse

from src.server.app.rag_evaluation import load_evaluation_cases

CASES_PATH = Path(__file__).parent / "fixtures" / "rag_eval" / "cases.sokolova.v1.json"
PDF_QUESTIONS = [
    "Кто ректор?",
    "Что такое академический отпуск и почему его можно взять?",
    "Какие документы нужно нести в военкомат для отсрочки?",
    "Как пересдать экзамен, пропущенный по уважительной причине?",
    "Какие уважительные причины бывают?",
    "Что такое пересдача при неудовлетворительной оценке за экзамен и как она происходит?",
    "Что делать если не сдал экзамен?",
    "Что такое летняя практика?",
    "Что такое электронный студенческий билет?",
    "Где электронный студенческий билет доступен?",
    "Что такое КУД?",
    "Где найти расписание?",
    "Что такое независимый экзамен и какие они бывают?",
    "Какие документы нужны первокурснику для поступления?",
]


def test_sokolova_cases_preserve_all_pdf_questions_and_reference_provenance():
    cases = load_evaluation_cases(CASES_PATH)

    assert [case.question for case in cases] == PDF_QUESTIONS
    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        assert case.source_pdf_page == 12
        assert case.reference_answer
        assert case.reference_checked_at == "2026-09-22"
        assert case.minimum_answer_points
        assert case.reference_source_urls
        assert all(
            urlparse(url).hostname in {"www.hse.ru", "ba.hse.ru"}
            for url in case.reference_source_urls
        )


def test_sokolova_cases_keep_known_corpus_gaps_explicit():
    cases = {case.id: case for case in load_evaluation_cases(CASES_PATH)}

    assert cases["sokolova-01-rector"].expected_documents == []
    assert cases["sokolova-14-admission-documents"].expected_documents == []
    card_access_case = cases["sokolova-10-electronic-student-card-access"]
    assert card_access_case.reference_answer is not None
    assert card_access_case.reference_note is not None
    assert "MAX" in card_access_case.reference_answer
    assert "локальном DOCX" in card_access_case.reference_note
