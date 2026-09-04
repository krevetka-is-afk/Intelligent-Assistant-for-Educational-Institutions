import pytest

from src.server.app.question_validation import (
    QuestionValidationError,
    normalize_question,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("hello", "hello", id="ordinary"),
        pytest.param(" hello\n", "hello", id="trim-whitespace"),
        pytest.param("a" * 500, "a" * 500, id="at-limit"),
    ],
)
def test_normalize_question(value, expected):
    assert normalize_question(value) == expected


error_type_msg = [
    ["invalid_question", "Question must be a non-empty string"],
    ["question_too_long", "Question must not exceed 500 characters"],
]
error_msg = []


@pytest.mark.parametrize(
    ("value", "expected_code", "expected_message"),
    [
        pytest.param(None, error_type_msg[0][0], error_type_msg[0][1], id="missing-question"),
        pytest.param("\n\t", error_type_msg[0][0], error_type_msg[0][1], id="blank-question"),
        pytest.param(123, error_type_msg[0][0], error_type_msg[0][1], id="wrong-type"),
        pytest.param("a" * 501, error_type_msg[1][0], error_type_msg[1][1], id="over-limit"),
        pytest.param(
            " " + "a" * 501 + " ",
            error_type_msg[1][0],
            error_type_msg[1][1],
            id="trimmed-over-limit",
        ),
    ],
)
def test_question_validation_error(value, expected_code, expected_message):
    with pytest.raises(QuestionValidationError) as exc_info:
        normalize_question(value)

    assert exc_info.value.code == expected_code
    assert str(exc_info.value) == expected_message  # exc_info.value.public_message
