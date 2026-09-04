QUESTION_MAX_LENGTH = 500


class QuestionValidationError(ValueError):
    code: str
    public_message: str

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


def normalize_question(value: object) -> str:
    if not isinstance(value, str):
        raise QuestionValidationError(
            code="invalid_question",
            public_message="Question must be a non-empty string",
        )
    value = value.strip()
    if not value:
        raise QuestionValidationError(
            code="invalid_question",
            public_message="Question must be a non-empty string",
        )
    if len(value) > QUESTION_MAX_LENGTH:
        raise QuestionValidationError(
            code="question_too_long",
            public_message="Question must not exceed 500 characters",
        )
    return value
