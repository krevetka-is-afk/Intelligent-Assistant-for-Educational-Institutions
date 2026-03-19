import io
import os

import pytest
from PIL import Image

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from src.bot.handlers import common


def test_read_image_uses_russian_and_english_tesseract(monkeypatch):
    captured = {}

    def fake_image_to_string(image, *, lang):
        captured["image"] = image
        captured["lang"] = lang
        return "  Расписание \n\n exam   "

    monkeypatch.setattr(common.pytesseract, "image_to_string", fake_image_to_string)

    image = Image.new("RGB", (8, 8), color="white")
    text = common.read_image(image)

    assert captured["lang"] == "rus+eng"
    assert captured["image"].size == (8, 8)
    assert text == "Расписание\n\nexam"


def test_read_pdf_concatenates_extracted_pages(monkeypatch):
    class _Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class _Reader:
        def __init__(self, file_obj):
            assert isinstance(file_obj, io.BytesIO)
            self.pages = [_Page("Стр. 1"), _Page(None), _Page("Page 2")]

    monkeypatch.setattr(common, "PdfReader", _Reader)

    text = common.read_PDF(b"%PDF-test")

    assert text == "Стр. 1\n\nPage 2"


def test_validate_pdf_size_rejects_files_larger_than_20_mb():
    with pytest.raises(common.PDFTooLargeError):
        common.validate_pdf_size(common.MAX_PDF_SIZE_BYTES + 1)


def test_prepare_text_for_api_normalizes_and_truncates():
    text = "line 1   \n\n\nline 2 and a very long ending"

    prepared = common.prepare_text_for_api(text, max_length=20)

    assert prepared == "line 1\n\nline 2..."


def test_prepare_text_for_api_rejects_empty_text():
    with pytest.raises(common.EmptyExtractedTextError):
        common.prepare_text_for_api(" \n\t ")
