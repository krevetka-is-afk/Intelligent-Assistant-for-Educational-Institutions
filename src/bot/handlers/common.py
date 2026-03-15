from __future__ import annotations

import io
import re
from typing import Any, BinaryIO

import pytesseract
from PIL import Image, ImageOps
from PyPDF2 import PdfReader

DEFAULT_OCR_LANG = "rus+eng"
API_QUESTION_MAX_LENGTH = 500
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024


class MediaProcessingError(RuntimeError):
    """Raised when image or PDF text extraction fails."""


class PDFTooLargeError(MediaProcessingError):
    """Raised when an uploaded PDF exceeds the allowed size."""


class EmptyExtractedTextError(MediaProcessingError):
    """Raised when extracted text is empty after normalization."""


def normalize_extracted_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", " ").replace("\x0c", "\n")
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in normalized.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(stripped_line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def prepare_text_for_api(text: str, *, max_length: int = API_QUESTION_MAX_LENGTH) -> str:
    normalized = normalize_extracted_text(text)
    if not normalized:
        raise EmptyExtractedTextError("Extracted text is empty.")

    if len(normalized) <= max_length:
        return normalized

    suffix = "..."
    if max_length <= len(suffix):
        return normalized[:max_length]

    truncated = normalized[: max_length - len(suffix)].rstrip()
    split_at = max(truncated.rfind(" "), truncated.rfind("\n"))
    if split_at >= int((max_length - len(suffix)) * 0.7):
        truncated = truncated[:split_at].rstrip()
    return f"{truncated}{suffix}"


def validate_pdf_size(file_size: int, *, max_size: int = MAX_PDF_SIZE_BYTES) -> None:
    if file_size > max_size:
        raise PDFTooLargeError(
            f"PDF file is too large: {file_size} bytes, maximum allowed is {max_size} bytes."
        )


def _to_pil_image(image_np: Any) -> Image.Image:
    if isinstance(image_np, Image.Image):
        image = image_np
    else:
        try:
            image = Image.fromarray(image_np)
        except Exception as exc:  # pragma: no cover - Pillow raises several types here.
            raise MediaProcessingError("Could not convert image payload for OCR.") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "P"}:
        image = image.convert("RGB")
    return image


def read_image(image_np: Any) -> str:
    """
    Extract text from an image payload using Tesseract OCR.

    The function accepts a numpy array-like payload or an already created PIL image.
    """

    image = _to_pil_image(image_np)
    try:
        text = pytesseract.image_to_string(image, lang=DEFAULT_OCR_LANG)
    except (pytesseract.TesseractError, pytesseract.TesseractNotFoundError) as exc:
        raise MediaProcessingError("Tesseract OCR failed while parsing the image.") from exc

    return normalize_extracted_text(text)


def _to_binary_stream(pdf_data: bytes | bytearray | BinaryIO) -> BinaryIO:
    if isinstance(pdf_data, (bytes, bytearray)):
        return io.BytesIO(pdf_data)

    if hasattr(pdf_data, "seek"):
        pdf_data.seek(0)
    return pdf_data


def read_PDF(pdf_data: bytes | bytearray | BinaryIO) -> str:
    """
    Extract text from a PDF payload.

    The function accepts bytes, bytearray, BytesIO, or another binary file-like object.
    """

    try:
        pdf_reader = PdfReader(_to_binary_stream(pdf_data))
        pages = [page.extract_text() or "" for page in pdf_reader.pages]
    except Exception as exc:
        raise MediaProcessingError("PDF text extraction failed.") from exc

    return normalize_extracted_text("\n\n".join(filter(None, pages)))
