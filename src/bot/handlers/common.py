import io
import logging

import pytesseract
from PIL import Image
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def read_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang="rus+eng")
        return text.strip()
    except Exception as e:
        logger.error("OCR error: %s", e)
        return ""


def read_PDF(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)
        return "\n".join(pages_text).strip()
    except Exception as e:
        logger.error("PDF read error: %s", e)
        return ""
