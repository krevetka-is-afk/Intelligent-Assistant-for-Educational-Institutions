FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/bot/ .

CMD ["python", "bot.py"]
