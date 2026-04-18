FROM python:3.12-slim AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

ENV UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /_project

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

COPY ./src ./src
COPY ./app_runtime.py ./app_runtime.py

COPY VERSION ./

RUN --mount=type=cache,target=/root/.cache/uv \
    sed -Ei "s/^(version = \")0\.0\.0(\")$/\1$(cat VERSION)\2/" pyproject.toml && \
    uv sync --no-dev --no-editable --frozen

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall --python /app/bin/python torch torchvision torchaudio 2>/dev/null || true \
 && find /app/lib/python3.12/site-packages -maxdepth 1 \( -name 'nvidia*' -o -name 'triton*' \) -exec rm -rf {} + 2>/dev/null || true \
 && uv pip install --python /app/bin/python torch --index-url https://download.pytorch.org/whl/cpu
RUN rm -rf /app/lib/python3.12/site-packages/torch/include \
    /app/lib/python3.12/site-packages/torch/test \
    /app/lib/python3.12/site-packages/torch/share \
    2>/dev/null || true \
 && find /app/lib/python3.12/site-packages -depth -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true \
 && du -sh /app /app/lib/python3.12/site-packages 2>/dev/null || true


FROM python:3.12-slim AS runtime-base

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH=/app/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
 && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid ${APP_GID} appgroup && \
    adduser --disabled-password --gecos '' \
      --uid ${APP_UID} \
      --gid ${APP_GID} \
      --home /app appuser

FROM runtime-base AS server

COPY --from=build --chown=appuser:appgroup /app /app
COPY --from=build --chown=appuser:appgroup /_project/src/server /workspace/src/server
RUN mkdir -p /data_and_documents
COPY --from=build --chown=appuser:appgroup /_project/app_runtime.py /workspace/app_runtime.py
RUN chmod -R a+rX /workspace

WORKDIR /workspace

EXPOSE 8000

CMD ["/app/bin/python", "-m", "uvicorn", "src.server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime-base AS bot

COPY --from=build --chown=appuser:appgroup /app /app
COPY --from=build --chown=appuser:appgroup /_project/src/bot /workspace/src/bot
COPY --from=build --chown=appuser:appgroup /_project/app_runtime.py /workspace/app_runtime.py
RUN chmod -R a+rX /workspace

WORKDIR /workspace

CMD ["/app/bin/python", "-m", "src.bot.bot"]

FROM runtime-base AS client

ENV PATH=/app/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY --from=build --chown=appuser:appgroup /app /app
COPY --from=build --chown=appuser:appgroup /_project/src/client /workspace/src/client
COPY --from=build --chown=appuser:appgroup /_project/app_runtime.py /workspace/app_runtime.py
RUN chmod -R a+rX /workspace

WORKDIR /workspace

EXPOSE 8501

CMD ["/app/bin/python", "-m", "streamlit", "run", "src/client/app/streamlit_app.py"]
