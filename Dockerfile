############################
# Build stage (shared)
############################
FROM python:3.12-slim AS build

ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
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

# COPY src/server /server
# COPY src/client /client
COPY ./src ./src
COPY ./data_and_documents ./data_and_documents
COPY ./app_runtime.py ./app_runtime.py

COPY VERSION ./

RUN --mount=type=cache,target=/root/.cache/uv \
    sed -Ei "s/^(version = \")0\.0\.0(\")$/\1$(cat VERSION)\2/" pyproject.toml && \
    uv sync --no-dev --no-editable --frozen


############################
# Runtime base stage
############################
FROM python:3.12-slim AS runtime-base

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH=/app/bin:$PATH \
    PYTHONUNBUFFERED=1

# hadolint ignore=DL3008
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

############################
# Runtime stage: server
############################
FROM runtime-base AS server

COPY --from=build --chown=appuser:appgroup /app /app
COPY --from=build --chown=appuser:appgroup /_project/src/server /server
COPY --from=build --chown=appuser:appgroup /_project/data_and_documents /data_and_documents
COPY --from=build --chown=appuser:appgroup /_project/app_runtime.py /server/app_runtime.py
RUN chmod -R a+rX /app /server

WORKDIR /server
# USER appuser # if uncomment cause to fail

EXPOSE 8000

CMD ["/app/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

############################
# Runtime stage: bot
############################
FROM runtime-base AS bot

COPY --from=build --chown=appuser:appgroup /app /app
COPY --from=build --chown=appuser:appgroup /_project/src/bot /bot
COPY --from=build --chown=appuser:appgroup /_project/app_runtime.py /bot/app_runtime.py
RUN chmod -R a+rX /app /bot

WORKDIR /bot

CMD ["/app/bin/python", "bot.py"]

############################
# Runtime stage: client
############################
FROM runtime-base AS client

ENV PATH=/app/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY --from=build --chown=appuser:appgroup /app /app
COPY --from=build --chown=appuser:appgroup /_project/src/client /client
COPY --from=build --chown=appuser:appgroup /_project/app_runtime.py /client/app_runtime.py
RUN chmod -R a+rX /app /client

WORKDIR /client
# USER appuser

EXPOSE 8501

CMD ["/app/bin/python", "-m", "streamlit", "run", "app/streamlit_app.py"]
