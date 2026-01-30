############################
# Build stage (shared)
############################
FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

ENV UV_PYTHON=python3.11 \
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
COPY src ./src

COPY VERSION ./

RUN --mount=type=cache,target=/root/.cache/uv \
    sed -Ei "s/^(version = \")0\.0\.0(\")$/\1$(cat VERSION)\2/" pyproject.toml && \
    uv sync --no-dev --no-editable --frozen


############################
# Runtime stage: server
############################
FROM python:3.11-slim AS server

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH=/app/bin:$PATH \
    PYTHONUNBUFFERED=1

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
 && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid ${APP_GID} appgroup && \
    adduser --disabled-password --gecos '' \
      --uid ${APP_UID} \
      --gid ${APP_GID} \
      --home /app appuser

COPY --from=build --chown=appuser:appgroup /app /app
RUN chmod -R a+rX /app

WORKDIR /app
# USER appuser # if uncomment cause to fail

EXPOSE 8000

CMD ["/app/bin/python", "-m", "uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

############################
# Runtime stage: client
############################
FROM python:3.11-slim AS client

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH=/app/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN addgroup --gid ${APP_GID} appgroup && \
    adduser --disabled-password --gecos '' \
      --uid ${APP_UID} \
      --gid ${APP_GID} \
      --home /app appuser

COPY --from=build --chown=appuser:appgroup /app /app
RUN chmod -R a+rX /app

# WORKDIR /app
# USER appuser

EXPOSE 8501

CMD ["/app/bin/python", "-m", "streamlit", "run", "client/app/streamlit_app.py"]
