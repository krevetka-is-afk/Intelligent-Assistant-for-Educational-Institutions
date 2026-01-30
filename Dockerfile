############################
# Build stage
############################
FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential=12.9 \
        gcc=4:12.2.0-3 && \
    rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

# uv config
ENV UV_PYTHON=python3.11 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /_project

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

COPY src/ src/
COPY VERSION ./

RUN --mount=type=cache,target=/root/.cache/uv \
    sed -Ei "s/^(version = \")0\.0\.0(\")$/\1$(cat VERSION)\2/" pyproject.toml && \
    uv sync --no-dev --no-editable --frozen


############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH=/app/bin:$PATH \
    PYTHONOPTIMIZE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5=15.15-0+deb12u1 \
        gosu=1.14-1 && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --gid ${APP_GID} appgroup && \
    adduser --disabled-password --gecos '' \
      --uid ${APP_UID} \
      --gid ${APP_GID} \
      --home /app \
      appuser

COPY --from=build --chown=appuser:appgroup /app /app

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health').read()"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
