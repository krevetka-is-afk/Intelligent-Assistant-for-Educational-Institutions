FROM python:3.11-slim AS build
WORKDIR /build

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir --prefix=/install --requirement requirements.txt

COPY . .

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

ARG APP_UID=1000
ARG APP_GID=1000
RUN addgroup --gid ${APP_GID} appgroup && \
    adduser --disabled-password --gecos '' --uid ${APP_UID} --gid ${APP_GID} --home /app \
      --shell /usr/sbin/nologin appuser && \
    mkdir -p /app/data && chown -R appuser:appgroup /app/data /app

WORKDIR /app

COPY --from=build --chown=appuser:appgroup /install /usr/local

COPY --from=build --chown=appuser:appgroup /build .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys,urllib.request as u; u.urlopen('http://127.0.0.1:8000/health').read(); sys.exit(0)" || exit 1

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
