#!/usr/bin/env sh
# Собирает .env.deploy из переменных CI/CD GitLab (Settings → CI/CD → Variables).
# Обязательные: BOT_TOKEN, API_KEY, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB,
#               WEB_BOOTSTRAP_ADMIN_TOKEN
# Остальные — см. значения по умолчанию ниже.

set -eu

OUT="${CI_PROJECT_DIR:-.}/.env.deploy"
: > "$OUT"

append() {
  printf '%s=%s\n' "$1" "$2" >> "$OUT"
}

for v in BOT_TOKEN API_KEY POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB WEB_BOOTSTRAP_ADMIN_TOKEN; do
  eval "val=\${$v-}"
  if [ -z "$val" ]; then
    echo "render-env.sh: ERROR — задайте CI/CD variable: $v" >&2
    exit 1
  fi
done

if [ -n "${DATABASE_URL:-}" ]; then
  FINAL_DATABASE_URL="$DATABASE_URL"
else
  FINAL_DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}"
fi

append APP_ENV "${APP_ENV:-production}"
append LOG_LEVEL "${LOG_LEVEL:-INFO}"
append API_KEY "$API_KEY"
append SHOW_SOURCES "${SHOW_SOURCES:-1}"
append WEB_BOOTSTRAP_ADMIN_TOKEN "$WEB_BOOTSTRAP_ADMIN_TOKEN"
append PREPARE_RAG_ON_STARTUP "${PREPARE_RAG_ON_STARTUP:-1}"
append AUTO_INDEX_ON_STARTUP "${AUTO_INDEX_ON_STARTUP:-1}"

append OLLAMA_HOST "${OLLAMA_HOST:-http://host.docker.internal:11434}"
append LLM_MODEL "${LLM_MODEL:-mistral:7b}"
append HF_EMBEDDING_MODEL "${HF_EMBEDDING_MODEL:-cointegrated/rubert-tiny2}"
append CHROMA_COLLECTION_NAME "${CHROMA_COLLECTION_NAME:-edu_documents}"
append VECTOR_DB_DIR "${VECTOR_DB_DIR:-/data}"
append DOCUMENTS_DIR "${DOCUMENTS_DIR:-/data_and_documents}"
append DOCUMENTS_HOST_PATH "${DOCUMENTS_HOST_PATH:-./data_and_documents}"
append RAG_TOP_K "${RAG_TOP_K:-4}"
append RAG_TOTAL_TIMEOUT_SECONDS "${RAG_TOTAL_TIMEOUT_SECONDS:-420}"
append LLM_TIMEOUT_SECONDS "${LLM_TIMEOUT_SECONDS:-360}"
append BOT_API_TIMEOUT_SECONDS "${BOT_API_TIMEOUT_SECONDS:-480}"

SERVER_PUBLISH_PORT="${SERVER_PUBLISH_PORT:-8000}"
append SERVER_PUBLISH_PORT "$SERVER_PUBLISH_PORT"
append API_BASE_URL "${API_BASE_URL:-http://localhost:${SERVER_PUBLISH_PORT}}"
append BOT_TOKEN "$BOT_TOKEN"

append POSTGRES_DB "$POSTGRES_DB"
append POSTGRES_USER "$POSTGRES_USER"
append POSTGRES_PASSWORD "$POSTGRES_PASSWORD"
append DATABASE_URL "$FINAL_DATABASE_URL"
append WEB_AUTH_DATABASE_URL "${WEB_AUTH_DATABASE_URL:-sqlite+aiosqlite:////data/web_auth.db}"

echo "render-env.sh: записан $OUT"
