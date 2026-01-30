#!/bin/sh
set -e
# Ensure the vector DB directory exists and is writable by the app user.
# When Docker mounts a volume here, it is often root-owned; this fixes permissions.
VECTOR_DB_DIR="${VECTOR_DB_DIR:-/app/chrome_langchain_db}"
mkdir -p "$VECTOR_DB_DIR"
chown -R appuser:appgroup "$VECTOR_DB_DIR"
exec gosu appuser "$@"
