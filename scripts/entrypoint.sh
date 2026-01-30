#!/bin/sh
set -e
# With read_only + volume, chown on the mount is often blocked (seccomp/mount).
# Run the app as root so it can write to VECTOR_DB_DIR (/data) without chown.
exec "$@"
