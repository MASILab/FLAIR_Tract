#!/bin/bash
set -euo pipefail

EXTERNAL_JOBID="$1"

HOST="${EXTERNAL_JOBID%%:*}"
PID="${EXTERNAL_JOBID##*:}"

# Validate PID is numeric
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    exit 0
fi

# Kill process group to catch child processes
ssh "$HOST" "kill -TERM -$PID 2>/dev/null || kill -9 $PID 2>/dev/null || true"
