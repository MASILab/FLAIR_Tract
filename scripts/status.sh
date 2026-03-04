#!/bin/bash
set -euo pipefail

EXTERNAL_JOBID="$1"

HOST="${EXTERNAL_JOBID%%:*}"
PID="${EXTERNAL_JOBID##*:}"

# Validate PID is numeric
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "failed"
    exit 0
fi

# Check if process exists
if ssh "$HOST" "kill -0 $PID 2>/dev/null" 2>/dev/null; then
    echo "running"
else
    # Process completed - check for error logs
    if ssh "$HOST" "grep -q -E 'Error|Failed|Traceback|exit code' /tmp/job_${PID}_*.log 2>/dev/null"; then
        echo "failed"
    else
        echo "success"
    fi
fi
