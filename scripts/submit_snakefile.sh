#!/bin/bash
# submit.sh - Cluster submission wrapper for GNU Parallel + SSH

JOBSCRIPT="$1"
MACHINES=("beatles.vuds.vanderbilt.edu" "swift.masi.vanderbilt.edu" "goat.vuds.vanderbilt.edu" "masi-bulgarov.vuds.vanderbilt.edu" "masi-celgate2.vuds.vanderbilt.edu" "masi-lostgirl.vuds.vanderbilt.edu")  # Edit with your hosts
LOGFILE="/tmp/snakemake_submit.log"

# Select host
# Deterministic selection using md5 hash
# 16# converts hex string to decimal in Bash arithmetic
HASH=$(echo "$JOBSCRIPT" | md5sum | cut -c1-8)
HOST_IDX=$((16#$HASH % ${#MACHINES[@]}))
TARGET_HOST="${MACHINES[$HOST_IDX]}"

echo "[$(date)] Submitting $JOBSCRIPT to $TARGET_HOST (hash: $HASH, idx: $HOST_IDX)" >> "$LOGFILE"

# CRITICAL: Redirect all FDs and use setsid to fully detach
# SSH will return immediately after launching the background job
REMOTE_PID=$(ssh -n "$TARGET_HOST" "
    cd $PWD &&
    setsid bash -c '
        source /absolute/path/to/.venv/bin/activate
        nohup bash $JOBSCRIPT </dev/null >/tmp/job_\$\$_\$(date +%s).log 2>&1 &
        echo \$!
    '
" 2>&1) || {
    echo "[$(date)] SSH failed for $TARGET_HOST" >> "$LOGFILE"
    exit 1
}

# Validate PID is numeric
if ! [[ "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
    echo "[$(date)] Invalid PID received: $REMOTE_PID" >> "$LOGFILE"
    exit 1
fi

echo "[$(date)] Remote PID: $REMOTE_PID" >> "$LOGFILE"
echo "${TARGET_HOST}:${REMOTE_PID}"
