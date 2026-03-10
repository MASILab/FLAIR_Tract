#!/bin/bash
# submit.sh - Cluster submission wrapper for GNU Parallel + SSH

JOBSCRIPT="$1"
MACHINES=("swift.masi.vanderbilt.edu,goat.vuds.vanderbilt.edu,MASI-BULGAROV.vuds.vanderbilt.edu,masi-celgate2.vuds.vanderbilt.edu,masi-51.vuds.vanderbilt.edu")  # Edit with your hosts
LOGFILE="/tmp/snakemake_submit.log"

echo "[$(date)] Submitting $JOBSCRIPT" >> "$LOGFILE"

REMOTE_OUTPUT=$(parallel --sshlogin "$MACHINES" \
    --load 10% \
    --jobs 1 \
    --timeout 10 \
    --ssh "ssh -n" \
    "cd $PWD && setsid bash -c '
        nohup bash $JOBSCRIPT </dev/null >/tmp/job_\$\$_\$(date +%s).log 2>&1 &
        echo \$!
    '" \
    ::: "run" 2>&1) || {
    echo "[$(date)] Parallel submission failed: $REMOTE_OUTPUT" >> "$LOGFILE"
    exit 1
}

# Parallel output format is "host: PID"
# Validate we got a valid response
if [[ -z "$REMOTE_OUTPUT" ]]; then
    echo "[$(date)] No output from Parallel" >> "$LOGFILE"
    exit 1
fi

# Parse host and PID from Parallel output
TARGET_HOST=$(echo "$REMOTE_OUTPUT" | cut -d: -f1)
REMOTE_PID=$(echo "$REMOTE_OUTPUT" | cut -d: -f2 | xargs)

# Validate PID is numeric
if ! [[ "$REMOTE_PID" =~ ^[0-9]+$ ]]; then
    echo "[$(date)] Invalid PID received: $REMOTE_PID" >> "$LOGFILE"
    exit 1
fi

echo "[$(date)] Submitted to $TARGET_HOST (PID: $REMOTE_PID)" >> "$LOGFILE"
echo "${TARGET_HOST}:${REMOTE_PID}"
