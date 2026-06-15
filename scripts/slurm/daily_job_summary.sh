#!/bin/bash
# Daily SLURM Job Summary Script
# Runs via cron at 9am CET, summarizes overnight jobs, pushes to GitHub
#
# Cron entry (add via `crontab -e`):
#   0 9 * * 1-5 /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/scripts/slurm/daily_job_summary.sh
#
# This script:
#   1. Queries sacct for jobs completed in the last 24 hours
#   2. Parses log files for metrics (loss, RMSE, etc.)
#   3. Writes a JSON summary to job_reports/
#   4. Commits and pushes to GitHub (triggers remote routine)

set -e

PROJECT_DIR="/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA"
REPORTS_DIR="${PROJECT_DIR}/job_reports"
LOGS_DIR="${PROJECT_DIR}/logs"
SUMMARY_FILE="${REPORTS_DIR}/daily_summary.json"

mkdir -p "$REPORTS_DIR"

# Get timestamp for report
REPORT_DATE=$(date +%Y-%m-%d)
REPORT_TIME=$(date +%H:%M:%S)

# Query jobs from last 24 hours
echo "Querying SLURM jobs from last 24 hours..."
YESTERDAY=$(date -d "yesterday" +%Y-%m-%dT00:00:00)

# Get job info: JobID, JobName, State, ExitCode, Start, End, Elapsed
JOBS_RAW=$(sacct --starttime="$YESTERDAY" \
    --format=JobID,JobName%50,State,ExitCode,Start,End,Elapsed,NodeList%20 \
    --noheader --parsable2 \
    --user="$USER" \
    | grep -v "\.batch" | grep -v "\.extern" || true)

# Initialize JSON
echo "{" > "$SUMMARY_FILE"
echo "  \"report_date\": \"$REPORT_DATE\"," >> "$SUMMARY_FILE"
echo "  \"report_time\": \"$REPORT_TIME\"," >> "$SUMMARY_FILE"
echo "  \"jobs\": [" >> "$SUMMARY_FILE"

FIRST_JOB=true
COMPLETED_COUNT=0
FAILED_COUNT=0
RUNNING_COUNT=0

while IFS='|' read -r jobid jobname state exitcode start end elapsed nodelist; do
    [ -z "$jobid" ] && continue

    # Skip if not a main job (avoid sub-jobs)
    [[ "$jobid" == *"."* ]] && continue

    # Add comma separator
    if [ "$FIRST_JOB" = true ]; then
        FIRST_JOB=false
    else
        echo "," >> "$SUMMARY_FILE"
    fi

    # Determine status
    STATUS="unknown"
    case "$state" in
        COMPLETED) STATUS="completed"; ((COMPLETED_COUNT++)) ;;
        FAILED|NODE_FAIL|TIMEOUT|CANCELLED*) STATUS="failed"; ((FAILED_COUNT++)) ;;
        RUNNING|PENDING) STATUS="running"; ((RUNNING_COUNT++)) ;;
    esac

    # Try to find log file and extract metrics
    METRICS="{}"
    ERROR_MSG=""

    # Look for log file matching job name or ID
    LOG_FILE=$(find "$LOGS_DIR" -name "*${jobid}*" -o -name "*${jobname}*" 2>/dev/null | head -1)

    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        # Extract final metrics from log
        FINAL_LOSS=$(grep -oP "loss[=:]\s*\K[0-9.e+-]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        FINAL_RMSE=$(grep -oP "rmse[=:]\s*\K[0-9.e+-]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        FINAL_R2=$(grep -oP "r2[=:]\s*\K[0-9.e+-]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        EPOCH=$(grep -oP "epoch[=:]\s*\K[0-9]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")

        # Build metrics JSON
        METRICS="{"
        [ -n "$FINAL_LOSS" ] && METRICS="${METRICS}\"loss\": $FINAL_LOSS,"
        [ -n "$FINAL_RMSE" ] && METRICS="${METRICS}\"rmse\": $FINAL_RMSE,"
        [ -n "$FINAL_R2" ] && METRICS="${METRICS}\"r2\": $FINAL_R2,"
        [ -n "$EPOCH" ] && METRICS="${METRICS}\"epoch\": $EPOCH,"
        METRICS="${METRICS%,}}"  # Remove trailing comma
        [ "$METRICS" = "{}" ] || METRICS="$METRICS"

        # If failed, try to get error message
        if [ "$STATUS" = "failed" ]; then
            ERROR_MSG=$(tail -50 "$LOG_FILE" 2>/dev/null | grep -iE "error|exception|traceback|failed" | tail -3 | tr '\n' ' ' | cut -c1-200 || echo "")
        fi
    fi

    # Write job entry
    cat >> "$SUMMARY_FILE" << JOBENTRY
    {
      "job_id": "$jobid",
      "job_name": "$jobname",
      "status": "$STATUS",
      "state": "$state",
      "exit_code": "$exitcode",
      "start": "$start",
      "end": "$end",
      "elapsed": "$elapsed",
      "metrics": $METRICS,
      "error": "$(echo "$ERROR_MSG" | sed 's/"/\\"/g')"
    }
JOBENTRY

done <<< "$JOBS_RAW"

# Close JSON
cat >> "$SUMMARY_FILE" << EOF

  ],
  "summary": {
    "total": $((COMPLETED_COUNT + FAILED_COUNT + RUNNING_COUNT)),
    "completed": $COMPLETED_COUNT,
    "failed": $FAILED_COUNT,
    "running": $RUNNING_COUNT
  }
}
EOF

echo "Summary written to $SUMMARY_FILE"

# Commit and push to GitHub
cd "$PROJECT_DIR"
git add "$SUMMARY_FILE"
git commit -m "Daily job summary: $REPORT_DATE" --allow-empty || true
git push origin HEAD || echo "Warning: Could not push to remote"

echo "Done! Summary pushed to GitHub."
