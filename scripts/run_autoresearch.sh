#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-15}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-$ROOT/state/autoresearch}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_PATH:-$LOG_DIR/autoresearch-${TIMESTAMP}.log}"
LATEST_LINK="$LOG_DIR/latest.log"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_PATH")" "$LATEST_LINK"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] autoresearch bootstrap"
echo "root=$ROOT"
echo "interval_seconds=$INTERVAL_SECONDS"
echo "restart_delay_seconds=$RESTART_DELAY_SECONDS"
echo "log_path=$LOG_PATH"

source /home/staff/jiayining/miniconda3/etc/profile.d/conda.sh
conda activate kaggle-agent

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "$ROOT"

python -m kaggle_agent.cli doctor
python -m kaggle_agent.cli status || true

cycle=0
while true; do
    cycle=$((cycle + 1))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervisor_cycle=$cycle starting watch loop"
    if python -m kaggle_agent.cli watch --interval-seconds "$INTERVAL_SECONDS" --iterations 0; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watch loop exited cleanly"
    else
        rc=$?
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watch loop crashed rc=$rc"
        python -m kaggle_agent.cli status || true
        sleep "$RESTART_DELAY_SECONDS"
    fi
done
