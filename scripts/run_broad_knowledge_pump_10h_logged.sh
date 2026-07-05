#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/worldpgt/experiments/knowledge_pump_v1/logs"
LOG_FILE="$LOG_DIR/broad_pump_10h.log"
PID_FILE="$LOG_DIR/broad_pump_10h.pid"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"
exec >> "$LOG_FILE" 2>&1

echo "[10h-pump-launch] launcher_pid=$$ start $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
cd "$ROOT_DIR"
exec /usr/bin/caffeinate -dimsu "$ROOT_DIR/scripts/run_broad_knowledge_pump_10h.sh"
