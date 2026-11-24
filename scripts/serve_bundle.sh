#!/usr/bin/env bash
# Serve the app prototype's APK and model bundle on the home network (D-034): plain HTTP from this account,
# port 8765 by default. The app checks every bundle file against the bundle's SHA-256 manifest.
#   bash scripts/serve_bundle.sh          start (no-op if already running)
#   bash scripts/serve_bundle.sh stop     stop (exact PID from the pid file)
set -euo pipefail
: "${TRANSFERNER_DATA:?source ~/ml-env.sh first}"
PORT="${PORT:-8765}"
DIR="$TRANSFERNER_DATA/bundle"
PIDF="$TRANSFERNER_DATA/ops/serve_bundle.pid"
LOG="$TRANSFERNER_DATA/ops/logs/serve_bundle.log"
PY="$(dirname "$0")/../.venv/bin/python"
running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
if [[ "${1:-}" == "stop" ]]; then
    running && kill "$(cat "$PIDF")" && echo "stopped $(cat "$PIDF")"; rm -f "$PIDF"; exit 0
fi
if running; then echo "already serving (pid $(cat "$PIDF"))"; exit 0; fi
mkdir -p "$(dirname "$LOG")"
nohup "$PY" -m http.server "$PORT" --bind 0.0.0.0 --directory "$DIR" >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "serving $DIR on port $PORT (pid $(cat "$PIDF")), log $LOG"
