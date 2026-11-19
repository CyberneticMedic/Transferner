#!/usr/bin/env bash
# Run the rest of the data + BioCLIP pipeline once the given download parent PIDs exit.
# Usage: scripts/after_download.sh <pid> [<pid> ...]   (logs to $TRANSFERNER_DATA/logs/after_download.log)
set -euo pipefail
source "$HOME/ml-env.sh"
cd "$(dirname "$0")/.."
for pid in "$@"; do tail --pid="$pid" -f /dev/null; done
step() { echo "=== $(date '+%F %T') $*"; "$@"; }
step python -m transferner.data.download --merge-only
step python -m transferner.data.dedup
step python -m transferner.data.splits
step python -m transferner.train.embed_text
step python -m transferner.train.embed_images --force
step python -m transferner.train.linear_probe
step python -m transferner.train.calibrate
step python -m transferner.eval.evaluate
echo "=== $(date '+%F %T') done"
