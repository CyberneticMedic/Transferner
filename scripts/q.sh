#!/usr/bin/env bash
# GPU job queue for the overnight loop (scripts/overnight.sh). One job per line in
# $TRANSFERNER_DATA/ops/queue.txt; every change goes through flock so the runner and agent
# rounds never race. Jobs must be `python -m transferner.<module> [plain args]` - no shell.
#
#   scripts/q.sh add "python -m transferner.eval.efficiency --name fp16"   # append
#   scripts/q.sh add --front "python -m ..."                               # run next
#   scripts/q.sh list | status | pop | remove <line-number>
set -euo pipefail
source "$HOME/ml-env.sh"
OPS="$TRANSFERNER_DATA/ops"
mkdir -p "$OPS/logs"
Q="$OPS/queue.txt"; touch "$Q" "$OPS/done.tsv"
exec 9>"$OPS/queue.lock"

valid() {
    local re='^python -m transferner\.[a-z0-9_.]+( .*)?$'
    if [[ ! "$1" =~ $re ]] || printf '%s' "$1" | grep -q "[;|&\`\$<>()\\\\'\"]"; then
        echo "rejected (must be 'python -m transferner.<module> [plain args]', no shell syntax): $1" >&2
        return 1
    fi
}

case "${1:-status}" in
    add)
        shift; front=0
        [[ "${1:-}" == "--front" ]] && { front=1; shift; }
        cmd="$*"; valid "$cmd"
        flock 9
        if (( front )); then { echo "$cmd"; cat "$Q"; } > "$Q.tmp" && mv "$Q.tmp" "$Q"; else echo "$cmd" >> "$Q"; fi
        echo "queued: $cmd" ;;
    pop)
        flock 9
        line=$(grep -v -m1 -E '^\s*(#|$)' "$Q" || true)
        [[ -z "$line" ]] && exit 1
        awk -v l="$line" 'done || $0 != l {print; next} {done=1}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
        echo "$line" ;;
    remove)
        flock 9; sed -i "${2}d" "$Q"; cat -n "$Q" ;;
    list)
        cat -n "$Q" ;;
    status)
        echo "runner: $(cat "$OPS/runner.state" 2>/dev/null || echo 'not running')"
        echo "queued: $(grep -cv -E '^\s*(#|$)' "$Q" || true) job(s)"; cat -n "$Q" | head -20
        echo "recent:"; tail -n 8 "$OPS/done.tsv" | cut -f1-4
        [[ -f "$OPS/STOP" ]] && echo "STOP file present: loop will stop" || true ;;
    *) echo "usage: q.sh add [--front] CMD | pop | list | status | remove N" >&2; exit 2 ;;
esac
