#!/usr/bin/env bash
# Overnight loop: a GPU job runner plus bounded, unattended Claude Code rounds.
#
#   scripts/overnight.sh start    # detached (setsid): survives closing the terminal / this session
#   scripts/overnight.sh status   # runner, queue, rounds
#   scripts/overnight.sh stop     # touch STOP: finish the current job and round, then exit
#
# Runner: pops `python -m transferner...` jobs from $TRANSFERNER_DATA/ops/queue.txt (scripts/q.sh)
#   and runs each from the latest *committed* code of $BRANCH in a separate worktree, so edits in
#   the main checkout never reach a running job. Results tables still land in the main checkout.
# Rounds: `claude -p ops/prompt.md --permission-mode auto` (same safety classifier as an
#   interactive auto-mode session). A round starts when the GPU queue drains (at most every
#   MIN_GAP s) or every ROUND_EVERY s while it is busy, so CPU work overlaps GPU work.
#   At most MAX_ROUNDS rounds; each is killed after ROUND_TIMEOUT s; nothing starts after UNTIL.
# Knobs (env): MAX_ROUNDS=8 ROUND_EVERY=5400 MIN_GAP=900 ROUND_TIMEOUT=5400 UNTIL=10:00
set -uo pipefail
source "$HOME/ml-env.sh"
REPO="$HOME/Transferner"
OPS="$TRANSFERNER_DATA/ops"
WT="$OPS/runner-worktree"
BRANCH=setup/data-pipeline
PY="$REPO/.venv/bin/python"
CLAUDE="$HOME/.local/bin/claude"
MAX_ROUNDS=${MAX_ROUNDS:-8}
ROUND_EVERY=${ROUND_EVERY:-5400}
MIN_GAP=${MIN_GAP:-900}
ROUND_TIMEOUT=${ROUND_TIMEOUT:-5400}
UNTIL=${UNTIL:-10:00}
mkdir -p "$OPS/logs"

log() { echo "$(date '+%F %T') $*" >> "$OPS/overnight.log"; }
queue_empty() { ! grep -q -v -E '^\s*(#|$)' "$OPS/queue.txt" 2>/dev/null; }
runner_idle() { [[ "$(cut -d' ' -f1 "$OPS/runner.state" 2>/dev/null)" == idle ]]; }

runner() {
    local n=0 cmd sha lf rc
    [[ -d "$WT" ]] || git -C "$REPO" worktree add --detach "$WT" "$BRANCH" >> "$OPS/overnight.log" 2>&1
    while [[ ! -f "$OPS/STOP" ]] && (( $(date +%s) < END )); do
        if ! cmd=$("$REPO/scripts/q.sh" pop 2>/dev/null); then
            echo "idle $(date '+%T')" > "$OPS/runner.state"; sleep 20; continue
        fi
        n=$((n + 1))
        git -C "$WT" checkout -q --detach -f "$BRANCH"
        sha=$(git -C "$WT" rev-parse --short HEAD)
        lf="$OPS/logs/job-$(date +%m%d-%H%M%S)-$n.log"
        echo "busy $(date '+%T') $sha $cmd" > "$OPS/runner.state"
        log "job start [$sha] $cmd -> $lf"
        read -ra a <<< "$cmd"
        (cd "$WT" && PYTHONPATH="$WT/src" TRANSFERNER_RESULTS="$REPO/results" "$PY" "${a[@]:1}") > "$lf" 2>&1
        rc=$?
        printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$rc" "$sha" "$cmd" "$lf" >> "$OPS/done.tsv"
        log "job end rc=$rc $cmd"
    done
    echo "stopped $(date '+%T')" > "$OPS/runner.state"
}

round() {
    local i=$1 lf prompt
    lf="$OPS/logs/round-$i-$(date +%m%d-%H%M).log"
    prompt=$(sed -e "s/{ROUND}/$i/g; s/{MAX_ROUNDS}/$MAX_ROUNDS/g; s/{UNTIL}/$(date -d "@$END" '+%F %H:%M')/g" \
        "$REPO/ops/prompt.md")
    log "round $i start -> $lf"
    (cd "$REPO" && timeout "$ROUND_TIMEOUT" "$CLAUDE" -p "$prompt" --permission-mode auto --output-format text < /dev/null) \
        > "$lf" 2>&1
    log "round $i end rc=$?"
}

run() {
    END=$(date -d "today $UNTIL" +%s)
    (( END <= $(date +%s) )) && END=$(date -d "tomorrow $UNTIL" +%s)
    echo $$ > "$OPS/overnight.pid"
    rm -f "$OPS/STOP"
    log "loop start: until $(date -d "@$END" '+%F %H:%M'), max $MAX_ROUNDS rounds"
    runner &
    local rp=$! rounds=0 last=0 now
    sleep 30
    while [[ ! -f "$OPS/STOP" ]] && (( rounds < MAX_ROUNDS )) && (( $(date +%s) < END )); do
        now=$(date +%s)
        if { queue_empty && runner_idle && (( now - last >= MIN_GAP )); } || (( now - last >= ROUND_EVERY )); then
            rounds=$((rounds + 1))
            round "$rounds"
            last=$(date +%s)
            if grep -q '^STATUS: DONE' "$REPO/ops/OVERNIGHT.md" && queue_empty; then
                log "agent reported STATUS: DONE"; break
            fi
        fi
        sleep 60
    done
    while [[ ! -f "$OPS/STOP" ]] && (( $(date +%s) < END )) && ! { queue_empty && runner_idle; }; do
        sleep 60                                       # rounds exhausted: let queued jobs finish
    done
    touch "$OPS/STOP"
    wait "$rp"
    rm -f "$OPS/STOP" "$OPS/overnight.pid"
    log "loop end after $rounds rounds"
}

case "${1:-status}" in
    start)
        if [[ -f "$OPS/overnight.pid" ]] && kill -0 "$(cat "$OPS/overnight.pid")" 2>/dev/null; then
            echo "already running (pid $(cat "$OPS/overnight.pid"))"; exit 1
        fi
        setsid nohup "$0" run > "$OPS/overnight.out" 2>&1 < /dev/null &
        sleep 2; echo "started (pid $(cat "$OPS/overnight.pid" 2>/dev/null)); log: $OPS/overnight.log" ;;
    run) run ;;
    stop) touch "$OPS/STOP"; echo "STOP requested: current job and round finish first" ;;
    status)
        [[ -f "$OPS/overnight.pid" ]] && echo "loop pid $(cat "$OPS/overnight.pid")" || echo "loop not running"
        "$REPO/scripts/q.sh" status
        echo "--- log"; tail -n 12 "$OPS/overnight.log" 2>/dev/null ;;
    *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
