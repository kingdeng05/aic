#!/usr/bin/env bash
# /tmp/run_iter.sh — one iteration of cheatcode data collection.
#
# Stdout (last line):
#   OUTCOME=<HOLD|TIMEOUT|PHANTOM_KEY|LAUNCH_FAIL> REPO=<repo_id>
# Exit code: 0=HOLD, 1=TIMEOUT, 2=PHANTOM_KEY, 3=LAUNCH_FAIL.
#
# Steps:
#   1. setsid distrobox-enter `aic_eval` and run launch_randomized_episode.sh
#   2. Wait up to 60s for the printed `[record]` block in stderr+stdout
#   3. Wait up to 120s for `Tared FT sensor at home pose` (homing complete)
#   4. Extract the pixi-record command via awk; rewrite if needed
#   5. Run pixi-record on host; tail rec.log for HOLD / arrow / Esc
#   6. On HOLD → 5s grace + pynput Right (save). On TIMEOUT → pynput Left
#      (discard). On phantom keypress → leave to record's own discard path.
#   7. If outcome != HOLD, rm -rf the dataset directory
#   8. Tear down sim (broad pkill via distrobox)

set -uo pipefail

# CLI arg: target slot (defaults to nic_card_mount_0).
# Usage: run_iter.sh [<slot>]
SLOT="${1:-nic_card_mount_0}"

WS=/home/sai/ws_aic_challenge
SIM_OUT=/tmp/sim.out
REC_LOG=/tmp/rec.log
TIMEOUT=60
GRACE=5

teardown() {
    distrobox enter aic_eval -- pkill -KILL -f \
        'ros2 launch|aic_gz_bringup|gz sim|rmw_zenohd|aic_adapter|component_container|robot_state_publisher|tf2_ros|topic_tools|rviz2' \
        >/dev/null 2>&1 || true
}

press_key() {
    pixi run python -c "from pynput.keyboard import Controller, Key; import time; c=Controller(); c.press(Key.${1}); time.sleep(0.05); c.release(Key.${1})" >/dev/null 2>&1 || true
}

: > "$SIM_OUT"
: > "$REC_LOG"

# 1. Launch sim
setsid distrobox enter aic_eval -- bash -lc \
    "cd $WS && ./src/aic/aic_bringup/scripts/launch_randomized_episode.sh ${SLOT}_present:=true --random-home-offset=0.06" \
    >"$SIM_OUT" 2>&1 &
SIM_PID=$!

# 2. Wait for [record] block (up to 60s)
for _ in $(seq 1 120); do
    if grep -q "dataset.reset_time_s=0" "$SIM_OUT" 2>/dev/null; then break; fi
    if ! kill -0 "$SIM_PID" 2>/dev/null; then break; fi
    sleep 0.5
done
if ! grep -q "dataset.reset_time_s=0" "$SIM_OUT" 2>/dev/null; then
    teardown
    echo "OUTCOME=LAUNCH_FAIL REPO=-"
    exit 3
fi

# 3. Wait for homing (up to 30s)
for _ in $(seq 1 60); do
    if grep -q "Tared FT sensor at home pose" "$SIM_OUT" 2>/dev/null; then break; fi
    if ! kill -0 "$SIM_PID" 2>/dev/null; then break; fi
    sleep 0.5
done

# 4. Extract pixi cmd
RECORD_CMD=$(awk '
    /pixi run aic-record/ {capture=1}
    capture {
        line=$0
        sub(/\\$/, "", line)
        sub(/^[[:space:]]+/, "", line)
        sub(/[[:space:]]+$/, "", line)
        if (length(line)) printf "%s ", line
    }
    /--dataset.reset_time_s=0/ {capture=0; exit}
' "$SIM_OUT")
REPO_ID=$(echo "$RECORD_CMD" | sed -nE 's/.*--dataset\.repo_id=([^ ]+).*/\1/p')

if [[ -z "$RECORD_CMD" || -z "$REPO_ID" ]]; then
    teardown
    echo "OUTCOME=LAUNCH_FAIL REPO=-"
    exit 3
fi

# 5. Run pixi-record
cd "$WS/src/aic"
bash -c "$RECORD_CMD" >"$REC_LOG" 2>&1 &
REC_PID=$!

# 6. Wait for HOLD or any keypress, with TIMEOUT
WAITED=0
HOLD=0
KEYPRESS=0
while kill -0 "$REC_PID" 2>/dev/null; do
    if grep -q "insertion success, entering HOLD" "$REC_LOG" 2>/dev/null; then
        HOLD=1; break
    fi
    if grep -qE "arrow key pressed|Escape key pressed" "$REC_LOG" 2>/dev/null; then
        KEYPRESS=1; break
    fi
    if [[ $WAITED -ge $TIMEOUT ]]; then break; fi
    WAITED=$((WAITED + 1))
    sleep 1
done

# 7. Send key based on outcome
if [[ $HOLD -eq 1 ]]; then
    sleep $GRACE
    press_key right
    OUTCOME=HOLD
elif [[ $KEYPRESS -eq 1 ]]; then
    OUTCOME=PHANTOM_KEY
else
    press_key left
    OUTCOME=TIMEOUT
fi

# 8. Wait briefly for record to exit (encoder finalize takes a few seconds)
for _ in $(seq 1 30); do
    kill -0 "$REC_PID" 2>/dev/null || break
    sleep 1
done
kill -KILL "$REC_PID" 2>/dev/null || true

# 9. Delete dataset on non-HOLD outcomes
if [[ "$OUTCOME" != HOLD ]]; then
    rm -rf "$HOME/.cache/huggingface/lerobot/$REPO_ID" 2>/dev/null || true
fi

# 10. Teardown sim
teardown

# 11. Report
echo "OUTCOME=$OUTCOME REPO=$REPO_ID"
case "$OUTCOME" in
    HOLD)        exit 0 ;;
    TIMEOUT)     exit 1 ;;
    PHANTOM_KEY) exit 2 ;;
    *)           exit 3 ;;
esac
