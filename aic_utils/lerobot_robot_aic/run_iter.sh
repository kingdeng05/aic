#!/usr/bin/env bash
# run_iter.sh — one iteration of cheatcode data collection.
#
# Stdout (last line):
#   OUTCOME=<HOLD|TIMEOUT|LAUNCH_FAIL> REPO=<repo_id>
# Exit code: 0=HOLD, 1=TIMEOUT, 3=LAUNCH_FAIL.
#
# Steps:
#   1. setsid distrobox-enter `aic_eval` and run launch_randomized_episode.sh
#      headless (gazebo_gui:=false launch_rviz:=false)
#   2. Wait up to 60s for the printed `[record]` block in stderr+stdout
#   3. Wait up to 30s for `Tared FT sensor at home pose` (homing complete)
#   4. Extract the pixi-record command via awk
#   5. Run pixi-record on host with `env -u DISPLAY` (kills pynput so real
#      user keystrokes can't reach lerobot's listener — see aic_record.py's
#      file-sentinel polling thread for the new save/discard control path)
#   6. On HOLD → 5s grace + write "save" to $SENTINEL.
#      On TIMEOUT  → write "discard" to $SENTINEL.
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

# Path to the file-based event sentinel that aic_record.py polls.
# Writing 'save' / 'discard' / 'stop' here triggers the matching event
# inside the record loop without using keyboard input. Overridable via
# AIC_RECORD_SENTINEL.
SENTINEL="${AIC_RECORD_SENTINEL:-/tmp/aic_record_event}"
export AIC_RECORD_SENTINEL="$SENTINEL"
rm -f "$SENTINEL" 2>/dev/null || true

send_event() {
    # write event command to sentinel; aic-record's poll thread picks it up
    echo "$1" > "$SENTINEL"
}

: > "$SIM_OUT"
: > "$REC_LOG"

# 1. Launch sim (headless: no gz gui, no rviz)
setsid distrobox enter aic_eval -- bash -lc \
    "cd $WS && ./src/aic/aic_bringup/scripts/launch_randomized_episode.sh ${SLOT}_present:=true --random-home-offset=0.06 gazebo_gui:=false launch_rviz:=false" \
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

# 5. Run pixi-record on the host. `env -u DISPLAY` makes pynput's keyboard
# listener unavailable (is_headless() in lerobot returns True), so real
# user keystrokes on the workstation can't reach aic-record. Control is
# instead via the file sentinel ($SENTINEL).
cd "$WS/src/aic"
env -u DISPLAY bash -c "$RECORD_CMD" >"$REC_LOG" 2>&1 &
REC_PID=$!

# 6. Wait for HOLD or TIMEOUT (no keypress path — pynput is disabled)
WAITED=0
HOLD=0
while kill -0 "$REC_PID" 2>/dev/null; do
    if grep -q "insertion success, entering HOLD" "$REC_LOG" 2>/dev/null; then
        HOLD=1; break
    fi
    if [[ $WAITED -ge $TIMEOUT ]]; then break; fi
    WAITED=$((WAITED + 1))
    sleep 1
done

# 7. Signal aic-record via file sentinel (no keyboard).
if [[ $HOLD -eq 1 ]]; then
    sleep $GRACE
    send_event save
    OUTCOME=HOLD
else
    send_event discard
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
    HOLD)    exit 0 ;;
    TIMEOUT) exit 1 ;;
    *)       exit 3 ;;
esac
