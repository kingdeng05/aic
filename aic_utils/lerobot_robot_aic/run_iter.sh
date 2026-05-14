#!/usr/bin/env bash
# run_iter.sh — one iteration of cheatcode data collection.
#
# Multi-slot scene: every launch spawns all 5 NIC card mounts and both SC
# ports. The orchestrator picks a *random* (mount, port) target per iter
# from the requested connector family and conditions the policy via the
# task-id one-hot appended to obs.state by the controller.
#
# Usage:
#   run_iter.sh --connector=nic    # random NIC target (1 of 10)
#   run_iter.sh --connector=sc     # random SC  target (1 of 2)
#   run_iter.sh nic | sc           # legacy positional shortcut
#
# Stdout (last line):
#   OUTCOME=<HOLD|TIMEOUT|LAUNCH_FAIL> CONNECTOR=<nic|sc>
#       TARGET=<mount>/<port> REPO=<repo_id>
# Exit code: 0=HOLD, 1=TIMEOUT, 3=LAUNCH_FAIL.
#
# Steps:
#   1. Pick a random target from the connector's table.
#   2. setsid distrobox-enter `aic_eval` and run launch_randomized_episode.sh
#      headless (gazebo_gui:=false launch_rviz:=false) with --all-slots-present
#      --randomize-task-board, the picked --target and matching --plug-name /
#      --cable-type.
#   3. Wait up to 60s for the printed `[record]` block.
#   4. Wait up to 30s for `Tared FT sensor at home pose`.
#   5. Extract the pixi-record command (includes --robot.task_target_* so the
#      controller emits the matching one-hot).
#   6. Run pixi-record on host with `env -u DISPLAY` (pynput off; control via
#      file sentinel — see aic_record.py).
#   7. On HOLD → 5s grace + write "save" to $SENTINEL.
#      On TIMEOUT  → write "discard" to $SENTINEL.
#   8. If outcome != HOLD, rm -rf the dataset directory.
#   9. Teardown sim (broad pkill via distrobox).

set -uo pipefail

# --- Argument parsing ----------------------------------------------------
CONNECTOR=""
for arg in "$@"; do
    case "$arg" in
        --connector=*)  CONNECTOR="${arg#*=}" ;;
        nic|sc)         CONNECTOR="$arg" ;;
        # Legacy compat: accept "<slot>" but only use it to infer connector.
        nic_card_mount_*) CONNECTOR="nic" ;;
        sc_port_*)        CONNECTOR="sc"  ;;
        *) echo "run_iter.sh: unknown arg: $arg" >&2; exit 2 ;;
    esac
done
CONNECTOR="${CONNECTOR:-nic}"  # default to NIC

# --- Random target pick from the connector's table -----------------------
PICK=$(CONNECTOR="$CONNECTOR" python3 - <<'PYEOF'
import os, random
nic_targets = [
    ("nic_card_mount_0", "sfp_port_0"),
    ("nic_card_mount_0", "sfp_port_1"),
    ("nic_card_mount_1", "sfp_port_0"),
    ("nic_card_mount_1", "sfp_port_1"),
    ("nic_card_mount_2", "sfp_port_0"),
    ("nic_card_mount_2", "sfp_port_1"),
    ("nic_card_mount_3", "sfp_port_0"),
    ("nic_card_mount_3", "sfp_port_1"),
    ("nic_card_mount_4", "sfp_port_0"),
    ("nic_card_mount_4", "sfp_port_1"),
]
sc_targets = [
    # port_name is "sc_port" (cheatcode auto-appends "_link" to build
    # the TF frame "task_board/sc_port_X/sc_port_link"). Must match the
    # TASK_ID_TABLE keys in aic_robot_aic_controller.py.
    ("sc_port_0", "sc_port"),
    ("sc_port_1", "sc_port"),
]
connector = os.environ["CONNECTOR"]
if connector == "nic":
    pool = nic_targets
    cable = "sfp_sc_cable"
    plug  = "sfp_tip"
elif connector == "sc":
    pool = sc_targets
    cable = "sfp_sc_cable_reversed"
    plug  = "sc_tip"
else:
    raise SystemExit(f"unknown connector: {connector}")
mount, port = random.choice(pool)
print(f"{mount} {port} {cable} {plug}")
PYEOF
)
read -r TARGET_MOUNT TARGET_PORT CABLE_TYPE PLUG_NAME <<< "$PICK"
TARGET="${TARGET_MOUNT}/${TARGET_PORT}"

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

# 1. Launch sim. Multi-slot scene with all NIC mounts + SC ports present.
# Env-var knobs:
#   AIC_RANDOMIZE_TASK_BOARD=0  → skip task-board pose randomization (default 1)
#   AIC_GZ_GUI=1                → show gz GUI + rviz window (default 0 = headless)
TASK_BOARD_FLAG=""
if [[ "${AIC_RANDOMIZE_TASK_BOARD:-1}" != "0" ]]; then
    TASK_BOARD_FLAG="--randomize-task-board"
fi
if [[ "${AIC_GZ_GUI:-0}" == "1" ]]; then
    GUI_FLAGS="gazebo_gui:=true launch_rviz:=true"
else
    GUI_FLAGS="gazebo_gui:=false launch_rviz:=false"
fi
echo "[iter] connector=$CONNECTOR target=$TARGET cable=$CABLE_TYPE plug=$PLUG_NAME task_board_rand=${AIC_RANDOMIZE_TASK_BOARD:-1} gz_gui=${AIC_GZ_GUI:-0}" >&2
setsid distrobox enter aic_eval -- bash -lc \
    "cd $WS && ./src/aic/aic_bringup/scripts/launch_randomized_episode.sh \
        --all-slots-present $TASK_BOARD_FLAG \
        --target=$TARGET --plug-name=$PLUG_NAME --cable-type=$CABLE_TYPE \
        --random-home-offset=0.06 \
        $GUI_FLAGS" \
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
# AIC_KEEP_DISCARDED=1 → save the partial episode on TIMEOUT for inspection
# (smoke tests). Production: discard.
if [[ $HOLD -eq 1 ]]; then
    sleep $GRACE
    send_event save
    OUTCOME=HOLD
elif [[ "${AIC_KEEP_DISCARDED:-0}" == "1" ]]; then
    send_event save  # save partial frames for schema inspection
    OUTCOME=TIMEOUT
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

# 9. Delete dataset on non-HOLD outcomes (skip if AIC_KEEP_DISCARDED=1 — used
# for smoke tests / schema verification where we want the partial parquet).
if [[ "$OUTCOME" != HOLD && "${AIC_KEEP_DISCARDED:-0}" != "1" ]]; then
    rm -rf "$HOME/.cache/huggingface/lerobot/$REPO_ID" 2>/dev/null || true
fi

# 10. Teardown sim
teardown

# 11. Report
echo "OUTCOME=$OUTCOME CONNECTOR=$CONNECTOR TARGET=$TARGET REPO=$REPO_ID"
case "$OUTCOME" in
    HOLD)    exit 0 ;;
    TIMEOUT) exit 1 ;;
    *)       exit 3 ;;
esac
