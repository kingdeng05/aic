#!/usr/bin/env bash
# run_iter_sc.sh — one iteration of cheatcode data collection for SC-port
# (or any task-board slot), driven from the host with the aic_eval docker
# container.
#
# Differences from run_iter.sh (Sai's version, distrobox + ~/ws_aic_challenge):
#   * Uses `docker exec aic_eval` instead of `distrobox enter aic_eval`.
#   * Workspace is ${AIC_WS:-$HOME/ws_aic} on host and inside the container
#     (bind-mounted under the same path).
#   * Default slot is sc_port_0, and launch_randomized_episode.sh now picks
#     plug/port/cable_type from the target.
#   * Headless: sim runs with gazebo_gui:=false launch_rviz:=false.
#   * No pynput key injection (this host has no DISPLAY). Instead we cap the
#     record loop with --dataset.episode_time_s=${RECORD_TIME_S} so it
#     terminates naturally, then grep the log to decide keep vs delete.
#
# Stdout (last line):
#   OUTCOME=<HOLD|TIMEOUT|LAUNCH_FAIL> REPO=<repo_id>
# Exit code: 0=HOLD, 1=TIMEOUT, 3=LAUNCH_FAIL.
#
# Usage:
#   run_iter_sc.sh [<slot>] [--time-s=<sec>] [--log-dir=<dir>]
#   Examples:
#     run_iter_sc.sh                  # sc_port_0, 30s record window
#     run_iter_sc.sh sc_port_1
#     run_iter_sc.sh nic_card_mount_0 --time-s=20

set -uo pipefail

SLOT="sc_port_0"
RECORD_TIME_S=30
LOG_DIR="/tmp"
HOME_OFFSET="0.06"
KEEP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time-s=*)        RECORD_TIME_S="${1#*=}"; shift ;;
        --log-dir=*)       LOG_DIR="${1#*=}";       shift ;;
        --home-offset=*)   HOME_OFFSET="${1#*=}";   shift ;;
        --keep)            KEEP=1;                  shift ;;
        --*)               echo "unknown flag: $1" >&2; exit 2 ;;
        *)                 SLOT="$1";               shift ;;
    esac
done
mkdir -p "$LOG_DIR"

CONTAINER="${AIC_EVAL_CONTAINER:-aic_eval}"
WS="${AIC_WS:-$HOME/ws_aic}"
SIM_OUT="${LOG_DIR}/sim_${SLOT}_$$.out"
REC_LOG="${LOG_DIR}/rec_${SLOT}_$$.log"
# Hard wall clock cap so a stuck record never blocks the batch.
WALL_CAP_S=$(( RECORD_TIME_S * 3 + 60 ))

teardown() {
    docker exec -u fuheng "$CONTAINER" pkill -KILL -f \
        'ros2 launch|aic_gz_bringup|gz sim|rmw_zenohd|aic_adapter|component_container|robot_state_publisher|tf2_ros|topic_tools|rviz2|home_robot' \
        >/dev/null 2>&1 || true
}

: > "$SIM_OUT"
: > "$REC_LOG"

# 1. Launch sim inside aic_eval (headless).
docker exec -u fuheng "$CONTAINER" bash -lc \
    "cd $WS && ./src/aic/aic_bringup/scripts/launch_randomized_episode.sh ${SLOT}_present:=true --headless --random-home-offset=${HOME_OFFSET}" \
    >"$SIM_OUT" 2>&1 &
SIM_PID=$!

# 2. Wait for [record] block (the printed pixi-record command).
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

# 3. Wait for homing complete (FT tare), up to 60s.
for _ in $(seq 1 120); do
    if grep -q "Tared FT sensor at home pose" "$SIM_OUT" 2>/dev/null; then break; fi
    if ! kill -0 "$SIM_PID" 2>/dev/null; then break; fi
    sleep 0.5
done

# 4. Extract the printed pixi-record command and overwrite episode_time_s.
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

# Replace the default 600s episode cap with the caller-specified record window.
RECORD_CMD=$(echo "$RECORD_CMD" | sed -E "s/--dataset\.episode_time_s=[0-9]+/--dataset.episode_time_s=${RECORD_TIME_S}/")

# 5. Run pixi-record on the host. Strip DISPLAY/WAYLAND so lerobot's pynput
# keyboard handler can't attach to the host X server. Without this, stray
# keystrokes the user types in any window can trigger "Left arrow key pressed
# - rerecord the last episode" and abort the batch episode mid-run.
cd "$WS/src/aic"
env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY bash -c "$RECORD_CMD" >"$REC_LOG" 2>&1 &
REC_PID=$!

# 6. Wait for record to finish naturally; enforce a hard wall-time cap.
WAITED=0
while kill -0 "$REC_PID" 2>/dev/null; do
    if [[ $WAITED -ge $WALL_CAP_S ]]; then
        kill -KILL "$REC_PID" 2>/dev/null || true
        break
    fi
    WAITED=$((WAITED + 1))
    sleep 1
done

# 7. Classify outcome. First gate: the cheatcode must have actually printed
#    "CheatCodeTeleop: insertion success" — without this, the plug never
#    reached the success threshold and the dataset is a failed demo even if
#    the action plateaued (the trajectory parks once max_lift_retries is
#    exhausted, fooling a plateau-only quality check). Second gate: enough
#    frames, plateau at end, and all 3 camera videos present.
DATASET_DIR="$HOME/.cache/huggingface/lerobot/$REPO_ID"
QUALITY=""
if [[ ! -d "$DATASET_DIR/data" ]]; then
    OUTCOME=LAUNCH_FAIL
elif ! grep -q "CheatCodeTeleop: insertion success" "$REC_LOG" 2>/dev/null; then
    QUALITY="NO_INSERT_SUCCESS"
    OUTCOME=BAD
else
    QUALITY=$(REPO_ID="$REPO_ID" pixi run --manifest-path "$WS/src/aic/pixi.toml" python <<'PYEOF'
import glob, json, os, sys
import pandas as pd
import numpy as np
root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{os.environ['REPO_ID']}")
data_glob = sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))
if not data_glob:
    print("MISSING_DATA"); sys.exit()
df = pd.concat([pd.read_parquet(p) for p in data_glob])
n = len(df)
if n < 600:
    print(f"SHORT n={n}"); sys.exit()
a = np.stack(df["action"].values)
# Trailing-stationary frames (motion < 0.5mm/step on commanded position).
da = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1)
tail = 0
for i in range(len(da) - 1, -1, -1):
    if da[i] < 5e-4:
        tail += 1
    else:
        break
# Require trajectory to plateau (cheatcode reached terminal commanded pose),
# else descend probably did not finish — treat as low quality.
if tail < 30:  # at least 1 s of plateau
    print(f"NO_PLATEAU n={n} tail={tail}"); sys.exit()
videos_ok = all(
    any(glob.glob(f"{root}/videos/observation.images.{cam}/**/*.mp4", recursive=True))
    for cam in ("left_camera", "center_camera", "right_camera")
)
if not videos_ok:
    print(f"MISSING_VIDEOS n={n}"); sys.exit()
print(f"OK n={n} tail={tail}")
PYEOF
)
    case "$QUALITY" in
        OK*)  OUTCOME=OK ;;
        *)    OUTCOME=BAD ;;
    esac
fi

# 8. Discard dataset on non-OK outcomes (unless --keep).
if [[ "$OUTCOME" != OK && $KEEP -eq 0 ]]; then
    rm -rf "$DATASET_DIR" 2>/dev/null || true
fi

# 9. Teardown sim.
teardown

# 10. Report.
echo "OUTCOME=$OUTCOME REPO=$REPO_ID QUALITY=\"$QUALITY\""
case "$OUTCOME" in
    OK)          exit 0 ;;
    BAD)         exit 1 ;;
    LAUNCH_FAIL) exit 3 ;;
    *)           exit 3 ;;
esac
