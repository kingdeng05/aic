#!/usr/bin/env bash
#
# launch_randomized_episode.sh
#
# Wrapper around `ros2 launch aic_bringup aic_gz_bringup.launch.py` that:
#   1. Auto-samples translation/yaw within docs ranges for any task-board slot
#      passed with `<slot>_present:=true` (unless --no-randomize is set or the
#      caller supplied an explicit value for that key).
#   2. With --all-slots-present, injects nic_card_mount_{0..4}_present:=true
#      AND sc_port_{0,1}_present:=true so a single launch produces a scene
#      with every NIC + SC slot visible (per-slot pose still randomized).
#   3. With --randomize-task-board, samples task_board_x/y/z (slight jitter)
#      and task_board_yaw ∈ [−π, +π] (full 360° sweep) for visual variety.
#   4. Generates a unique dataset repo_id (timestamp-based) for one-shot
#      single-episode capture, since /episode_reset is not used.
#   5. Prints the matching `pixi run aic-record` command. The target slot +
#      port are passed both to the teleop (target_module_name/port_name) AND
#      the robot (task_target_module/task_target_port) so the controller can
#      append the matching task-id one-hot to obs.state.
#   6. Sources the workspace, starts rmw_zenohd, and execs the launch.
#
# Usage:
#   launch_randomized_episode.sh \
#     [<launch arg=val>...] \
#     [--target=<mount>[/<port>]] [--plug-name=<name>] \
#     [--cable-type=<type>] [--episode-time-s=<int>] \
#     [--dataset-prefix=<s>] [--seed=<int>] \
#     [--all-slots-present] [--randomize-task-board] \
#     [--no-randomize] [--dry-run] \
#     [-- <extra ros2 launch args>]
#
# Examples:
#   # Multi-slot NIC scene, random NIC target:
#   ./launch_randomized_episode.sh --all-slots-present --randomize-task-board \
#       --target=nic_card_mount_2/sfp_port_1 --plug-name=sfp_tip
#
#   # Multi-slot SC scene, SC target:
#   ./launch_randomized_episode.sh --all-slots-present --randomize-task-board \
#       --cable-type=sfp_sc_cable_reversed --target=sc_port_0/sc_port_link \
#       --plug-name=sc_tip
#
# The <launch arg=val> items use the same `key:=value` syntax as `ros2 launch`.

set -euo pipefail

TARGET=""                 # raw arg, may be "<mount>" or "<mount>/<port>"
TARGET_MOUNT=""
TARGET_PORT=""
PLUG_NAME="sfp_tip"       # default: NIC head plug. SC head should pass sc_tip.
CABLE_TYPE="sfp_sc_cable"
EPISODE_TIME_S=600
DATASET_PREFIX="cheatcode"
SEED=$RANDOM
RANDOMIZE=1
RANDOM_HOME_OFFSET="0.06"
ALL_SLOTS=0
RANDOMIZE_TASK_BOARD=0
DRY_RUN=0
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target=*)               TARGET="${1#*=}";              shift ;;
        --plug-name=*)            PLUG_NAME="${1#*=}";           shift ;;
        --cable-type=*)           CABLE_TYPE="${1#*=}";          shift ;;
        --episode-time-s=*)       EPISODE_TIME_S="${1#*=}";      shift ;;
        --dataset-prefix=*)       DATASET_PREFIX="${1#*=}";      shift ;;
        --seed=*)                 SEED="${1#*=}";                shift ;;
        --random-home-offset=*)   RANDOM_HOME_OFFSET="${1#*=}";  shift ;;
        --all-slots-present)      ALL_SLOTS=1;                   shift ;;
        --randomize-task-board)   RANDOMIZE_TASK_BOARD=1;        shift ;;
        --no-randomize)           RANDOMIZE=0;                   shift ;;
        --dry-run)                DRY_RUN=1;                     shift ;;
        --)                       shift; LAUNCH_ARGS+=("$@"); break ;;
        *:=*)                     LAUNCH_ARGS+=("$1");           shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Parse --target=<mount>[/<port>]. If port is omitted we keep TARGET_PORT
# empty; downstream code will fall back to a per-mount default.
if [[ -n "$TARGET" ]]; then
    if [[ "$TARGET" == */* ]]; then
        TARGET_MOUNT="${TARGET%%/*}"
        TARGET_PORT="${TARGET#*/}"
    else
        TARGET_MOUNT="$TARGET"
        TARGET_PORT=""
    fi
fi

# --all-slots-present injects every NIC + SC slot if the caller hasn't
# already passed an explicit *_present arg for it.
if [[ $ALL_SLOTS -eq 1 ]]; then
    declare -a SLOTS_TO_PRESENT=(
        nic_card_mount_0 nic_card_mount_1 nic_card_mount_2
        nic_card_mount_3 nic_card_mount_4
        sc_port_0 sc_port_1
    )
    EXISTING=" ${LAUNCH_ARGS[*]:-} "
    for slot in "${SLOTS_TO_PRESENT[@]}"; do
        if [[ "$EXISTING" != *" ${slot}_present:="* ]]; then
            LAUNCH_ARGS+=("${slot}_present:=true")
        fi
    done
fi

if [[ $RANDOMIZE -eq 1 ]]; then
    SAMPLED=$(LAUNCH_ARGS_STR="${LAUNCH_ARGS[*]:-}" SEED="$SEED" python3 - <<'PYEOF'
import os, random, re

seed = int(os.environ["SEED"])
rng = random.Random(seed)

# Sampling ranges from docs/task_board_description.md.
RANGES = {
    "nic_card_mount":  {"translation": (-0.0215, 0.0234), "yaw": (-0.1745, 0.1745)},
    "sc_port":         {"translation": (-0.06,   0.055)},
    "lc_mount_rail":   {"translation": (-0.09425, 0.09425), "yaw": (-1.047, 1.047)},
    "sfp_mount_rail":  {"translation": (-0.09425, 0.09425), "yaw": (-1.047, 1.047)},
    "sc_mount_rail":   {"translation": (-0.09425, 0.09425), "yaw": (-1.047, 1.047)},
}

args = os.environ.get("LAUNCH_ARGS_STR", "").split()
explicit = {a.split(":=", 1)[0] for a in args if ":=" in a}
present_slots = sorted({
    m.group(1) for a in args
    for m in [re.match(r"^([a-z_]+_\d+)_present:=true$", a)] if m
})

out = []
for slot in present_slots:
    family = re.sub(r"_\d+$", "", slot)
    for key, (lo, hi) in RANGES.get(family, {}).items():
        full = f"{slot}_{key}"
        if full not in explicit:
            out.append(f"{full}:={rng.uniform(lo, hi):.6f}")
print(" ".join(out))
PYEOF
)
    if [[ -n "$SAMPLED" ]]; then
        # shellcheck disable=SC2206
        LAUNCH_ARGS+=($SAMPLED)
    fi
fi

if [[ -z "$TARGET_MOUNT" ]]; then
    for arg in "${LAUNCH_ARGS[@]}"; do
        if [[ "$arg" =~ ^([a-z_]+_[0-9]+)_present:=true$ ]]; then
            TARGET_MOUNT="${BASH_REMATCH[1]}"
            break
        fi
    done
fi
# Per-mount default port if caller didn't specify one.
if [[ -n "$TARGET_MOUNT" && -z "$TARGET_PORT" ]]; then
    case "$TARGET_MOUNT" in
        nic_card_mount_*) TARGET_PORT="sfp_port_0" ;;
        sc_port_*)        TARGET_PORT="sc_port_link" ;;
        *)                TARGET_PORT="sfp_port_0" ;;
    esac
fi

# Sample task-board pose if requested. task_board_x/y/z/yaw are live launch
# args (defaults: x=0.15, y=-0.20, z=1.14, yaw=3.1415 rad — declared in
# aic_gz_bringup.launch.py:667-706). Slight DELTA jitter on xyz around those
# defaults; yaw is a full 360° sweep absolute (cyclic).
TASK_BOARD_SUMMARY=""
if [[ $RANDOMIZE_TASK_BOARD -eq 1 ]]; then
    TB_OUT=$(SEED="$SEED" python3 - <<'PYEOF'
import os, math, random
rng = random.Random(int(os.environ["SEED"]) ^ 0xC0DE)
TB_DEFAULT_X, TB_DEFAULT_Y, TB_DEFAULT_Z = 0.15, -0.20, 1.14
dx = rng.uniform(-0.02, 0.02)
dy = rng.uniform(-0.02, 0.02)
dz = rng.uniform(-0.005, 0.005)
x = TB_DEFAULT_X + dx
y = TB_DEFAULT_Y + dy
z = TB_DEFAULT_Z + dz
yaw = rng.uniform(-math.pi, math.pi)
print(f"task_board_x:={x:.6f} task_board_y:={y:.6f} task_board_z:={z:.6f} task_board_yaw:={yaw:.6f}")
print("---SUMMARY---")
print(f"  Δx = {dx:+.4f} m   Δy = {dy:+.4f} m   Δz = {dz:+.4f} m   yaw = {yaw:+.4f} rad ({math.degrees(yaw):+.2f} deg)")
print(f"  abs: x = {x:+.4f}  y = {y:+.4f}  z = {z:+.4f}")
PYEOF
)
    TB_ARGS_LINE="${TB_OUT%%$'\n---SUMMARY---'*}"
    TASK_BOARD_SUMMARY="${TB_OUT##*---SUMMARY---$'\n'}"
    # shellcheck disable=SC2206
    LAUNCH_ARGS+=($TB_ARGS_LINE)
else
    TASK_BOARD_SUMMARY="  disabled (no --randomize-task-board)"
fi

# Sample home offset (dx, dy, dz) once and forward to home_robot.py via
# home_x/y/z launch params. Cable is NOT shifted: at launch the gripper homes
# to its default position first, the cable spawns at the matching default world
# pose, CablePlugin attaches plug→gripper cleanly, and only then does
# home_robot.py translate the gripper by d (~10s later via the launch event
# handler). The cable rides along on the welded joint, so the plug stays
# seated in the gripper at the new home.
HOME_OFFSET_ARGS=()
HOME_OFFSET_SUMMARY=""
if [[ "$RANDOM_HOME_OFFSET" != "0" && "$RANDOM_HOME_OFFSET" != "0.0" ]]; then
    HOME_PY_OUT=$(SEED="$SEED" OFFSET="$RANDOM_HOME_OFFSET" python3 - <<'PYEOF'
import os, random
rng = random.Random(int(os.environ["SEED"]))
m = float(os.environ["OFFSET"])
# NOMINAL_HOME_POS from aic_bringup/scripts/_reset_helper.py.
home_xyz = (-0.3719, 0.1943, 0.3286)
dx, dy, dz = rng.uniform(-m, m), rng.uniform(-m, m), rng.uniform(-m, m)
args = [
    "home_on_startup:=true",
    f"home_x:={home_xyz[0]+dx:.6f}",
    f"home_y:={home_xyz[1]+dy:.6f}",
    f"home_z:={home_xyz[2]+dz:.6f}",
]
summary = (
    f"  dx = {dx:+.4f} m   dy = {dy:+.4f} m   dz = {dz:+.4f} m\n"
    f"  TCP home = ({home_xyz[0]+dx:+.4f}, {home_xyz[1]+dy:+.4f}, {home_xyz[2]+dz:+.4f})\n"
    f"  cable    = launch defaults (welded joint moves with gripper)"
)
print(" ".join(args))
print("---SUMMARY---")
print(summary)
PYEOF
)
    HOME_OFFSET_ARGS_LINE="${HOME_PY_OUT%%$'\n---SUMMARY---'*}"
    HOME_OFFSET_SUMMARY="${HOME_PY_OUT##*---SUMMARY---$'\n'}"
    # shellcheck disable=SC2206
    HOME_OFFSET_ARGS=($HOME_OFFSET_ARGS_LINE)
else
    HOME_OFFSET_ARGS=("home_on_startup:=false")
    HOME_OFFSET_SUMMARY="  disabled (--random-home-offset=0)"
fi

TS=$(date +%s)
REPO_ID="local/${DATASET_PREFIX}-${TS}"

SCENE_SUMMARY=$(LAUNCH_ARGS_STR="${LAUNCH_ARGS[*]:-}" python3 - <<'PYEOF'
import math, os, re
from collections import defaultdict

args = os.environ.get("LAUNCH_ARGS_STR", "").split()
slots = defaultdict(dict)
for a in args:
    m = re.match(r"^([a-z_]+_\d+)_(present|translation|roll|pitch|yaw):=(.+)$", a)
    if m:
        slot, key, val = m.group(1), m.group(2), m.group(3)
        slots[slot][key] = val

if not slots:
    print("  <empty board — no _present slots>")
else:
    for slot in sorted(slots):
        keys = slots[slot]
        if keys.get("present", "false").lower() != "true":
            continue
        parts = [f"{slot}:"]
        for key in ("translation", "roll", "pitch", "yaw"):
            if key in keys:
                v = float(keys[key])
                if key == "translation":
                    parts.append(f"  {key:11s} = {v:+.4f} m")
                else:
                    parts.append(f"  {key:11s} = {v:+.4f} rad ({math.degrees(v):+.2f} deg)")
        print("\n".join(parts))
PYEOF
)

cat >&2 <<EOF
[sim] seed=${SEED}
[sim] dataset=${REPO_ID}
[sim] cable_type=${CABLE_TYPE}  plug_name=${PLUG_NAME}
[sim] target=${TARGET_MOUNT:-<unset>}/${TARGET_PORT:-<unset>}
[sim] scene:
${SCENE_SUMMARY}
[sim] task board (--randomize-task-board=${RANDOMIZE_TASK_BOARD}):
${TASK_BOARD_SUMMARY}
[sim] home offset (--random-home-offset=${RANDOM_HOME_OFFSET}):
${HOME_OFFSET_SUMMARY}
[record] paste this in another terminal:
  pixi run aic-record \\
    --robot.type=aic_controller --robot.id=aic \\
    --teleop.type=cheatcode --teleop.id=aic \\
    --teleop.cable_name=cable_0 --teleop.plug_name=${PLUG_NAME} \\
    --teleop.target_module_name=${TARGET_MOUNT} --teleop.port_name=${TARGET_PORT} \\
    --teleop.approach_noise_xyz_m=0 \\
    --teleop.descent_noise_xyz_m=0 \\
    --teleop.approach_rot_noise_deg=0 \\
    --robot.teleop_target_mode=pose --robot.teleop_frame_id=gripper/tcp \\
    --robot.task_target_module=${TARGET_MOUNT} --robot.task_target_port=${TARGET_PORT} \\
    --dataset.repo_id=${REPO_ID} \\
    --dataset.single_task="Insert SFP cable into ${TARGET_MOUNT}/${TARGET_PORT}" \\
    --dataset.push_to_hub=false --play_sounds=false --display_data=false \\
    --dataset.num_episodes=1 --dataset.episode_time_s=${EPISODE_TIME_S} --dataset.reset_time_s=0
EOF

LAUNCH_CMD=(ros2 launch aic_bringup aic_gz_bringup.launch.py
    ground_truth:=true
    start_aic_engine:=false
    spawn_task_board:=true
    spawn_cable:=true
    "cable_type:=${CABLE_TYPE}"
    attach_cable_to_gripper:=true
    "${HOME_OFFSET_ARGS[@]}"
    "${LAUNCH_ARGS[@]}")

if [[ $DRY_RUN -eq 1 ]]; then
    printf '[sim] would run:' >&2
    printf ' %q' "${LAUNCH_CMD[@]}" >&2
    printf '\n' >&2
    exit 0
fi

# colcon's setup.bash references COLCON_TRACE without a default, which
# trips `set -u`. Relax nounset for the source + exec tail.
set +u
# shellcheck disable=SC1091
source ~/ws_aic_challenge/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run rmw_zenoh_cpp rmw_zenohd &
sleep 2
exec "${LAUNCH_CMD[@]}"
