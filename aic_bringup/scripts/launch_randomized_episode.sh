#!/usr/bin/env bash
#
# launch_randomized_episode.sh
#
# Wrapper around `ros2 launch aic_bringup aic_gz_bringup.launch.py` that:
#   1. Auto-samples translation/yaw within docs ranges for any task-board slot
#      passed with `<slot>_present:=true` (unless --no-randomize is set or the
#      caller supplied an explicit value for that key).
#   2. Generates a unique dataset repo_id (timestamp-based) for one-shot
#      single-episode capture, since /episode_reset is not used.
#   3. Prints the matching `pixi run aic-record` command for the second
#      terminal.
#   4. Sources the workspace, starts rmw_zenohd, and execs the launch.
#
# Usage:
#   launch_randomized_episode.sh \
#     [<launch arg=val>...] \
#     [--target=<slot>] [--cable-type=<type>] [--episode-time-s=<int>] \
#     [--dataset-prefix=<s>] [--seed=<int>] [--no-randomize] [--dry-run]
#     [-- <extra ros2 launch args>]
#
# The <launch arg=val> items use the same `key:=value` syntax as `ros2 launch`.

set -euo pipefail

TARGET=""
CABLE_TYPE="sfp_sc_cable"
EPISODE_TIME_S=600
DATASET_PREFIX="cheatcode"
SEED=$RANDOM
RANDOMIZE=1
RANDOM_HOME_OFFSET="0.06"
DRY_RUN=0
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target=*)             TARGET="${1#*=}";              shift ;;
        --cable-type=*)         CABLE_TYPE="${1#*=}";          shift ;;
        --episode-time-s=*)     EPISODE_TIME_S="${1#*=}";      shift ;;
        --dataset-prefix=*)     DATASET_PREFIX="${1#*=}";      shift ;;
        --seed=*)               SEED="${1#*=}";                shift ;;
        --random-home-offset=*) RANDOM_HOME_OFFSET="${1#*=}";  shift ;;
        --no-randomize)         RANDOMIZE=0;                   shift ;;
        --dry-run)              DRY_RUN=1;                     shift ;;
        --)                     shift; LAUNCH_ARGS+=("$@"); break ;;
        *:=*)                   LAUNCH_ARGS+=("$1");           shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

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

if [[ -z "$TARGET" ]]; then
    for arg in "${LAUNCH_ARGS[@]}"; do
        if [[ "$arg" =~ ^([a-z_]+_[0-9]+)_present:=true$ ]]; then
            TARGET="${BASH_REMATCH[1]}"
            break
        fi
    done
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
[sim] target=${TARGET:-<none — pass <slot>_present:=true>}
[sim] scene:
${SCENE_SUMMARY}
[sim] home offset (--random-home-offset=${RANDOM_HOME_OFFSET}):
${HOME_OFFSET_SUMMARY}
[record] paste this in another terminal:
  pixi run aic-record \\
    --robot.type=aic_controller --robot.id=aic \\
    --teleop.type=cheatcode --teleop.id=aic \\
    --teleop.cable_name=cable_0 --teleop.plug_name=sfp_tip \\
    --teleop.target_module_name=${TARGET} --teleop.port_name=sfp_port_0 \\
    --robot.teleop_target_mode=pose --robot.teleop_frame_id=gripper/tcp \\
    --dataset.repo_id=${REPO_ID} \\
    --dataset.single_task="Insert SFP SC cable into NIC card port" \\
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
