#!/usr/bin/env bash
# run_eval_iter.sh — one eval iteration.
# Usage: run_eval_iter.sh <translation_m> <yaw_rad> <iter_dir>
# Side effect: edits sample_config.yaml IN PLACE.
# Final stdout: RESULT translation=<t> yaw=<y> trial1_tier3=<score> outcome=<SUCCESS|FAIL|FAIL_WRONG_PORT|TIMEOUT>

set -uo pipefail
T=$1; Y=$2; D=$3
mkdir -p "$D"

WS=/home/sai/ws_aic_challenge
SRC=$WS/src/aic
CFG=$SRC/aic_engine/config/sample_config.yaml
CKPT=$SRC/outputs/train/nic_card_mount_0_merged_trimmed_rot6d_slim_chunk25/checkpoints/100000/pretrained_model
SCORE_FILE=$HOME/aic_results/scoring.yaml

# 1. Edit sample_config.yaml in place: trial_1 nic_rail_0 entity_pose
python3 - "$CFG" "$T" "$Y" <<'PY'
import sys, yaml
p, t, y = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with open(p) as f: c = yaml.safe_load(f)
c['trials']['trial_1']['scene']['task_board']['nic_rail_0']['entity_pose']['translation'] = t
c['trials']['trial_1']['scene']['task_board']['nic_rail_0']['entity_pose']['yaw'] = y
with open(p, 'w') as f: yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False)
PY

# 2. Kill any leftover aic_model / sim from a previous iter so the engine doesn't
# see duplicate 'aic_model' nodes. Clear scoring.yaml so we detect this iter's write.
# Patterns chosen to be unique to the actual binaries (cmdline starts with the
# full python path, not literal 'python', so we anchor on the unique
# 'lib/aic_model/aic_model' binary path and the pixi wrapper).
for pid in $(pgrep -f 'lib/aic_model/aic_model' 2>/dev/null); do kill -KILL "$pid" 2>/dev/null || true; done
for pid in $(pgrep -f 'pixi run ros2 run aic_model' 2>/dev/null); do kill -KILL "$pid" 2>/dev/null || true; done
distrobox enter aic_eval -- pkill -KILL -f \
    'ros2 launch|aic_gz_bringup|gz sim|rmw_zenohd|aic_adapter|component_container|robot_state_publisher|tf2_ros|rviz2|aic_engine' \
    >/dev/null 2>&1 || true
sleep 3
rm -f "$SCORE_FILE"

# 3. Run sim + engine in distrobox. Use ';' instead of '&&' so the `&` only
# backgrounds zenohd; the source/export persist into the outer shell so
# `ros2 launch` is found on PATH.
setsid distrobox enter aic_eval -- bash -lc \
    "source $WS/install/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp; ros2 run rmw_zenoh_cpp rmw_zenohd >/dev/null 2>&1 & sleep 2; exec ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=false start_aic_engine:=true aic_engine_config_file:=$CFG" \
    >"$D/sim.log" 2>&1 &

# 4. Run policy on host (exact user command)
cd "$SRC"
setsid env AIC_ACT_BLANK_IMAGES=0 AIC_ACT_MODEL_PATH="$CKPT" \
    pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=my_policy_node.RunACT \
    >"$D/policy.log" 2>&1 &

# 5. Wait up to 15 min for scoring.yaml (engine writes once after all 3 trials)
for _ in $(seq 1 900); do
    [[ -s "$SCORE_FILE" ]] && break
    sleep 1
done

# 6. Capture
[[ -s "$SCORE_FILE" ]] && cp "$SCORE_FILE" "$D/scoring.yaml"

# 7. Teardown (same as pre-iter cleanup; ensures next iter starts fresh)
for pid in $(pgrep -f 'lib/aic_model/aic_model' 2>/dev/null); do kill -KILL "$pid" 2>/dev/null || true; done
for pid in $(pgrep -f 'pixi run ros2 run aic_model' 2>/dev/null); do kill -KILL "$pid" 2>/dev/null || true; done
distrobox enter aic_eval -- pkill -KILL -f \
    'ros2 launch|aic_gz_bringup|gz sim|rmw_zenohd|aic_adapter|component_container|robot_state_publisher|tf2_ros|rviz2|aic_engine' \
    >/dev/null 2>&1 || true
sleep 2

# 8. Parse + report
if [[ -s "$D/scoring.yaml" ]]; then
    PARSED=$(python3 -c "
import yaml
d = yaml.safe_load(open('$D/scoring.yaml'))
s = float(d['trial_1']['tier_3']['score'])
o = 'SUCCESS' if s >= 75 else ('FAIL_WRONG_PORT' if s < 0 else 'FAIL')
print(f'{s:.4f} {o}')
")
    SCORE=$(echo "$PARSED" | awk '{print $1}')
    OUTCOME=$(echo "$PARSED" | awk '{print $2}')
else
    SCORE=NA; OUTCOME=TIMEOUT
fi
echo "RESULT translation=$T yaw=$Y trial1_tier3=$SCORE outcome=$OUTCOME"
