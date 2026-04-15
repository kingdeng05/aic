"""
Transform LeRobot dataset: replace velocity-twist action with next-frame joint positions.

This converts the action representation from 6-dim bang-bang velocity commands
(which don't work well with ACT) to 7-dim absolute joint position targets
(which match what the original ACT paper uses).

For each frame t, the new action = observation.state[t+1][19:26] (next frame joint positions).
For the last frame of each episode, action = current frame joint positions (hold pose).
"""

import json
import shutil
from pathlib import Path
import glob
import pandas as pd
import numpy as np

SRC = Path("/home/fuheng/.cache/huggingface/lerobot/kingdeng05/aic-insert-demo-2")
DST = Path("/home/fuheng/.cache/huggingface/lerobot/kingdeng05/aic-insert-demo-2-joints")

# Clean and create destination
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

# Copy everything first, then overwrite the parts we need
print(f"Copying {SRC} -> {DST}")
shutil.copytree(SRC, DST, dirs_exist_ok=True)

# Transform parquet files
data_dir = DST / "data" / "chunk-000"
parquet_files = sorted(glob.glob(str(data_dir / "*.parquet")))
print(f"Transforming {len(parquet_files)} parquet files")

JOINT_SLICE = slice(19, 26)  # 7 joint positions in observation.state

for pf in parquet_files:
    df = pd.read_parquet(pf)

    # Sort by episode then frame to ensure correct ordering for next-frame lookup
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    states = np.stack(df["observation.state"].values)
    joint_positions = states[:, JOINT_SLICE].astype(np.float32)  # (N, 7)

    # For each frame, action = next frame's joint positions (same episode)
    new_actions = np.zeros((len(df), 7), dtype=np.float32)
    for ep in df["episode_index"].unique():
        mask = (df["episode_index"] == ep).values
        idxs = np.where(mask)[0]
        ep_joints = joint_positions[idxs]
        # Shift by +1: action[t] = joints[t+1]
        ep_new = np.empty_like(ep_joints)
        ep_new[:-1] = ep_joints[1:]
        ep_new[-1] = ep_joints[-1]  # last frame: hold
        new_actions[idxs] = ep_new

    # Replace action column (must store as list-of-arrays for parquet compatibility)
    df["action"] = [row for row in new_actions]

    df.to_parquet(pf, index=False)
    print(f"  {Path(pf).name}: {len(df)} rows transformed")

# Update info.json
info_path = DST / "meta" / "info.json"
with open(info_path) as f:
    info = json.load(f)

info["features"]["action"] = {
    "dtype": "float32",
    "names": [
        "joint_positions.0",
        "joint_positions.1",
        "joint_positions.2",
        "joint_positions.3",
        "joint_positions.4",
        "joint_positions.5",
        "joint_positions.6",
    ],
    "shape": [7],
}

with open(info_path, "w") as f:
    json.dump(info, f, indent=4)
print(f"Updated info.json: action shape 6 -> 7")

# Recompute stats.json for the action feature
print("Recomputing action stats...")
all_actions = []
for pf in parquet_files:
    df = pd.read_parquet(pf)
    all_actions.append(np.stack(df["action"].values))
all_actions = np.concatenate(all_actions)  # (N, 7)

stats_path = DST / "meta" / "stats.json"
with open(stats_path) as f:
    stats = json.load(f)

stats["action"] = {
    "mean": all_actions.mean(axis=0).tolist(),
    "std": all_actions.std(axis=0).tolist() if all_actions.std(axis=0).min() > 0 else (all_actions.std(axis=0) + 1e-6).tolist(),
    "min": all_actions.min(axis=0).tolist(),
    "max": all_actions.max(axis=0).tolist(),
    "count": [len(all_actions)],
}

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=4)
print(f"Updated stats.json with new action stats")
print(f"  action mean: {stats['action']['mean']}")
print(f"  action std:  {stats['action']['std']}")
print(f"  action min:  {stats['action']['min']}")
print(f"  action max:  {stats['action']['max']}")

print(f"\nDone! New dataset at: {DST}")
print(f"To train: --dataset.repo_id=kingdeng05/aic-insert-demo-2-joints")
print(f"(LeRobot will load from local cache first)")
