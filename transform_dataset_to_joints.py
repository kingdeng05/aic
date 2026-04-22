"""
Transform LeRobot dataset: replace the recorded action with next-frame joint positions.

Many teleop datasets record action as either a 6-dim velocity twist or a 7-dim
absolute TCP pose. Neither target works well with ACT out of the box. This
script rewrites the `action` column as 7-dim absolute joint positions, pulled
from `observation.state[19:26]` at the next frame (`action[t] = joints[t+1]`).
For the final frame of each episode, action = current joints (hold pose).

Usage:
    python transform_dataset_to_joints.py \\
        --src <user>/<source-repo-id> \\
        --dst <user>/<dest-repo-id>

Paths resolve under ~/.cache/huggingface/lerobot/<user>/<repo_id> by default.
Pass --prefix to override.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREFIX = Path.home() / ".cache" / "huggingface" / "lerobot"
JOINT_SLICE = slice(19, 26)  # 7 joint positions inside observation.state


def resolve(repo_id: str, prefix: Path) -> Path:
    if "/" not in repo_id:
        sys.exit(f"error: expected '<user>/<repo-id>', got '{repo_id}'")
    return prefix / repo_id


def transform(src: Path, dst: Path) -> None:
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"error: {src} is not a LeRobot dataset (missing meta/info.json)")

    if dst.exists():
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Copying {src} -> {dst}")
    shutil.copytree(src, dst)

    data_dir = dst / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    print(f"Transforming {len(parquet_files)} parquet files")

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        states = np.stack(df["observation.state"].values)
        state_dim = states.shape[1]
        if state_dim < JOINT_SLICE.stop:
            sys.exit(
                f"error: observation.state has {state_dim} dims, "
                f"needs >= {JOINT_SLICE.stop} to slice joints 19:26"
            )
        joint_positions = states[:, JOINT_SLICE].astype(np.float32)  # (N, 7)

        new_actions = np.zeros((len(df), 7), dtype=np.float32)
        for ep in df["episode_index"].unique():
            idxs = np.where((df["episode_index"] == ep).values)[0]
            ep_joints = joint_positions[idxs]
            ep_new = np.empty_like(ep_joints)
            ep_new[:-1] = ep_joints[1:]
            ep_new[-1] = ep_joints[-1]
            new_actions[idxs] = ep_new

        df["action"] = [row for row in new_actions]
        df.to_parquet(pf, index=False)
        print(f"  {pf.name}: {len(df)} rows transformed")

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {
        "dtype": "float32",
        "names": [f"joint_positions.{i}" for i in range(7)],
        "shape": [7],
    }
    info_path.write_text(json.dumps(info, indent=4))
    print("Updated info.json: action now 7-dim joint positions")

    print("Recomputing action stats...")
    all_actions = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_actions.append(np.stack(df["action"].values))
    all_actions = np.concatenate(all_actions)

    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    else:
        stats = {}

    std = all_actions.std(axis=0)
    stats["action"] = {
        "mean": all_actions.mean(axis=0).tolist(),
        "std": std.tolist() if std.min() > 0 else (std + 1e-6).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "count": [len(all_actions)],
    }
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"  action mean: {stats['action']['mean']}")
    print(f"  action std : {stats['action']['std']}")
    print(f"  action min : {stats['action']['min']}")
    print(f"  action max : {stats['action']['max']}")

    print(f"\nDone. New dataset at: {dst}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="source repo_id, e.g. sai-chand-gubbala/cheatcode-teleop-sfpsc-nic-30")
    p.add_argument("--dst", required=True, help="destination repo_id, e.g. kingdeng05/cheatcode-teleop-sfpsc-nic-30-joints")
    p.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX,
                   help=f"cache prefix (default: {DEFAULT_PREFIX})")
    args = p.parse_args()

    src = resolve(args.src, args.prefix)
    dst = resolve(args.dst, args.prefix)
    transform(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
