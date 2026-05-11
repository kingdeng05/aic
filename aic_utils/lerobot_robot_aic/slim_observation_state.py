#!/usr/bin/env python3
"""Slim observation.state down to non-leaky channels (tcp_velocity + wrench).

Source layout (rot6d dataset, observation.state shape=34):
   0..2    tcp_pose.position.{x,y,z}                <- DROP (leaks position)
   3..8    tcp_pose.rot6d.{0..5}                    <- DROP (leaks orientation, near-constant)
   9..11   tcp_velocity.linear.{x,y,z}              <- KEEP
  12..14   tcp_velocity.angular.{x,y,z}             <- KEEP
  15..17   tcp_error.{x,y,z}                        <- DROP (commanded - current = next action)
  18..20   tcp_error.{rx,ry,rz}                     <- DROP (same)
  21..27   joint_positions.{0..6}                   <- DROP (R^2=0.84 -> 0.9999 leak)
  28..30   wrench.force.{x,y,z}                     <- KEEP
  31..33   wrench.torque.{x,y,z}                    <- KEEP

Output observation.state shape: 12  [tcp_velocity (6) + wrench (6)]
Action is NOT modified (stays 9D rot6d).

Usage:
  pixi run python aic_utils/lerobot_robot_aic/slim_observation_state.py \
      [src_name] [out_name]
defaults: nic_card_mount_0_merged_trimmed_rot6d   <src>_slim
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "local"
SRC_NAME = sys.argv[1] if len(sys.argv) > 1 else "nic_card_mount_0_merged_trimmed_rot6d"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else f"{SRC_NAME}_slim"
SRC = ROOT / SRC_NAME
OUT = ROOT / OUT_NAME

VIDEO_KEYS = [
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
]


def aggregate_stats(values: dict[str, np.ndarray]) -> dict:
    out = {}
    for feat, cat in values.items():
        if cat.ndim == 1:
            cat = cat[:, None]
        out[feat] = {
            "min": cat.min(axis=0).tolist(),
            "max": cat.max(axis=0).tolist(),
            "mean": cat.mean(axis=0).tolist(),
            "std": cat.std(axis=0).tolist(),
            "count": [int(cat.shape[0])],
            "q01": np.quantile(cat, 0.01, axis=0).tolist(),
            "q10": np.quantile(cat, 0.10, axis=0).tolist(),
            "q50": np.quantile(cat, 0.50, axis=0).tolist(),
            "q90": np.quantile(cat, 0.90, axis=0).tolist(),
            "q99": np.quantile(cat, 0.99, axis=0).tolist(),
        }
    return out


def find_indices(state_names: list[str], group_prefixes: list[str]) -> tuple[list[int], list[str]]:
    """Return (kept_indices, kept_names) — order preserved as in source."""
    idx, kept = [], []
    for i, n in enumerate(state_names):
        if any(n.startswith(p) for p in group_prefixes):
            idx.append(i)
            kept.append(n)
    return idx, kept


def main() -> None:
    if OUT.exists():
        sys.exit(f"output {OUT} already exists; refusing to overwrite")
    if not SRC.is_dir():
        sys.exit(f"source {SRC} does not exist")

    print(f"slimming {SRC} -> {OUT}")

    (OUT / "data/chunk-000").mkdir(parents=True)
    (OUT / "meta/episodes/chunk-000").mkdir(parents=True)
    for k in VIDEO_KEYS:
        (OUT / f"videos/{k}/chunk-000").mkdir(parents=True)

    # --- Load source info & data ---
    info = json.loads((SRC / "meta/info.json").read_text())
    state_names = list(info["features"]["observation.state"]["names"])
    state_shape = info["features"]["observation.state"]["shape"][0]
    print(f"  source obs.state: shape={state_shape}, names={len(state_names)}")
    assert len(state_names) == state_shape

    KEEP_PREFIXES = ["tcp_velocity.", "wrench."]
    keep_idx, keep_names = find_indices(state_names, KEEP_PREFIXES)
    print(f"  keeping {len(keep_idx)} dims: {keep_names}")
    assert len(keep_idx) == 12, f"expected 12 kept dims, got {len(keep_idx)}"

    data = pd.read_parquet(SRC / "data/chunk-000/file-000.parquet")
    n = len(data)
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    assert state.shape == (n, state_shape), f"unexpected state shape {state.shape}"
    state_new = state[:, keep_idx].astype(np.float32)
    print(f"  frames={n}  obs.state {state.shape} -> {state_new.shape}")

    # Action is unchanged.
    action = np.stack(data["action"].to_numpy()).astype(np.float32)
    print(f"  action shape (unchanged): {action.shape}")

    # --- Replace state column, keep action as-is, write parquet ---
    data["observation.state"] = list(state_new)
    data["action"] = list(action)  # ensure float32 dtype is preserved
    data.to_parquet(OUT / "data/chunk-000/file-000.parquet", index=False)
    print(f"  wrote data parquet")

    # --- info.json ---
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "names": keep_names,
        "shape": [len(keep_idx)],
    }
    (OUT / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"  wrote info.json (obs.state shape={info['features']['observation.state']['shape']})")

    # --- stats.json ---
    src_stats = json.loads((SRC / "meta/stats.json").read_text())
    new_stats = dict(src_stats)  # start with source (preserves action + image stats)
    # Slice obs.state stats by index — this is exact and avoids re-aggregation noise.
    src_state_stats = src_stats["observation.state"]
    sliced_state_stats = {}
    for k, v in src_state_stats.items():
        if k == "count":
            sliced_state_stats[k] = list(v)
        else:
            arr = np.asarray(v)
            sliced_state_stats[k] = arr[keep_idx].tolist()
    new_stats["observation.state"] = sliced_state_stats
    (OUT / "meta/stats.json").write_text(json.dumps(new_stats, indent=4))
    print(f"  wrote stats.json (sliced obs.state stats from source)")

    # --- Pass-through copies ---
    shutil.copy2(SRC / "meta/tasks.parquet", OUT / "meta/tasks.parquet")
    shutil.copy2(SRC / "meta/episodes/chunk-000/file-000.parquet",
                 OUT / "meta/episodes/chunk-000/file-000.parquet")
    for k in VIDEO_KEYS:
        shutil.copy2(SRC / f"videos/{k}/chunk-000/file-000.mp4",
                     OUT / f"videos/{k}/chunk-000/file-000.mp4")
    print(f"  copied videos and episode/task metadata")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
