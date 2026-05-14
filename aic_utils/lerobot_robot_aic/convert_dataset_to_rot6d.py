#!/usr/bin/env python3
"""Convert a LeRobot dataset's quaternion rotations to 6D (Zhou et al.) representation.

Source dataset layout (e.g. nic_card_mount_0_merged_trimmed):
  action          : [px, py, pz, qw, qx, qy, qz]                          (7D)
  observation.state: [tcp_pose.position.{x,y,z}, tcp_pose.orientation.{x,y,z,w},
                      tcp_velocity{6}, tcp_error{6}, joint_positions{7},
                      wrench{6}, task_id_one_hot{12}]                     (44D)

Output:
  action          : [px, py, pz, rot6d.0..rot6d.5]                        (9D)
  observation.state: [tcp_pose.position{3}, tcp_pose.rot6d{6}, tcp_velocity{6},
                      tcp_error{6}, joint_positions{7}, wrench{6},
                      task_id_one_hot{12}]                                 (46D)

The 6D rotation representation is the first two columns of the rotation matrix,
flattened column-wise: [R[:,0]; R[:,1]] (Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks").

Usage:
  pixi run python aic_utils/lerobot_robot_aic/convert_dataset_to_rot6d.py \
      [src_name] [out_name]
defaults: nic_card_mount_0_merged_trimmed  <src>_rot6d
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "local"
SRC_NAME = sys.argv[1] if len(sys.argv) > 1 else "nic_card_mount_0_merged_trimmed"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else f"{SRC_NAME}_rot6d"
SRC = ROOT / SRC_NAME
OUT = ROOT / OUT_NAME

VIDEO_KEYS = [
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
]


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """q: (..., 4) in wxyz order. Returns (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened column-wise. (..., 3, 3) -> (..., 6)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: (..., 6) -> (..., 3, 3). Inverse of matrix_to_rot6d (up to noise)."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) in wxyz order. Branchless via per-element selection."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(R.shape[:-2] + (4,), dtype=R.dtype)
    # Standard 4-case formulation; vectorized via masks.
    pos_t = t > 0
    s = np.where(pos_t, np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2, 0)
    qw = np.where(pos_t, 0.25 * s, 0)
    qx = np.where(pos_t, (m[..., 2, 1] - m[..., 1, 2]) / np.where(s == 0, 1, s), 0)
    qy = np.where(pos_t, (m[..., 0, 2] - m[..., 2, 0]) / np.where(s == 0, 1, s), 0)
    qz = np.where(pos_t, (m[..., 1, 0] - m[..., 0, 1]) / np.where(s == 0, 1, s), 0)
    # Fallback: pick the largest diagonal.
    not_t = ~pos_t
    d0 = (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2]) & not_t
    d1 = (m[..., 1, 1] > m[..., 0, 0]) & (m[..., 1, 1] >= m[..., 2, 2]) & not_t
    d2 = not_t & ~d0 & ~d1
    s0 = np.sqrt(np.maximum(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-12)) * 2
    s1 = np.sqrt(np.maximum(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], 1e-12)) * 2
    s2 = np.sqrt(np.maximum(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], 1e-12)) * 2
    qw = np.where(d0, (m[..., 2, 1] - m[..., 1, 2]) / s0, qw)
    qx = np.where(d0, 0.25 * s0, qx)
    qy = np.where(d0, (m[..., 0, 1] + m[..., 1, 0]) / s0, qy)
    qz = np.where(d0, (m[..., 0, 2] + m[..., 2, 0]) / s0, qz)
    qw = np.where(d1, (m[..., 0, 2] - m[..., 2, 0]) / s1, qw)
    qx = np.where(d1, (m[..., 0, 1] + m[..., 1, 0]) / s1, qx)
    qy = np.where(d1, 0.25 * s1, qy)
    qz = np.where(d1, (m[..., 1, 2] + m[..., 2, 1]) / s1, qz)
    qw = np.where(d2, (m[..., 1, 0] - m[..., 0, 1]) / s2, qw)
    qx = np.where(d2, (m[..., 0, 2] + m[..., 2, 0]) / s2, qx)
    qy = np.where(d2, (m[..., 1, 2] + m[..., 2, 1]) / s2, qy)
    qz = np.where(d2, 0.25 * s2, qz)
    q[..., 0] = qw
    q[..., 1] = qx
    q[..., 2] = qy
    q[..., 3] = qz
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


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


def convert_actions(action: np.ndarray) -> np.ndarray:
    """(N, 7) [px,py,pz,qw,qx,qy,qz] -> (N, 9) [px,py,pz,rot6d]."""
    pos = action[:, :3]
    quat_wxyz = action[:, 3:7]
    R = quat_wxyz_to_matrix(quat_wxyz)
    rot6d = matrix_to_rot6d(R)
    return np.concatenate([pos, rot6d], axis=-1).astype(np.float32)


def convert_states(state: np.ndarray) -> np.ndarray:
    """(N, 44) with quaternion in xyzw order at indices 3..6 -> (N, 46) with rot6d.

    The trailing 12 dims (task_id_one_hot.0..11) ride along via the
    `state[:, 7:]` tail slice — unaffected by the rot6d rewrite.
    """
    quat_xyzw = state[:, 3:7]
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    R = quat_wxyz_to_matrix(quat_wxyz)
    rot6d = matrix_to_rot6d(R)
    return np.concatenate([state[:, :3], rot6d, state[:, 7:]], axis=-1).astype(np.float32)


def roundtrip_check(action_orig: np.ndarray, action_new: np.ndarray) -> None:
    """Recover R from rot6d, recover quat, compare to original quat (up to sign)."""
    rot6d = action_new[:, 3:9]
    R = rot6d_to_matrix(rot6d)
    q_back = matrix_to_quat_wxyz(R)
    q_orig = action_orig[:, 3:7]
    q_orig = q_orig / np.linalg.norm(q_orig, axis=-1, keepdims=True)
    err_pos = np.linalg.norm(q_back - q_orig, axis=-1)
    err_neg = np.linalg.norm(q_back + q_orig, axis=-1)
    err = np.minimum(err_pos, err_neg)
    print(f"  action quat round-trip: max={err.max():.3e}  mean={err.mean():.3e}  p99={np.quantile(err, 0.99):.3e}")


def main() -> None:
    if OUT.exists():
        sys.exit(f"output {OUT} already exists; refusing to overwrite")
    if not SRC.is_dir():
        sys.exit(f"source {SRC} does not exist")

    print(f"converting {SRC} -> {OUT}")

    (OUT / "data/chunk-000").mkdir(parents=True)
    (OUT / "meta/episodes/chunk-000").mkdir(parents=True)
    for k in VIDEO_KEYS:
        (OUT / f"videos/{k}/chunk-000").mkdir(parents=True)

    # --- Load source data ---
    src_data_path = SRC / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(src_data_path)
    n = len(data)
    print(f"  frames: {n}")

    action = np.stack(data["action"].to_numpy()).astype(np.float64)
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    assert action.shape == (n, 7), f"unexpected action shape {action.shape}"
    assert state.shape == (n, 44), f"unexpected state shape {state.shape} (expected 44 = 32 base + 12 task one-hot)"

    # --- Convert ---
    action_new = convert_actions(action)
    state_new = convert_states(state)
    roundtrip_check(action, action_new.astype(np.float64))

    # --- Replace columns (each row holds a 1D float32 numpy array) ---
    data["action"] = list(action_new)
    data["observation.state"] = list(state_new)
    data.to_parquet(OUT / "data/chunk-000/file-000.parquet", index=False)
    print(f"  wrote data parquet")

    # --- info.json ---
    info = json.loads((SRC / "meta/info.json").read_text())
    info["features"]["action"] = {
        "dtype": "float32",
        "names": [
            "position.x", "position.y", "position.z",
            "rot6d.0", "rot6d.1", "rot6d.2",
            "rot6d.3", "rot6d.4", "rot6d.5",
        ],
        "shape": [9],
    }
    old_state_names = info["features"]["observation.state"]["names"]
    new_state_names = (
        list(old_state_names[:3])
        + [f"tcp_pose.rot6d.{i}" for i in range(6)]
        + list(old_state_names[7:])
    )
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "names": new_state_names,
        "shape": [46],
    }
    (OUT / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"  wrote info.json (action shape={info['features']['action']['shape']}, "
          f"obs.state shape={info['features']['observation.state']['shape']})")

    # --- stats.json ---
    src_stats = json.loads((SRC / "meta/stats.json").read_text())
    stat_features = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    stat_values: dict[str, np.ndarray] = {}
    for f in stat_features:
        if f == "action":
            stat_values[f] = action_new.astype(np.float64)
        elif f == "observation.state":
            stat_values[f] = state_new.astype(np.float64)
        else:
            stat_values[f] = data[f].to_numpy().astype(np.float64)
    new_stats = aggregate_stats(stat_values)
    # Pass image stats through unchanged.
    for k in VIDEO_KEYS:
        if k in src_stats:
            new_stats[k] = src_stats[k]
    (OUT / "meta/stats.json").write_text(json.dumps(new_stats, indent=4))
    print(f"  wrote stats.json")

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
