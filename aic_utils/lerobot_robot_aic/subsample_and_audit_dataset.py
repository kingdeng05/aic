#!/usr/bin/env python3
"""Subsample N episodes from a v3.0 LeRobot dataset and re-run the rot6d audit plots.

The source mp4s are not modified or re-encoded. The new dataset references the
same `videos/.../file-000.mp4` files but the meta keeps only the kept episodes'
`from_timestamp`/`to_timestamp` ranges, so the loader seeks into the same mp4
but at the kept episodes' positions only.

Usage:
  pixi run python aic_utils/lerobot_robot_aic/subsample_and_audit_dataset.py \
      [src_name] [out_name] [n_keep] [seed] [diag_dir]
defaults:
  src_name  = nic_card_mount_0_merged_trimmed_rot6d_slim
  out_name  = nic_card_mount_0_merged_trimmed_rot6d_slim_50ep
  n_keep    = 50
  seed      = 0
  diag_dir  = <repo>/diagnostics/rot6d_audit_50ep
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "local"
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../src/aic

SRC_NAME = sys.argv[1] if len(sys.argv) > 1 else "nic_card_mount_0_merged_trimmed_rot6d_slim"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "nic_card_mount_0_merged_trimmed_rot6d_slim_50ep"
N_KEEP = int(sys.argv[3]) if len(sys.argv) > 3 else 50
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
DIAG_DIR = Path(sys.argv[5]) if len(sys.argv) > 5 else REPO_ROOT / "diagnostics" / "rot6d_audit_50ep"

SRC = ROOT / SRC_NAME
OUT = ROOT / OUT_NAME

VIDEO_KEYS = [
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
]

ACTION_NAMES = ["px (m)", "py (m)", "pz (m)",
                "rot6d.0", "rot6d.1", "rot6d.2",
                "rot6d.3", "rot6d.4", "rot6d.5"]


def aggregate_scalar_stats(values: dict[str, np.ndarray]) -> dict:
    out = {}
    for feat, cat in values.items():
        if cat.ndim == 1:
            cat = cat[:, None]
        out[feat] = {
            "min":   cat.min(axis=0).tolist(),
            "max":   cat.max(axis=0).tolist(),
            "mean":  cat.mean(axis=0).tolist(),
            "std":   cat.std(axis=0).tolist(),
            "count": [int(cat.shape[0])],
            "q01":   np.quantile(cat, 0.01, axis=0).tolist(),
            "q10":   np.quantile(cat, 0.10, axis=0).tolist(),
            "q50":   np.quantile(cat, 0.50, axis=0).tolist(),
            "q90":   np.quantile(cat, 0.90, axis=0).tolist(),
            "q99":   np.quantile(cat, 0.99, axis=0).tolist(),
        }
    return out


def pool_image_stats(ep_df: pd.DataFrame, video_keys: list[str]) -> dict:
    """Pool per-episode image stats across kept episodes (count-weighted)."""
    out = {}
    for k in video_keys:
        n      = ep_df[f"stats/{k}/count"].apply(lambda v: int(np.asarray(v).flatten()[0])).to_numpy().astype(np.float64)
        means  = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/mean"]])
        stds   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/std"]])
        mins   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/min"]])
        maxs   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/max"]])
        q01s   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/q01"]])
        q10s   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/q10"]])
        q50s   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/q50"]])
        q90s   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/q90"]])
        q99s   = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in ep_df[f"stats/{k}/q99"]])

        N = float(n.sum())
        mean = (n[:, None] * means).sum(axis=0) / N
        var  = stds ** 2
        pooled_var = (n[:, None] * (var + means * means)).sum(axis=0) / N - mean * mean
        std  = np.sqrt(np.maximum(pooled_var, 0.0))

        def chw(v: np.ndarray) -> list:
            return v.reshape(3, 1, 1).tolist()

        out[k] = {
            "min":   chw(mins.min(axis=0)),
            "max":   chw(maxs.max(axis=0)),
            "mean":  chw(mean),
            "std":   chw(std),
            "count": [int(N)],
            "q01":   chw((n[:, None] * q01s).sum(axis=0) / N),
            "q10":   chw((n[:, None] * q10s).sum(axis=0) / N),
            "q50":   chw((n[:, None] * q50s).sum(axis=0) / N),
            "q90":   chw((n[:, None] * q90s).sum(axis=0) / N),
            "q99":   chw((n[:, None] * q99s).sum(axis=0) / N),
        }
    return out


def subsample(src: Path, out: Path, n_keep: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if out.exists():
        sys.exit(f"output {out} already exists; refusing to overwrite")
    if not src.is_dir():
        sys.exit(f"source dataset not found: {src}")

    info = json.loads((src / "meta/info.json").read_text())
    total_eps = int(info["total_episodes"])
    if n_keep > total_eps:
        sys.exit(f"asked for {n_keep} episodes but source only has {total_eps}")

    rng = np.random.default_rng(seed)
    kept = sorted(rng.choice(total_eps, size=n_keep, replace=False).tolist())
    print(f"keeping {n_keep}/{total_eps} episodes (seed={seed}); first 10: {kept[:10]} ... last 5: {kept[-5:]}")

    data = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
    ep   = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")

    # ---- filter rows to kept episodes ----
    keep_mask = data["episode_index"].isin(kept)
    data = data[keep_mask].reset_index(drop=True)
    ep = ep[ep["episode_index"].isin(kept)].sort_values("episode_index").reset_index(drop=True)
    assert len(ep) == n_keep

    # ---- renumber episodes 0..n_keep-1 ----
    old_to_new = {old: new for new, old in enumerate(kept)}
    data["episode_index"] = data["episode_index"].map(old_to_new).astype(np.int64)
    ep["episode_index"]   = ep["episode_index"].map(old_to_new).astype(np.int64)

    # ---- renumber global `index` (0..total-1) and per-episode `dataset_*_index` ----
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    data["index"] = np.arange(len(data), dtype=np.int64)

    # build new dataset_from/to_index from the (renumbered) episode lengths
    cum = 0
    from_idx, to_idx = [], []
    for length in ep["length"].astype(np.int64):
        from_idx.append(cum)
        to_idx.append(cum + int(length))
        cum += int(length)
    ep["dataset_from_index"] = np.asarray(from_idx, dtype=np.int64)
    ep["dataset_to_index"]   = np.asarray(to_idx,   dtype=np.int64)
    # data/chunk_index, data/file_index, videos/.../chunk_index, file_index, from/to_timestamp
    # remain valid because we keep the source mp4 + parquet files unchanged.
    ep["data/chunk_index"] = np.int64(0)
    ep["data/file_index"]  = np.int64(0)
    ep["meta/episodes/chunk_index"] = np.int64(0)
    ep["meta/episodes/file_index"]  = np.int64(0)

    # ---- write new layout ----
    (out / "data/chunk-000").mkdir(parents=True)
    (out / "meta/episodes/chunk-000").mkdir(parents=True)
    for k in VIDEO_KEYS:
        (out / f"videos/{k}/chunk-000").mkdir(parents=True)
        # hard-link the mp4 to avoid duplicating ~700MB; fall back to copy.
        src_mp4 = src / f"videos/{k}/chunk-000/file-000.mp4"
        dst_mp4 = out / f"videos/{k}/chunk-000/file-000.mp4"
        try:
            dst_mp4.hardlink_to(src_mp4)
        except OSError:
            shutil.copy2(src_mp4, dst_mp4)

    shutil.copy2(src / "meta/tasks.parquet", out / "meta/tasks.parquet")
    data.to_parquet(out / "data/chunk-000/file-000.parquet", index=False)
    ep.to_parquet(out / "meta/episodes/chunk-000/file-000.parquet", index=False)

    # ---- recompute scalar stats from kept frames; pool image stats from kept eps ----
    stat_features = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    stat_values: dict[str, np.ndarray] = {}
    for f in stat_features:
        if f in ("action", "observation.state"):
            stat_values[f] = np.stack(data[f].to_numpy()).astype(np.float64)
        else:
            stat_values[f] = data[f].to_numpy().astype(np.float64)
    stats = aggregate_scalar_stats(stat_values)
    stats.update(pool_image_stats(ep, VIDEO_KEYS))
    (out / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    new_info = info.copy()
    new_info["total_episodes"] = n_keep
    new_info["total_frames"]   = int(len(data))
    new_info["splits"]         = {"train": f"0:{n_keep}"}
    (out / "meta/info.json").write_text(json.dumps(new_info, indent=4))

    print(f"wrote subsampled dataset: {n_keep} episodes, {len(data)} frames -> {out}")
    return data, ep, new_info


def audit(data: pd.DataFrame, ep: pd.DataFrame, info: dict, diag_dir: Path) -> None:
    diag_dir.mkdir(parents=True, exist_ok=True)
    fps = int(info["fps"])
    actions = np.stack(data["action"].to_numpy()).astype(np.float64)   # (N, 9)
    states  = np.stack(data["observation.state"].to_numpy()).astype(np.float64)  # (N, 12)
    n_eps = int(info["total_episodes"])
    cmap  = plt.get_cmap("viridis")

    # per-episode index slices (in the subsampled dataset frames)
    slices = []
    for new_idx in range(n_eps):
        m = data["episode_index"].to_numpy() == new_idx
        idxs = np.where(m)[0]
        slices.append(idxs)

    # ===== 01: action histograms =====
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for i, ax in enumerate(axes.flat):
        v = actions[:, i]
        std = float(v.std())
        rng = float(v.max() - v.min())
        mean = float(v.mean())
        ax.hist(v, bins=60, color="steelblue")
        ax.axvline(mean, color="red", ls="--", label=f"mean={mean:.4f}")
        ax.set_title(f"{ACTION_NAMES[i]}    std={std:.4f}    range={rng:.4f}")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Action histograms — note rot6d dims with tiny std (MEAN_STD blow-up)")
    fig.tight_layout()
    fig.savefig(diag_dir / "01_action_hist.png", dpi=120)
    plt.close(fig)

    # ===== 02: action xyz trajectories (colored by episode) =====
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = [("top-down (x,y)", 0, 1), ("side (x,z)", 0, 2), ("side (y,z)", 1, 2)]
    for ax, (title, a, b) in zip(axes, titles):
        for ei, idxs in enumerate(slices):
            color = cmap(ei / max(n_eps - 1, 1))
            ax.plot(actions[idxs, a], actions[idxs, b], color=color, lw=0.5, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("p" + "xyz"[a]); ax.set_ylabel("p" + "xyz"[b])
        ax.grid(alpha=0.3)
    fig.suptitle("Action xyz trajectories — colour by episode")
    fig.tight_layout()
    fig.savefig(diag_dir / "02_action_xyz_trajectories.png", dpi=120)
    plt.close(fig)

    # ===== 03: episode means vs frames =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, a, b) in zip(axes, [("top-down: episode means / starts vs all frames", 0, 1),
                                        ("side: episode means / starts vs all frames",     0, 2)]):
        ax.scatter(actions[:, a], actions[:, b], s=2, color="lightgray", alpha=0.4, label="all frames")
        means_a, means_b, start_a, start_b = [], [], [], []
        for idxs in slices:
            means_a.append(float(actions[idxs, a].mean()))
            means_b.append(float(actions[idxs, b].mean()))
            start_a.append(float(actions[idxs[0], a]))
            start_b.append(float(actions[idxs[0], b]))
        ax.scatter(means_a, means_b, s=22, color="red", label="episode mean")
        ax.scatter(start_a, start_b, s=24, marker="x", color="green", label="episode start")
        ax.set_title(title); ax.set_xlabel("p" + "xyz"[a]); ax.set_ylabel("p" + "xyz"[b])
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("if red dots cluster at one point, vision can't recover episode identity from xy alone")
    fig.tight_layout()
    fig.savefig(diag_dir / "03_episode_means_vs_frames.png", dpi=120)
    plt.close(fig)

    # ===== 04: z over time + distance to endpoint =====
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ei, idxs in enumerate(slices):
        color = cmap(ei / max(n_eps - 1, 1))
        t = np.arange(len(idxs)) / fps
        axes[0].plot(t, actions[idxs, 2], color=color, lw=0.6, alpha=0.7)
        endpoint = actions[idxs[-1], :3]
        d = np.linalg.norm(actions[idxs, :3] - endpoint, axis=1) * 1000.0  # mm
        axes[1].plot(t, d, color=color, lw=0.6, alpha=0.7)
    axes[0].set_title("z over time per episode (descent profile)")
    axes[0].set_ylabel("pz (m)"); axes[0].grid(alpha=0.3)
    axes[1].set_title("distance from each frame to its episode's final pose")
    axes[1].set_ylabel("‖xyz - endpoint‖ (mm)")
    axes[1].set_xlabel("seconds since episode start")
    axes[1].axhline(20, color="red",    ls="--", label="20 mm")
    axes[1].axhline(5,  color="orange", ls="--", label="5 mm")
    axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "04_z_over_time_and_dist_to_endpoint.png", dpi=120)
    plt.close(fig)

    # ===== 05: rot6d histograms (state has no rot6d in the slim variant; fall back to action.rot6d) =====
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, ax in enumerate(axes.flat):
        v = actions[:, 3 + i]
        std = float(v.std()); mean = float(v.mean())
        ax.hist(v, bins=60, color="orange")
        ax.axvline(mean, color="red", ls="--")
        ax.set_title(f"obs.state.rot6d.{i}    std={std:.4f}")
    fig.suptitle("Observation.state rot6d histograms (TCP orientation across whole dataset) — slim state has no rot6d, showing action.rot6d as proxy")
    fig.tight_layout()
    fig.savefig(diag_dir / "05_state_rot6d_hist.png", dpi=120)
    plt.close(fig)

    # ===== 06: step size distributions =====
    dxyz_mm, drot6d = [], []
    for idxs in slices:
        if len(idxs) < 2:
            continue
        a = actions[idxs]
        dxyz_mm.append(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1) * 1000.0)
        drot6d.append(np.linalg.norm(np.diff(a[:, 3:9], axis=0), axis=1))
    dxyz_mm = np.concatenate(dxyz_mm)
    drot6d  = np.concatenate(drot6d)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(dxyz_mm, bins=120, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set_title(f"xyz step  median={np.median(dxyz_mm):.2f} mm  p99={np.quantile(dxyz_mm, 0.99):.2f} mm")
    axes[0].set_xlabel("‖Δxyz‖ between consecutive frames (mm)"); axes[0].set_ylabel("count"); axes[0].grid(alpha=0.3)
    axes[1].hist(drot6d, bins=120, color="orange")
    axes[1].set_yscale("log")
    axes[1].set_title(f"rot6d step  median={np.median(drot6d):.5f}  p99={np.quantile(drot6d, 0.99):.5f}")
    axes[1].set_xlabel("‖Δrot6d‖ between consecutive frames"); axes[1].set_ylabel("count"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "06_step_size_distribution.png", dpi=120)
    plt.close(fig)

    # ===== 07: episode lengths =====
    lengths = ep["length"].to_numpy().astype(np.int64)
    durations = lengths / fps
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(durations, bins=24, color="seagreen")
    ax.set_title(f"episode lengths  n={n_eps}  total={int(lengths.sum())} frames  median={np.median(durations):.1f}s")
    ax.set_xlabel("episode duration (s)"); ax.set_ylabel("count"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "07_episode_lengths.png", dpi=120)
    plt.close(fig)

    print(f"wrote 7 plots -> {diag_dir}")
    # silence unused-state warning
    _ = states


def main() -> None:
    data, ep, info = subsample(SRC, OUT, N_KEEP, SEED)
    audit(data, ep, info, DIAG_DIR)


if __name__ == "__main__":
    main()
