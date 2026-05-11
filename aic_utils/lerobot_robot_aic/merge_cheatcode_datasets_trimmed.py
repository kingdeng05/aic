#!/usr/bin/env python3
"""Merge per-episode cheatcode-* LeRobot datasets after trimming trailing HOLD frames.

Trimming rule (per episode):
  motion[i] = ||action_position[i+1] - action_position[i]||_2
  HOLD start = first index k from the end such that motion[i] < EPS for all i >= k
  trimmed_length = min(original_length, HOLD_start + TAIL_BUFFER)

Sources are not modified. Output folder must not already exist.

Usage:
  pixi run python aic_utils/lerobot_robot_aic/merge_cheatcode_datasets_trimmed.py \
      [out_name] [eps_meters] [tail_buffer_frames]
defaults: nic_card_mount_0_merged_trimmed 0.0005 30
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "local"
OUT_NAME = sys.argv[1] if len(sys.argv) > 1 else "nic_card_mount_0_merged_trimmed"
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005       # 0.5 mm
TAIL_BUFFER = int(sys.argv[3]) if len(sys.argv) > 3 else 30      # 1 s @ 30 fps
OUT = ROOT / OUT_NAME

VIDEO_KEYS = [
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
]


def find_sources() -> list[Path]:
    srcs = sorted(
        (p for p in ROOT.glob("cheatcode-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return [
        s for s in srcs
        if (s / "data/chunk-000/file-000.parquet").is_file()
        and (s / "meta/info.json").is_file()
        and (s / "meta/tasks.parquet").is_file()
        and (s / "meta/episodes/chunk-000/file-000.parquet").is_file()
        and all((s / f"videos/{k}/chunk-000/file-000.mp4").is_file() for k in VIDEO_KEYS)
    ]


def find_trim_index(actions: np.ndarray, eps: float, tail_buffer: int) -> int:
    """Return number of frames to keep (length after trim)."""
    pos = actions[:, :3]
    motion = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # length n-1
    # Find last index with significant motion.
    moving = motion >= eps
    if not moving.any():
        # Pathological: no motion at all. Keep at least tail_buffer frames.
        return min(len(actions), tail_buffer)
    last_motion = int(np.where(moving)[0].max()) + 1  # +1: motion[i] is i->i+1, last motion ends at i+1
    return min(len(actions), last_motion + tail_buffer)


def ffmpeg_concat(input_paths: list[Path], output_path: Path) -> None:
    """Lossless concat (no trim, no re-encode). Trimming happens via episode timestamp metadata."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in input_paths:
            f.write(f"file '{p.resolve()}'\n")
        list_path = f.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                str(output_path),
            ],
            check=True,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


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


def aggregate_image_stats(srcs: list[Path], video_keys: list[str]) -> dict:
    out = {}
    for k in video_keys:
        ms, vs, mins, maxs, q01s, q10s, q50s, q90s, q99s, counts = [], [], [], [], [], [], [], [], [], []
        for s in srcs:
            stat = json.loads((s / "meta/stats.json").read_text())[k]
            counts.append(int(np.asarray(stat["count"]).flatten()[0]))
            ms.append(np.asarray(stat["mean"], dtype=np.float64).reshape(-1))
            vs.append(np.asarray(stat["std"], dtype=np.float64).reshape(-1) ** 2)
            mins.append(np.asarray(stat["min"], dtype=np.float64).reshape(-1))
            maxs.append(np.asarray(stat["max"], dtype=np.float64).reshape(-1))
            q01s.append(np.asarray(stat["q01"], dtype=np.float64).reshape(-1))
            q10s.append(np.asarray(stat["q10"], dtype=np.float64).reshape(-1))
            q50s.append(np.asarray(stat["q50"], dtype=np.float64).reshape(-1))
            q90s.append(np.asarray(stat["q90"], dtype=np.float64).reshape(-1))
            q99s.append(np.asarray(stat["q99"], dtype=np.float64).reshape(-1))
        n = np.asarray(counts, dtype=np.float64)
        N = float(n.sum())
        M = np.stack(ms)
        V = np.stack(vs)
        mean = (n[:, None] * M).sum(axis=0) / N
        pooled_var = (n[:, None] * (V + M * M)).sum(axis=0) / N - mean * mean
        std = np.sqrt(np.maximum(pooled_var, 0.0))

        def shape_chw(v: np.ndarray) -> list:
            return v.reshape(3, 1, 1).tolist()

        out[k] = {
            "min": shape_chw(np.stack(mins).min(axis=0)),
            "max": shape_chw(np.stack(maxs).max(axis=0)),
            "mean": shape_chw(mean),
            "std": shape_chw(std),
            "count": [int(N)],
            "q01": shape_chw((n[:, None] * np.stack(q01s)).sum(axis=0) / N),
            "q10": shape_chw((n[:, None] * np.stack(q10s)).sum(axis=0) / N),
            "q50": shape_chw((n[:, None] * np.stack(q50s)).sum(axis=0) / N),
            "q90": shape_chw((n[:, None] * np.stack(q90s)).sum(axis=0) / N),
            "q99": shape_chw((n[:, None] * np.stack(q99s)).sum(axis=0) / N),
        }
    return out


def main() -> None:
    if OUT.exists():
        sys.exit(f"output {OUT} already exists; refusing to overwrite")

    srcs = find_sources()
    if not srcs:
        sys.exit(f"no valid cheatcode-* datasets found in {ROOT}")
    print(f"merging {len(srcs)} episodes -> {OUT}  (eps={EPS} m, tail_buffer={TAIL_BUFFER} frames)")

    (OUT / "data/chunk-000").mkdir(parents=True)
    (OUT / "meta/episodes/chunk-000").mkdir(parents=True)
    for k in VIDEO_KEYS:
        (OUT / f"videos/{k}/chunk-000").mkdir(parents=True)

    shutil.copy2(srcs[0] / "meta/tasks.parquet", OUT / "meta/tasks.parquet")
    info_template = json.loads((srcs[0] / "meta/info.json").read_text())
    fps = int(info_template["fps"])

    # --- Pass 1: trim, rewrite per-episode data, build episode rows ---
    # Videos are concatenated lossless (full source duration), so video timestamps are based on
    # cumulative *full* source duration; trimming is reflected only in `to_timestamp` per episode
    # and in `dataset_*_index` (which controls how many frames the dataloader can index).
    data_frames: list[pd.DataFrame] = []
    episode_rows: list[pd.DataFrame] = []
    cumulative_frames = 0
    cumulative_video_time = 0.0  # cumulative *full* source duration (for video offsets)
    trim_log = []  # (orig, trimmed, dropped)

    for new_idx, src in enumerate(srcs):
        data = pd.read_parquet(src / "data/chunk-000/file-000.parquet")
        actions = np.stack(data["action"].to_numpy()).astype(np.float64)
        n_orig = len(data)
        n_keep = find_trim_index(actions, EPS, TAIL_BUFFER)
        trim_log.append((n_orig, n_keep, n_orig - n_keep))

        data = data.iloc[:n_keep].copy()
        data["episode_index"] = np.int64(new_idx)
        data["index"] = np.arange(cumulative_frames, cumulative_frames + n_keep, dtype=np.int64)
        data_frames.append(data)

        ep = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
        ep = ep.copy()
        ep["episode_index"] = np.int64(new_idx)
        ep["length"] = np.int64(n_keep)
        ep["dataset_from_index"] = np.int64(cumulative_frames)
        ep["dataset_to_index"] = np.int64(cumulative_frames + n_keep)
        ep["data/chunk_index"] = np.int64(0)
        ep["data/file_index"] = np.int64(0)
        full_duration = n_orig / fps
        kept_duration = n_keep / fps
        for k in VIDEO_KEYS:
            ep[f"videos/{k}/chunk_index"] = np.int64(0)
            ep[f"videos/{k}/file_index"] = np.int64(0)
            ep[f"videos/{k}/from_timestamp"] = float(cumulative_video_time)
            ep[f"videos/{k}/to_timestamp"] = float(cumulative_video_time + kept_duration)
        ep["meta/episodes/chunk_index"] = np.int64(0)
        ep["meta/episodes/file_index"] = np.int64(0)
        episode_rows.append(ep)

        cumulative_frames += n_keep
        cumulative_video_time += full_duration

    # --- Print trim distribution before any heavy work ---
    arr = np.array(trim_log)
    orig_total = int(arr[:, 0].sum())
    kept_total = int(arr[:, 1].sum())
    drop_total = int(arr[:, 2].sum())
    print(
        f"  total frames: orig={orig_total} kept={kept_total} dropped={drop_total} "
        f"({100 * drop_total / orig_total:.1f}%)"
    )
    drops = arr[:, 2]
    print(
        f"  per-episode dropped frames -- min={drops.min()} median={int(np.median(drops))} "
        f"mean={drops.mean():.1f} max={drops.max()}"
    )
    # show a few extremes
    by_drop = sorted(range(len(srcs)), key=lambda i: trim_log[i][2])
    print("  smallest trims:", [(srcs[i].name, trim_log[i]) for i in by_drop[:3]])
    print("  largest  trims:", [(srcs[i].name, trim_log[i]) for i in by_drop[-3:]])

    print(f"  writing combined data parquet: {cumulative_frames} frames")
    pd.concat(data_frames, ignore_index=True).to_parquet(OUT / "data/chunk-000/file-000.parquet", index=False)
    pd.concat(episode_rows, ignore_index=True).to_parquet(OUT / "meta/episodes/chunk-000/file-000.parquet", index=False)

    # --- Concat full source videos lossless (-c copy); trimming is in metadata ---
    for k in VIDEO_KEYS:
        srcs_mp4 = [s / f"videos/{k}/chunk-000/file-000.mp4" for s in srcs]
        out_mp4 = OUT / f"videos/{k}/chunk-000/file-000.mp4"
        print(f"  concat {len(srcs_mp4)} mp4s -> {out_mp4.relative_to(OUT)}")
        ffmpeg_concat(srcs_mp4, out_mp4)

    # --- Stats from trimmed data ---
    print("  computing aggregate stats")
    combined = pd.concat(data_frames, ignore_index=True)
    stat_features = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    stat_values: dict[str, np.ndarray] = {}
    for f in stat_features:
        if f in ("action", "observation.state"):
            stat_values[f] = np.stack(combined[f].to_numpy()).astype(np.float64)
        else:
            stat_values[f] = combined[f].to_numpy().astype(np.float64)
    stats = aggregate_stats(stat_values)
    # Note: image stats still aggregated from source (per-episode stats; trimming a small tail of
    # near-static frames barely shifts pixel stats — acceptable approximation).
    stats.update(aggregate_image_stats(srcs, VIDEO_KEYS))
    with open(OUT / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    info = info_template.copy()
    info["total_episodes"] = len(srcs)
    info["total_frames"] = cumulative_frames
    info["splits"] = {"train": f"0:{len(srcs)}"}
    with open(OUT / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"done: {len(srcs)} episodes, {cumulative_frames} frames "
          f"(video {cumulative_video_time:.1f}s of source) -> {OUT}")


if __name__ == "__main__":
    main()
