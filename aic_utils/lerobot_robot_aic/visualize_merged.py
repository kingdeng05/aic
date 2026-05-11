#!/usr/bin/env python3
"""Visualize a local LeRobot dataset with rerun, forcing the pyav video backend.

The default `lerobot-dataset-viz` picks torchcodec when the package is importable,
but in this pixi env torchcodec fails at runtime (FFmpeg ABI mismatch). pyav works.

Usage:
    pixi run python aic_utils/lerobot_robot_aic/visualize_merged.py \
        --root /home/sai/.cache/huggingface/lerobot/local/nic_card_mount_0_merged \
        --episode-index 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--repo-id", default=None, help="defaults to local/<root.name>")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--save", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mode", default="local")
    p.add_argument("--grpc-port", type=int, default=9876)
    p.add_argument("--web-port", type=int, default=9090)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    repo_id = args.repo_id or f"local/{args.root.name}"
    logging.basicConfig(level=logging.INFO)

    ds = LeRobotDataset(
        repo_id,
        episodes=[args.episode_index],
        root=args.root,
        video_backend="pyav",
    )
    print(f"loaded {repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames")

    visualize_dataset(
        ds,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
        save=bool(args.save),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
