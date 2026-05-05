#!/usr/bin/env python3
"""Export one camera from a LeRobot dataset to a single MP4 (all frames, all episodes)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def chw_float_to_bgr_uint8(t: torch.Tensor):
    rgb = (t * 255).clamp(0, 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_episode_label(frame_bgr: np.ndarray, episode_index: int) -> None:
    """In-place: top-left label with dark backing for readability."""
    label = f"episode {episode_index}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    x0, y0 = 8, 8
    x1, y1 = x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (32, 32, 32), -1)
    cv2.putText(
        frame_bgr,
        label,
        (x0 + pad, y0 + th + pad),
        font,
        scale,
        (240, 240, 240),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--camera-key",
        default=None,
        help="Feature key (e.g. observation.images.cam_center). Default: first key containing 'center'.",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument(
        "--no-episode-overlay",
        action="store_true",
        help="Do not draw episode index on frames.",
    )
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.root)
    keys = ds.meta.camera_keys
    if args.camera_key is not None:
        cam = args.camera_key
        if cam not in keys:
            raise SystemExit(f"--camera-key must be one of:\n{keys}")
    else:
        found = [k for k in keys if "center" in k.lower()]
        cam = found[0] if found else keys[0]

    print("camera_keys:", keys)
    print("using:", cam)

    sample = ds[0][cam]
    h, w = int(sample.shape[1]), int(sample.shape[2])
    fps = float(ds.fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open; check OpenCV and path.")

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    for batch in tqdm(loader, desc="encoding"):
        imgs = batch[cam]
        ep = batch["episode_index"]
        for i in range(imgs.shape[0]):
            frame = chw_float_to_bgr_uint8(imgs[i])
            if not args.no_episode_overlay:
                draw_episode_label(frame, int(ep[i].item()))
            writer.write(frame)
    writer.release()
    print("Wrote", args.output.resolve())


if __name__ == "__main__":
    main()
