#!/usr/bin/env python
"""Download a LeRobot dataset into the local HF cache."""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


def download(repo_id: str, revision: str | None, force: bool) -> Path:
    local_dir = HF_LEROBOT_HOME / repo_id
    if local_dir.exists() and not force:
        info_path = local_dir / "meta" / "info.json"
        if info_path.exists():
            print(f"[skip] already present at {local_dir}")
            return local_dir

    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )
    return local_dir


def summarize(local_dir: Path) -> None:
    info_path = local_dir / "meta" / "info.json"
    if not info_path.exists():
        print("[warn] no meta/info.json found")
        return
    info = json.loads(info_path.read_text())
    print("---")
    print(f"codebase_version : {info.get('codebase_version')}")
    print(f"robot_type       : {info.get('robot_type')}")
    print(f"episodes         : {info.get('total_episodes')}")
    print(f"frames           : {info.get('total_frames')}")
    print(f"fps              : {info.get('fps')}")
    feats = info.get("features", {})
    a = feats.get("action", {}).get("shape")
    s = feats.get("observation.state", {}).get("shape")
    cams = [k for k in feats if k.startswith("observation.images.")]
    print(f"action shape     : {a}")
    print(f"state shape      : {s}")
    print(f"cameras          : {cams}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id", help="e.g. kingdeng05/aic-insert-demo-5-joints")
    p.add_argument("--revision", default=None,
                   help="git tag/branch/sha. Omit to let HF resolve main.")
    p.add_argument("--force", action="store_true",
                   help="re-download even if cache is populated")
    args = p.parse_args()

    local_dir = download(args.repo_id, args.revision, args.force)
    summarize(local_dir)
    print(f"\nready at: {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
