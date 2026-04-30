"""Drop near-static, no-contact frames from a LeRobot joint-action dataset.

A frame is DROPPED only if all three are true:
  - ||action[t] - joints[t]|| < delta_min                (no committed motion)
  - |tared F.z|  < force_keep_min                        (no axial contact)
  - max |tared torque|  < torque_keep_min                (no rotational contact)

This keeps every frame the demonstrator was committing to action OR feeling
contact, and drops only "I'm sitting here thinking" hesitation frames. First
and last few frames per episode are always kept so episode boundaries have
clean anchors.

Assumes:
  - 32-dim observation.state with joints at indices [19:26], wrench at [26:32]
    (force xyz then torque xyz).
  - 7-dim action in joint positions (alphabetical, matches state[19:26] order).
  - Episodes are wholly contained within a single data parquet file.

Usage:
    python filter_static_frames.py \\
        --src kingdeng05/aic-insert-demo-6-joints \\
        --dst kingdeng05/aic-insert-demo-6-joints-filtered
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREFIX = Path.home() / ".cache" / "huggingface" / "lerobot"

TCP_Z_INDEX = 2
JOINT_SLICE = slice(19, 26)
FORCE_SLICE = slice(26, 29)
TORQUE_SLICE = slice(29, 32)


def resolve(repo_id: str, prefix: Path) -> Path:
    if "/" not in repo_id:
        sys.exit(f"error: expected '<user>/<repo>', got '{repo_id}'")
    return prefix / repo_id


def filter_dataset(
    src: Path,
    dst: Path,
    delta_min: float,
    force_keep_min: float,
    torque_keep_min: float,
    always_keep_n: int,
    insertion_z_max: float,
    insertion_delta_min: float,
) -> None:
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"error: {src} is not a LeRobot dataset")
    if dst.exists():
        print(f"removing {dst}")
        shutil.rmtree(dst)
    print(f"copying {src} -> {dst}")
    shutil.copytree(src, dst)

    data_files = sorted((dst / "data").glob("chunk-*/file-*.parquet"))
    print(f"\nfiltering {len(data_files)} data parquet(s)")

    # Phase 1: per-file row filter, re-index frame_index per episode.
    # Episode contiguous-row check happens on the filtered slice.
    per_file_dfs = []
    total_in = 0
    total_out = 0
    per_episode_stats = []  # (ep_idx, n_in, n_out, file_path)

    for pf in data_files:
        df = pd.read_parquet(pf)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        states = np.stack(df["observation.state"].values)
        actions = np.stack(df["action"].values)

        joints = states[:, JOINT_SLICE]
        force = states[:, FORCE_SLICE]
        torque = states[:, TORQUE_SLICE]
        tcp_z = states[:, TCP_Z_INDEX]

        delta = np.linalg.norm(actions - joints, axis=1)        # (N,)
        max_force = np.abs(force[:, 2])                          # |F.z|
        max_torque = np.max(np.abs(torque), axis=1)              # |τ| infty-norm

        keep_motion = delta >= delta_min
        keep_contact = (max_force >= force_keep_min) | (max_torque >= torque_keep_min)
        keep_mask = keep_motion | keep_contact

        # Engagement-band override: in the insertion zone (low z), the contact
        # rule keeps every "user fumbling at port" frame — those dilute the
        # action distribution to ~zero and the policy collapses to "stay put"
        # exactly where it needs to commit. Force-drop frames here that aren't
        # actually committing motion, even with contact.
        in_insertion = tcp_z <= insertion_z_max
        not_committing = delta < insertion_delta_min
        force_drop = in_insertion & not_committing
        keep_mask &= ~force_drop

        # Force-keep first/last N of every episode regardless.
        for ep in df["episode_index"].unique():
            idxs = np.where((df["episode_index"] == ep).values)[0]
            keep_mask[idxs[:always_keep_n]] = True
            keep_mask[idxs[-always_keep_n:]] = True

        for ep in df["episode_index"].unique():
            idxs = np.where((df["episode_index"] == ep).values)[0]
            n_in = len(idxs)
            n_out = int(keep_mask[idxs].sum())
            per_episode_stats.append((int(ep), n_in, n_out, pf.name))

        df_out = df[keep_mask].reset_index(drop=True)

        # Re-index frame_index per episode (0..k-1) within this file.
        df_out["frame_index"] = (
            df_out.groupby("episode_index").cumcount().astype("int64")
        )

        total_in += len(df)
        total_out += len(df_out)
        per_file_dfs.append((pf, df_out))

        print(f"  {pf.name}: {len(df):>6,} -> {len(df_out):>6,} rows  "
              f"(kept {100 * len(df_out) / max(1, len(df)):.1f}%)")

    print(f"\nGLOBAL: {total_in:,} -> {total_out:,} rows "
          f"(dropped {total_in - total_out:,}, "
          f"kept {100 * total_out / total_in:.1f}%)")

    # Phase 2: assign global `index` 0..N-1, keep file boundaries fixed.
    cur = 0
    for pf, df in per_file_dfs:
        df["index"] = np.arange(cur, cur + len(df), dtype=np.int64)
        cur += len(df)
        df.to_parquet(pf, index=False)

    # Phase 3: update episode metadata. Need new dataset_from_index /
    # dataset_to_index (global index range), length, etc., per episode.
    print("\nupdating meta/episodes/...")
    # Build episode -> (new length, file_path) map by reading what we just wrote.
    ep_info = {}  # episode_index -> dict
    for pf, df in per_file_dfs:
        for ep in df["episode_index"].unique():
            ep_df = df[df["episode_index"] == ep]
            from_idx = int(ep_df["index"].min())
            to_idx = int(ep_df["index"].max()) + 1  # exclusive
            length = len(ep_df)
            ep_info[int(ep)] = dict(
                length=length,
                dataset_from_index=from_idx,
                dataset_to_index=to_idx,
            )

    em_files = sorted((dst / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    dropped_eps = []
    for emf in em_files:
        em = pd.read_parquet(emf)
        n_before = len(em)
        # Drop empty episodes (length 0 or not in ep_info).
        rows_to_keep = []
        for _, row in em.iterrows():
            ep = int(row["episode_index"])
            if ep not in ep_info or ep_info[ep]["length"] == 0:
                dropped_eps.append(ep)
                continue
            row = row.copy()
            row["length"] = ep_info[ep]["length"]
            row["dataset_from_index"] = ep_info[ep]["dataset_from_index"]
            row["dataset_to_index"] = ep_info[ep]["dataset_to_index"]
            rows_to_keep.append(row)
        if rows_to_keep:
            new_em = pd.DataFrame(rows_to_keep)
            new_em.to_parquet(emf, index=False)
        else:
            new_em = pd.DataFrame(columns=em.columns)
            new_em.to_parquet(emf, index=False)
        print(f"  {emf.name}: {n_before} -> {len(new_em)} episodes")

    if dropped_eps:
        print(f"  dropped empty episodes: {dropped_eps}")

    # Phase 4: update info.json
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    old_total = info["total_frames"]
    old_eps = info["total_episodes"]
    new_eps = old_eps - len(dropped_eps)
    info["total_frames"] = total_out
    info["total_episodes"] = new_eps
    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{new_eps}"
    info_path.write_text(json.dumps(info, indent=4))
    print(f"\ninfo.json: total_frames {old_total:,} -> {total_out:,}, "
          f"total_episodes {old_eps} -> {new_eps}")

    # Phase 5: recompute action stats (others stay; lerobot will recompute on
    # train if needed).
    all_actions = np.concatenate(
        [np.stack(pd.read_parquet(pf)["action"].values) for pf, _ in per_file_dfs]
    )
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    else:
        stats = {}
    std = all_actions.std(axis=0)
    stats["action"] = dict(
        mean=all_actions.mean(axis=0).tolist(),
        std=std.tolist() if std.min() > 0 else (std + 1e-6).tolist(),
        min=all_actions.min(axis=0).tolist(),
        max=all_actions.max(axis=0).tolist(),
        count=[int(len(all_actions))],
    )
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"\naction stats refreshed: count={len(all_actions):,}")
    print(f"  mean: {[round(v, 4) for v in stats['action']['mean']]}")
    print(f"  std : {[round(v, 4) for v in stats['action']['std']]}")

    # Phase 6: per-episode summary
    print("\n=== per-episode result ===")
    print(f"{'ep':>3}  {'n_in':>6}  {'n_out':>6}  kept%")
    by_ep = {}
    for ep, n_in, n_out, _ in per_episode_stats:
        if ep not in by_ep:
            by_ep[ep] = [0, 0]
        by_ep[ep][0] += n_in
        by_ep[ep][1] += n_out
    for ep in sorted(by_ep.keys()):
        n_in, n_out = by_ep[ep]
        print(f"{ep:>3}  {n_in:>6}  {n_out:>6}  "
              f"{100 * n_out / max(1, n_in):5.1f}%")

    print(f"\nDone. Filtered dataset at: {dst}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="source repo_id")
    p.add_argument("--dst", required=True, help="destination repo_id")
    p.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    p.add_argument("--delta-min", type=float, default=5e-4,
                   help="rad — drop if action-vs-joints L2 below this")
    p.add_argument("--force-keep-min", type=float, default=2.0,
                   help="N — keep if |tared F.z| above this")
    p.add_argument("--torque-keep-min", type=float, default=0.5,
                   help="Nm — keep if max(|tared τ|) above this")
    p.add_argument("--always-keep-n", type=int, default=3,
                   help="always keep first/last N frames of every episode")
    p.add_argument("--insertion-z-max", type=float, default=0.25,
                   help="m — TCP z below this counts as insertion/engagement zone")
    p.add_argument("--insertion-delta-min", type=float, default=1e-3,
                   help="rad — within insertion zone, drop frames below this "
                        "even if contact is present (kills 'fumbling at port' dilution)")
    args = p.parse_args()

    src = resolve(args.src, args.prefix)
    dst = resolve(args.dst, args.prefix)
    filter_dataset(
        src,
        dst,
        delta_min=args.delta_min,
        force_keep_min=args.force_keep_min,
        torque_keep_min=args.torque_keep_min,
        always_keep_n=args.always_keep_n,
        insertion_z_max=args.insertion_z_max,
        insertion_delta_min=args.insertion_delta_min,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
