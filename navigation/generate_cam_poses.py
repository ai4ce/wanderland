#!/usr/bin/env python3
"""
Generate per-episode camera pose sequences for UrbanSim VLN navigation data.

For a given scene id, this script loads the generated navigation episodes,
filters out under/over-length paths, determines key turning points based on
the reference path, and produces a sequence of camera poses (position +
quaternion) suitable for first-person rendering.

Key behaviours
--------------
* Paths with <=5 gt locations AND <=2 reference waypoints are skipped.
* Paths with >9 reference waypoints AND >300 gt locations are skipped.
* Camera positions follow gt_locations with a fixed vertical offset
  (cam_height, default 1.2m, subtracting along +Y).
* Between keypoints (matched from reference_path), the camera translates while
  keeping its current yaw. At a keypoint the camera produces an in-place turn:
  first a frame with the incoming heading, then one or more frames rotating
  toward the next segment, each yaw step limited to max_turn_deg (≤90°).
* If no keypoints are matched (excluding start/end), the episode is skipped.
* The first frame uses the episode's provided start_rotation; the second frame
  (if possible) keeps the same position but looks toward the next waypoint.
* Output is a JSON per episode at nav/<out-subdir>/episode_<id>.json unless
  --dry-run is specified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Coordinate convention: X/Z horizontal plane, -Y is upward.
UP_WORLD = np.array([0.0, -1.0, 0.0], dtype=np.float64)


# ---------- Quaternion / rotation utilities ----------

def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w] (Hamilton, scalar last)."""
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def quat_to_forward(q_xyzw: np.ndarray) -> np.ndarray:
    """Return horizontal forward vector (XZ plane) for camera quaternion."""
    x, y, z, w = q_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    fwd = -R[:, 2]
    fwd[1] = 0.0
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return fwd / n


def yaw_from_forward(fwd: np.ndarray) -> float:
    """Map forward vector on XZ plane to yaw angle (rad) using -Z forward convention."""
    return math.atan2(fwd[0], -fwd[2])


def forward_from_yaw(theta: float) -> np.ndarray:
    """Return forward unit vector on XZ for yaw theta."""
    return np.array([math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float64)


def wrap_to_pi(val: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (val + math.pi) % (2.0 * math.pi) - math.pi


def lookat_quat_yaw_only(
    current: np.ndarray,
    target: np.ndarray,
    *,
    up: np.ndarray = UP_WORLD,
    prev_q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate quaternion looking from current to target, clamping pitch to zero."""
    v = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    v[1] = 0.0
    n = np.linalg.norm(v)
    if n < 1e-6:
        if prev_q is not None:
            return prev_q
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    v /= n
    z = -v
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        if prev_q is not None:
            return prev_q
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    x /= nx
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return rotmat_to_quat_xyzw(R)


def split_yaw_turns(
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    max_step_deg: float,
) -> List[np.ndarray]:
    """Split yaw rotation from q_from to q_to into <= max_step_deg increments."""
    f_old = quat_to_forward(q_from)
    f_new = quat_to_forward(q_to)
    th_old = yaw_from_forward(f_old)
    th_new = yaw_from_forward(f_new)
    delta = wrap_to_pi(th_new - th_old)
    if abs(delta) < 1e-6:
        return [q_to]
    max_step = math.radians(max(1e-3, max_step_deg))
    steps = max(1, int(math.ceil(abs(delta) / max_step)))
    step_angle = delta / steps
    result: List[np.ndarray] = []
    for k in range(1, steps + 1):
        th_k = th_old + step_angle * k
        fwd = forward_from_yaw(th_k)
        z = -fwd
        x = np.cross(UP_WORLD, z)
        nx = np.linalg.norm(x)
        if nx < 1e-6:
            result.append(result[-1] if result else q_from)
            continue
        x /= nx
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)
        result.append(rotmat_to_quat_xyzw(R))
    return result


# ---------- Helpers ----------

def match_keypoint_indices(
    locs: np.ndarray,
    ref_path: Sequence[Sequence[float]],
    *,
    radius: float,
) -> tuple[List[int], bool]:
    """Match reference path points to nearest gt_location indices within radius.

    Returns indices (possibly empty) and a flag indicating whether every
    reference point was matched within the radius.
    """
    if not ref_path:
        return [], True
    locs_np = np.asarray(locs, dtype=np.float64)
    indices: List[int] = []
    radius_sq = radius * radius
    all_matched = True
    for ref in ref_path:
        ref_np = np.asarray(ref, dtype=np.float64)
        d_sq = np.sum((locs_np - ref_np) ** 2, axis=1)
        idx = int(np.argmin(d_sq))
        if d_sq[idx] <= radius_sq:
            indices.append(idx)
        else:
            all_matched = False
            break
    return sorted(set(indices)), all_matched


def should_skip_episode(episode: dict) -> bool:
    """Apply length-based filters to skip unsuitable episodes."""
    locs = episode.get("gt_locations", [])
    ref = episode.get("reference_path", [])
    if len(locs) <= 5 and len(ref) <= 2:
        return True
    if len(ref) > 9 and len(locs) > 300:
        return True
    return False


def generate_cam_sequence(
    episode: dict,
    *,
    cam_height: float,
    key_radius: float,
    max_turn_deg: float,
) -> Optional[List[dict]]:
    """Generate camera pose sequence for a single episode."""
    if should_skip_episode(episode):
        return None

    locs = np.asarray(episode["gt_locations"], dtype=np.float64)
    if len(locs) < 2:
        return None

    positions = locs.copy()
    positions[:, 1] -= cam_height

    ref_path = episode.get("reference_path", [])
    key_idxs_all, matched_all = match_keypoint_indices(locs, ref_path, radius=key_radius)
    if not matched_all:
        return None
    N = len(positions)

    key_idxs = [idx for idx in key_idxs_all if 0 < idx < N - 1]
    key_set = set(key_idxs)

    frames: List[dict] = []

    q0 = np.asarray(episode["start_rotation"], dtype=np.float64)
    q0 /= np.linalg.norm(q0) + 1e-12
    frames.append(
        {
            "position": positions[0].tolist(),
            "quaternion_xyzw": q0.tolist(),
            "note": "start_given",
        }
    )

    if N >= 2:
        q1 = lookat_quat_yaw_only(positions[0], positions[1], prev_q=q0)
        frames.append(
            {
                "position": positions[0].tolist(),
                "quaternion_xyzw": q1.tolist(),
                "note": "start_look_next",
            }
        )
        q_cur = q1
        start_idx = 1
    else:
        q_cur = q0
        start_idx = 0
    for i in range(start_idx, N):
        if i in key_set:
            frames.append(
                {
                    "position": positions[i].tolist(),
                    "quaternion_xyzw": q_cur.tolist(),
                    "note": "key_pre_turn",
                }
            )
            if i + 1 < N:
                q_target = lookat_quat_yaw_only(positions[i], positions[i + 1], prev_q=q_cur)
                turn_steps = split_yaw_turns(
                    q_cur,
                    q_target,
                    max_step_deg=max_turn_deg,
                )
                if turn_steps:
                    for q_step in turn_steps[:-1]:
                        frames.append(
                            {
                                "position": positions[i].tolist(),
                                "quaternion_xyzw": q_step.tolist(),
                                "note": "key_turn_step",
                            }
                        )
                    frames.append(
                        {
                            "position": positions[i].tolist(),
                            "quaternion_xyzw": turn_steps[-1].tolist(),
                            "note": "key_post_turn",
                        }
                    )
                    q_cur = turn_steps[-1]
            continue

        frames.append(
            {
                "position": positions[i].tolist(),
                "quaternion_xyzw": q_cur.tolist(),
                "note": "move",
            }
        )

    return frames


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate camera pose sequences from VLN episodes for a scene."
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene identifier (e.g., 1EoD__hPvywH8eDKTNzpjbHTAqI0nKXol)",
    )
    parser.add_argument(
        "--paths-root",
        default="paths",
        help="Root directory containing per-scene navigation outputs.",
    )
    parser.add_argument(
        "--out-subdir",
        default="cam_paths",
        help="Subdirectory under nav/ where pose sequences are saved.",
    )
    parser.add_argument(
        "--key-radius",
        type=float,
        default=0.75,
        help="Matching radius (meters) when mapping reference_path points to gt_locations.",
    )
    parser.add_argument(
        "--cam-height",
        type=float,
        default=1.2,
        help="Camera height offset applied along +Y (positions[:,1] -= cam_height).",
    )
    parser.add_argument(
        "--max-turn-deg",
        type=float,
        default=10.0,
        help="Maximum yaw change per frame at keypoints (degrees). Smaller values yield slower turns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output files; only report statistics.",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of episode ids to process (others skipped).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_nav_dir = Path(args.paths_root) / args.scene / "nav"
    episodes_path = scene_nav_dir / "episodes.json"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"episodes.json not found at {episodes_path}")

    with episodes_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    if not episodes:
        print(f"[warn] No episodes found for scene {args.scene}")
        return

    target_ids = set(args.episode_ids) if args.episode_ids is not None else None

    out_dir = scene_nav_dir / args.out_subdir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    for episode in episodes:
        ep_id = episode.get("episode_id")
        if target_ids is not None and ep_id not in target_ids:
            continue

        seq = generate_cam_sequence(
            episode,
            cam_height=args.cam_height,
            key_radius=args.key_radius,
            max_turn_deg=args.max_turn_deg,
        )
        if seq is None:
            skipped += 1
            continue

        kept += 1
        if args.dry_run:
            print(f"[dry-run] scene={args.scene}, episode={ep_id}, frames={len(seq)}")
            continue

        out_path = out_dir / f"episode_{ep_id:04d}.json"
        payload = {
            "scene_id": episode.get("scene_id"),
            "episode_id": ep_id,
            "frames": seq,
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"[write] scene={args.scene}, episode={ep_id}, frames={len(seq)}, path= {out_path}")

    print(
        f"[done] scene={args.scene}, written={kept}, skipped={skipped}, "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
