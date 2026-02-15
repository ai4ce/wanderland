#!/usr/bin/env python3
"""
Generate navigation episodes (draft VLN-CE style) for UrbanSim scenes.

For a single scene (provided via command line), this script:
  - Loads camera poses from the scene's transforms.json
  - Reads image lists from nvs_split and builds eligible frames
  - Strategy (default: origin_extremes): use only val frames; pick the frame
    closest to origin as start and farthest as goal; also add the reversed pair
    as a second episode when distinct
  - Writes per-scene episodes to the specified output directory (defaults to
    the pinhole scene's own nav folder)

Notes
  - This is an initial random sampler. Customize `sample_pairs` for better strategies
    (e.g., distance constraints, visibility checks, graph-aware sampling).
  - Fields beyond start/goal are populated with simple placeholders to keep the JSON
    self-contained: geodesic_distance is straight-line distance; reference_path and
    gt_locations are [start, goal]; instruction is a templated string; gt_steps is
    distance/step_size rounded up.
  - Paths may be provided explicitly or fall back to project-default roots.
"""

from __future__ import annotations

import json
import math
import os
import random
import argparse
import re
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np


def _load_colmap_module():
    module_path = (
        PROJECT_ROOT
        / "reconstruction"
        / "gsplat"
        / "examples"
        / "datasets"
        / "read_write_model.py"
    )
    loader = SourceFileLoader("_colmap_rw", str(module_path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


colmap_io = _load_colmap_module()


# ===== Default roots (used only when caller supplies a scene identifier instead of a path) =====
# For open-source usage, prefer passing explicit `--pinhole-scene-dir` + `--raw-scene-dir`.
DEFAULT_EXTRACTED_ROOT = Path(os.environ.get("WANDERLAND_EXTRACTED_ROOT", "data/extracted/data"))
DEFAULT_PINHOLE_ROOT = Path(os.environ.get("WANDERLAND_PINHOLE_ROOT", "data/pinhole"))


# ===== Generation config (tweak as needed) =====
EPISODES_PER_SCENE = int(os.environ.get("NAV_EPISODES_PER_SCENE", 20))
RANDOM_SEED = int(os.environ.get("NAV_RANDOM_SEED", 42))
GOAL_RADIUS_METERS = float(os.environ.get("NAV_GOAL_RADIUS", 3.0))
STEP_SIZE_METERS = float(os.environ.get("NAV_STEP_SIZE", 0.25))
OVERWRITE = os.environ.get("NAV_OVERWRITE", "0") == "1"
NAV_STRATEGY = os.environ.get("NAV_STRATEGY", "origin_extremes")  # or "random"


VIEW_ID_PATTERN = re.compile(
    r"(?i)(?:^|[/\\])(?P<cam>left|right)[_/\\]?(?P<stamp>\d+)\.(?P<ext>jpg|jpeg|png)$"
)


def canonicalize_view_id(path: str) -> str | None:
    """Normalize camera file path to form 'left_<timestamp>'."""
    normalized = path.strip().replace("\\", "/")
    match = VIEW_ID_PATTERN.search(normalized)
    if not match:
        return None
    cam = match.group("cam").lower()
    stamp = match.group("stamp")
    return f"{cam}_{stamp}"


# ===== Utilities =====
def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Convert rotation matrix to quaternion [x, y, z, w] (Hamilton, scalar last).

    Args:
        R: (3,3) rotation matrix
    Returns:
        List [x, y, z, w]
    """
    # Robust conversion (Shoemake) with trace branching
    t = np.trace(R)
    if t > 0:
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
    # Normalize
    q /= np.linalg.norm(q) + 1e-12
    return q.tolist()


def load_colmap_poses(pinhole_scene_dir: Path) -> Dict[str, Tuple[List[float], List[float]]]:
    """Load camera poses from COLMAP binaries within the processed scene."""

    def add_from_bin(bin_path: Path, output: Dict[str, Tuple[List[float], List[float]]]):
        if not bin_path.is_file():
            return
        images = colmap_io.read_images_binary(str(bin_path))
        for image in images.values():
            canonical = canonicalize_view_id(image.name)
            if canonical is None:
                continue
            R = colmap_io.qvec2rotmat(image.qvec)
            C = (-R.T @ image.tvec).tolist()
            q = np.array([image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]], dtype=np.float64)
            q /= np.linalg.norm(q) + 1e-12
            output[canonical] = (C, q.tolist())

    poses: Dict[str, Tuple[List[float], List[float]]] = {}
    sparse_root = pinhole_scene_dir / "sparse"
    if not sparse_root.is_dir():
        return poses

    for subdir in sorted(sparse_root.iterdir()):
        if not subdir.is_dir():
            continue
        add_from_bin(subdir / "images.bin", poses)
        add_from_bin(subdir / "images_val.bin", poses)

    # Fallback to root when sparse directories absent but binaries exist directly
    add_from_bin(sparse_root / "images.bin", poses)
    add_from_bin(sparse_root / "images_val.bin", poses)

    return poses


def load_split_frames(scene_dir: Path, splits: Sequence[str] = ("train", "val")) -> List[str]:
    """Load image paths from the requested splits (relative like 'left/123.jpg')."""
    split_dir = scene_dir / "nvs_split"
    frames: List[str] = []
    for split in splits:
        p = split_dir / f"{split}.txt"
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            rel = line.strip()
            if not rel:
                continue
            canonical = canonicalize_view_id(rel)
            if canonical:
                frames.append(canonical)
    # De-duplicate preserving order
    seen = set()
    uniq: List[str] = []
    for f in frames:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def select_origin_extremes(
    frame_ids: Sequence[str], poses: Dict[str, Tuple[List[float], List[float]]]
) -> Tuple[str, str, float]:
    """Pick nearest-to-origin and farthest-from-origin frames and return their IDs and distance."""
    if not frame_ids:
        raise ValueError("No frame_ids provided")
    # Build list of (norm, frame) only for frames with pose
    scored: List[Tuple[float, str]] = []
    for fid in frame_ids:
        if fid in poses:
            pos = np.asarray(poses[fid][0], dtype=np.float64)
            scored.append((float(np.linalg.norm(pos)), fid))
    if len(scored) < 2:
        fid = scored[0][1]
        return fid, fid, 0.0
    scored.sort(key=lambda x: x[0])
    start_fid = scored[0][1]
    goal_fid = scored[-1][1]
    dist = euclidean(poses[start_fid][0], poses[goal_fid][0]) if start_fid != goal_fid else 0.0
    return start_fid, goal_fid, dist


def sample_random_pairs_with_constraints(
    frame_ids: Sequence[str],
    poses: Dict[str, Tuple[List[float], List[float]]],
    rng: random.Random,
    count: int,
    baseline_distance: float | None,
    forbidden_pairs: set[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    if len(frame_ids) < 2 or count <= 0:
        return []

    min_dist = baseline_distance / 3.0 if baseline_distance else 0.0
    sep_threshold = max(baseline_distance / 10.0, 5.0) if baseline_distance else 5.0

    selected: List[Tuple[str, str]] = []
    max_attempts = max(1000, count * 50)
    attempts = 0

    available = list(frame_ids)

    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        s_id, g_id = rng.sample(available, 2)
        if s_id == g_id:
            continue
        pair = (s_id, g_id)
        if pair in forbidden_pairs or (g_id, s_id) in forbidden_pairs:
            continue

        dist = euclidean(poses[s_id][0], poses[g_id][0])
        if dist < min_dist:
            continue

        satisfied = True
        for prev_s, prev_g in selected:
            start_sep = euclidean(poses[s_id][0], poses[prev_s][0])
            goal_sep = euclidean(poses[g_id][0], poses[prev_g][0])
            if start_sep + goal_sep < sep_threshold:
                satisfied = False
                break
        if not satisfied:
            continue

        selected.append(pair)

    return selected


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(va - vb))


def build_episode(
    episode_id: int,
    scene: str,
    start_pose: Tuple[List[float], List[float]],
    goal_pose: Tuple[List[float], List[float]],
) -> Dict:
    start_pos, start_quat = start_pose
    goal_pos, _ = goal_pose

    dist = euclidean(start_pos, goal_pos)
    steps = max(1, int(math.ceil(dist / STEP_SIZE_METERS)))

    return {
        "episode_id": episode_id,
        "scene_id": f"Urbansim/{scene}",
        "start_position": start_pos,
        "start_rotation": start_quat,
        "geodesic_distance": max(dist, 1e-6),  # keep > 0 for downstream
        "goals": [
            {
                "position": goal_pos,
                "radius": GOAL_RADIUS_METERS,
            }
        ],
        # Placeholders to satisfy consumers that expect fields to exist
        "instruction": "Navigate from the start position to the goal position.",
        "reference_path": [start_pos, goal_pos],
        "gt_locations": [start_pos, goal_pos],
        "gt_steps": steps,
    }


def generate_for_scene(
    scene_id: str,
    extracted_scene_dir: Path,
    pinhole_scene_dir: Path,
    output_scene_dir: Path,
    rng: random.Random,
) -> Tuple[int, Path]:
    poses = load_colmap_poses(pinhole_scene_dir)
    # Use only val frames for origin-extremes strategy, else use union
    if NAV_STRATEGY == "origin_extremes":
        split_frames = load_split_frames(extracted_scene_dir, ("val",))
    else:
        split_frames = load_split_frames(extracted_scene_dir, ("train", "val"))
    # Intersect frames with available poses (keep order)
    frame_ids = [f for f in split_frames if f in poses]

    out_dir = output_scene_dir / "nav"
    out_path = out_dir / "episodes.json"

    if out_path.exists() and not OVERWRITE:
        return 0, out_path

    if len(frame_ids) < 2:
        # Not enough frames to form an episode
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"episodes": []}, indent=2), encoding="utf-8")
        return 0, out_path

    episodes: List[Dict] = []
    baseline_distance: float | None = None
    forbidden_pairs: set[Tuple[str, str]] = set()

    if len(frame_ids) >= 1:
        if len(frame_ids) == 1:
            fid = frame_ids[0]
            start_pose = poses[fid]
            goal_pose = poses[fid]
            episodes.append(build_episode(1, scene_id, start_pose, goal_pose))
            baseline_distance = 0.0
        else:
            s_id, g_id, baseline_distance = select_origin_extremes(frame_ids, poses)
            start_pose = poses[s_id]
            goal_pose = poses[g_id]
            episodes.append(build_episode(1, scene_id, start_pose, goal_pose))
            forbidden_pairs.add((s_id, g_id))
            if s_id != g_id:
                episodes.append(build_episode(2, scene_id, goal_pose, start_pose))
                forbidden_pairs.add((g_id, s_id))

    if NAV_STRATEGY != "origin_extremes":
        remaining = max(0, EPISODES_PER_SCENE - len(episodes))
        random_pairs = sample_random_pairs_with_constraints(
            frame_ids,
            poses,
            rng,
            remaining,
            baseline_distance,
            forbidden_pairs,
        )

        eid = len(episodes) + 1
        for s_id, g_id in random_pairs:
            start_pose = poses[s_id]
            goal_pose = poses[g_id]
            episodes.append(build_episode(eid, scene_id, start_pose, goal_pose))
            eid += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"episodes": episodes}, indent=2), encoding="utf-8")
    return len(episodes), out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate navigation episodes for a single scene")
    parser.add_argument(
        "scene",
        nargs="?",
        help="Path to the pinhole scene directory or scene identifier (optional if --pinhole-scene-dir is set)",
    )
    parser.add_argument(
        "--pinhole-scene-dir",
        type=Path,
        default=None,
        help="Explicit pinhole scene directory (overrides positional scene resolution)",
    )
    parser.add_argument(
        "--raw-scene-dir",
        type=Path,
        default=None,
        help="Explicit raw/extracted scene directory containing nvs_split/ (overrides auto detection)",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Override the scene_id used in output (defaults to pinhole dir name or positional scene string).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root directory where processed scenes will be written (scene name appended)",
    )
    args = parser.parse_args()

    if args.pinhole_scene_dir is not None:
        pinhole_scene_dir = args.pinhole_scene_dir.expanduser()
        if args.scene_id is not None:
            scene_id = args.scene_id
        else:
            scene_id = pinhole_scene_dir.name
        extracted_scene_dir = args.raw_scene_dir.expanduser() if args.raw_scene_dir is not None else None
    else:
        if args.scene is None:
            raise SystemExit("Missing scene argument. Provide <scene> or set --pinhole-scene-dir.")
        scene_arg = Path(args.scene).expanduser()
        if scene_arg.is_dir():
            pinhole_scene_dir = scene_arg
            scene_id = args.scene_id or pinhole_scene_dir.name
            dataset_root = pinhole_scene_dir.parent.parent
            extracted_scene_dir = dataset_root / "extracted" / "data" / scene_id
            if args.raw_scene_dir is not None:
                extracted_scene_dir = args.raw_scene_dir.expanduser()
            if not extracted_scene_dir.is_dir():
                # fallback to default root if relative resolution fails
                extracted_scene_dir = DEFAULT_EXTRACTED_ROOT / scene_id
        else:
            scene_id = args.scene_id or args.scene
            pinhole_scene_dir = DEFAULT_PINHOLE_ROOT / scene_id
            extracted_scene_dir = args.raw_scene_dir.expanduser() if args.raw_scene_dir is not None else (DEFAULT_EXTRACTED_ROOT / scene_id)

    if not pinhole_scene_dir.is_dir():
        raise FileNotFoundError(f"Pinhole scene directory not found: {pinhole_scene_dir}")
    if extracted_scene_dir is None or not extracted_scene_dir.is_dir():
        raise FileNotFoundError(f"Extracted scene directory not found: {extracted_scene_dir}")

    if args.output_root:
        output_scene_dir = args.output_root.expanduser() / scene_id
    else:
        output_scene_dir = pinhole_scene_dir

    rng = random.Random(RANDOM_SEED)
    episodes_count, out_path = generate_for_scene(
        scene_id,
        extracted_scene_dir,
        pinhole_scene_dir,
        output_scene_dir,
        rng,
    )
    print(f"[nav] scene={scene_id} episodes={episodes_count} -> {out_path}")


if __name__ == "__main__":
    main()
