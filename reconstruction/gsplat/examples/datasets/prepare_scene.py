#!/usr/bin/env python3
"""Unified scene preparation pipeline for gsplat fisheye datasets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_SKY_POINTS = 50_000

try:
    import laspy
except ImportError:  # pragma: no cover - handled at runtime
    laspy = None

from fisheye_to_colmap import (
    convert_fisheye_to_colmap,
    create_alignment_visualization,
)
from fisheye_to_pinhole_colmap import convert_fisheye_to_pinhole_colmap
from read_write_model import Point3D, read_points3D_binary, write_points3D_binary


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_las_pointcloud(
    dataset_dir: Path,
    rng: np.random.Generator,
    max_points: Optional[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load LAS point cloud, apply transform, and optionally downsample."""
    if laspy is None:
        raise RuntimeError("laspy is required for point cloud generation. Please install it.")

    las_path = dataset_dir / "colorized.las"
    if not las_path.exists():
        print(f"[prepare_scene] No colorized.las found at {las_path}; skipping point cloud generation.")
        return None

    print(f"[prepare_scene] Loading LAS point cloud: {las_path}")
    las = laspy.read(str(las_path))

    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        red = (las.red / 65535.0).astype(np.float32)
        green = (las.green / 65535.0).astype(np.float32)
        blue = (las.blue / 65535.0).astype(np.float32)
        cols = np.vstack([red, green, blue]).T
    else:
        print("[prepare_scene] No color info in LAS; defaulting to white")
        cols = np.ones((len(pts), 3), dtype=np.float32)

    # Apply fisheye pipeline coordinate transform [y, -x, z]
    pts_transformed = np.stack([pts[:, 1], -1*pts[:, 2], -1*pts[:, 0]], axis=1)

    colors_255 = (cols * 255).astype(np.uint8)

    if max_points is not None and len(pts_transformed) > max_points:
        indices = rng.choice(len(pts_transformed), max_points, replace=False)
        pts_transformed = pts_transformed[indices]
        colors_255 = colors_255[indices]
        print(f"[prepare_scene] Downsampled LAS to {len(pts_transformed)} points (cap={max_points})")

    print(f"[prepare_scene] Loaded {len(pts_transformed)} points from LAS")
    return pts_transformed, colors_255


def _write_points3d_binary(points: np.ndarray, colors: np.ndarray, path: Path) -> int:
    """Write COLMAP points3D.bin from numpy arrays."""
    empty_ids = np.empty(0, dtype=np.int32)
    point_dict: Dict[int, Point3D] = {}
    for idx in range(points.shape[0]):
        point_dict[idx + 1] = Point3D(
            id=idx + 1,
            xyz=np.asarray(points[idx], dtype=np.float64),
            rgb=np.asarray(colors[idx], dtype=np.uint8),
            error=1.0,
            image_ids=empty_ids,
            point2D_idxs=empty_ids,
        )

    write_points3D_binary(point_dict, str(path))
    print(f"[prepare_scene] Wrote {len(point_dict)} points to {path}")
    return len(point_dict)


def _sample_training_points(
    points: np.ndarray,
    colors: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or len(points) <= limit:
        return points.copy(), colors.copy()

    indices = rng.choice(len(points), limit, replace=False)
    indices.sort()
    sampled_points = points[indices].copy()
    sampled_colors = colors[indices].copy()
    print(f"[prepare_scene] Sampled training cloud: {len(sampled_points)} points (cap={limit})")
    return sampled_points, sampled_colors


def _append_sky_points_array(
    points: np.ndarray,
    colors: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_points <= 0 or len(points) == 0:
        print("[prepare_scene] Sky point addition skipped (num_points <= 0 or empty cloud).")
        return points, colors

    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center[None, :], axis=1).max()
    if radius <= 0:
        print("[prepare_scene] Degenerate scene radius; sky points skipped.")
        return points, colors

    directions = rng.normal(size=(num_points, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_dirs = directions / norms
    sky_points = center[None, :] + radius * unit_dirs
    sky_colors = np.full((num_points, 3), 255, dtype=np.uint8)

    augmented_points = np.concatenate([points, sky_points], axis=0)
    augmented_colors = np.concatenate([colors, sky_colors], axis=0)
    print(f"[prepare_scene] Added {num_points} sky points (radius={radius:.3f}).")
    return augmented_points, augmented_colors


def _run_depth_generation(
    data_dir: Path,
    camera_model: str,
    data_factor: int,
    init_opa: float,
    init_scale: float,
    depth_uniform_fallback: bool,
) -> None:
    examples_dir = Path(__file__).resolve().parent.parent
    script = examples_dir / "generate_init_depth_maps.py"

    if not script.exists():
        raise FileNotFoundError(f"Could not find generate_init_depth_maps.py at {script}")

    depth_result_dir = data_dir / "prep_depth"
    _ensure_dir(depth_result_dir)

    cmd = [
        sys.executable,
        str(script),
        "--data_dir",
        str(data_dir),
        "--data_factor",
        str(data_factor),
        "--camera_model",
        camera_model,
        "--init_opa",
        str(init_opa),
        "--init_scale",
        str(init_scale),
        "--result_dir",
        str(depth_result_dir),
        "--disable_viewer",
    ]

    if depth_uniform_fallback:
        cmd.append("--depth_uniform_fallback")

    print("[prepare_scene] Generating dense depth maps (this may take a while)...")
    subprocess.run(cmd, check=True, cwd=str(examples_dir))
    print("[prepare_scene] Dense depth generation completed.")


def _points_from_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    points3d = read_points3D_binary(str(path))
    if not points3d:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

    keys = sorted(points3d.keys())
    pts = np.stack([points3d[k].xyz for k in keys]).astype(np.float64)
    cols = np.stack([points3d[k].rgb for k in keys]).astype(np.uint8)
    return pts, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified fisheye-to-COLMAP preparation pipeline")
    parser.add_argument("--dataset_dir", required=True, help="Input fisheye dataset root")
    parser.add_argument("--out_dir", required=True, help="Output directory for COLMAP data")
    parser.add_argument(
        "--camera_model",
        choices=["pinhole", "fisheye"],
        default="pinhole",
        help="Target camera model for COLMAP conversion",
    )
    parser.add_argument(
        "--full_points_cap",
        type=int,
        default=0,
        help="Optional cap for the full point cloud (0=unlimited)",
    )
    parser.add_argument(
        "--train_points_cap",
        type=int,
        default=2_000_000,
        help="Cap for the training point cloud (0=skip downsample)",
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=1,
        help="Dataset downsample factor for depth generation",
    )
    parser.add_argument(
        "--depth_uniform_fallback",
        action="store_true",
        help="Enable uniform depth fallback for unobserved pixels",
    )
    parser.add_argument(
        "--init_opa",
        type=float,
        default=1.0,
        help="Initial opacity passed to depth renderer",
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=0.5,
        help="Initial scale passed to depth renderer",
    )
    parser.add_argument(
        "--add_sky_points",
        action="store_true",
        help="Append uniformly sampled sky points on the scene bounding sphere",
    )
    parser.add_argument(
        "--num_sky_points",
        type=int,
        default=DEFAULT_SKY_POINTS,
        help="Number of sky points to add when --add_sky_points is set",
    )
    parser.add_argument("--skip_convert", action="store_true", help="Skip fisheye->COLMAP conversion")
    parser.add_argument(
        "--skip_pointcloud",
        action="store_true",
        help="Skip regenerating the point cloud from LAS",
    )
    parser.add_argument("--skip_depths", action="store_true", help="Skip dense depth generation")
    parser.add_argument(
        "--skip_downsample",
        action="store_true",
        help="Skip training point cloud downsampling",
    )
    parser.add_argument(
        "--save_full_points3d",
        action="store_true",
        help="Save a copy of the full point cloud as points3D_full.bin",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs when possible")
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for subsampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    sparse_dir = out_dir / "sparse" / "0"
    points_path = sparse_dir / "points3D.bin"
    backup_path = sparse_dir / "points3D_full.bin"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    _ensure_dir(out_dir)
    rng = np.random.default_rng(args.seed)

    # Step 1: COLMAP conversion (skip LAS/visualization here; handled below).
    if not args.skip_convert:
        if args.camera_model == "pinhole":
            convert_fisheye_to_pinhole_colmap(
                str(dataset_dir),
                str(out_dir),
                include_las_points=False,
                generate_visualization=False,
            )
        else:
            convert_fisheye_to_colmap(
                str(dataset_dir),
                str(out_dir),
                include_las_points=False,
                generate_visualization=False,
            )
    else:
        if not sparse_dir.exists():
            raise FileNotFoundError(
                "Sparse directory missing; cannot skip conversion without existing COLMAP data."
            )
        print("[prepare_scene] Skipping conversion step (using existing COLMAP data).")

    full_points: Optional[np.ndarray] = None
    full_colors: Optional[np.ndarray] = None
    sky_points: Optional[np.ndarray] = None
    sky_colors: Optional[np.ndarray] = None

    # Step 2: Load LAS and write full point cloud if requested.
    if not args.skip_pointcloud:
        cap = args.full_points_cap if args.full_points_cap > 0 else None
        loaded = _load_las_pointcloud(dataset_dir, rng, cap)
        if loaded is not None:
            full_points, full_colors = loaded
            _ensure_dir(sparse_dir)
            _write_points3d_binary(full_points, full_colors, points_path)

            if args.save_full_points3d:
                shutil.copy2(points_path, backup_path)
                print(f"[prepare_scene] Copied full point cloud to {backup_path}")
        else:
            if not points_path.exists():
                raise FileNotFoundError(
                    "No LAS data and points3D.bin missing; cannot proceed."
                )
    else:
        print("[prepare_scene] Skipping LAS point cloud regeneration.")

    # If we skipped saving full earlier but need backup for later steps, create it now if requested.
    if args.save_full_points3d and points_path.exists() and not backup_path.exists():
        shutil.copy2(points_path, backup_path)
        print(f"[prepare_scene] Created backup of existing point cloud at {backup_path}")

    # Ensure we have base point arrays for later stages.
    if full_points is None or full_colors is None:
        source_for_base = backup_path if backup_path.exists() else points_path
        if source_for_base.exists():
            print(f"[prepare_scene] Loading existing point cloud from {source_for_base}")
            full_points, full_colors = _points_from_binary(source_for_base)
        else:
            full_points = np.empty((0, 3), dtype=np.float64)
            full_colors = np.empty((0, 3), dtype=np.uint8)

    # Prepare point cloud for depth rendering (append sky points before rasterization).
    depth_points = full_points.copy()
    depth_colors = full_colors.copy()
    if args.add_sky_points:
        depth_points, depth_colors = _append_sky_points_array(
            depth_points, depth_colors, max(1, args.num_sky_points), rng
        )
        num_added = depth_points.shape[0] - full_points.shape[0]
        if num_added > 0:
            sky_points = depth_points[-num_added:].copy()
            sky_colors = depth_colors[-num_added:].copy()
        else:
            sky_points = np.empty((0, 3), dtype=np.float64)
            sky_colors = np.empty((0, 3), dtype=np.uint8)
        _write_points3d_binary(depth_points, depth_colors, points_path)
        print(
            f"[prepare_scene] Wrote sky-augmented point cloud ({depth_points.shape[0]} points) for depth generation."
        )
    else:
        print("[prepare_scene] Sky point addition skipped before depth generation.")

    # Step 3: Dense depth generation uses the full point cloud currently in points3D.bin.
    if not args.skip_depths:
        _run_depth_generation(
            data_dir=out_dir,
            camera_model=args.camera_model,
            data_factor=args.data_factor,
            init_opa=args.init_opa,
            init_scale=args.init_scale,
            depth_uniform_fallback=args.depth_uniform_fallback,
        )
    else:
        print("[prepare_scene] Skipping dense depth generation step.")

    # Step 4: Prepare training point cloud (downsample + sky points after depth generation).
    if full_points.size == 0:
        raise ValueError("Training point cloud is empty; check input data and LAS availability.")

    if args.skip_downsample or args.train_points_cap <= 0:
        train_points = full_points.copy()
        train_colors = full_colors.copy()
        print("[prepare_scene] Using full point cloud for training (no downsample).")
    else:
        train_points, train_colors = _sample_training_points(
            full_points, full_colors, args.train_points_cap, rng
        )

    if args.add_sky_points:
        if sky_points is not None and sky_points.size > 0:
            train_points = np.concatenate([train_points, sky_points], axis=0)
            train_colors = np.concatenate([train_colors, sky_colors], axis=0)
            print(
                f"[prepare_scene] Reused {sky_points.shape[0]} sky points for training cloud."
            )
        else:
            train_points, train_colors = _append_sky_points_array(
                train_points, train_colors, max(1, args.num_sky_points), rng
            )

    _write_points3d_binary(train_points, train_colors, points_path)

    # Step 5: Visualization after sky point augmentation.
    if len(train_points) > 0:
        create_alignment_visualization(
            str(dataset_dir),
            str(out_dir),
            points_xyz=train_points,
        )

    print("[prepare_scene] Scene preparation completed successfully.")


if __name__ == "__main__":
    main()
