#!/usr/bin/env python3
"""
Standalone mesh reconstruction pipeline for COLMAP-style outputs.

Given a sparse point cloud and camera poses (typically COLMAP `points3D.bin`
and `images.bin`), this script filters the cloud around the camera frustum,
produces a collision cloud, and runs a density-field + marching cubes
reconstruction (DSMC) without writing intermediate point clouds to disk.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import pycolmap
import torch
from scipy.ndimage import gaussian_filter
from skimage import measure


def _points_dict_to_arrays(points_dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert COLMAP Point3D dict into XYZ and RGB arrays."""
    if not points_dict:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    xyz = []
    rgb = []
    for pt in points_dict.values():
        xyz.append(np.array([pt.x, pt.y, pt.z], dtype=np.float64))
        rgb.append(np.array(pt.color, dtype=np.float64) / 255.0)
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float64)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from .ply, points3D.bin, or points3D.txt."""
    ext = os.path.splitext(path)[1].lower()
    print(f"[mesh] Loading point cloud: {path}")
    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)
    elif ext == ".bin":
        reconstruction = pycolmap.Reconstruction(os.path.dirname(path))
        xyz, rgb = _points_dict_to_arrays(reconstruction.points3D)
        pcd = o3d.geometry.PointCloud()
        if xyz.size:
            pcd.points = o3d.utility.Vector3dVector(xyz)
            if rgb.size:
                pcd.colors = o3d.utility.Vector3dVector(rgb)
    elif ext == ".txt":
        reconstruction = pycolmap.Reconstruction(os.path.dirname(path))
        xyz, rgb = _points_dict_to_arrays(reconstruction.points3D)
        pcd = o3d.geometry.PointCloud()
        if xyz.size:
            pcd.points = o3d.utility.Vector3dVector(xyz)
            if rgb.size:
                pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        raise ValueError(f"Unsupported point cloud file: {path}")

    if not pcd.has_points():
        raise RuntimeError(f"Empty point cloud loaded from {path}")
    print(f"[mesh] Point count = {len(pcd.points)}")
    return pcd


def load_camera_centers(images_path: str) -> np.ndarray:
    """Load camera centers from COLMAP images file (.bin or .txt)."""
    ext = os.path.splitext(images_path)[1].lower()
    print(f"[mesh] Loading camera poses: {images_path}")
    
    reconstruction = pycolmap.Reconstruction(os.path.dirname(images_path))
    
    if not reconstruction.images:
        raise RuntimeError(f"No registered images found in {images_path}")

    centers = []
    for img in reconstruction.images.values():
        centers.append(img.projection_center())
    centers_np = np.asarray(centers, dtype=np.float64)
    print(f"[mesh] Camera centers loaded: {centers_np.shape[0]}")
    return centers_np


def uniform_downsample(pcd: o3d.geometry.PointCloud, every_k: int) -> o3d.geometry.PointCloud:
    """Uniformly downsample every_k points; returns new point cloud."""
    if every_k is None or every_k <= 1:
        return pcd
    print(f"[mesh] Uniform down-sampling every {every_k} points")
    return pcd.uniform_down_sample(every_k)


def filter_points_near_cameras(
    pcd: o3d.geometry.PointCloud,
    camera_centers: np.ndarray,
    horiz_thresh: float,
    vert_thresh: float,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Keep points within an expanded bounding box around camera centers."""
    xyz = np.asarray(pcd.points)
    if xyz.size == 0:
        raise RuntimeError("Input point cloud has no points for filtering.")
    if camera_centers.size == 0:
        raise RuntimeError("Camera centers array is empty.")

    bounds_min = camera_centers.min(axis=0).copy()
    bounds_max = camera_centers.max(axis=0).copy()

    # Convention: X/Z lie on the horizontal plane, -Y points upward.
    bounds_min[0] -= horiz_thresh
    bounds_max[0] += horiz_thresh
    bounds_min[2] -= horiz_thresh
    bounds_max[2] += horiz_thresh

    bounds_min[1] -= vert_thresh/8
    bounds_max[1] += vert_thresh

    keep_mask = np.all((xyz >= bounds_min) & (xyz <= bounds_max), axis=1)
    kept_points = xyz[keep_mask]
    if kept_points.size == 0:
        raise RuntimeError(
            "No points remain after near-camera filtering. Adjust thresholds.")

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(kept_points)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape[0] == xyz.shape[0]:
            filtered.colors = o3d.utility.Vector3dVector(colors[keep_mask])

    print(
        f"[mesh] Filtered {len(kept_points)} points near cameras (ratio={len(kept_points)/len(pcd.points):.2f})")

    if voxel_size and voxel_size > 0.0:
        filtered = filtered.voxel_down_sample(voxel_size=voxel_size)
        print(
            f"[mesh] Voxel down-sampled collision cloud to {len(filtered.points)} points (voxel={voxel_size} m)")

    return filtered


def reconstruct_mesh_from_points(
    points: np.ndarray,
    output_path: str,
    *,
    voxel_size_xyz: Sequence[float],
    padding: float,
    iso_level: Optional[float],
    gaussian_sigma: Optional[Sequence[float]],
    decimation_target: int,
    smoothing_iterations: int,
    cleanup_min_triangles: int,
) -> str:
    """Run DSMC reconstruction directly from numpy points."""
    if points.size == 0:
        raise RuntimeError(
            "Mesh reconstruction received an empty point array.")

    pts = np.asarray(points, dtype=np.float64)
    minb = pts.min(axis=0) - padding
    maxb = pts.max(axis=0) + padding
    vx, vy, vz = np.minimum(np.array(voxel_size_xyz), (maxb - minb) / 250.0)
    dims = np.ceil((maxb - minb) / np.array([vx, vy, vz])).astype(int) + 1
    nx, ny, nz = map(int, dims)
    if nx * ny * nz <= 0:
        raise RuntimeError("Invalid voxel grid dimensions computed for DSMC.")

    print(
        f"[mesh] Voxel grid dims = ({nx}, {ny}, {nz}), padding = {padding:.2f}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mesh] Computing density field on {device}")

    pts_t = torch.from_numpy(pts).to(device)
    minb_t = torch.from_numpy(minb).to(device)
    vs_t = torch.tensor([vx, vy, vz], dtype=pts_t.dtype, device=device)

    idx = ((pts_t - minb_t) / vs_t).floor().long()
    idx[:, 0].clamp_(0, nx - 1)
    idx[:, 1].clamp_(0, ny - 1)
    idx[:, 2].clamp_(0, nz - 1)

    flat = idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2]
    density_flat = torch.zeros(nx * ny * nz, dtype=torch.int32, device=device)
    ones = torch.ones_like(flat, dtype=torch.int32)
    density_flat.scatter_add_(0, flat, ones)
    density = density_flat.reshape(nx, ny, nz).cpu().numpy()

    dens_max = density.max()
    dens_mean = density.mean()
    print(f"[mesh] Density stats: max = {dens_max}, mean = {dens_mean:.2f}")
    if dens_max == 0:
        raise RuntimeError(
            "Density field is empty (all zeros); cannot reconstruct mesh.")

    level = float(dens_mean) if iso_level is None else float(iso_level)
    print(f"[mesh] Using iso_level = {level}")

    if gaussian_sigma is not None:
        print(f"[mesh] Gaussian smoothing sigma = {tuple(gaussian_sigma)}")
        density = gaussian_filter(density, sigma=tuple(gaussian_sigma))

    print("[mesh] Running Marching Cubes…")
    volume = density.transpose(0, 1, 2)
    verts, faces, normals, _ = measure.marching_cubes(
        volume=volume,
        level=level,
        spacing=(vx, vy, vz),
    )
    verts += minb
    print(f"[mesh] Extracted raw mesh: {len(verts)} verts, {len(faces)} faces")

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()

    if decimation_target and decimation_target > 0:
        print(f"[mesh] Quadric decimation → target ≃ {decimation_target} tris")
        mesh = mesh.simplify_quadric_decimation(decimation_target)

    if smoothing_iterations > 0:
        print(f"[mesh] Taubin smoothing ({smoothing_iterations} iterations)")
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=smoothing_iterations)

    if cleanup_min_triangles > 0:
        print(f"[mesh] Removing clusters < {cleanup_min_triangles} triangles")
        tc, cluster_n, _ = mesh.cluster_connected_triangles()
        tc_np = np.asarray(tc, dtype=np.int64)
        cluster_n_np = np.asarray(cluster_n, dtype=np.int64)
        if tc_np.size and cluster_n_np.size:
            largest = int(np.argmax(cluster_n_np))
            mask = (tc_np != largest).tolist()
            mesh.remove_triangles_by_mask(mask)
            mesh.remove_unreferenced_vertices()

    print(
        f"[mesh] Final mesh → {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(
        output_path,
        mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=False,
        compressed=False,
    )
    print(f"[mesh] Mesh saved to: {output_path}")
    return output_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a DSMC mesh from a sparse point cloud and COLMAP images file."
    )
    parser.add_argument("--point-cloud", required=True,
                        help="Path to point cloud (.ply, points3D.bin, points3D.txt).")
    parser.add_argument("--images", required=True,
                        help="Path to COLMAP images file (.bin or .txt) providing camera poses.")
    parser.add_argument("--output-mesh", required=True,
                        help="Destination mesh path (e.g. meshes/scene.obj).")

    parser.add_argument("--every-k", type=int, default=1,
                        help="Uniform point-cloud down-sample factor (<=1 disables).")
    parser.add_argument("--filter-horiz", type=float, default=4.0,
                        help="Horizontal padding (m) around camera centers.")
    parser.add_argument("--filter-vert", type=float, default=1.5,
                        help="Vertical padding (m) around camera centers.")
    parser.add_argument("--filter-voxel", type=float, default=0.0,
                        help="Voxel size (m) for collision cloud down-sample.")

    parser.add_argument(
        "--voxel-size-xyz",
        type=float,
        nargs=3,
        default=(0.08, 0.04, 0.08),
        metavar=("VX", "VY", "VZ"),
        help="Voxel size (m) along XYZ for density grid.",
    )
    parser.add_argument("--padding", type=float, default=0.10,
                        help="Bounding-box padding (m) applied before voxelization.")
    parser.add_argument("--iso-level", type=float, default=None,
                        help="Marching cubes iso-level (defaults to density mean).")
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Optional Gaussian sigma for density smoothing.",
    )
    parser.add_argument(
        "--decimation-target",
        type=int,
        default=2_500_000,
        help="Target triangle count for quadric decimation (<=0 disables).",
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=5,
        help="Number of Taubin smoothing iterations (0 disables).",
    )
    parser.add_argument(
        "--cleanup-min-triangles",
        type=int,
        default=50,
        help="Drop connected components smaller than this triangle count (<=0 disables).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    pcd = load_point_cloud(args.point_cloud)
    if args.every_k and args.every_k > 1:
        pcd = uniform_downsample(pcd, args.every_k)

    camera_centers = load_camera_centers(args.images)
    collision_pcd = filter_points_near_cameras(
        pcd,
        camera_centers,
        horiz_thresh=args.filter_horiz,
        vert_thresh=args.filter_vert,
        voxel_size=args.filter_voxel,
    )

    reconstruct_mesh_from_points(
        np.asarray(collision_pcd.points),
        args.output_mesh,
        voxel_size_xyz=tuple(args.voxel_size_xyz),
        padding=args.padding,
        iso_level=args.iso_level,
        gaussian_sigma=tuple(
            args.gaussian_sigma) if args.gaussian_sigma else None,
        decimation_target=args.decimation_target,
        smoothing_iterations=args.smoothing_iterations,
        cleanup_min_triangles=args.cleanup_min_triangles,
    )


if __name__ == "__main__":
    main()
