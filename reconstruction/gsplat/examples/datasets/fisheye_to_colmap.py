#!/usr/bin/env python3
"""
Fisheye Dataset to COLMAP Format Converter

Converts fisheye datasets with transforms.json and nvs_split to COLMAP format.
Based on urbansim_raw.py utilities with proper coordinate transformations.

Usage:
    python fisheye_to_colmap.py /path/to/dataset /path/to/output
"""

import os
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from scipy.spatial.transform import Rotation as R

# Import tqdm with fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"Processing {desc}...")
        return iterable

# Import COLMAP writing utilities
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from read_write_model import (
    Camera, BaseImage as Image, Point3D,
    write_cameras_binary, write_images_binary, write_points3D_binary
)

# Optional point cloud processing
import laspy
import matplotlib.pyplot as plt

# Coordinate transformation parameters (from user-provided config)
COLMAP_GLOBAL_TRANSFORM = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

COLMAP_GLOBAL_ROTATION = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

def normalize_frame_filename(frame: Dict) -> str:
    """Normalize a frame's filename to match nvs_split and on-disk files.

    - Unify path separators to '/'
    - If extension is .png (any case), coerce to .jpg (datasets store JPGs)
    - Flatten subpaths: 'left/foo.jpg' -> 'left_foo.jpg'
    """
    file_path = frame['file_path'].replace('\\\\', '/').replace('\\', '/')
    if file_path.lower().endswith('.png'):
        file_path = os.path.splitext(file_path)[0] + '.jpg'
    filename = os.path.basename(file_path)
    if '/' in file_path:
        return file_path.replace('/', '_')
    if not filename.startswith('left_') and file_path.startswith('left'):
        return file_path.replace('\\', '_')
    return filename

def load_train_val_split(dataset_dir: str) -> Tuple[Set[str], Set[str]]:
    """
    Load train/validation split from nvs_split directory.
    Based on urbansim_raw.py pattern.
    """
    nvs_split_dir = os.path.join(dataset_dir, "nvs_split")
    train_file = os.path.join(nvs_split_dir, "train.txt")
    val_file = os.path.join(nvs_split_dir, "val.txt")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Train/val split files not found in {nvs_split_dir}")

    # Load training images
    with open(train_file, 'r') as f:
        train_images = {line.strip() for line in f if line.strip()}

    # Load validation images
    with open(val_file, 'r') as f:
        val_images = {line.strip() for line in f if line.strip()}

    print(f"Loaded split: {len(train_images)} train, {len(val_images)} val images")
    return train_images, val_images

def load_transforms_json(transforms_path: str) -> Dict:
    """Load transforms.json file."""
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")

    with open(transforms_path, 'r') as f:
        data = json.load(f)

    if 'frames' not in data:
        raise ValueError("No 'frames' found in transforms.json")

    print(f"Loaded {len(data['frames'])} frames from transforms.json")
    return data

def setup_colmap_directories(output_dir: str) -> Dict[str, str]:
    """
    Create COLMAP directory structure.
    Copied from urbansim_raw.py:setup_colmap_directories
    """
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    images_dir = os.path.join(output_dir, "images")

    directories = {
        "sparse_dir": sparse_dir,
        "images_dir": images_dir,
    }

    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created {name}: {path}")

    return directories

def write_cameras_file(frames: List[Dict], sparse_dir: str):
    """
    Write COLMAP cameras file with fisheye camera parameters.
    Handles dual-camera setups with separate left and right camera intrinsics.
    """
    if not frames:
        raise ValueError("No frames provided")

    # Separate frames by camera (left/right)
    left_frames = []
    right_frames = []

    for frame in frames:
        file_path = frame['file_path'].replace('\\\\', '/').replace('\\', '/')
        if file_path.startswith('left') or 'left' in file_path:
            left_frames.append(frame)
        elif file_path.startswith('right') or 'right' in file_path:
            right_frames.append(frame)

    cameras = {}

    # Create left camera (ID=1)
    if left_frames:
        left_frame = left_frames[0]
        cameras[1] = Camera(
            id=1,
            model="OPENCV_FISHEYE",
            width=left_frame['w'],
            height=left_frame['h'],
            params=np.array([
                left_frame['fl_x'], left_frame['fl_y'],
                left_frame['cx'], left_frame['cy'],
                left_frame['k1'], left_frame['k2'],
                left_frame['k3'], left_frame['k4']
            ])
        )
        print(f"Added left fisheye camera (ID=1): {left_frame['w']}x{left_frame['h']}")
        print(f"  Left intrinsics: fx={left_frame['fl_x']:.1f}, fy={left_frame['fl_y']:.1f}, cx={left_frame['cx']:.1f}, cy={left_frame['cy']:.1f}")

    # Create right camera (ID=2)
    if right_frames:
        right_frame = right_frames[0]
        cameras[2] = Camera(
            id=2,
            model="OPENCV_FISHEYE",
            width=right_frame['w'],
            height=right_frame['h'],
            params=np.array([
                right_frame['fl_x'], right_frame['fl_y'],
                right_frame['cx'], right_frame['cy'],
                right_frame['k1'], right_frame['k2'],
                right_frame['k3'], right_frame['k4']
            ])
        )
        print(f"Added right fisheye camera (ID=2): {right_frame['w']}x{right_frame['h']}")
        print(f"  Right intrinsics: fx={right_frame['fl_x']:.1f}, fy={right_frame['fl_y']:.1f}, cx={right_frame['cx']:.1f}, cy={right_frame['cy']:.1f}")

    if not cameras:
        raise ValueError("No valid cameras found in frames")

    # Write using binary format
    cam_file = os.path.join(sparse_dir, 'cameras.bin')
    print(f"Writing {cam_file} with {len(cameras)} cameras in binary format")

    write_cameras_binary(cameras, cam_file)
    print(f"Successfully wrote {cam_file}")

def write_images_file(
    frames: List[Dict],
    src_dir: str,
    sparse_dir: str,
    images_dir: str,
    train_images: Set[str],
    val_images: Set[str],
    start_id: int = 0,
    out_name: str = "images",
):
    """
    Write COLMAP images file and copy corresponding images.
    Based on urbansim_raw.py:write_images_file with coordinate transformations.
    """
    images_file = os.path.join(sparse_dir, f"{out_name}.bin")
    print(f"Writing {images_file} with {len(frames)} frames in binary format")

    # Apply global transformations (from urbansim_raw.py)
    global_trans = COLMAP_GLOBAL_TRANSFORM
    global_rot = COLMAP_GLOBAL_ROTATION

    # Create COLMAP Image objects
    images = {}
    processed_count = 0

    for idx, frame in enumerate(tqdm(frames, desc=f"Processing {out_name}")):
        # Extract normalized filename and unified file_path for side detection
        file_path = frame['file_path'].replace('\\\\', '/').replace('\\', '/')
        filename = normalize_frame_filename(frame)

        # Filter based on train/val split
        # For "images.bin", include all images that are in either train or val splits
        # For "images_val.bin", include only validation images
        if out_name == "images":
            if filename not in train_images and filename not in val_images:
                continue
        elif out_name == "images_val" and filename not in val_images:
            continue

        img_id = start_id + processed_count
        processed_count += 1

        # Apply transformations (from urbansim_raw.py:write_images_file)
        trans = np.array(frame['transform_matrix'])
        trans[:3, :3] = trans[:3, :3] @ global_rot
        trans = global_trans @ trans
        cam_pose = np.linalg.inv(trans)

        # Extract pose with coordinate system alignment
        # Apply R_x(-90°) rotation to align coordinate systems (from urbansim_raw.py)
        Rx_m90 = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0,-1, 0]], dtype=float)  # R_x(-90°)

        R_wc_fixed = cam_pose[:3, :3] @ Rx_m90
        tvec = cam_pose[:3, 3]                      # Translation unchanged

        q = R.from_matrix(R_wc_fixed).as_quat()     # [x,y,z,w]
        qvec = np.array([q[3], q[0], q[1], q[2]])   # Convert to COLMAP [w,x,y,z]

        # Copy image file
        src_path = os.path.join(src_dir, 'fisheye', filename)
        dst_path = os.path.join(images_dir, filename)

        if not os.path.exists(dst_path):  # Avoid duplicate copying
            shutil.copy(src_path, dst_path)

        # Determine camera ID based on left/right
        camera_id = 1  # Default to left camera
        if file_path.startswith('right') or 'right' in file_path:
            camera_id = 2  # Right camera

        # Create COLMAP Image object
        images[img_id] = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,  # Use appropriate camera ID (1=left, 2=right)
            name=filename,
            xys=np.empty((0, 2)),  # Empty point observations
            point3D_ids=np.empty(0, dtype=int)  # Empty point IDs
        )

    # Write using binary format
    write_images_binary(images, images_file)

    print(f"Successfully wrote {images_file} with {len(images)} images")

def write_colmap_points3d_file(
    dataset_dir: str,
    sparse_dir: str,
    max_points: Optional[int] = 3_000_000,
):
    """Write COLMAP points3D file from colorized.las if available."""
    las_path = os.path.join(dataset_dir, "colorized.las")

    if not os.path.exists(las_path):
        print("No colorized.las found, skipping points3D generation")
        return  # Not an error

    print(f"Loading point cloud from {las_path}")

    # Read LAS file (from urbansim_raw.py:load_pointcloud_from_las)
    las = laspy.read(las_path)

    # Extract coordinates and colors
    pts = np.vstack([las.x, las.y, las.z]).T

    # Color data in LAS is typically 16-bit, convert to 0-1 range
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        red = (las.red / 65535.0).astype(np.float32)
        green = (las.green / 65535.0).astype(np.float32)
        blue = (las.blue / 65535.0).astype(np.float32)
        cols = np.vstack([red, green, blue]).T
    else:
        print("No color information found in LAS file, using white")
        cols = np.ones((len(pts), 3), dtype=np.float32)

    print(f"Loaded {len(pts)} points from LAS file")

    # Apply coordinate transformation to match camera pose transformation R_x(-90°)
    # (from urbansim_raw.py:load_pointcloud_from_las)
    pts_transformed = np.stack([pts[:, 1], -1 * pts[:, 2], -1 * pts[:, 0]], axis=1)

    # Convert colors from 0-1 to 0-255 range
    colors_255 = (cols * 255).astype(np.uint8)

    if max_points is not None and len(pts_transformed) > max_points:
        indices = np.random.choice(len(pts_transformed), max_points, replace=False)
        pts_transformed = pts_transformed[indices]
        colors_255 = colors_255[indices]
        print(f"Downsampled to {len(pts_transformed)} points (cap={max_points})")

    # Create COLMAP Point3D objects (from urbansim_raw.py:write_colmap_points3d_file)
    points3D = {}
    for i, (pt, col) in enumerate(zip(pts_transformed, colors_255)):
        point_id = i + 1  # COLMAP uses 1-based indexing
        x, y, z = pt
        r, g, b = col
        error = 1.0  # Default reprojection error

        points3D[point_id] = Point3D(
            id=point_id,
            xyz=np.array([x, y, z]),
            rgb=np.array([r, g, b]),
            error=error,
            image_ids=np.empty(0, dtype=int),  # No track information
            point2D_idxs=np.empty(0, dtype=int)  # No track information
        )

    # Write using binary format
    points3d_file = os.path.join(sparse_dir, "points3D.bin")
    write_points3D_binary(points3D, points3d_file)

    print(f"Successfully wrote {points3d_file} with {len(points3D)} points")

def load_camera_centers_from_binary(images_bin_path: str, num_samples: int = 500) -> np.ndarray:
    """
    Load camera centers from COLMAP images.bin file.
    Based on urbansim_raw.py:load_camera_centers_from_binary
    """
    from read_write_model import read_images_binary

    if not os.path.exists(images_bin_path):
        print(f"Images file not found: {images_bin_path}")
        return np.empty((0, 3))

    print(f"Loading camera centers from binary file: {images_bin_path}")

    # Load images from binary file
    images = read_images_binary(images_bin_path)

    centers = []
    image_ids = list(images.keys())

    # Sample images if too many
    if len(image_ids) > num_samples:
        import random
        image_ids = random.sample(image_ids, num_samples)

    for image_id in image_ids:
        img = images[image_id]

        # Parse quaternion and translation
        qvec = img.qvec  # [w, x, y, z] from COLMAP format
        tvec = img.tvec  # [tx, ty, tz]

        # Convert quaternion to rotation matrix
        R_wc = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()  # Convert to [x,y,z,w]
        t_wc = np.array(tvec)

        # Calculate camera center: C = -R^T * t
        C = -R_wc.T @ t_wc
        centers.append(C)

    if centers:
        print(f"Loaded {len(centers)} camera centers from binary file")
        return np.asarray(centers, dtype=float)
    else:
        print(f"No valid camera poses found in {images_bin_path}")
        return np.empty((0, 3))

def sample_points_for_visualization(points: np.ndarray, num_pts: int = 4000) -> np.ndarray:
    """
    Sample points from point cloud for visualization.
    Based on urbansim_raw.py:sample_points_for_visualization
    """
    if len(points) <= num_pts:
        return points

    # Random sampling
    indices = np.random.choice(len(points), num_pts, replace=False)
    return points[indices]

def plot_three_views(points: np.ndarray, cams_train: np.ndarray, cams_val: np.ndarray, save_path: str):
    """
    Create three-view visualization plot (Front X-Z, Side Y-Z, Top X-Y).
    Based on urbansim_raw.py:plot_three_views
    """
    print("Creating three-view visualization...")

    plt.figure(figsize=(15, 5))

    def _subplot(idx, x, y, title):
        ax = plt.subplot(1, 3, idx)
        if len(points) > 0:
            ax.scatter(points[:, x], points[:, y], s=3, c='tab:red', alpha=0.6, label='Points')
        if len(cams_train) > 0:
            ax.scatter(cams_train[:, x], cams_train[:, y], s=6, c='tab:green', label='Cams (train)')
        if len(cams_val) > 0:
            ax.scatter(cams_val[:, x], cams_val[:, y], s=6, c='tab:blue', label='Cams (val)')
        ax.set_xlabel(['X','Y','Z'][x])
        ax.set_ylabel(['X','Y','Z'][y])
        ax.set_title(title)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

    _subplot(1, 0, 2, 'Front  (X–Z)')
    _subplot(2, 1, 2, 'Side   (Y–Z)')
    _subplot(3, 0, 1, 'Top    (X–Y)')

    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()

    print(f'Visualization saved: {save_path}')

def create_alignment_visualization(
    dataset_dir: str,
    output_dir: str,
    points_xyz: Optional[np.ndarray] = None,
):
    """Create visualization to verify point cloud and camera alignment."""
    print("=== Creating alignment visualization ===")

    sparse_dir = os.path.join(output_dir, "sparse", "0")

    # Load camera centers from binary files
    images_train_path = os.path.join(sparse_dir, "images.bin")
    images_val_path = os.path.join(sparse_dir, "images_val.bin")

    cams_train = load_camera_centers_from_binary(images_train_path, 800)
    cams_val = load_camera_centers_from_binary(images_val_path, 800)

    # Load and sample points from LAS file if available
    las_path = os.path.join(dataset_dir, "colorized.las")
    points = np.empty((0, 3))

    if points_xyz is not None:
        points = sample_points_for_visualization(points_xyz, 4000)
        print(f"Using provided point cloud for visualization ({len(points_xyz)} points)")
    elif os.path.exists(las_path):
        print(f"Loading points from {las_path} for visualization")
        las = laspy.read(las_path)
        pts = np.vstack([las.x, las.y, las.z]).T

        # Apply the same coordinate transformation as in point cloud processing
        pts_transformed = np.stack([pts[:, 1], -1 * pts[:, 2], -1 * pts[:, 0]], axis=1)

        # Sample points for visualization
        points = sample_points_for_visualization(pts_transformed, 4000)
        print(f"Sampled {len(points)} points for visualization")

    # Create visualization
    vis_path = os.path.join(output_dir, "camera_pointcloud_alignment.png")
    plot_three_views(points, cams_train, cams_val, vis_path)
    print("✓ Created alignment visualization")
    print(f"Visualization shows:")
    print(f"  - {len(points)} point cloud points (red)")
    print(f"  - {len(cams_train)} training cameras (green)")
    print(f"  - {len(cams_val)} validation cameras (blue)")

def convert_fisheye_to_colmap(
    dataset_dir: str,
    output_dir: str,
    include_las_points: bool = True,
    generate_visualization: bool = True,
    max_points: Optional[int] = 3_000_000,
):
    """
    Convert fisheye dataset to COLMAP format.
    Main conversion function based on urbansim_raw.py:generate_colmap_format.
    """
    print(f"Converting fisheye dataset to COLMAP format")
    print(f"Input: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Format: Binary (.bin)")

    # Load train/validation split
    train_images, val_images = load_train_val_split(dataset_dir)

    # Load transforms.json
    transforms_path = os.path.join(dataset_dir, "transforms.json")
    transforms_data = load_transforms_json(transforms_path)
    frames = transforms_data['frames']

    # Setup directories (from urbansim_raw.py)
    dirs = setup_colmap_directories(output_dir)

    # Write cameras file
    write_cameras_file(frames, dirs["sparse_dir"])

    # Filter frames for train/val sets
    def get_normalized_filename(frame):
        return normalize_frame_filename(frame)

    train_frames = [f for f in frames if get_normalized_filename(f) in train_images]
    val_frames = [f for f in frames if get_normalized_filename(f) in val_images]
    # All frames that exist in either train or val splits
    all_split_frames = train_frames + val_frames

    print(f"Train frames: {len(train_frames)}, Val frames: {len(val_frames)}, Total split frames: {len(all_split_frames)}")

    # Write all images in splits to images.bin (both train and val)
    write_images_file(
        all_split_frames, dataset_dir, dirs["sparse_dir"], dirs["images_dir"],
        train_images, val_images, start_id=0, out_name="images"
    )

    # Write validation images file separately for nvs_split mode
    write_images_file(
        val_frames, dataset_dir, dirs["sparse_dir"], dirs["images_dir"],
        train_images, val_images, start_id=0, out_name="images_val"
    )

    # Write points3D file from colorized.las if available
    if include_las_points:
        write_colmap_points3d_file(dataset_dir, dirs["sparse_dir"], max_points=max_points)

    if generate_visualization:
        create_alignment_visualization(dataset_dir, output_dir)

    print(f"\\n✓ Conversion completed successfully!")
    print(f"COLMAP files created in: {dirs['sparse_dir']}")
    print(f"- cameras.bin")
    print(f"- images.bin")
    print(f"- images_val.bin")
    if os.path.exists(os.path.join(dirs['sparse_dir'], 'points3D.bin')):
        print(f"- points3D.bin")
    print(f"Images copied to: {dirs['images_dir']}")
    print(f"\\nYou can now use this with gsplat:")
    print(f"python examples/simple_trainer.py default \\\\")
    print(f"    --data_dir {output_dir} \\\\")
    print(f"    --camera_model fisheye \\\\")
    print(f"    --with_ut True")

def main():
    parser = argparse.ArgumentParser(description="Convert fisheye dataset to COLMAP format")
    parser.add_argument("dataset_dir", help="Path to fisheye dataset directory")
    parser.add_argument("output_dir", help="Path to output COLMAP directory")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        return 1

    # Check required files
    required_files = [
        "transforms.json",
        "nvs_split/train.txt",
        "nvs_split/val.txt",
        "fisheye"
    ]

    for req_file in required_files:
        path = os.path.join(args.dataset_dir, req_file)
        if not os.path.exists(path):
            print(f"Error: Required file/directory not found: {path}")
            return 1

    # Run conversion
    convert_fisheye_to_colmap(args.dataset_dir, args.output_dir)
    return 0

if __name__ == "__main__":
    exit(main())
