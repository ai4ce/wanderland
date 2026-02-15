#!/usr/bin/env python3
"""
Fisheye Dataset -> Pinhole + COLMAP Converter

Reads fisheye dataset with transforms.json and nvs_split, uses
the top-level `undistort_camera_model` as the target PINHOLE
intrinsic, and remaps fisheye images to pinhole images via
OpenCV fisheye undistort/rectify. Then writes COLMAP binary
files (PINHOLE cameras + images) pointing to the undistorted
images. Points3D and alignment visualization are reused from
the fisheye pipeline.

Usage:
    python examples/datasets/fisheye_to_pinhole_colmap.py /path/to/dataset /path/to/output
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Set, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Make local read_write_model importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from read_write_model import (
    Camera, BaseImage as Image,
    write_cameras_binary, write_images_binary
)

# Reuse helpers and constants from the fisheye pipeline
from fisheye_to_colmap import (
    load_train_val_split,
    load_transforms_json,
    setup_colmap_directories,
    create_alignment_visualization,
    write_colmap_points3d_file,
    COLMAP_GLOBAL_TRANSFORM,
    COLMAP_GLOBAL_ROTATION,
)


def normalize_frame_filename(frame: Dict) -> str:
    """Normalize a frame's file_path to match nvs_split naming.

    Examples:
    - "left/foo.jpg" -> "left_foo.jpg"
    - "left\\foo.jpg" -> "left_foo.jpg"
    - already flat names stay unchanged.
    """
    file_path = frame["file_path"].replace("\\\\", "/").replace("\\", "/")
    # If extension is .png (case-insensitive), force it to .jpg to match datasets
    if file_path.lower().endswith(".png"):
        file_path = os.path.splitext(file_path)[0] + ".jpg"
    filename = os.path.basename(file_path)
    if "/" in file_path:
        return file_path.replace("/", "_")
    # Handle backslash-only path that started with 'left' or 'right'
    if not filename.startswith("left_") and file_path.startswith("left"):
        return file_path.replace("\\", "_")
    if not filename.startswith("right_") and file_path.startswith("right"):
        return file_path.replace("\\", "_")
    return filename


DEFAULT_PINHOLE_WIDTH = 800
DEFAULT_PINHOLE_HEIGHT = 800
DEFAULT_PINHOLE_FOV_DEG = 120.0


def _make_default_pinhole(width: int, height: int, fov_deg: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Construct a PINHOLE intrinsic matrix assuming symmetric FOV."""

    half_fov = np.deg2rad(fov_deg) * 0.5
    focal = 0.5 * float(width) / np.tan(half_fov)
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    K = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, (int(width), int(height))


def read_target_pinhole(
    transforms_data: Dict,
    override_intrinsic: Optional[np.ndarray] = None,
    override_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resolve target pinhole K and image size, with optional overrides."""

    if override_intrinsic is not None and override_size is not None:
        return override_intrinsic.astype(np.float64), override_size

    # Default configuration avoids relying on transforms.json content.
    return _make_default_pinhole(
        width=DEFAULT_PINHOLE_WIDTH,
        height=DEFAULT_PINHOLE_HEIGHT,
        fov_deg=DEFAULT_PINHOLE_FOV_DEG,
    )


def get_fisheye_intrinsics(frame: Dict) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build OpenCV fisheye K and D from a frame entry."""
    fx = float(frame["fl_x"])
    fy = float(frame["fl_y"])
    cx = float(frame["cx"])
    cy = float(frame["cy"])
    k1 = float(frame["k1"]) ; k2 = float(frame["k2"]) ; k3 = float(frame["k3"]) ; k4 = float(frame["k4"]) 
    w = int(frame["w"]) ; h = int(frame["h"])  # original image size

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)  # (4,1)
    return K, D, (w, h)


def split_frames_by_camera(frames: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate frames into left and right lists based on file_path.
    A frame is considered right if 'right' appears in its file_path.
    """
    left_frames, right_frames = [], []
    for f in frames:
        fp = f["file_path"].replace("\\\\", "/").replace("\\", "/")
        if fp.startswith("right") or ("/right" in fp) or ("right" in fp):
            right_frames.append(f)
        else:
            left_frames.append(f)
    return left_frames, right_frames


def build_rectify_maps(
    K_tgt: np.ndarray,
    size_tgt: Tuple[int, int],
    K_fish: np.ndarray,
    D_fish: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create undistort/rectify maps for one camera side."""
    W_tgt, H_tgt = size_tgt
    R_rect = np.eye(3, dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_fish, D_fish, R_rect, K_tgt, (W_tgt, H_tgt), cv2.CV_16SC2
    )
    return map1, map2


def _read_image_size_if_exists(dataset_dir: str, frame: Dict) -> Optional[Tuple[int, int]]:
    """Try to read actual image size from disk for this frame; returns (W,H) or None."""
    fn = normalize_frame_filename(frame)
    path = os.path.join(dataset_dir, "fisheye", fn)
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)


def _scale_K_for_image_size(K: np.ndarray, frame_wh: Tuple[int, int], actual_wh: Tuple[int, int]) -> np.ndarray:
    """Scale intrinsics matrix K from (w,h) in frame to actual (W,H) from disk.

    Assumes uniform scale s_w ~= s_h. Returns scaled K'.
    """
    w0, h0 = frame_wh
    w1, h1 = actual_wh
    s_w = float(w1) / float(w0) if w0 > 0 else 1.0
    s_h = float(h1) / float(h0) if h0 > 0 else 1.0
    s = 0.5 * (s_w + s_h)
    Kp = K.copy().astype(np.float64)
    Kp[0, 0] *= s
    Kp[1, 1] *= s
    Kp[0, 2] *= s
    Kp[1, 2] *= s
    return Kp


def undistort_and_save_images(
    frames: List[Dict],
    dataset_dir: str,
    images_dir: str,
    maps_by_side: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_images: Set[str],
    val_images: Set[str],
) -> None:
    """Undistort images covered by train/val splits and save into images_dir.

    Input images are read from `<dataset_dir>/fisheye/<normalized_filename>`
    Output images are written to `<images_dir>/<normalized_filename>`.
    """
    os.makedirs(images_dir, exist_ok=True)

    # Use union of train and val sets
    keep_set = set(train_images) | set(val_images)

    for frame in frames:
        filename = normalize_frame_filename(frame)
        if filename not in keep_set:
            continue

        # Decide side
        fp = frame["file_path"].replace("\\\\", "/").replace("\\", "/")
        side = "right" if (fp.startswith("right") or ("/right" in fp) or ("right" in fp)) else "left"

        src_path = os.path.join(dataset_dir, "fisheye", filename)
        dst_path = os.path.join(images_dir, filename)

        if os.path.exists(dst_path):
            continue

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {src_path}")

        map1, map2 = maps_by_side[side]
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imwrite(dst_path, undistorted)


def write_pinhole_cameras_file(
    K_tgt: np.ndarray,
    size_tgt: Tuple[int, int],
    has_left: bool,
    has_right: bool,
    sparse_dir: str,
) -> None:
    """Write COLMAP cameras.bin with PINHOLE cameras for left/right."""
    fx, fy = float(K_tgt[0, 0]), float(K_tgt[1, 1])
    cx, cy = float(K_tgt[0, 2]), float(K_tgt[1, 2])
    W_tgt, H_tgt = size_tgt

    cameras = {}
    if has_left:
        cameras[1] = Camera(
            id=1,
            model="PINHOLE",
            width=W_tgt,
            height=H_tgt,
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )
    if has_right:
        cameras[2] = Camera(
            id=2,
            model="PINHOLE",
            width=W_tgt,
            height=H_tgt,
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )

    cam_file = os.path.join(sparse_dir, "cameras.bin")
    write_cameras_binary(cameras, cam_file)
    print(f"Wrote PINHOLE cameras to {cam_file}")


def write_images_file_pinhole(
    frames: List[Dict],
    dataset_dir: str,
    sparse_dir: str,
    images_dir: str,
    train_images: Set[str],
    val_images: Set[str],
    out_name: str = "images",
):
    """Write COLMAP images file referencing the undistorted pinhole images.

    This mirrors fisheye_to_colmap.write_images_file but omits copying
    and uses PINHOLE camera IDs (1=left, 2=right) based on file_path.
    """
    images_file = os.path.join(sparse_dir, f"{out_name}.bin")
    print(f"Writing {images_file} with frames filtered by split")

    global_trans = COLMAP_GLOBAL_TRANSFORM
    global_rot = COLMAP_GLOBAL_ROTATION

    images = {}
    processed_count = 0

    for frame in frames:
        file_path = frame["file_path"].replace("\\\\", "/").replace("\\", "/")
        filename = normalize_frame_filename(frame)

        if out_name == "images":
            if filename not in train_images and filename not in val_images:
                continue
        elif out_name == "images_val":
            if filename not in val_images:
                continue

        img_id = processed_count
        processed_count += 1

        trans = np.array(frame["transform_matrix"], dtype=np.float64)
        trans[:3, :3] = trans[:3, :3] @ global_rot
        trans = global_trans @ trans
        cam_pose = np.linalg.inv(trans)

        # Keep same alignment convention as original fisheye pipeline
        Rx_m90 = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0,-1, 0]], dtype=float)  # R_x(-90°)
        R_wc_fixed = cam_pose[:3, :3] @ Rx_m90
        tvec = cam_pose[:3, 3]
        q = R.from_matrix(R_wc_fixed).as_quat()  # [x,y,z,w]
        qvec = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # [w,x,y,z]

        camera_id = 2 if (file_path.startswith("right") or ("/right" in file_path) or ("right" in file_path)) else 1

        images[img_id] = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=filename,
            xys=np.empty((0, 2)),
            point3D_ids=np.empty(0, dtype=int),
        )

    write_images_binary(images, images_file)
    print(f"Wrote {images_file} with {len(images)} images")


def convert_fisheye_to_pinhole_colmap(
    dataset_dir: str,
    output_dir: str,
    include_las_points: bool = True,
    generate_visualization: bool = True,
    max_points: Optional[int] = 3_000_000,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    target_fx: Optional[float] = None,
    target_fy: Optional[float] = None,
    target_cx: Optional[float] = None,
    target_cy: Optional[float] = None,
) -> None:
    print("Converting fisheye dataset to PINHOLE + COLMAP format")
    print(f"Input:  {dataset_dir}")
    print(f"Output: {output_dir}")

    # Load splits and transforms
    train_images, val_images = load_train_val_split(dataset_dir)
    transforms_path = os.path.join(dataset_dir, "transforms.json")
    transforms_data = load_transforms_json(transforms_path)
    frames: List[Dict] = transforms_data["frames"]

    override_intrinsic = None
    override_size = None
    override_set = [target_width, target_height, target_fx, target_fy, target_cx, target_cy]
    if any(v is not None for v in override_set):
        if not all(v is not None for v in override_set):
            raise ValueError(
                "All target pinhole parameters (width, height, fx, fy, cx, cy) must be provided together."
            )
        override_intrinsic = np.array(
            [[float(target_fx), 0.0, float(target_cx)], [0.0, float(target_fy), float(target_cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        override_size = (int(target_width), int(target_height))
        print(
            "[Pinhole Override] Using custom target intrinsics and size: "
            f"{override_size[0]}x{override_size[1]}, fx={override_intrinsic[0,0]:.2f}, "
            f"fy={override_intrinsic[1,1]:.2f}, cx={override_intrinsic[0,2]:.2f}, cy={override_intrinsic[1,2]:.2f}"
        )

    # Target pinhole intrinsics
    K_tgt, size_tgt = read_target_pinhole(transforms_data, override_intrinsic, override_size)
    W_tgt, H_tgt = size_tgt
    print(f"Target PINHOLE: {W_tgt}x{H_tgt}, fx={K_tgt[0,0]:.2f}, fy={K_tgt[1,1]:.2f}, cx={K_tgt[0,2]:.2f}, cy={K_tgt[1,2]:.2f}")

    # Setup output directories
    dirs = setup_colmap_directories(output_dir)
    images_dir = dirs["images_dir"]
    sparse_dir = dirs["sparse_dir"]

    # Split frames per side and take representative frame per side for K/D
    left_frames, right_frames = split_frames_by_camera(frames)
    maps_by_side: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if len(left_frames) > 0:
        K_fL, D_fL, (wL, hL) = get_fisheye_intrinsics(left_frames[0])
        # Infer actual image size from disk and scale K accordingly if needed
        actual_size_L = _read_image_size_if_exists(dataset_dir, left_frames[0])
        if actual_size_L is not None and (actual_size_L != (wL, hL)):
            K_fL = _scale_K_for_image_size(K_fL, (wL, hL), actual_size_L)
            print(f"[Left] Scaled input detected: frame ({wL},{hL}) -> actual {actual_size_L}")
        else:
            actual_size_L = (wL, hL)
            print(f"[Left] Using frame-declared size as actual: {actual_size_L}")
        print(f"[Left] K used: fx={K_fL[0,0]:.4f}, fy={K_fL[1,1]:.4f}, cx={K_fL[0,2]:.2f}, cy={K_fL[1,2]:.2f}")
        maps_by_side["left"] = build_rectify_maps(K_tgt, size_tgt, K_fL, D_fL)
        print("Prepared undistort maps for LEFT camera")
    if len(right_frames) > 0:
        K_fR, D_fR, (wR, hR) = get_fisheye_intrinsics(right_frames[0])
        actual_size_R = _read_image_size_if_exists(dataset_dir, right_frames[0])
        if actual_size_R is not None and (actual_size_R != (wR, hR)):
            K_fR = _scale_K_for_image_size(K_fR, (wR, hR), actual_size_R)
            print(f"[Right] Scaled input detected: frame ({wR},{hR}) -> actual {actual_size_R}")
        else:
            actual_size_R = (wR, hR)
            print(f"[Right] Using frame-declared size as actual: {actual_size_R}")
        print(f"[Right] K used: fx={K_fR[0,0]:.4f}, fy={K_fR[1,1]:.4f}, cx={K_fR[0,2]:.2f}, cy={K_fR[1,2]:.2f}")
        maps_by_side["right"] = build_rectify_maps(K_tgt, size_tgt, K_fR, D_fR)
        print("Prepared undistort maps for RIGHT camera")

    # Undistort and save images into output images dir
    undistort_and_save_images(
        frames, dataset_dir, images_dir, maps_by_side, train_images, val_images
    )

    # Write PINHOLE cameras
    write_pinhole_cameras_file(
        K_tgt, size_tgt, has_left=len(left_frames) > 0, has_right=len(right_frames) > 0, sparse_dir=sparse_dir
    )

    # Filter frames by split for image bins
    def norm_name(f: Dict) -> str:
        return normalize_frame_filename(f)

    train_frames = [f for f in frames if norm_name(f) in train_images]
    val_frames = [f for f in frames if norm_name(f) in val_images]
    all_split_frames = train_frames + val_frames

    # Write images.bin (train + val) and images_val.bin (val only)
    write_images_file_pinhole(all_split_frames, dataset_dir, sparse_dir, images_dir, train_images, val_images, out_name="images")
    write_images_file_pinhole(val_frames, dataset_dir, sparse_dir, images_dir, train_images, val_images, out_name="images_val")

    if include_las_points:
        write_colmap_points3d_file(dataset_dir, sparse_dir, max_points=max_points)

    if generate_visualization:
        create_alignment_visualization(dataset_dir, output_dir)

    print("\n✓ Conversion to PINHOLE completed!")
    print(f"COLMAP files at: {sparse_dir}")
    print("- cameras.bin (PINHOLE)")
    print("- images.bin")
    print("- images_val.bin")
    if os.path.exists(os.path.join(sparse_dir, "points3D.bin")):
        print("- points3D.bin")
    print(f"Images at: {images_dir}")
    print("\nTrain with gsplat (pinhole):")
    print(f"python examples/simple_trainer.py default \\")
    print(f"    --data_dir {output_dir} \\")
    print(f"    --camera_model pinhole ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert fisheye dataset to PINHOLE + COLMAP format")
    parser.add_argument("dataset_dir", help="Path to fisheye dataset directory")
    parser.add_argument("output_dir", help="Path to output COLMAP directory")
    parser.add_argument("--target_width", type=int, help="Override target pinhole width")
    parser.add_argument("--target_height", type=int, help="Override target pinhole height")
    parser.add_argument("--target_fx", type=float, help="Override target pinhole fx")
    parser.add_argument("--target_fy", type=float, help="Override target pinhole fy")
    parser.add_argument("--target_cx", type=float, help="Override target pinhole cx")
    parser.add_argument("--target_cy", type=float, help="Override target pinhole cy")
    args = parser.parse_args()

    convert_fisheye_to_pinhole_colmap(
        args.dataset_dir,
        args.output_dir,
        target_width=args.target_width,
        target_height=args.target_height,
        target_fx=args.target_fx,
        target_fy=args.target_fy,
        target_cx=args.target_cx,
        target_cy=args.target_cy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
