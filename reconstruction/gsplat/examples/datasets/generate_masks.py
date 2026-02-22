#!/usr/bin/env python3
"""
Generate per-image training masks by composing:

Fisheye:
  final = raw_mask ∧ ellipse_mask ∧ (¬ seg_mask)

Pinhole:
  final = undistorted(raw_mask) ∧ (¬ seg_mask)

Inputs:
- --data_dir: original dataset root containing fisheye/ and fisheye_mask/
- --out_dir: prepared COLMAP directory containing images/ and sparse/0/{images*.txt,cameras.txt}
- --camera_model: 'fisheye' or 'pinhole'
- --weights: YOLO-seg weights file(s) for dynamic objects
- --classes: class ids to treat as dynamic (remove)
- --fov: fisheye ellipse FOV (radians). Set in fisheye mode; pinhole mode disables ellipse.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import imageio.v2 as imageio

import cv2  # type: ignore

from img_utils import generate_image_masks, ensure_yolo_weights
from fisheye_to_pinhole_colmap import (
    get_fisheye_intrinsics,
    read_target_pinhole,
    build_rectify_maps,
    _read_image_size_if_exists,
    _scale_K_for_image_size,
)
from read_write_model import (
    read_images_binary,
    read_cameras_binary,
    write_cameras_text,
)


def _read_split_names(sparse_dir: str) -> List[str]:
    names: List[str] = []
    for base in ("images", "images_val"):
        txt = os.path.join(sparse_dir, f"{base}.txt")
        binp = os.path.join(sparse_dir, f"{base}.bin")
        if os.path.isfile(txt):
            with open(txt, "r") as f:
                for ln in f:
                    if not ln.strip() or ln.startswith("#"):
                        continue
                    names.append(ln.split()[-1])
        elif os.path.isfile(binp):
            imgs = read_images_binary(binp)
            names.extend([img.name for img in imgs.values()])
    # deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def _compose_and_save(dst_path: str, masks: Sequence[np.ndarray]) -> None:
    out = np.ones_like(masks[0], dtype=bool)
    for m in masks:
        out &= m.astype(bool)
    np.save(dst_path, out)


def _load_raw_mask(dataset_dir: str, name: str) -> Optional[np.ndarray]:
    raw_path = os.path.join(dataset_dir, "fisheye_mask", os.path.splitext(name)[0] + ".png")
    if not os.path.isfile(raw_path):
        return None
    m = imageio.imread(raw_path)
    if m.ndim == 3:
        # take first channel if needed
        m = m[..., 0]
    return (m > 0).astype(np.bool_)


def _split_sides(frames: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    left_frame = None
    right_frame = None
    for fr in frames:
        fp = fr.get("file_path", "").replace("\\\\", "/").replace("\\", "/")
        if fp.startswith("right") or ("/right" in fp) or ("right" in fp):
            if right_frame is None:
                right_frame = fr
        else:
            if left_frame is None:
                left_frame = fr
        if left_frame is not None and right_frame is not None:
            break
    return left_frame, right_frame


def _side_from_name(name: str) -> str:
    n = name.replace("\\", "/")
    return "right" if ("/right" in n or n.startswith("right") or "right_" in n) else "left"


def _build_maps_for_pinhole(dataset_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    tf_path = os.path.join(dataset_dir, "transforms.json")
    with open(tf_path, "r") as f:
        data = json.load(f)
    frames: List[Dict] = data.get("frames", [])
    left_frame, right_frame = _split_sides(frames)
    if left_frame is None and right_frame is None:
        raise RuntimeError("No frames found to build rectify maps.")

    # Target pinhole intrinsics/size (mirrors fisheye_to_pinhole_colmap defaults)
    K_tgt, size_tgt = read_target_pinhole(data)

    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if left_frame is not None:
        # Scale intrinsics if actual on-disk resolution differs from metadata.
        K_f, D_f, _ = get_fisheye_intrinsics(left_frame)
        actual_size = _read_image_size_if_exists(dataset_dir, left_frame)
        declared_size = (int(left_frame.get("w", 0)), int(left_frame.get("h", 0)))
        if actual_size is not None and declared_size != (0, 0) and actual_size != declared_size:
            K_f = _scale_K_for_image_size(K_f, declared_size, actual_size)
        maps["left"] = build_rectify_maps(K_tgt=K_tgt, size_tgt=size_tgt, K_fish=K_f, D_fish=D_f)
    if right_frame is not None:
        K_f, D_f, _ = get_fisheye_intrinsics(right_frame)
        actual_size = _read_image_size_if_exists(dataset_dir, right_frame)
        declared_size = (int(right_frame.get("w", 0)), int(right_frame.get("h", 0)))
        if actual_size is not None and declared_size != (0, 0) and actual_size != declared_size:
            K_f = _scale_K_for_image_size(K_f, declared_size, actual_size)
        maps["right"] = build_rectify_maps(K_tgt=K_tgt, size_tgt=size_tgt, K_fish=K_f, D_fish=D_f)
    return maps


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-image training masks (raw ∧ ellipse ∧ ¬seg)")
    ap.add_argument("--data_dir", required=True, help="Original dataset root with fisheye/ and fisheye_mask/")
    ap.add_argument("--out_dir", required=True, help="Prepared COLMAP directory (contains images/ and sparse/0/")
    ap.add_argument("--camera_model", choices=["fisheye", "pinhole"], required=True)
    ap.add_argument("--weights", nargs="+", default=[
        "datasets/models/yolo11n-seg.pt",
        "datasets/models/yolo11x-seg.pt",
    ])
    ap.add_argument("--classes", nargs="+", type=int, default=[0, 1, 3, 16])
    ap.add_argument("--fov", type=float, default=1.884, help="Ellipse FOV (radians) for fisheye")
    args = ap.parse_args()

    dataset_dir = args.data_dir
    out_dir = args.out_dir
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    images_dir = os.path.join(out_dir, "images")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")

    names = _read_split_names(sparse_dir)
    if not names:
        raise FileNotFoundError(f"No image names found in {sparse_dir}/images*.txt")

    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    ensure_yolo_weights(args.weights)

    # Ensure cameras.txt exists (img_utils relies on it); if only .bin exists, write a text view
    if not os.path.isfile(cameras_txt) and os.path.isfile(cameras_bin):
        cams = read_cameras_binary(cameras_bin)
        write_cameras_text(cams, cameras_txt)

    # Run segmentation (+ ellipse for fisheye) first to create base masks under out_dir/masks
    # For fisheye: include ellipse with provided FOV
    # For pinhole: disable ellipse by passing fov=None (makes ellipse mask all True)
    mode = args.camera_model
    if mode == "fisheye":
        for split in ("images.txt", "images_val.txt"):
            path = os.path.join(sparse_dir, split)
            if not os.path.isfile(path):
                # If .txt missing but .bin exists, synthesize a text file for img_utils
                binp = os.path.join(sparse_dir, os.path.splitext(split)[0] + ".bin")
                if os.path.isfile(binp):
                    imgs = read_images_binary(binp)
                    with open(path, "w") as f:
                        for i, img in imgs.items():
                            # Minimal line format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                            # We only need CAMERA_ID and NAME (last two tokens) in img_utils
                            f.write(f"{img.id} 0 0 0 1 0 0 0 {img.camera_id} {img.name}\n\n")
                else:
                    continue
            generate_image_masks(
                images_txt=path,
                cameras_txt=cameras_txt,
                images_dir=images_dir,
                masks_dir=masks_dir,
                seg_model_paths=args.weights,
                valid_cls=tuple(args.classes),
                fov=float(args.fov),
            )
        # Compose with raw masks
        for name in names:
            base_path = os.path.join(masks_dir, name + ".npy")
            if not os.path.isfile(base_path):
                # If missing (e.g., name only in val/train), skip
                continue
            base_mask = np.load(base_path).astype(bool)
            raw = _load_raw_mask(dataset_dir, name)
            if raw is None:
                final = base_mask
            else:
                # Expect same resolution for fisheye copy → direct AND
                if raw.shape != base_mask.shape:
                    # Resize raw to image size (nearest)
                    raw_u8 = raw.astype(np.uint8) * 255
                    raw = cv2.resize(raw_u8, (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                final = base_mask & raw
            np.save(base_path, final)
    else:
        # Pinhole: segmentation only (ellipse disabled)
        for split in ("images.txt", "images_val.txt"):
            path = os.path.join(sparse_dir, split)
            if not os.path.isfile(path):
                binp = os.path.join(sparse_dir, os.path.splitext(split)[0] + ".bin")
                if os.path.isfile(binp):
                    imgs = read_images_binary(binp)
                    with open(path, "w") as f:
                        for i, img in imgs.items():
                            f.write(f"{img.id} 0 0 0 1 0 0 0 {img.camera_id} {img.name}\n\n")
                else:
                    continue
            generate_image_masks(
                images_txt=path,
                cameras_txt=cameras_txt,
                images_dir=images_dir,
                masks_dir=masks_dir,
                seg_model_paths=args.weights,
                valid_cls=tuple(args.classes),
                fov=None,  # disable ellipse
            )
        # Build rectify maps to undistort raw masks into pinhole space
        maps_by_side = _build_maps_for_pinhole(dataset_dir)
        # Compose with raw_undistorted
        # Load each saved (~seg) and AND with undistorted raw
        for name in names:
            base_path = os.path.join(masks_dir, name + ".npy")
            if not os.path.isfile(base_path):
                continue
            base_mask = np.load(base_path).astype(bool)  # this is ~seg
            raw = _load_raw_mask(dataset_dir, name)
            if raw is None:
                final = base_mask
            else:
                # undistort raw using the correct side map
                side = _side_from_name(name)
                if side not in maps_by_side:
                    # fallback: use left map
                    side = "left"
                map1, map2 = maps_by_side[side]
                raw_u8 = raw.astype(np.uint8) * 255
                und = cv2.remap(raw_u8, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                und = und > 0
                # Ensure size match the pinhole image
                if und.shape != base_mask.shape:
                    und = cv2.resize(und.astype(np.uint8) * 255, (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                final = base_mask & und
            np.save(base_path, final)

    print(f"[generate_masks] Done. Masks saved under {masks_dir}")


if __name__ == "__main__":
    main()
