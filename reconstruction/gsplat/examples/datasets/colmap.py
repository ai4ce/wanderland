import json
import os
from typing import Any, Dict, List, Optional
from typing_extensions import Literal

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
    imageio.imwrite(resized_path, resized_image)
    return resized_dir

def _resize_mask_folder(mask_dir: str, resized_dir: str, factor: int) -> str:
    """Nearest-neighbor style downsample for boolean .npy masks.

    If `resized_dir` already exists, return it. Otherwise read each .npy mask under
    `mask_dir`, take strided subsampling [::factor, ::factor], and save to `resized_dir`.
    """
    if os.path.exists(resized_dir):
        return resized_dir
    print(f"Downscaling masks by {factor}x from {mask_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    for fn in os.listdir(mask_dir):
        if not fn.endswith(".npy"):
            continue
        src = os.path.join(mask_dir, fn)
        dst = os.path.join(resized_dir, fn)
        try:
            m = np.load(src)
        except Exception:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        m_ds = m[::factor, ::factor]
        np.save(dst, m_ds.astype(bool))
    return resized_dir


def _resize_depth_folder(depth_dir: str, resized_dir: str, factor: int) -> str:
    """Downsample .pt dense depth maps by an integer factor using pooling.

    Creates `resized_dir` from `depth_dir` if needed. Each file is expected to be
    a single 2D tensor saved via `torch.save` with shape [H, W].
    """
    if os.path.exists(resized_dir):
        return resized_dir
    print(f"Downscaling depth maps by {factor}x from {depth_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    for fn in os.listdir(depth_dir):
        if not fn.endswith(".pt"):
            continue
        src = os.path.join(depth_dir, fn)
        dst = os.path.join(resized_dir, fn)
        depth = torch.load(src)
        # Pool to avoid aliasing; expect float tensor [H, W]
        depth_ds = (
            torch.nn.functional.max_pool2d(
                depth.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=factor,
                stride=factor,
            )
            .squeeze(0)
            .squeeze(0)
        )
        torch.save(depth_ds, dst)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        split_mode: Literal["nvs_split", "test_every"] = "test_every",
        nvs_split_profile: Literal["default", "interp_val", "extrap_val", "full_train"] = "default",
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.split_mode = split_mode
        # nvs_split_profile controls how we *derive* the final train/val split when split_mode="nvs_split".
        #
        # Background: our prepared scenes store two COLMAP image sets:
        #   - images.bin      : "base_train" views
        #   - images_val.bin  : "base_val" views
        #
        # Different downstream tasks want different semantics for "train" vs "val":
        #   - default    : Debugging / quick sanity checks.
        #                Uses COLMAP's base splits directly:
        #                  train = base_train
        #                  val   = base_val
        #   - interp_val : Experiments (interpolation-style evaluation).
        #                Split within base_train:
        #                  val   = every 8th view from base_train (offset 0)
        #                  train = remaining 7/8 of base_train
        #                base_val is not used.
        #   - extrap_val : Experiments (extrapolation / novel-view evaluation).
        #                Train on 7/8 of base_train, evaluate on base_val:
        #                  train = base_train minus every 8th view (offset 0)
        #                  val   = base_val
        #   - full_train : Final export only (maximize training views).
        #                Train on base_train + base_val, but keep evaluation on base_val:
        #                  train = base_train ∪ base_val
        #                  val   = base_val
        #
        # Notes:
        # - The holdout rule is intentionally fixed (every=8, offset=0) to keep the interface minimal.
        # - We also force base_train/base_val to be disjoint (if overlap exists, treat it as val).
        self.nvs_split_profile = nvs_split_profile

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # Per-image masks directory (prepared by prepare_scene.sh)
        # We map each COLMAP image name `n` to mask path `<mask_dir>/<n>.npy` directly
        # without changing basename or extension to avoid mismatch.
        mask_dir = os.path.join(data_dir, "masks")
        image_masks = [None] * len(image_names)
        if os.path.isdir(mask_dir):
            image_masks = [os.path.join(mask_dir, f"{n}.npy") for n in image_names]

        # Optional dense depth directory alongside images.
        depth_root = os.path.join(data_dir, "depths")
        if os.path.isdir(depth_root):
            if factor > 1 and not self.extconf["no_factor_suffix"]:
                depth_dir = depth_root + f"_{factor}"
                if not os.path.exists(depth_dir):
                    depth_dir = _resize_depth_folder(depth_root, depth_dir, factor)
            else:
                depth_dir = depth_root
            self.depth_dir = depth_dir
        else:
            self.depth_dir = None

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.image_masks = image_masks  # List[str|None], per-image .npy mask paths
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Load split indices for nvs_split mode
        self.split_indices = {}
        if self.split_mode == "nvs_split":
            print(f"[Parser] Using nvs_split from COLMAP files.")
            name2idx = {name: i for i, name in enumerate(self.image_names)}

            for tag in ["train", "val"]:
                suffix = "" if tag == "train" else "_val"
                # Try binary format first, then text format
                bin_path = os.path.join(colmap_dir, f"images{suffix}.bin")
                txt_path = os.path.join(colmap_dir, f"images{suffix}.txt")

                if os.path.isfile(bin_path):
                    print(f"[Parser] Loading {bin_path} for {tag} split.")
                    from .read_write_model import read_images_binary
                    images_data = read_images_binary(bin_path)
                    names = [img.name for img in images_data.values()]
                elif os.path.isfile(txt_path):
                    print(f"[Parser] Loading {txt_path} for {tag} split.")
                    with open(txt_path, 'r') as f:
                        lines = [line for line in f if line.strip() and not line.startswith('#')]
                    names = [line.split()[-1] for line in lines]
                else:
                    raise FileNotFoundError(f"split_mode='nvs_split' requires {bin_path} or {txt_path}")

                self.split_indices[tag] = np.array(
                    [name2idx[name] for name in names if name in name2idx], dtype=np.int32
                )

            # Base splits from COLMAP artifacts.
            base_train = self.split_indices.get("train", np.array([], dtype=np.int32))
            base_val = self.split_indices.get("val", np.array([], dtype=np.int32))

            # Ensure no overlap: if an image exists in both splits, treat it as val.
            base_val_set = set(map(int, base_val.tolist()))
            base_train_list = [int(i) for i in base_train.tolist() if int(i) not in base_val_set]
            base_train = np.array(base_train_list, dtype=np.int32)

            self.base_split_indices = {"train": base_train, "val": base_val}

            # Derive effective splits based on profile.
            profile = self.nvs_split_profile
            # Keep holdout rule simple and stable unless you explicitly change the code:
            # hold out every 8th frame from base_train, starting at offset 0.
            every = 8
            offset = 0

            holdout = np.array(base_train_list[offset::every], dtype=np.int32) if base_train_list else np.array([], dtype=np.int32)
            holdout_set = set(map(int, holdout.tolist()))
            train_7_8 = np.array([i for i in base_train_list if int(i) not in holdout_set], dtype=np.int32)

            if profile == "default":
                train_idx = base_train
                val_idx = base_val
            elif profile == "interp_val":
                # 7/8 of base_train for training, 1/8 of base_train for evaluation.
                train_idx = train_7_8
                val_idx = holdout
            elif profile == "extrap_val":
                # 7/8 of base_train for training, base_val for evaluation (holdout is unused).
                train_idx = train_7_8
                val_idx = base_val
            elif profile == "full_train":
                # Production/full-data mode:
                # - Train on all known frames (base_train + base_val).
                # - Keep val as base_val for compatibility with existing evaluation/reporting code.
                base_train_set = set(map(int, base_train.tolist()))
                merged = base_train.tolist() + [int(i) for i in base_val.tolist() if int(i) not in base_train_set]
                train_idx = np.array(merged, dtype=np.int32)
                val_idx = base_val
            else:
                raise ValueError(f"Unknown nvs_split_profile: {profile}")

            self.split_indices["train"] = train_idx
            self.split_indices["val"] = val_idx

            print(
                f"[Parser] nvs_split_profile={profile} base_train={len(base_train)} base_val={len(base_val)} "
                f"-> train={len(train_idx)} val={len(val_idx)} (holdout_every=8 offset=0)"
            )

            print(f"[Parser] nvs_split: train={len(self.split_indices.get('train', []))}, "
                  f"val={len(self.split_indices.get('val', []))}")


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        load_dense_depths: bool = False,
        split_mode: Literal["nvs_split", "test_every"] = "test_every",
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_dense_depths = load_dense_depths

        if self.load_depths and self.load_dense_depths and getattr(self.parser, "depth_dir", None) is None:
            print("[Dataset] load_dense_depths=True but no depths/ directory found; falling back to sparse depth.")

        if split_mode == "nvs_split":
            if not hasattr(parser, 'split_indices') or split not in parser.split_indices:
                raise ValueError(f"split_mode='nvs_split' but split '{split}' not found in parser")
            self.indices = parser.split_indices[split]
            print(f"[Dataset] Using nvs_split with {len(self.indices)} images for split '{split}'.")
        else:
            indices = np.arange(len(parser.image_names))
            if split == "train":
                self.indices = indices[indices % parser.test_every != 0]
            else:
                self.indices = indices[indices % parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        # Load per-image mask if provided
        mask = None
        if hasattr(self.parser, "image_masks") and self.parser.image_masks[index] is not None:
            try:
                mask_arr = np.load(self.parser.image_masks[index])
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
                mask = mask_arr.astype(bool)
            except Exception:
                mask = None

        if len(params) > 0:
            # Images are distorted. Undistort them consistently with the mask.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            # Undistort image with bilinear filtering
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            # Undistort mask with nearest to preserve binary values
            if mask is not None:
                mask = cv2.remap(
                    mask.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST
                ).astype(bool)
            # Crop to undistorted ROI
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
            if mask is not None:
                mask = mask[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        # After all crops, apply mask to GT image for strict alignment with UrbanSim
        if mask is not None:
            image[~mask] = 0
            data["image"] = torch.from_numpy(image).float()
            data["mask"] = torch.from_numpy(mask).bool()

        alpha_paths = getattr(self.parser, "alpha_mask_paths", None)
        if alpha_paths is not None:
            alpha_mask = imageio.imread(alpha_paths[index]).astype(np.float32)
            if alpha_mask.ndim == 2:
                alpha_mask = alpha_mask[..., None]
            if self.patch_size is not None:
                alpha_mask = alpha_mask[y : y + self.patch_size, x : x + self.patch_size]
            data["alpha_mask"] = torch.from_numpy(alpha_mask / 255.0).float()

        if self.load_depths:
            def _add_sparse_depth() -> None:
                worldtocams = np.linalg.inv(camtoworlds)
                image_name = self.parser.image_names[index]
                # Some scenes can have missing point-index entries for specific images
                # (e.g., partial COLMAP reconstructions or filtering differences).
                # In that case, skip sparse depth for this image instead of crashing.
                try:
                    point_indices = self.parser.point_indices[image_name]
                except KeyError:
                    return
                if point_indices is None or len(point_indices) == 0:
                    return
                points_world = self.parser.points[point_indices]
                points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
                depths = points_cam[:, 2]  # (M,)
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
                data["points"] = torch.from_numpy(points).float()
                data["depths"] = torch.from_numpy(depths).float()

            # Dense per-pixel supervision when explicitly requested and available.
            # If an individual depth file is missing, fall back to sparse depth for that image.
            if self.load_dense_depths and getattr(self.parser, "depth_dir", None):
                img_name = self.parser.image_names[index]
                base = os.path.splitext(img_name)[0]
                depth_path = os.path.join(self.parser.depth_dir, f"{base}.pt")
                if os.path.isfile(depth_path):
                    data["depth_map"] = torch.load(depth_path)
                else:
                    # Avoid crashing training due to partial depth generation.
                    _add_sparse_depth()
            else:
                _add_sparse_depth()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
