# imgmask_utils.py
import os
import shutil
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
import cv2
import urllib.request


def write_images_txt(
    frames,
    src_dir,
    sparse_dir,
    images_dir,
    global_trans,
    global_rot,
    start_id=0,
    out_name="images.txt",
):
    """Apply transforms, copy images, and write COLMAP images.txt."""
    images_txt = os.path.join(sparse_dir, out_name)
    with open(images_txt, "w") as f:
        for idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
            img_id = start_id + idx
            trans = np.array(frame['transform_matrix'])
            trans[:3, :3] = trans[:3, :3] @ global_rot
            trans = global_trans @ trans
            cam_pose = np.linalg.inv(trans)

            tvec = cam_pose[:3, 3]
            q = R.from_matrix(cam_pose[:3, :3]).as_quat()
            qvec = [q[3], q[0], q[1], q[2]]

            src_name = frame['file_path'].replace('\\', '/')
            dst_name = src_name.replace('/', '_')
            try:
                shutil.copy(
                    os.path.join(src_dir, 'camera', src_name),
                    os.path.join(images_dir, dst_name),
                )
            except Exception:
                shutil.copy(
                    os.path.join(src_dir, 'images', src_name),
                    os.path.join(images_dir, dst_name),
                )

            cam_id = 1 if 'left' in src_name else 2
            elems = [img_id] + qvec + tvec.tolist() + [cam_id, dst_name]
            f.write(" ".join(map(str, elems)) + "\n\n")


def write_cameras_txt(frames, sparse_dir):
    """Write fisheye intrinsics for left/right cameras."""
    cam_file = os.path.join(sparse_dir, 'cameras.txt')
    left_cam = right_cam = None
    for frm in frames:
        path = frm['file_path']
        if path.startswith('left') and left_cam is None:
            left_cam = frm
        elif path.startswith('right') and right_cam is None:
            right_cam = frm
        if left_cam and right_cam:
            break

    with open(cam_file, 'w') as f:
        if left_cam:
            f.write(
                f"1 OPENCV_FISHEYE {left_cam['w']} {left_cam['h']} "
                f"{left_cam['fl_x']} {left_cam['fl_y']} {left_cam['cx']} {left_cam['cy']} "
                f"{left_cam['k1']} {left_cam['k2']} {left_cam['k3']} {left_cam['k4']}\n"
            )
        if right_cam:
            f.write(
                f"2 OPENCV_FISHEYE {right_cam['w']} {right_cam['h']} "
                f"{right_cam['fl_x']} {right_cam['fl_y']} {right_cam['cx']} {right_cam['cy']} "
                f"{right_cam['k1']} {right_cam['k2']} {right_cam['k3']} {right_cam['k4']}\n"
            )


def generate_image_masks(images_txt, cameras_txt, images_dir, masks_dir,
                         seg_model_paths, valid_cls=(0, 1, 3, 16), fov=np.pi * 0.6):
    """
    Generate boolean masks for each image by combining an ellipse from intrinsics
    and removing pixels covered by segmentation models.

    Output: save one .npy mask (H,W,bool) per image under masks_dir with filename `${image_name}.npy`.
    """
    seg_models = [YOLO(p) for p in seg_model_paths]

    cam_intr = {}
    with open(cameras_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # id, model, w, h, fx, fy, cx, cy, ...
            cam_id = int(parts[0])
            if parts[1] in ("SIMPLE_PINHOLE", "PINHOLE"):
                # No distortion; still record intrinsics
                w, h = int(parts[2]), int(parts[3])
                fx, fy, cx, cy = map(float, parts[4:8])
            else:
                w, h = int(parts[2]), int(parts[3])
                fx, fy, cx, cy = map(float, parts[4:8])
            cam_intr[cam_id] = (w, h, fx, fy, cx, cy)

    img_cam = {}
    with open(images_txt, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            tokens = line.split()
            img_cam[tokens[-1]] = int(tokens[-2])

    os.makedirs(masks_dir, exist_ok=True)

    for fname, cam_id in tqdm(img_cam.items(), desc='Generating masks'):
        img_path = os.path.join(images_dir, fname)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            # Skip silently if image is missing
            continue
        h, w = orig_img.shape[:2]

        width, height, fx, fy, cx, cy = cam_intr[cam_id]
        scale_x = w / max(1, width)
        scale_y = h / max(1, height)
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y

        # Ellipse mask centered at (cx, cy)
        if fov is None:
            ellipse_mask = np.ones((h, w), dtype=bool)
        else:
            a, b = fx * (fov / 2.0), fy * (fov / 2.0)
            ys = np.arange(h)[:, None].astype(np.float32)
            xs = np.arange(w)[None, :].astype(np.float32)
            ellipse_mask = (((xs - cx) ** 2) / (a ** 2 + 1e-8) + ((ys - cy) ** 2) / (b ** 2 + 1e-8)) <= 1.0

        # Segmentation mask (union of specified classes)
        seg_mask = np.zeros((h, w), dtype=bool)
        for model in seg_models:
            res = model.predict(source=orig_img, conf=0.25, save=False,
                                show_labels=False, show_conf=False,
                                show_boxes=False, verbose=False)
            if not res or res[0].masks is None:
                continue
            mask_data = res[0].masks.data.cpu().numpy()
            cls_ids = res[0].boxes.cls.cpu().numpy().astype(int)
            # filter by classes
            mask_data = mask_data[np.isin(cls_ids, valid_cls)]
            in_shape = mask_data.shape[1:]
            # Fit into original image shape while preserving aspect
            ratio = min(in_shape[0] / h, in_shape[1] / w)
            new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
            dw, dh = int((in_shape[1] - new_w) / 2), int((in_shape[0] - new_h) / 2)
            for m in mask_data:
                pm = m[dh:dh + new_h, dw:dw + new_w]
                rm = cv2.resize(pm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                seg_mask |= rm.astype(bool)

        final_mask = ellipse_mask & (~seg_mask)
        mask_path = os.path.join(masks_dir, fname + '.npy')
        np.save(mask_path, final_mask)

    print(f"Generated masks saved to {masks_dir}")


def ensure_yolo_weights(paths):
    """
    Check that each path in `paths` exists, otherwise download from predefined URLs.
    """
    download_map = {
        "datasets/models/yolo11n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "datasets/models/yolo11x-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
        "../../datasets/models/yolo11n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "../../datasets/models/yolo11x-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
    }
    # normalize and ensure parent dir
    for p in paths:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    for p in paths:
        if not os.path.exists(p):
            url = download_map.get(p)
            if url is None:
                # Try fallback on basename match
                base = os.path.basename(p)
                for key, val in download_map.items():
                    if os.path.basename(key) == base:
                        url = val
                        break
            if url is None:
                raise FileNotFoundError(f"No download URL configured for missing weight: {p}")
            print(f"Downloading {os.path.basename(p)} …")
            urllib.request.urlretrieve(url, p)
            print(f"Saved → {p}")
