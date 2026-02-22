# Reconstruction

This module contains the **Wanderland reconstruction pipeline** (3D Gaussian Splatting training + utilities).

## Installation

```bash
cd reconstruction
uv sync
source .venv/bin/activate
```

`gsplat` is vendored under `reconstruction/gsplat/` and needs to be installed separately (it builds CUDA extensions):

```bash
uv pip install -e ./gsplat --no-build-isolation
```

## Quick Start

### 1) Download one scene

```bash
cd ../download
uv sync
source .venv/bin/activate

# Download a small subset first
python download.py --modality nvs --scene-list eval_scenes_v1.txt --count 1 --output ../wanderland_data
```

### 2) Train 3DGS on the downloaded scene

Run from `reconstruction/` (e.g., `cd ../reconstruction`):

```bash
SCENE_ID="YOUR_SCENE_ID"
DATA_DIR="../wanderland_data/${SCENE_ID}"

python train_3dgs.py \
  --data_dir "${DATA_DIR}" \
  --result_dir "results/${SCENE_ID}" \
  --data_factor 1
```

### 3) (Optional) Generate dense depth maps for supervision

```bash
SCENE_ID="YOUR_SCENE_ID"
DATA_DIR="../wanderland_data/${SCENE_ID}"

python generate_depths.py \
  --data_dir "${DATA_DIR}" \
  --result_dir "results/${SCENE_ID}" \
  --data_factor 1
```

## Notes

- If you want to use GPU, ensure your CUDA toolkit + driver are set up correctly before installing `gsplat`.
- Some scripts expect `nvs_split/train.txt` and `nvs_split/val.txt` to exist in the scene directory (download `--modality nvs` or `--modality full`).
- To export an Isaac/Omniverse USDZ from a trained checkpoint, see `data_processing/README.md` (PLY→USDZ uses optional NVIDIA 3DGRUT).
