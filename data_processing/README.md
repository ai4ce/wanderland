# Data Processing Pipeline

This directory contains standalone scripts for post-processing 3D Gaussian Splatting (3DGS) outputs. **Note:** Due to patent considerations, we are not releasing our 3DGS training pipeline. These scripts demonstrate the data processing steps that occur after 3DGS model training.

## Installation

Requires Python 3.12. Install dependencies using `uv`:

```bash
cd data_processing
uv sync
source .venv/bin/activate
```

## Scripts

### 1. ckpt2ply.py

Export a 3DGS gsplat checkpoint to PLY format.

**Usage:**
```bash
python ckpt2ply.py --ckpt path/to/checkpoint.pt --out output.ply
```

**Options:**
- `--ckpt`: Path to 3DGS checkpoint file (`.pt`)
- `--out`: Output PLY file path
- `--alpha_thr`: Opacity threshold for filtering (default: 0.0)

### 2. pcd2mesh.py

Generate a collision mesh from COLMAP sparse point cloud and camera poses.

**Usage:**
```bash
python pcd2mesh.py \
  --point-cloud sparse/points3D.bin \
  --images sparse/images.bin \
  --output-mesh output.obj
```

**Key Options:**
- `--point-cloud`: Path to point cloud (`.ply`, `points3D.bin`, or `points3D.txt`)
- `--images`: Path to COLMAP images file (`.bin` or `.txt`)
- `--output-mesh`: Output mesh path
- `--filter-horiz`: Horizontal padding around cameras (default: 4.0m)
- `--filter-vert`: Vertical padding around cameras (default: 1.5m)
- `--decimation-target`: Target triangle count (default: 2,500,000)

### 3. scene2usdz.py

Compose a USD/USDZ scene from a collision mesh and 3DGS USDZ asset.

**Usage:**
```bash
python scene2usdz.py mesh.obj gaussian_splat.usdz output.usdz
```

**Options:**
- `mesh`: Path to collision mesh (OBJ format)
- `gs_usdz`: Path to 3DGS USDZ asset
- `output`: Output USD/USDZ file
- `--translate X Y Z`: Translation offset (default: 0 0 0)
- `--rotate RX RY RZ`: Rotation in degrees (default: 0 0 0)
- `--scale SX SY SZ`: Scale factors (default: 1 1 1)
