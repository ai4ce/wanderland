# Wanderland Dataset - Download Tool

Download tool for the Wanderland dataset from HuggingFace. The Wanderland dataset contains 435 urban scenes with fisheye images, undistorted images, 3D reconstructions, and navigation data for embodied AI research.

> **Note on Dataset Release (v1)**: The current public release contains **435 scenes**, which is a subset of the total scenes referenced in our paper. We have reserved some scenes for future research and challenge development. Wanderland is an ongoing project, and we are actively working to expand the dataset to **over 1,000 scenes** with additional modalities and annotations. Stay tuned for updates!

## Quick Start

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Download evaluation scenes for novel view synthesis
python download.py --modality nvs --scene-list eval_scenes_v1.txt

# Download first 5 training scenes for 3D reconstruction
python download.py --modality 3d --scene-list train_scenes_v1.txt --count 5

# Download the entire dataset with full modalities
python download.py --modality full --all
```
For more downloading options and examples, see the [Usage Examples](#usage-examples) section below, or run `python download.py --help`.

## Dataset Overview

The Wanderland dataset is available on HuggingFace at [`ai4ce/wanderland`](https://huggingface.co/datasets/ai4ce/wanderland). Each scene contains:

- **Images**: Original fisheye images and undistorted pinhole-projection images (800×800, 90° FOV)
- **Masks**: Validity masks for both fisheye and undistorted images
- **3D Data**: Point clouds, 3D Gaussian Splatting (3DGS) reconstructions, and COLMAP sparse models
- **Camera Parameters**: Intrinsics, extrinsics, and transformations in `transforms.json`
- **Splits**: Train/validation splits for novel view synthesis (per-scene image splits)
- **Navigation**: Isaac Sim compatible scene files (USDZ) and episode configurations

## Download Modalities

The download tool supports four modalities optimized for different tasks:

### `3d` - 3D Reconstruction
Downloads minimal data for 3D reconstruction methods:
- `images.tar.gz` - Undistorted images (800×800 PNG)
- `images_mask.tar.gz` - Validity masks for images
- `sparse/` - COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)

**Use case**: Training/testing 3D reconstruction algorithms (e.g., NeRF, 3DGS, MVS)

### `nvs` - Novel View Synthesis
Downloads 3D modality plus additional data for rendering:
- All files from `3d` modality
- `raw_pcd.ply` - Dense point cloud with RGB colors
- `3dgs.ply` - Pre-trained 3D Gaussian Splatting model
- `nvs_split/` - Train/validation image splits (train.txt, val.txt)

**Use case**: Novel view synthesis, view interpolation, 3D scene rendering

### `navigation` - Navigation Tasks
Downloads only navigation-specific files for Isaac Sim:
- `scene.usdz` - USD scene file compatible with Isaac Sim
- `episodes.json` - Navigation episode configurations

**Use case**: Embodied AI navigation, Isaac Sim simulation

### `full` - Complete Dataset
Downloads all available data for a scene (all files above plus fisheye images).

## Dataset Structure

After downloading, each scene will have the following structure:

```
wanderland_data/
└── <scene_name>/
    ├── fisheye.tar.gz           # Original fisheye images (JPG) [full only]
    ├── fisheye_mask.tar.gz      # Fisheye image masks (PNG) [full only]
    ├── images/                  # Undistorted images (PNG, 800×800) [extracted from tar.gz]
    ├── images_mask/             # Undistorted masks (PNG, 800×800) [extracted from tar.gz]
    ├── raw_pcd.ply              # Dense point cloud (PLY format) [nvs/full]
    ├── 3dgs.ply                 # 3D Gaussian Splatting model [nvs/full]
    ├── transforms.json          # Camera parameters and transformations [full]
    ├── scene.usdz               # Isaac Sim scene file [navigation/full]
    ├── episodes.json            # Navigation episodes [navigation/full]
    ├── sparse/                  # COLMAP sparse reconstruction [3d/nvs/full]
    │   └── 0/
    │       ├── cameras.bin      # Camera intrinsics
    │       ├── images.bin       # Camera poses
    │       └── points3D.bin     # Sparse 3D points
    └── nvs_split/               # Train/validation splits [nvs/full]
        ├── train.txt            # Training image list
        └── val.txt              # Validation image list
```

## File Descriptions

### Image Data

**`images/`** (extracted from images.tar.gz)
- Undistorted pinhole-projection images
- Format: PNG files, 800×800 pixels
- Field of View: 90 degrees
- Naming: `{camera_side}_{timestamp}.png` (e.g., `left_1234567890.png`, `right_1234567890.png`)
- Dual camera setup from left and right fisheye cameras

**`images_mask/`** (extracted from images_mask.tar.gz)
- Validity masks for undistorted images
- Format: PNG files (grayscale), 800×800 pixels
- Black pixels indicate invalid regions (outside fisheye FOV)

**`fisheye/`** and **`fisheye_mask/`** (full modality only)
- Original fisheye images and masks before undistortion
- Format: JPG (images), PNG (masks)

### 3D Reconstruction Data

**`raw_pcd.ply`**
- Dense 3D point cloud of the scene
- Format: PLY (Polygon File Format)
- Contains: XYZ coordinates and RGB color information
- Coordinate system: Aligned with COLMAP camera poses

**`3dgs.ply`**
- Pre-trained 3D Gaussian Splatting model
- Format: PLY with Gaussian parameters (position, scale, rotation, color)
- Ready for real-time rendering with 3DGS viewers

**`sparse/0/`**
- COLMAP sparse reconstruction in binary format
- **`cameras.bin`**: Camera intrinsics (PINHOLE model: fx=fy=400, cx=cy=400 for 800×800 images)
- **`images.bin`**: Camera poses (rotation quaternion + translation)
- **`points3D.bin`**: Sparse 3D points from structure-from-motion

### Camera Parameters

**`transforms.json`**
- Complete camera parameters for all frames
- JSON structure:
```json
{
  "frames": [
    {
      "file_path": "left/timestamp.png",
      "transform_matrix": [[...], [...], [...], [...]],  // 4×4 camera-to-world
      "w": 1920,                    // Original fisheye width
      "h": 1080,                    // Original fisheye height
      "fl_x": 500.0,                // Fisheye focal length X
      "fl_y": 500.0,                // Fisheye focal length Y
      "cx": 960.0,                  // Fisheye principal point X
      "cy": 540.0,                  // Fisheye principal point Y
      "k1": -0.5, "k2": 0.2,        // Fisheye distortion coefficients
      "k3": -0.1, "k4": 0.05
    }
  ]
}
```

### Navigation Data

**`scene.usdz`**
- USD (Universal Scene Description) format scene file
- Compatible with NVIDIA Isaac Sim
- Contains 3D geometry, materials, and scene structure

**`episodes.json`**
- Navigation episode configurations
- Defines start/goal positions and trajectories for navigation tasks

### Data Splits

**`nvs_split/`** (Per-scene image splits)
- **`train.txt`**: List of training images for this scene
- **`val.txt`**: List of validation images for this scene
- Format: One filename per line (e.g., `left_1234567890.png`)
- Purpose: Split images within a scene for novel view synthesis

**Note**: This is different from scene-level splits (train_scenes_v1.txt / eval_scenes_v1.txt) which divide scenes for 3D reconstruction benchmarking.

## Scene Splits

The dataset provides two types of splits:

### Scene-Level Splits (3D Reconstruction)
Used to divide scenes into training and evaluation sets for 3D reconstruction benchmarks.

- **`train_scenes_v1.txt`**: 235 training scenes
- **`eval_scenes_v1.txt`**: 200 evaluation scenes

These splits correspond to the exact division used in the Wanderland paper.

### Image-Level Splits (Novel View Synthesis)
Each scene contains `nvs_split/train.txt` and `nvs_split/val.txt` that divide the images within that scene for novel view synthesis tasks.

## Usage Examples

### Example 1: Download Evaluation Scenes for NVS Benchmark
```bash
# Download all 200 evaluation scenes with NVS data
python download.py --modality nvs --scene-list eval_scenes_v1.txt
```

### Example 2: Download Training Scenes for 3D Reconstruction
```bash
# Download first 10 training scenes with 3D reconstruction data
python download.py --modality 3d --scene-list train_scenes_v1.txt --count 10
```

### Example 3: Download Navigation Data for Isaac Sim
```bash
# Download navigation files for all scenes
python download.py --modality navigation --all --output ../nav_data
```

### Example 4: Download Specific Scenes
```bash
# Download specific scenes by name
python download.py --modality full \
  --scenes 1-A_d1uPKpnDksrjY3UE23dUTC0odvnHu 1-j7j0xj8vB0uYpWlByv0PvyRCawz6WXH
```

### Example 5: Download Custom Scene List
```bash
# Create custom scene list
cat > my_scenes.txt << EOF
1-A_d1uPKpnDksrjY3UE23dUTC0odvnHu
10D5SVlAs1uRVz6KGIOb5qkmt8sM1dKpn
10Q-ZXPgoc6vor8qFSb90jCuNA72KonzW
EOF

# Download custom scenes
python download.py --modality nvs --scene-list my_scenes.txt
```

## Image Naming Convention

All images follow a consistent naming pattern:

- **Fisheye images**: `{camera_side}_{timestamp}.jpg`
- **Undistorted images**: `{camera_side}_{timestamp}.png`
- **Masks**: `{camera_side}_{timestamp}.png`
- **In transforms.json**: `{camera_side}/{timestamp}.png`

Where:
- `camera_side`: Either `left` or `right`
- `timestamp`: Unix timestamp or frame identifier

## Camera Models

### Fisheye Camera (Original)
- Distortion model: 4-parameter fisheye (k1, k2, k3, k4)
- Intrinsics vary by scene (see `transforms.json`)
- Resolution: Typically 2K

### Undistorted Camera (Processed)
- Model: PINHOLE (rectilinear projection)
- Intrinsics: fx=fy=400.0, cx=cy=400.0
- Resolution: 800×800 pixels
- Field of view: 90 degrees

## Coordinate System

- Camera poses follow COLMAP convention (camera-to-world transformation)
- Point cloud coordinates aligned with camera coordinate system
- Right-handed coordinate system
- Units: Meters
