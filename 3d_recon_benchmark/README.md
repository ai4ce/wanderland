# Wanderland-Recon: 3D Reconstruction Benchmark

A comprehensive pose estimation evaluation tool for COLMAP sparse reconstructions. This tool compares predicted camera poses against ground truth poses and computes standard metrics including ATE, RTE, and AUC@30.

## Features

- **Format Support**: Automatically detects and reads both COLMAP `.txt` and `.bin` formats
- **Comprehensive Metrics**:
  - ATE (Absolute Trajectory Error): Translation and rotation errors
  - RTE (Relative Trajectory Error): Pairwise translation and rotation errors
  - AUC@30: Area Under Curve metric for pose accuracy
  - Reconstruction completeness ratio
- **Pose Alignment**: Supports both SE(3) and SE(3)+scale alignment
- **COLMAP Convention**: Works with COLMAP sparse reconstruction format

## Installation

```bash
cd 3d_recon_benchmark
uv sync
```

## Usage

### Basic Usage

Evaluate pose estimation by comparing predicted and ground truth COLMAP reconstructions:

```bash
uv run python eval_pose.py <pred_dir> <gt_dir> [--output <output_file>]
```

### Arguments

- `pred_dir`: Directory containing prediction poses in COLMAP sparse format (must contain `images.txt` or `images.bin`)
- `gt_dir`: Directory containing ground truth poses in COLMAP sparse format (must contain `images.txt` or `images.bin`)
- `--output`, `-o`: (Optional) Output file for results (default: `pose_evaluation_results.txt`)

### Example

```bash
# Evaluate poses with default output file
uv run python eval_pose.py /path/to/predicted/sparse/0 /path/to/ground_truth/sparse/0

# Evaluate poses with custom output file
uv run python eval_pose.py /path/to/predicted/sparse/0 /path/to/ground_truth/sparse/0 --output my_results.txt
```

## Input Format

The script expects COLMAP sparse reconstruction format with either:
- `images.bin` (binary format)
- `images.txt` (text format)

The script automatically detects which format is available and reads accordingly. COLMAP sparse directories typically also contain `cameras.bin`/`cameras.txt` and `points3D.bin`/`points3D.txt`, but this script only requires the images file.

### Directory Structure Example

```
predicted/sparse/0/
├── cameras.bin
├── images.bin      # Required
└── points3D.bin

ground_truth/sparse/0/
├── cameras.bin
├── images.bin      # Required
└── points3D.bin
```
