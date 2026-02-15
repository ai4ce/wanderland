# Navigation

This module contains **navigation episode generation** and **(optional) video rendering** utilities.

## Installation

```bash
cd navigation
uv sync
source .venv/bin/activate
```

## Generate Episodes

Episodes are written under `<paths_root>/<scene_id>/nav/episodes.json`.

```bash
SCENE_ID="YOUR_SCENE_ID"
DATA_DIR="../wanderland_data/${SCENE_ID}"

python generate_nav_episodes.py \
  --pinhole-scene-dir "${DATA_DIR}" \
  --raw-scene-dir "${DATA_DIR}" \
  --output-root ../paths \
  --scene-id "${SCENE_ID}"
```

## Generate Camera Pose Sequences (optional)

```bash
SCENE_ID="YOUR_SCENE_ID"

python generate_cam_poses.py \
  --scene "${SCENE_ID}" \
  --paths-root ../paths
```

## Render Navigation Videos (run with the reconstruction environment)

Rendering uses `gsplat` and should be run in the **reconstruction** venv:

```bash
cd ../reconstruction
uv sync
source .venv/bin/activate
uv pip install -e ./gsplat --no-build-isolation

SCENE_ID="YOUR_SCENE_ID"
CKPT="/path/to/ckpt_*.pt"

python ../navigation/render_nav_videos.py \
  --scene "${SCENE_ID}" \
  --paths-root ../paths \
  --ckpt "${CKPT}" \
  --output-dir "../paths/${SCENE_ID}/nav/videos"
```
