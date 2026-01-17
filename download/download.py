#!/usr/bin/env python3

import os
import sys
import argparse
import tarfile
import shutil
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# Constants
DATASET_REPO = "ai4ce/wanderland"


def list_available_scenes():
    """Query HuggingFace API for available scenes."""
    print(f"Fetching available scenes from {DATASET_REPO}...")
    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")

    scenes = set()
    for file in files:
        # Look for files under data/{scene_name}/
        if file.startswith("data/"):
            parts = file.split("/")
            if len(parts) >= 2:
                scene_name = parts[1]
                scenes.add(scene_name)

    return sorted(list(scenes))


def download_file_from_hf(repo_id, filename, local_dir):
    """Download individual file from HuggingFace."""
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_dir,
        )
        return downloaded_path
    except Exception as e:
        print(f"Warning: Failed to download {filename}: {e}")
        return None


def download_directory_from_hf(repo_id, scene_name, directory_name, output_dir):
    """Download entire directory from HuggingFace using snapshot_download."""
    try:
        # Download only files matching the directory pattern
        pattern = f"data/{scene_name}/{directory_name}/**"

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=pattern,
            local_dir=output_dir,
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to download directory {directory_name}: {e}")
        return False


def untar_file(tar_path, extract_dir, remove_after=True):
    """Extract tar.gz archive and optionally remove it."""
    print(f"  Extracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)

        if remove_after:
            tar_path.unlink()

        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False


def download_scene(scene_name, modality, output_dir):
    """
    Download a scene based on the specified modality.
    
    Args:
        scene_name: Name of the scene to download
        modality: Download modality ('3d', 'nvs', 'navigation', or 'full')
        output_dir: Root output directory
    """
    print(f"\n{'='*60}")
    print(f"Downloading scene: {scene_name} (modality: {modality})")
    print(f"{'='*60}")

    scene_output_dir = output_dir / scene_name
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    if modality == "full":
        # Download entire scene directory
        print("Downloading full scene...")
        pattern = f"data/{scene_name}/**"

        try:
            snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                allow_patterns=pattern,
                local_dir=output_dir,
            )

            # Move files from data/{scene_name}/ to {scene_name}/
            downloaded_scene_dir = output_dir / "data" / scene_name

            if downloaded_scene_dir.exists():
                # Move all contents
                for item in downloaded_scene_dir.iterdir():
                    dest = scene_output_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))

                # Clean up empty data directory
                downloaded_scene_dir.rmdir()
                (output_dir / "data").rmdir()

            # Extract all tar.gz files
            for tar_file in scene_output_dir.glob("*.tar.gz"):
                untar_file(tar_file, scene_output_dir, remove_after=True)

            print(f"Full scene downloaded to {scene_output_dir}")

        except Exception as e:
            print(f"Error downloading full scene: {e}")
            return False

    elif modality == "navigation":
        # Download only navigation-related files for Isaac Sim
        files_to_download = [
            f"data/{scene_name}/scene.usdz",
            f"data/{scene_name}/episodes.json",
        ]

        # Download files
        for file_path in files_to_download:
            print(f"Downloading {Path(file_path).name}...")
            downloaded = download_file_from_hf(
                DATASET_REPO,
                file_path,
                output_dir
            )

            if downloaded:
                # Move from data/{scene_name}/ to {scene_name}/
                src = output_dir / file_path
                dst = scene_output_dir / Path(file_path).name

                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))

        # Clean up data directory structure
        data_dir = output_dir / "data" / scene_name
        if data_dir.exists() and not any(data_dir.iterdir()):
            data_dir.rmdir()
        data_parent = output_dir / "data"
        if data_parent.exists() and not any(data_parent.iterdir()):
            data_parent.rmdir()

        print(f"Navigation files downloaded to {scene_output_dir}")

    elif modality in ["3d", "nvs"]:
        # Download specific files based on modality
        files_to_download = []
        dirs_to_download = []

        # Common files for both modes
        files_to_download.extend([
            f"data/{scene_name}/images.tar.gz",
            f"data/{scene_name}/images_mask.tar.gz",
        ])
        dirs_to_download.append("sparse")

        if modality == "nvs":
            # Additional files for NVS mode
            files_to_download.extend([
                f"data/{scene_name}/raw_pcd.ply",
                f"data/{scene_name}/3dgs.ply",
            ])
            dirs_to_download.extend(["nvs_split"])

        # Download files
        for file_path in files_to_download:
            print(f"Downloading {Path(file_path).name}...")
            downloaded = download_file_from_hf(
                DATASET_REPO,
                file_path,
                output_dir
            )

            if downloaded:
                # Move from data/{scene_name}/ to {scene_name}/
                src = output_dir / file_path
                dst = scene_output_dir / Path(file_path).name

                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))

        # Download directories
        for dir_name in dirs_to_download:
            print(f"Downloading {dir_name}/ directory...")
            download_directory_from_hf(
                DATASET_REPO,
                scene_name,
                dir_name,
                output_dir
            )

            # Move from data/{scene_name}/{dir_name}/ to {scene_name}/{dir_name}/
            src_dir = output_dir / "data" / scene_name / dir_name
            dst_dir = scene_output_dir / dir_name

            if src_dir.exists():
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.move(str(src_dir), str(dst_dir))

        # Clean up data directory structure
        data_dir = output_dir / "data" / scene_name
        if data_dir.exists() and not any(data_dir.iterdir()):
            data_dir.rmdir()
        data_parent = output_dir / "data"
        if data_parent.exists() and not any(data_parent.iterdir()):
            data_parent.rmdir()

        # Extract tar.gz files
        for tar_file in scene_output_dir.glob("*.tar.gz"):
            untar_file(tar_file, scene_output_dir, remove_after=True)

        print(f"Scene downloaded to {scene_output_dir}")

    return True


def load_scene_list_from_file(file_path):
    """Load scene names from a text file (one per line, ignoring comments)."""
    with open(file_path, 'r') as f:
        scenes = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return scenes


def main():
    parser = argparse.ArgumentParser(
        description="Download Wanderland dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 5 scenes for 3D reconstruction
  python download.py --modality 3d --count 5

  # Download specific scenes for novel view synthesis
  python download.py --modality nvs --scenes scene_001 scene_002

  # Download all scenes (full data)
  python download.py --modality full --all

  # Download scenes from a list file for navigation tasks
  python download.py --modality navigation --scene-list train_scenes_v1.txt

  # Download evaluation scenes for benchmarking
  python download.py --modality nvs --scene-list eval_scenes_v1.txt --output ../wanderland_data
        """
    )

    # Modality selection
    parser.add_argument(
        "--modality",
        choices=["3d", "nvs", "navigation", "full"],
        required=True,
        help="Download modality: '3d' (images+masks+sparse for 3D reconstruction), "
             "'nvs' (3d+point clouds+splits for novel view synthesis), "
             "'navigation' (Isaac Sim files: scene.usdz+episodes.json), "
             "'full' (everything)"
    )

    # Output directory
    parser.add_argument(
        "--output",
        type=str,
        default="../wanderland_data",
        help="Output directory for downloaded data (default: ../wanderland_data)"
    )

    # Scene selection (mutually exclusive)
    scene_group = parser.add_mutually_exclusive_group(required=True)
    scene_group.add_argument(
        "--count",
        type=int,
        help="Download first N scenes"
    )
    scene_group.add_argument(
        "--scenes",
        nargs="+",
        help="Download specific scenes by name"
    )
    scene_group.add_argument(
        "--all",
        action="store_true",
        help="Download all available scenes"
    )
    scene_group.add_argument(
        "--scene-list",
        type=str,
        help="Path to text file containing scene names (one per line, # for comments)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(f"Wanderland Dataset Downloader")
    print(f"Modality: {args.modality}")
    print(f"Output: {output_dir.absolute()}")
    print("="*60)

    # Get list of available scenes
    available_scenes = list_available_scenes()
    print(f"Found {len(available_scenes)} available scenes")

    # Determine which scenes to download
    scenes_to_download = []

    if args.all:
        scenes_to_download = available_scenes
        print(f"Downloading all {len(scenes_to_download)} scenes")

    elif args.count:
        scenes_to_download = available_scenes[:args.count]
        print(f"Downloading first {len(scenes_to_download)} scenes")

    elif args.scenes:
        # Validate scene names
        invalid_scenes = []
        for scene in args.scenes:
            if scene in available_scenes:
                scenes_to_download.append(scene)
            else:
                invalid_scenes.append(scene)

        if invalid_scenes:
            print(f"Warning: Invalid scene names: {invalid_scenes}")

        if not scenes_to_download:
            print("Error: No valid scenes specified")
            sys.exit(1)

        print(f"Downloading {len(scenes_to_download)} specified scenes")

    elif args.scene_list:
        # Load scenes from file
        if not Path(args.scene_list).exists():
            print(f"Error: Scene list file not found: {args.scene_list}")
            sys.exit(1)

        requested_scenes = load_scene_list_from_file(args.scene_list)

        # Validate scene names
        invalid_scenes = []
        for scene in requested_scenes:
            if scene in available_scenes:
                scenes_to_download.append(scene)
            else:
                invalid_scenes.append(scene)

        if invalid_scenes:
            print(f"Warning: Invalid scene names in file: {invalid_scenes}")

        if not scenes_to_download:
            print("Error: No valid scenes in list file")
            sys.exit(1)

        print(f"Downloading {len(scenes_to_download)} scenes from list file")

    # Download each scene
    success_count = 0
    fail_count = 0

    for idx, scene_name in enumerate(scenes_to_download, start=1):
        print(f"\n[Scene {idx}/{len(scenes_to_download)}]")

        try:
            success = download_scene(scene_name, args.modality, output_dir)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            fail_count += 1

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successfully downloaded: {success_count} scenes")
    if fail_count > 0:
        print(f"Failed: {fail_count} scenes")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
