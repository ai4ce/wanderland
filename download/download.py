#!/usr/bin/env python3

import sys
import argparse
import csv
import tarfile
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# Constants
DATASET_REPO = "ai4ce/wanderland"
PUBLIC_MANIFEST_FILE = "wanderland_public_manifest.csv"


def clean_manifest_value(row, field):
    """Return a manifest field as a stripped string, treating missing values as empty."""
    return (row.get(field) or "").strip()


def load_public_manifest():
    """Load the public manifest that defines the currently released scenes."""
    print(f"Fetching public manifest from {DATASET_REPO}/{PUBLIC_MANIFEST_FILE}...")
    try:
        manifest_path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=PUBLIC_MANIFEST_FILE,
        )
    except Exception as e:
        print(f"Error: Failed to download public manifest: {e}")
        sys.exit(1)

    required_columns = {"scene_id", "quality_tier"}
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            print(
                "Error: Public manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
            sys.exit(1)

        for row in reader:
            scene_id = clean_manifest_value(row, "scene_id")
            if not scene_id:
                continue
            row["scene_id"] = scene_id
            row["quality_tier"] = clean_manifest_value(row, "quality_tier")
            rows.append(row)

    if not rows:
        print("Error: Public manifest contains no scenes")
        sys.exit(1)

    return rows


def describe_public_manifest(rows):
    """Print a concise summary of the loaded public manifest."""
    versions = sorted({clean_manifest_value(row, "manifest_version") for row in rows if clean_manifest_value(row, "manifest_version")})
    updated_at = sorted({clean_manifest_value(row, "manifest_updated_at") for row in rows if clean_manifest_value(row, "manifest_updated_at")})
    tiers = Counter(clean_manifest_value(row, "quality_tier") or "unspecified" for row in rows)

    print(f"Found {len(rows)} released scenes in public manifest")
    if versions:
        print(f"Manifest version: {', '.join(versions)}")
    if updated_at:
        print(f"Manifest updated at: {', '.join(updated_at)}")
    print("Quality tiers: " + ", ".join(f"{tier}={count}" for tier, count in sorted(tiers.items())))


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
  # Recommended: download showcase scenes for quick inspection
  python download.py --modality nvs --quality-tier showcase

  # Download evaluation-quality scenes for novel view synthesis
  python download.py --modality nvs --quality-tier evaluation_ready

  # Download first 5 training-ready scenes for 3D reconstruction
  python download.py --modality 3d --quality-tier training_ready --count 5

  # Download specific scenes for novel view synthesis
  python download.py --modality nvs --scenes scene_001 scene_002

  # Explicitly download every scene listed in the public manifest
  python download.py --modality full --all
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

    # Scene selection (mutually exclusive, with --count as an optional limiter)
    scene_group = parser.add_mutually_exclusive_group()
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
    scene_group.add_argument(
        "--quality-tier",
        type=str,
        help="Download scenes from a public manifest quality tier, e.g. showcase, "
             "evaluation_ready, or training_ready"
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Limit the selected scenes to the first N entries"
    )

    args = parser.parse_args()

    if args.quality_tier is not None:
        args.quality_tier = args.quality_tier.strip()
        if not args.quality_tier:
            parser.error("--quality-tier must be a non-empty string")

    if args.count is not None and args.count <= 0:
        parser.error("--count must be a positive integer")

    if not any([
        args.scenes,
        args.all,
        args.scene_list,
        args.quality_tier,
        args.count is not None,
    ]):
        parser.error(
            "choose a scene selector: --quality-tier, --scene-list, --scenes, --all, or --count"
        )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(f"Wanderland Dataset Downloader")
    print(f"Modality: {args.modality}")
    print(f"Output: {output_dir.absolute()}")
    print("="*60)

    # Get list of currently released scenes from the public manifest.
    manifest_rows = load_public_manifest()
    describe_public_manifest(manifest_rows)
    available_scenes = [row["scene_id"] for row in manifest_rows]
    available_scene_set = set(available_scenes)

    # Determine which scenes to download
    scenes_to_download = []

    if args.all:
        scenes_to_download = available_scenes
        print(f"Selected all {len(scenes_to_download)} manifest scenes")

    elif args.quality_tier:
        scenes_to_download = [
            row["scene_id"]
            for row in manifest_rows
            if row.get("quality_tier", "") == args.quality_tier
        ]

        if not scenes_to_download:
            available_tiers = sorted({
                clean_manifest_value(row, "quality_tier")
                for row in manifest_rows
                if clean_manifest_value(row, "quality_tier")
            })
            print(f"Error: No scenes found for quality tier '{args.quality_tier}'")
            print(f"Available quality tiers: {', '.join(available_tiers)}")
            sys.exit(1)

        print(f"Selected {len(scenes_to_download)} scenes from quality tier '{args.quality_tier}'")

    elif args.scenes:
        # Validate scene names
        invalid_scenes = []
        for scene in args.scenes:
            if scene in available_scene_set:
                scenes_to_download.append(scene)
            else:
                invalid_scenes.append(scene)

        if invalid_scenes:
            print(f"Warning: Scene names not found in public manifest: {invalid_scenes}")

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
            if scene in available_scene_set:
                scenes_to_download.append(scene)
            else:
                invalid_scenes.append(scene)

        if invalid_scenes:
            print(f"Warning: Scene names not found in public manifest: {invalid_scenes}")

        if not scenes_to_download:
            print("Error: No valid scenes in list file")
            sys.exit(1)

        print(f"Downloading {len(scenes_to_download)} scenes from list file")

    elif args.count:
        scenes_to_download = available_scenes[:args.count]
        print(f"Selected first {len(scenes_to_download)} manifest scenes")

    count_only_selection = args.count is not None and not any([
        args.scenes,
        args.all,
        args.scene_list,
        args.quality_tier,
    ])
    if args.count is not None and not count_only_selection:
        original_count = len(scenes_to_download)
        scenes_to_download = scenes_to_download[:args.count]
        print(f"Applying --count {args.count}: {len(scenes_to_download)} of {original_count} selected scenes")

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
