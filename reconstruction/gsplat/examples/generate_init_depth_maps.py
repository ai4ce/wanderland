#!/usr/bin/env python
"""Generate per-frame dense expected-depth maps for training supervision.

This script reuses the examples.simple_trainer Config/Runner to ensure camera
intrinsics/extrinsics match training. For each image in the dataset, it renders
RGB+ED (expected depth) once and saves a `[H, W]` float tensor to
`<data_dir>/depths/<image_basename>.pt`.
"""

import os
import torch
import tyro
import tqdm

from simple_trainer import Config, Runner


def main() -> None:
    # Parse CLI as simple_trainer Config for consistent dataset/cameras.
    cfg: Config = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    depths_dirname = os.environ.get("DEPTHS_DIRNAME", "depths").strip() or "depths"

    # Force render-only depth generation.
    cfg.max_steps = 0
    cfg.depth_loss = False
    cfg.disable_viewer = True
    cfg.export_init_splats = True
    if cfg.init_splats_path is None:
        cfg.init_splats_path = os.path.join(cfg.data_dir, depths_dirname, "full_init_splats.pt")
    cfg.init_splats_include_sh = False
    if cfg.depth_uniform_fallback:
        print("Using uniform opacity fallback for unobserved pixels.")

    # Initialize runner and parser (no training performed).
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    parser = runner.parser

    out_dir = os.path.join(cfg.data_dir, depths_dirname)
    os.makedirs(out_dir, exist_ok=True)

    only_split = (os.environ.get("DEPTH_ONLY_SPLIT", "").strip().lower() or "all")
    if only_split in ("train", "val"):
        if not hasattr(parser, "split_indices") or only_split not in parser.split_indices:
            raise RuntimeError(
                f"DEPTH_ONLY_SPLIT={only_split} requested but parser.split_indices missing; "
                f"did you prepare COLMAP nvs_split artifacts (images.bin/images_val.bin)?"
            )
        indices = list(map(int, parser.split_indices[only_split].tolist()))
    elif only_split in ("all", ""):
        indices = list(range(len(parser.image_names)))
    else:
        raise ValueError(f"Unknown DEPTH_ONLY_SPLIT={only_split!r} (expected all/train/val).")

    max_frames_s = os.environ.get("DEPTH_MAX_FRAMES", "").strip()
    if max_frames_s:
        max_frames = int(max_frames_s)
        if max_frames > 0:
            indices = indices[:max_frames]

    for idx in tqdm.tqdm(indices, desc="Generating depth maps"):
        img_name = parser.image_names[idx]
        # Per-frame camera pose and intrinsics
        c2w = torch.from_numpy(parser.camtoworlds[idx][None]).float().to(runner.device)
        cam_id = parser.camera_ids[idx]
        K = torch.from_numpy(parser.Ks_dict[cam_id][None]).float().to(runner.device)
        width, height = parser.imsize_dict[cam_id]

        # Render RGB+ED once at full resolution
        renders, alphas, _ = runner.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=width,
            height=height,
            sh_degree=0,
            render_mode="RGB+ED",
        )
        depth_primary = renders[0, ..., 3]
        depth_map = depth_primary.detach().cpu()

        if cfg.depth_uniform_fallback:
            alpha_acc = alphas[0]
            if alpha_acc.ndim > 2:
                alpha_acc = alpha_acc.squeeze(-1)
            coverage_mask = alpha_acc > 1e-4

            if not bool(coverage_mask.all()):
                with torch.no_grad():
                    original_opacities = runner.splats["opacities"].detach().clone()
                    init_opa_val = float(cfg.init_opa)
                    init_opa_val = min(max(init_opa_val, 1e-2), 1.0 - 1e-2)
                    logit_val = torch.logit(
                        torch.tensor(init_opa_val, device=runner.splats["opacities"].device)
                    )
                    runner.splats["opacities"].data.fill_(logit_val.item())

                fallback_renders, _, _ = runner.rasterize_splats(
                    camtoworlds=c2w,
                    Ks=K,
                    width=width,
                    height=height,
                    sh_degree=0,
                    render_mode="RGB+ED",
                )
                fallback_depth = fallback_renders[0, ..., 3]

                with torch.no_grad():
                    runner.splats["opacities"].data.copy_(original_opacities)

                fallback_cpu = fallback_depth.detach().cpu()
                hole_mask = ~coverage_mask.cpu()
                fallback_depth_mask = fallback_cpu >= 5.0
                fill_mask = hole_mask & fallback_depth_mask
                depth_map[fill_mask] = fallback_cpu[fill_mask]

        mask = torch.isfinite(depth_map)
        if mask.any():
            max_val = depth_map[mask].max()
            depth_map[~mask] = max_val
        else:
            depth_map[...] = 0.0

        base = os.path.splitext(img_name)[0]
        torch.save(depth_map.half(), os.path.join(out_dir, f"{base}.pt"))

    print(f"Saved dense depth maps to: {out_dir}")


if __name__ == "__main__":
    main()
