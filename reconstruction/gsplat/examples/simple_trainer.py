import inspect
import json
import math
import os
import random
import time
import wandb
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio.v2 as imageio
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from difix_pipeline import (
    export_difix_splat_snapshot,
    load_difix_depth_state,
    load_difix_pipeline,
    render_difix_depth,
)
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Split strategy for train/val data
    split_mode: Literal["nvs_split", "test_every"] = "nvs_split"
    # When split_mode="nvs_split", how to derive the effective train/val splits.
    #
    # Intended usage:
    # - default    : Debugging / sanity checks (uses COLMAP base splits)
    # - interp_val : Experiments (interpolation-style evaluation; val sampled from base_train)
    # - extrap_val : Experiments (extrapolation-style evaluation; val uses base_val)
    # - full_train : Final export only (train on base_train+base_val, still eval on base_val)
    nvs_split_profile: Literal["default", "interp_val", "extrap_val", "full_train"] = "default"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 8
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Interval for evaluation during training (0 to disable)
    eval_interval: int = 500
    # Number of frames to evaluate during training (None for all)
    eval_num_frames: Optional[int] = 64
    # When running eval-only via --ckpt, optionally save GT|render side-by-side PNGs.
    # Images are written under: <result_dir>/renders/val_step<STEP>_<IDX>.png
    eval_save_images: bool = False
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [5_000, 9_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 19_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Difix integration (optional)
    enable_difix: bool = False
    difix_model_id: str = "nvidia/difix_ref"
    difix_prompt: str = "remove degradation"
    difix_fix_steps: List[int] = field(default_factory=list)
    difix_num_inference_steps: int = 1
    difix_timestep: int = 199
    difix_guidance_scale: float = 0.0
    difix_use_fp16: bool = True
    difix_seed: Optional[int] = None
    difix_max_frames: Optional[int] = 64
    difix_rotation_weight: float = 1.0
    difix_translation_weight: float = 1.0
    difix_novel_sampling_prob: float = 0.3
    difix_novel_lambda: float = 0.3
    difix_viz_max: int = 16
    difix_use_val: bool = False
    difix_jitter_translation: float = 0.02
    difix_jitter_rotation_deg: float = 30.0
    difix_jitter_min_scale: float = 0.25

    # Snapshot export / reuse for high-fidelity depth rendering
    export_init_splats: bool = False
    init_splats_path: Optional[str] = None
    init_splats_dtype: Literal["float16", "float32"] = "float16"
    init_splats_include_sh: bool = False

    # wandb configuration
    use_wandb: bool = False
    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Use density-adaptive initialization for opacities
    use_adaptive_init_opacity: bool = True
    # Initial scale of GS
    init_scale: float = 1.0
    # Number of nearest neighbors used to initialize GS scale from SfM points.
    # Note: includes self-distance, so effective neighbors is (init_knn_k - 1).
    init_knn_k: int = 6
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 1e-8
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-7
    # LR for Gaussian scale factors
    scales_lr: float = 1e-3
    # LR for alpha blending weights
    opacities_lr: float = 2e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 0.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0
    # Scale regularization
    scale_reg: float = 0.01

    # Enable camera optimization.
    pose_opt: bool = True
    # Learning rate for camera optimization
    pose_opt_lr: float = 2e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-3
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss.
    #
    # UrbanSim_Recon default: enabled (uses depths saved by prepare_scene when available).
    # Disable with CLI: --no-depth_loss
    depth_loss: bool = True
    # Use dense depth supervision from per-pixel maps saved under data_dir/depths.
    dense_depth_loss: bool = True
    # Perform a secondary uniform-opacity render to fill missing dense depth pixels.
    depth_uniform_fallback: bool = False
    # Weight for depth loss
    # Default tuned around 1e-3; override via CLI: --depth_lambda <float>
    depth_lambda: float = 1e-3
    # Debug visualization for dense depth supervision
    debug_depth_viz: bool = False
    debug_viz_interval: int = 500

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    init_knn_k: int = 6,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 8,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    use_adaptive_opacity: bool = True,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # init_type chosen via function argument
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        print(f"Using {points.shape[0]} points from SfM.")
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size using kNN distances.
    # `knn` returns shape [N, K] including self at index 0 (distance ~ 0).
    init_knn_k = int(init_knn_k)
    if init_knn_k < 2:
        raise ValueError(f"init_knn_k must be >= 2, got {init_knn_k}")
    dist2_avg = (knn(points, init_knn_k)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    dist_avg = dist_avg[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    # Scheme D: density ~ 1 / dist^3, Beer–Lambert alpha = 1 - exp(-kappa * rho_norm)
    if use_adaptive_opacity:
        rho = (dist_avg + 1e-12).pow(-3)
        # Robust 5/95% quantiles on a subsample to avoid huge-tensor quantile issues
        q = torch.tensor([0.05, 0.95], device=rho.device)
        sample = rho
        max_q_samples = 1_000_000
        if rho.numel() > max_q_samples:
            idx = torch.randint(rho.numel(), (max_q_samples,), device=rho.device)
            sample = rho[idx]
        lo, hi = torch.quantile(sample, q)
        rho_n = ((rho - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
        kappa = 2.0
        opa_vals = 1.0 - torch.exp(-kappa * rho_n)
        mean_opa = float(opa_vals.mean().item())
        if mean_opa < 1e-8:
            opa_vals = torch.full((N,), float(init_opacity), device=rho.device)
        else:
            scale = float(init_opacity) / (mean_opa + 1e-8)
            opa_vals = (opa_vals * scale).clamp(1e-3, 1.0 - 1e-8)
    else:
        opa_vals = torch.full(
            (N,),
            float(init_opacity),
            device=points.device,
            dtype=points.dtype,
        ).clamp(1e-4, 1.0 - 1e-4)
    opacities = torch.logit(opa_vals)  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        self.difix_enabled = cfg.enable_difix
        self.difix_use_val = cfg.difix_use_val
        if self.difix_enabled and world_size > 1:
            if world_rank == 0:
                print("[Difix] Distributed training is not yet supported; disabling Difix.")
            self.difix_enabled = False
            self.difix_use_val = False
        self.difix_fix_steps = set(cfg.difix_fix_steps)
        self.difix_pipeline = None
        self.difix_interpolator = None
        self.difix_novelloaders = []
        self.difix_novelloaders_iter = []
        self.difix_novel_sample_count = 0
        self._sample_near_pose = None
        self.difix_depth_state = None

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            split_mode=cfg.split_mode,
            nvs_split_profile=cfg.nvs_split_profile,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            load_dense_depths=cfg.dense_depth_loss,
            split_mode=cfg.split_mode,
        )
        self.valset = Dataset(self.parser, split="val", split_mode=cfg.split_mode)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.difix_enabled:
            from difix_utils import CameraPoseInterpolator, sample_nearby_pose

            dtype = torch.float16 if (cfg.difix_use_fp16 and torch.cuda.is_available()) else torch.float32
            device = torch.device(self.device)
            try:
                self.difix_pipeline = load_difix_pipeline(
                    model_id=cfg.difix_model_id,
                    device=device,
                    torch_dtype=dtype,
                )
            except RuntimeError as exc:
                raise RuntimeError("Failed to initialize Difix pipeline. "
                                   "Refer to the README for installation instructions.") from exc

            self.difix_interpolator = CameraPoseInterpolator(
                rotation_weight=cfg.difix_rotation_weight,
                translation_weight=cfg.difix_translation_weight,
            )
            self.difix_render_dir = Path(cfg.result_dir) / "renders" / "difix"
            self.difix_render_dir.mkdir(parents=True, exist_ok=True)
            self._sample_near_pose = sample_nearby_pose

            snapshot_path = (
                Path(cfg.init_splats_path)
                if cfg.init_splats_path is not None
                else Path(cfg.data_dir) / "depths" / "full_init_splats.pt"
            )
            if snapshot_path.exists():
                self.difix_depth_state = load_difix_depth_state(snapshot_path)
                if self.difix_depth_state is None:
                    print(
                        f"[Difix] Failed to prepare high-fidelity depth state from {snapshot_path}; falling back to training splats."
                    )
            else:
                print(
                    f"[Difix] Initial splat snapshot not found at {snapshot_path}; novel depth rendering will use current splats."
                )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            init_knn_k=cfg.init_knn_k,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            use_adaptive_opacity=cfg.use_adaptive_init_opacity,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        if cfg.export_init_splats and world_rank == 0:
            snapshot_path = (
                Path(cfg.init_splats_path)
                if cfg.init_splats_path is not None
                else Path(cfg.data_dir) / "depths" / "full_init_splats.pt"
            )
            dtype = torch.float16 if cfg.init_splats_dtype == "float16" else torch.float32
            export_difix_splat_snapshot(
                self.splats,
                path=snapshot_path,
                dtype=dtype,
                include_sh=cfg.init_splats_include_sh,
            )

        # Initialize WandB
        if world_rank == 0 and cfg.use_wandb:
            wandb.init(
                project="gsplat-training",
                name=f"{os.path.basename(cfg.result_dir)}_{time.strftime('%Y%m%d_%H%M%S')}",
                config={
                    "num_gaussians": len(self.splats["means"]),
                    "strategy": type(cfg.strategy).__name__,
                    **{f"lr_{k}": opt.param_groups[0]["lr"] for k, opt in self.optimizers.items()},
                    **vars(cfg),
                },
            )
            wandb.watch(self.splats, log="gradients", log_freq=200)

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        # Enforce RGB-only when using eval3d backend (kernel only supports 3 channels).
        rm = kwargs.get("render_mode", None)
        if self.cfg.with_eval3d and rm in ("RGB+ED", "RGB+D", "ED", "D"):
            kwargs["render_mode"] = "RGB"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            # Avoid in-place on tensors needed for grad; follow UrbanSim pattern
            render_colors = render_colors.clone()
            render_alphas = render_alphas.clone()
            render_colors[~masks] = 0
            render_alphas[~masks] = 1
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            data = None
            is_novel_data = False
            novel_prob = cfg.difix_novel_sampling_prob
            if self.difix_enabled and self.difix_novel_sample_count > 0:
                total_samples = len(self.trainset) + self.difix_novel_sample_count
                if total_samples > 0:
                    ratio = self.difix_novel_sample_count / total_samples
                    novel_prob = min(0.7, max(0.3, ratio))
            if (
                self.difix_enabled
                and self.difix_novelloaders
                and random.random() < novel_prob
            ):
                loader = self.difix_novelloaders[-1]
                iterator = self.difix_novelloaders_iter[-1]
                try:
                    data = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    self.difix_novelloaders_iter[-1] = iterator
                    data = next(iterator)
                is_novel_data = True

            if data is None:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)
                is_novel_data = False

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            alpha_masks = (
                data["alpha_mask"].to(device) if "alpha_mask" in data else None
            )
            has_dense_depth = (
                cfg.depth_loss
                and cfg.dense_depth_loss
                and "depth_map" in data
            )
            depth_map = None
            if has_dense_depth:
                depth_map = data["depth_map"].to(device).float()

            # Sparse depth is used when dense depth is disabled, or as a fallback when a dense
            # depth file is missing for an individual image (common on shared HPC filesystems).
            has_sparse_depth = (
                cfg.depth_loss
                and (not cfg.dense_depth_loss or not has_dense_depth)
                and "points" in data
                and "depths" in data
            )
            points = depths_gt = None
            if has_sparse_depth:
                points = data["points"].to(device)
                depths_gt = data["depths"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if is_novel_data and alpha_masks is not None:
                if alpha_masks.dim() == 3:
                    alpha_masks = alpha_masks.unsqueeze(-1)
                colors = colors * (alpha_masks > 0.5).float()
                pixels = pixels * (alpha_masks > 0.5).float()

            # Note: We follow UrbanSim semantics — GT already masked in Dataset,
            # and renders are masked explicitly below (render_colors[~masks]=0,
            # render_alphas[~masks]=1). No additional colors/pixels multiplication here.

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            rgb_loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            depthloss = torch.tensor(0.0, device=device)
            if has_dense_depth and depths is not None:
                # Dense per-pixel inverse-depth supervision
                pred_depth = depths.squeeze(-1)  # [B, H, W]
                finite_pred = pred_depth[torch.isfinite(pred_depth)]
                if finite_pred.numel() > 0:
                    max_pred = finite_pred.max()
                    pred_depth[~torch.isfinite(pred_depth)] = max_pred
                else:
                    pred_depth[...] = 0.0

                valid = (depth_map > 0.3) & (pred_depth > 0.3)
                if valid.any():
                    eps = 1e-2
                    inv_pred = 1.0 / pred_depth.clamp(min=eps)
                    inv_gt = 1.0 / depth_map.clamp(min=eps)
                    depthloss = F.l1_loss(inv_pred[valid], inv_gt[valid]) * self.scene_scale
                # Debug visualization (optional)
                if (
                    world_rank == 0
                    and cfg.debug_depth_viz
                    and step % max(cfg.debug_viz_interval, 1) == 0
                ):
                    debug_dir = f"{cfg.result_dir}/debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    with torch.no_grad():
                        rgb_sbs = torch.cat([pixels[0], colors[0].clamp(0, 1)], dim=1)
                        eps_v = 1e-2
                        inv_gt_v = (1.0 / (depth_map).clamp(min=eps_v))*valid
                        inv_pred_v = (1.0 / (pred_depth).clamp(min=eps_v))*valid
                        valid_v = (depth_map > 0) & (pred_depth > 0)
                        inv_gt_vis = (inv_gt_v * valid_v.float()).cpu()
                        inv_pred_vis = (inv_pred_v * valid_v.float()).cpu()
                        def _norm01(x: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
                            if msk.any():
                                vals = x[msk.cpu()]
                                xmin = vals.min()
                                xmax = vals.max()
                            else:
                                xmin = x.min()
                                xmax = x.max()
                            denom = (xmax - xmin).clamp(min=1e-10)
                            return (x - xmin) / denom
                        inv_gt_n = _norm01(inv_gt_vis, valid_v)
                        inv_pred_n = _norm01(inv_pred_vis, valid_v)
                        diff_n = (inv_gt_n - inv_pred_n).abs()
                        # print("shape:", rgb_sbs.shape, inv_gt_n.shape, inv_pred_n.shape, diff_n.shape) # shape: torch.Size([1008, 1520, 3]) torch.Size([1, 1008, 760]) torch.Size([1, 1008, 760]) torch.Size([1, 1008, 760])
                        inv_gt_rgb = inv_gt_n.squeeze(0).unsqueeze(-1).repeat(1, 1, 3)
                        inv_pred_rgb = inv_pred_n.squeeze(0).unsqueeze(-1).repeat(1, 1, 3)
                        diff_rgb = diff_n.squeeze(0).unsqueeze(-1).repeat(1, 1, 3)
                        row = torch.cat([rgb_sbs.cpu(), inv_gt_rgb, inv_pred_rgb, diff_rgb], dim=1)
                        out = (row.clamp(0, 1).numpy() * 255).astype(np.uint8)
                        imageio.imwrite(os.path.join(debug_dir, f"step_{step:06d}.png"), out)
            elif cfg.depth_loss and has_sparse_depth and depths is not None:
                # Sparse COLMAP point supervision via disparity L1
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths_q = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths_q = depths_q.squeeze(3).squeeze(1)  # [1, M]
                disp = torch.where(depths_q > 0.0, 1.0 / depths_q, torch.zeros_like(depths_q))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                rgb_loss = rgb_loss + tvloss

            depth_term = depthloss * cfg.depth_lambda
            loss = rgb_loss + depth_term

            # regularizations
            # Allow negative values for ablations (anti-regularization).
            if cfg.opacity_reg != 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg != 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            if is_novel_data and self.difix_enabled:
                loss = loss * cfg.difix_novel_lambda

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            if is_novel_data and self.difix_enabled:
                desc += "novel batch| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)

                # WandB logging
                if cfg.use_wandb:
                    wandb.log({
                        "step": step,
                        "train/loss": loss.item(),
                        "train/l1": l1loss.item(),
                        "train/ssim": ssimloss.item(),
                        "train/num_GS": len(self.splats["means"]),
                        "train/memory_GB": mem,
                        **({"train/depth": depthloss.item()} if cfg.depth_loss else {}),
                        **({"train/tv": tvloss.item()} if cfg.use_bilateral_grid else {}),
                    }, step=step)
                    # optionally log the rendered image
                    if cfg.tb_save_image and step % (cfg.tb_every * 10) == 0:
                        wandb.log({"train/render": wandb.Image(canvas)}, step=step)

                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                opacities = torch.clamp(self.splats["opacities"], min=-4.0, max=9.0)

                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # periodic evaluation during training
            if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
                self.eval(step, num_samples=cfg.eval_num_frames, save_images=False)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if self.difix_enabled and step in self.difix_fix_steps and self.world_rank == 0:
                self.fix_with_difix(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def fix_with_difix(self, step: int) -> None:
        if not self.difix_enabled:
            return

        cfg = self.cfg
        print(f"[Difix] Running fixer at step {step}…")

        train_indices = np.asarray(getattr(self.trainset, "indices", np.arange(len(self.trainset))))
        if train_indices.size == 0:
            print("[Difix] No training frames available; skipping.")
            return

        max_frames = cfg.difix_max_frames if cfg.difix_max_frames is not None else len(train_indices)
        novel_items = []

        progress = min(step / max(cfg.max_steps, 1), 1.0)
        scale = cfg.difix_jitter_min_scale + (1.0 - cfg.difix_jitter_min_scale) * progress
        trans_jitter = cfg.difix_jitter_translation * scale
        rot_jitter = cfg.difix_jitter_rotation_deg * scale

        if self.difix_use_val:
            val_indices = np.asarray(getattr(self.valset, "indices", np.arange(len(self.valset))))
            if max_frames is not None:
                val_indices = val_indices[: max_frames]
            if val_indices.size == 0:
                print("[Difix] No validation frames available; skipping.")
                return
            train_poses = self.parser.camtoworlds[train_indices]
            val_poses = self.parser.camtoworlds[val_indices]
            assignments = self.difix_interpolator.find_nearest_assignments(train_poses, val_poses)
            for idx, assign in zip(val_indices, assignments):
                camera_id = self.parser.camera_ids[idx]
                novel_items.append(
                    {
                        "pose": self.parser.camtoworlds[idx].copy(),
                        "camera_id": camera_id,
                        "ref_idx": int(train_indices[assign]),
                    }
                )
        else:
            # Allow difix_max_frames to exceed the training set size by sampling with replacement.
            # This is useful for "oversampling" nearby-pose novel views (e.g., up to ~5x trainset)
            # when exploring how many repaired novel samples help/hurt training.
            #
            # We keep a safety cap to avoid runaway work on very large scenes.
            count_cap = min(1024, 5 * len(train_indices))
            count = min(max_frames, count_cap)
            if count <= 0:
                print("[Difix] Unable to sample novel poses; skipping.")
                return
            replace = count > len(train_indices)
            sampled_train = np.random.choice(train_indices, size=count, replace=replace)
            if self._sample_near_pose is None:
                raise RuntimeError("Difix pose sampler unavailable.")
            for idx in sampled_train:
                pose = self.parser.camtoworlds[idx]
                jitter_pose = self._sample_near_pose(
                    pose,
                    scene_scale=self.scene_scale,
                    translation_jitter=trans_jitter,
                    rotation_jitter_deg=rot_jitter,
                )
                camera_id = self.parser.camera_ids[idx]
                novel_items.append(
                    {
                        "pose": jitter_pose,
                        "camera_id": camera_id,
                        "ref_idx": int(idx),
                    }
                )

        if not novel_items:
            print("[Difix] No novel poses generated; skipping.")
            return

        render_base = self.difix_render_dir / f"{step:06d}"
        viz_dir = render_base / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        pred_images = []
        alpha_arrays = []
        depth_arrays = []

        for item in novel_items:
            pose_np = item["pose"].astype(np.float32)
            camera_id = item["camera_id"]
            camtoworld = torch.from_numpy(pose_np).float().to(self.device)
            K_np = self.parser.Ks_dict[camera_id]
            width, height = self.parser.imsize_dict[camera_id]

            Ks = torch.from_numpy(K_np).float().to(self.device).unsqueeze(0)
            camtoworlds = camtoworld.unsqueeze(0)
            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB+ED",
            )
            # Separate RGB and depth; only clamp RGB
            rgb = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depth = renders[..., 3]  # [1, H, W]

            if self.difix_depth_state is not None:
                depth_full = render_difix_depth(
                    self.difix_depth_state,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    camera_model=self.cfg.camera_model,
                    near_plane=self.cfg.near_plane,
                    far_plane=self.cfg.far_plane,
                    antialiased=True,
                )
                depth_map_tensor = depth_full.squeeze(0).detach().cpu()
            else:
                depth_map_tensor = depth.squeeze(0).detach().cpu()

            # Convert RGB to uint8 image
            color_np = (rgb.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            pred_images.append(Image.fromarray(color_np))

            alpha_np = alphas.squeeze(0).cpu().numpy()
            if alpha_np.ndim == 2:
                alpha_np = alpha_np[..., None]
            alpha_arrays.append(alpha_np.astype(np.float32))

            # Sanitize depth map (expected z-depth): replace non-finite with max finite
            finite_mask = torch.isfinite(depth_map_tensor)
            if finite_mask.any():
                max_val = depth_map_tensor[finite_mask].max()
                depth_map_tensor[~finite_mask] = max_val
            else:
                depth_map_tensor[...] = 0.0
            depth_arrays.append(depth_map_tensor.numpy().astype(np.float32))

        call_signature = inspect.signature(self.difix_pipeline.__call__)
        expects_ref = "ref_image" in call_signature.parameters

        entries = []
        viz_records = []
        base_seed = cfg.difix_seed

        for local_idx, pred_img in enumerate(pred_images):
            ref_idx = novel_items[local_idx]["ref_idx"]
            ref_image_path = self.parser.image_paths[ref_idx]
            ref_image = Image.open(ref_image_path).convert("RGB") if expects_ref else None

            generator = None
            if base_seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(base_seed + step * 1000 + local_idx)

            kwargs = {
                "prompt": cfg.difix_prompt,
                "image": pred_img,
                "num_inference_steps": cfg.difix_num_inference_steps,
                "timesteps": [cfg.difix_timestep],
                "guidance_scale": cfg.difix_guidance_scale,
            }
            if expects_ref:
                kwargs["ref_image"] = ref_image
            if generator is not None:
                kwargs["generator"] = generator

            result = self.difix_pipeline(**kwargs).images[0]
            result = result.resize(pred_img.size, Image.LANCZOS)
            fixed_img = result
            fixed_np = np.array(fixed_img, dtype=np.uint8)

            match = np.where(train_indices == ref_idx)[0]
            ref_dataset_idx = int(match[0]) if match.size > 0 else 0

            entries.append(
                {
                    "image_array": fixed_np.astype(np.float32),
                    "alpha_array": alpha_arrays[local_idx].copy(),
                    "K": self.parser.Ks_dict[novel_items[local_idx]["camera_id"]].copy(),
                    "camtoworld": novel_items[local_idx]["pose"].copy(),
                    "image_id": ref_dataset_idx,
                    # include dense expected-depth supervision for this novel view
                    "depth_map": depth_arrays[local_idx].copy(),
                }
            )
            viz_records.append(
                (
                    pred_img.copy(),
                    ref_image.copy() if ref_image is not None else None,
                    fixed_img.copy(),
                    alpha_arrays[local_idx].copy(),
                    depth_arrays[local_idx].copy(),
                )
            )

        if not entries:
            print("[Difix] No images were repaired.")
            return

        viz_count = min(len(viz_records), cfg.difix_viz_max)
        if viz_count > 0:
            for i in range(viz_count):
                pred_img, ref_img, fixed_img, alpha_arr, depth_arr = viz_records[i]
                width, height = pred_img.size
                pred_panel = pred_img.resize((width, height), Image.LANCZOS)
                if ref_img is None:
                    ref_panel = Image.new("RGB", (width, height), color=(128, 128, 128))
                else:
                    ref_panel = ref_img.resize((width, height), Image.LANCZOS)
                fixed_panel = fixed_img.resize((width, height), Image.LANCZOS)
                alpha_vis = np.clip(alpha_arr[..., 0] * 255.0, 0, 255).astype(np.uint8)
                alpha_panel = Image.fromarray(alpha_vis, mode="L")
                alpha_panel = ImageOps.autocontrast(alpha_panel).convert("RGB")
                alpha_panel = alpha_panel.resize((width, height), Image.NEAREST)

                depth_tensor = torch.from_numpy(depth_arr)
                valid_depth = depth_tensor > 0
                if valid_depth.any():
                    d_min = depth_tensor[valid_depth].min()
                    d_max = depth_tensor[valid_depth].max()
                    if (d_max - d_min) > 1e-8:
                        depth_norm = (depth_tensor - d_min) / (d_max - d_min)
                    else:
                        depth_norm = torch.zeros_like(depth_tensor)
                else:
                    depth_norm = torch.zeros_like(depth_tensor)
                depth_norm = depth_norm.unsqueeze(0).unsqueeze(-1).float()
                depth_col = apply_float_colormap(depth_norm.clamp(0.0, 1.0), colormap="turbo")[0]
                depth_vis = (depth_col.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                depth_panel = Image.fromarray(depth_vis)
                depth_panel = depth_panel.resize((width, height), Image.NEAREST)

                panel = Image.new("RGB", (width * 5, height))
                panel.paste(pred_panel, (0, 0))
                panel.paste(ref_panel, (width, 0))
                panel.paste(fixed_panel, (width * 2, 0))
                panel.paste(alpha_panel, (width * 3, 0))
                panel.paste(depth_panel, (width * 4, 0))
                panel.save((viz_dir / f"{i:04d}.png").as_posix())

        class _DifixNovelDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                item = self.items[idx]
                image = item["image_array"]
                data = {
                    "K": torch.from_numpy(item["K"]).float(),
                    "camtoworld": torch.from_numpy(item["camtoworld"]).float(),
                    "image": torch.from_numpy(image).float(),
                    "image_id": torch.tensor(item["image_id"], dtype=torch.long),
                }
                alpha = item.get("alpha_array")
                if alpha is not None:
                    if alpha.ndim == 2:
                        alpha = alpha[..., None]
                    data["alpha_mask"] = torch.from_numpy(alpha).float()
                depth_map = item.get("depth_map")
                if depth_map is not None:
                    # expected-depth per-pixel map [H, W]
                    data["depth_map"] = torch.from_numpy(depth_map).float()
                return data

        dataset = _DifixNovelDataset(entries)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
        )
        self.difix_novelloaders.append(dataloader)
        self.difix_novelloaders_iter.append(iter(dataloader))
        self.difix_novel_sample_count = sum(len(loader.dataset) for loader in self.difix_novelloaders)
        print(f"[Difix] Added {len(dataset)} repaired views to the training queue.")

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", num_samples: Optional[int] = None, save_images: bool = False):
        """Entry for evaluation."""
        eval_type = f"evaluation ({num_samples} frames)" if num_samples else "full evaluation"
        print(f"Running {eval_type}...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            # Limit evaluation frames if specified
            if num_samples is not None and i >= num_samples:
                break
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images if requested
                if save_images:
                    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                        canvas,
                    )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            num_evaluated = min(len(valloader), num_samples) if num_samples is not None else len(valloader)
            ellipse_time /= num_evaluated

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)

            # WandB eval logging
            if cfg.use_wandb:
                wandb.log({f"{stage}/{k}": v for k, v in stats.items()}, step=step)

            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            # Force RGB-only when eval3d is enabled to avoid channel mismatch in CUDA.
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode=("RGB" if cfg.with_eval3d else "RGB+ED"),
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            canvas_list = [colors]
            if not cfg.with_eval3d and renders.shape[-1] >= 4:
                depths = renders[..., 3:4]
                depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-10)
                canvas_list.append(depths.repeat(1, 1, 1, 3))

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        # If pose/app optimization was used during training, load their states so eval-only
        # renders match training-time evaluation.
        if cfg.pose_opt and "pose_adjust" in ckpts[0]:
            try:
                runner.pose_adjust.load_state_dict(ckpts[0]["pose_adjust"])
            except Exception as e:
                print(f"[eval-only] Warning: failed to load pose_adjust: {e}")
        if cfg.app_opt and "app_module" in ckpts[0]:
            try:
                runner.app_module.load_state_dict(ckpts[0]["app_module"])
            except Exception as e:
                print(f"[eval-only] Warning: failed to load app_module: {e}")

        runner.eval(step=step, num_samples=cfg.eval_num_frames, save_images=cfg.eval_save_images)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if world_rank == 0 and cfg.use_wandb:
        wandb.finish()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                # Best current baseline (by mean PSNR on larger runs) is:
                # - compare_v1_extrap_default_pd (baseline settings), plus
                # - depth_sweep_v1_extrap_lam5e-3 (depth_lambda tuned from 1e-3 -> 5e-3).
                init_opa=1.0,
                init_scale=0.6,
                random_bkgd=True,
                pose_opt=True,
                pose_opt_lr=2e-5,
                pose_opt_reg=1e-3,
                depth_loss=True,
                depth_lambda=5e-3,
                means_lr=1.6e-7,
                scales_lr=1e-3,
                opacities_lr=2e-2,
                strategy=DefaultStrategy(
                    verbose=True,
                    grow_scale3d = 0.005,
                    grow_grad2d = 0.00015,
                    prune_scale3d = 0.05,
                    prune_scale2d = 0.15,
                    refine_every = 500,
                    reset_every = 30000,
                    refine_stop_iter = 10000
                    ),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.max_steps > 0:
        cfg.use_adaptive_init_opacity = False

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
