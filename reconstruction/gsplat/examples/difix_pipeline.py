"""Utility helpers for Difix integration (pipeline + depth snapshots)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

from gsplat.rendering import rasterization


@dataclass
class DifixDepthState:
    """CPU-side storage with device caches for high-fidelity depth rendering."""

    storage: Dict[str, torch.Tensor]
    _device_cache: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def get(self, device: torch.device) -> Dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._device_cache:
            converted = {
                name: tensor.to(device=device, dtype=torch.float32)
                for name, tensor in self.storage.items()
                if name not in {"sh0", "shN"}
            }
            if "sh0" in self.storage and "shN" in self.storage:
                converted["colors"] = torch.cat(
                    [
                        self.storage["sh0"].to(device=device, dtype=torch.float32),
                        self.storage["shN"].to(device=device, dtype=torch.float32),
                    ],
                    dim=1,
                )
            else:
                n_pts = converted["means"].shape[0]
                converted["colors"] = torch.zeros(
                    (n_pts, 1, 3), device=device, dtype=torch.float32
                )
            self._device_cache[key] = converted
        return self._device_cache[key]


def _maybe_import_local_pipeline():
    root = Path(__file__).resolve().parents[2]
    difix_src = root / "Difix3D" / "src"
    if not difix_src.exists():
        return None
    path_str = str(difix_src)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        from pipeline_difix import DifixPipeline  # type: ignore
    except Exception:
        return None
    return DifixPipeline


def load_difix_pipeline(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    disable_progress_bar: bool = True,
) -> "DiffusionPipeline":
    """Return an initialized Difix diffusion pipeline on the requested device."""

    local_pipeline_cls = _maybe_import_local_pipeline()

    if local_pipeline_cls is not None:
        pipe = local_pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - informative failure path
            raise RuntimeError(
                "Difix integration requires the Hugging Face diffusers package. "
                "Install it via `pip install diffusers transformers accelerate`."
            ) from exc

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    if disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe


def export_difix_splat_snapshot(
    splats: torch.nn.ParameterDict,
    path: Path,
    dtype: torch.dtype = torch.float16,
    include_sh: bool = False,
) -> None:
    """Persist the full-geometry GS parameters for later Difix depth rendering."""

    cpu_dtype = dtype
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means = splats["means"].detach().cpu().to(cpu_dtype)
    scales = torch.exp(splats["scales"].detach().cpu()).to(cpu_dtype)
    quats = splats["quats"].detach().cpu().to(cpu_dtype)
    opacities = torch.sigmoid(splats["opacities"].detach().cpu()).to(cpu_dtype)

    payload: Dict[str, torch.Tensor] = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
    }

    if include_sh and "sh0" in splats and "shN" in splats:
        payload["sh0"] = splats["sh0"].detach().cpu().to(cpu_dtype)
        payload["shN"] = splats["shN"].detach().cpu().to(cpu_dtype)

    torch.save(payload, path)
    print(f"[Difix] Saved initial splat snapshot ({means.shape[0]} points) to {path}")


def load_difix_depth_state(path: Path) -> Optional[DifixDepthState]:
    """Load snapshot tensors for later transfer to the active device."""

    path = Path(path)
    if not path.exists():
        return None

    payload = torch.load(path, map_location="cpu")
    for key in ("means", "scales", "quats", "opacities"):
        if key not in payload:
            print(f"[Difix] Snapshot at {path} missing '{key}'; ignoring snapshot.")
            return None

    storage = {
        "means": payload["means"],
        "scales": payload["scales"],
        "quats": payload["quats"],
        "opacities": payload["opacities"],
    }
    if "sh0" in payload and "shN" in payload:
        storage["sh0"] = payload["sh0"]
        storage["shN"] = payload["shN"]
    return DifixDepthState(storage=storage)


def render_difix_depth(
    state: DifixDepthState,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    camera_model: str,
    near_plane: float,
    far_plane: float,
    antialiased: bool = True,
) -> torch.Tensor:
    """Render expected depth using the stored full-geometry state."""

    device = camtoworlds.device
    params = state.get(device)
    n_pts = params["means"].shape[-2]
    colors = torch.zeros((1, n_pts, 3), device=device, dtype=torch.float32)

    renders, _, _ = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"],
        opacities=params["opacities"],
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        render_mode="ED",
        rasterize_mode="antialiased" if antialiased else "classic",
        packed=False,
        sparse_grad=False,
        camera_model=camera_model,
        near_plane=near_plane,
        far_plane=far_plane,
    )
    return renders[..., 0]


__all__ = [
    "DifixDepthState",
    "export_difix_splat_snapshot",
    "load_difix_depth_state",
    "load_difix_pipeline",
    "render_difix_depth",
]
