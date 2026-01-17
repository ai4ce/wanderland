#!/usr/bin/env python3
"""
Export a Gaussian Splat checkpoint to a 3DGS-compatible PLY.

This is a robust exporter that ignores training-time masks and uses a
forgiving alpha threshold (default 0.0) to help diagnose "empty PLY" issues.

Usage:
  python scripts/export_ply_from_ckpt.py \
    --ckpt /path/to/ckpts/last.pt \
    --out  /path/to/out/model.ply \
    [--alpha_thr 0.0]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Optional, Tuple

import torch


def _get(sd: dict, *candidates: str) -> Optional[torch.Tensor]:
    for k in candidates:
        t = sd.get(k)
        if t is not None:
            return t
    return None


def _as_cpu_float(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().to(device="cpu", dtype=torch.float32)


def _linearize_opacity(opacities: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if opacities is None:
        return None
    # If already in [0,1], keep; otherwise treat as logits.
    if float(opacities.min()) < 0.0 or float(opacities.max()) > 1.0:
        opacities = opacities.sigmoid()
    return opacities


def _linearize_scales(scales: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scales is None:
        return None
    # 3DGS stores log-scales; exponentiate.
    return scales.exp()


def build_ply_header(n: int, n_rest: int) -> bytes:
    props = [
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(n_rest):
        props.append(f"property float f_rest_{i}")
    props.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
        ]
    )
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        *props,
        "end_header",
    ]
    return ("\n".join(header) + "\n").encode("ascii")


def export(ckpt_path: Path, out_path: Path, alpha_thr: float = 0.0) -> None:
    data = torch.load(str(ckpt_path), map_location="cpu")
    sd = data.get("state_dict", data)

    means = _as_cpu_float(_get(sd, "splats.means", "means"))
    scales = _as_cpu_float(_get(sd, "splats.scales", "scales"))
    quats = _as_cpu_float(_get(sd, "splats.quats", "quats"))
    opacities = _as_cpu_float(_get(sd, "splats.opacities", "opacities"))
    sh0 = _as_cpu_float(_get(sd, "splats.sh0", "sh0"))  # (N,1,3) or (N,3)
    shN = _as_cpu_float(_get(sd, "splats.shN", "shN"))  # (N,K,3)

    if means is None or means.numel() == 0:
        raise RuntimeError("Checkpoint has no 'means'—nothing to export.")

    N = means.shape[0]
    if scales is None or scales.shape[0] != N:
        scales = torch.zeros((N, 3), dtype=torch.float32)
    else:
        scales = _linearize_scales(scales)

    if quats is None or quats.shape[0] != N:
        quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(N, 1)

    if opacities is None or opacities.shape[0] != N:
        opacities = torch.ones((N, 1), dtype=torch.float32)
    else:
        opacities = _linearize_opacity(opacities).reshape(N, 1)

    # SH handling: f_dc (3), f_rest (K*3)
    if sh0 is None:
        f_dc = torch.zeros((N, 3), dtype=torch.float32)
    else:
        f_dc = sh0.reshape(N, -1)[:, :3]
    if shN is None:
        f_rest = torch.zeros((N, 0), dtype=torch.float32)
    else:
        f_rest = shN.reshape(N, -1)

    # Alpha filtering
    if alpha_thr > 0.0:
        mask = (opacities[:, 0] > alpha_thr)
    else:
        mask = torch.ones((N,), dtype=torch.bool)

    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    n_keep = int(idx.numel())
    if n_keep == 0:
        print("[export] Warning: mask removed all points; writing 0-vertex PLY.")

    means = means[idx]
    f_dc = f_dc[idx]
    f_rest = f_rest[idx]
    opacities = opacities[idx]
    scales = scales[idx]
    quats = quats[idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        header = build_ply_header(n_keep, f_rest.shape[1])
        f.write(header)
        pack = struct.Struct(
            "<" + "f" * (3 + 3 + f_rest.shape[1] + 1 + 3 + 4)).pack
        for i in range(n_keep):
            xyz = means[i].tolist()
            dc = f_dc[i].tolist()
            rest = f_rest[i].tolist() if f_rest.numel() else []
            a = [float(opacities[i, 0])]
            s = scales[i].tolist()
            r = quats[i].tolist()
            f.write(pack(*xyz, *dc, *rest, *a, *s, *r))
    print(f"[export] Wrote {n_keep} vertices to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export Gaussian checkpoint to PLY")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (pt)")
    ap.add_argument("--out", required=True, help="Output PLY path")
    ap.add_argument("--alpha_thr", type=float, default=0.0,
                    help="Opacity threshold (0 disables)")
    args = ap.parse_args()

    export(Path(args.ckpt), Path(args.out), alpha_thr=args.alpha_thr)


if __name__ == "__main__":
    main()
