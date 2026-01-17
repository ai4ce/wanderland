#!/usr/bin/env python3
"""Compose a simple USD/USDZ stage from a collider OBJ and a 3DGS USDZ asset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a single-scene USD/USDZ by referencing an existing 3DGS asset "
            "and converting an OBJ mesh into a collider."
        )
    )
    parser.add_argument("mesh", type=Path,
                        help="Path to the collider OBJ file")
    parser.add_argument(
        "gs_usdz",
        type=Path,
        help="Path to the 3D Gaussian Splatting USD or USDZ asset used for rendering",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Target USD, USDA, or USDZ file that will be written",
    )
    parser.add_argument(
        "--translate",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Translation in stage units applied to the collider",
    )
    parser.add_argument(
        "--rotate",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Rotation in degrees (XYZ order) applied to the collider",
    )
    parser.add_argument(
        "--scale",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Non-uniform scale applied to the collider",
    )
    return parser.parse_args(argv)


def _validate_path(path: Path, description: str) -> Path:
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"{description} not found: {expanded}")
    return expanded.resolve()


def _prepare_stage_and_output_paths(output: Path) -> Tuple[Path, Path]:
    """Return (stage_path, final_path) ensuring USD layers are authored to USD/USDA files."""
    resolved = output.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ext = resolved.suffix.lower()
    if ext in {".usd", ".usda"}:
        return resolved, resolved
    if ext == ".usdz":
        stage_path = resolved.with_suffix(".usda")
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        return stage_path, resolved
    raise ValueError("Output must end with .usd, .usda, or .usdz")


def _finalize_output(stage_path: Path, final_path: Path) -> Path:
    if stage_path == final_path:
        return stage_path
    UsdUtils.CreateNewUsdzPackage(str(stage_path), str(final_path))
    try:
        stage_path.unlink()
    except FileNotFoundError:
        pass
    return final_path


def _load_mesh_data(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh",
                        skip_materials=True, process=False)
    if mesh.is_empty or mesh.vertices is None or mesh.faces is None:
        raise ValueError(f"Unable to read mesh faces from {mesh_path}")
    if not mesh.is_winding_consistent:
        mesh.fix_normals()
    if hasattr(mesh, "triangles") and mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be Nx3")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError("Mesh faces must have at least three indices")
    return vertices, faces


def _np_to_vec3f_array(vertices: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])


def _build_collider(stage: Usd.Stage, mesh_data: Tuple[np.ndarray, np.ndarray]) -> None:
    vertices, faces = mesh_data
    collider_xform = UsdGeom.Xform.Define(stage, "/World/Collider")

    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Collider/Mesh")
    mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh_prim.CreateDoubleSidedAttr().Set(True)
    mesh_prim.CreatePointsAttr().Set(_np_to_vec3f_array(vertices))
    counts = np.full(len(faces), faces.shape[1], dtype=np.int32)
    mesh_prim.CreateFaceVertexCountsAttr().Set(Vt.IntArray(counts.tolist()))
    mesh_prim.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray(faces.reshape(-1).astype(np.int32).tolist())
    )
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    mesh_prim.CreateExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*mins.astype(float)),
                      Gf.Vec3f(*maxs.astype(float))])
    )

    # Collider is hidden from rendering but participates in physics collisions.
    UsdGeom.Imageable(mesh_prim.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())


def _attach_visual_reference(stage: Usd.Stage, asset_path: Path, transforms: dict) -> None:
    visual = UsdGeom.Xform.Define(stage, "/World/Visual")
    xformable = UsdGeom.Xformable(visual.GetPrim())
    xformable.AddScaleOp().Set(Gf.Vec3d(*transforms["scale"]))
    xformable.AddRotateXYZOp().Set(Gf.Vec3d(*transforms["rotate"]))
    xformable.AddTranslateOp().Set(Gf.Vec3d(*transforms["translate"]))
    visual.GetPrim().GetReferences().AddReference(str(asset_path))


def assemble_single_scene(args: argparse.Namespace) -> Path:
    mesh_path = _validate_path(args.mesh, "Collider mesh")
    gs_path = _validate_path(args.gs_usdz, "3DGS asset")
    stage_path, final_path = _prepare_stage_and_output_paths(args.output)

    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    transforms = {
        "translate": tuple(args.translate),
        "rotate": tuple(args.rotate),
        "scale": tuple(args.scale),
    }
    _attach_visual_reference(stage, gs_path, transforms)
    mesh_data = _load_mesh_data(mesh_path)
    _build_collider(stage, mesh_data)

    stage.GetRootLayer().Save()
    return _finalize_output(stage_path, final_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = assemble_single_scene(args)
    except Exception as exc:  # noqa: BLE001 - surface clear errors for CLI users
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] Wrote stage to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
