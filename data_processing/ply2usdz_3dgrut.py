#!/usr/bin/env python3
"""Convert a 3DGS-compatible PLY into an Isaac/Omniverse-compatible USDZ (via 3DGRUT).

This repo intentionally does NOT vendor 3DGRUT. Instead, we call its official exporter:
  python -m threedgrut.export.scripts.ply_to_usd <in.ply> --output_file <out.usdz>

You must run this with a Python that has 3DGRUT installed (see 3DGRUT README).
If 3DGRUT is not importable, this script prints a short, actionable error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 3DGS PLY to USDZ using NVIDIA 3DGRUT's exporter."
    )
    parser.add_argument("ply", type=Path, help="Input Gaussian PLY (e.g., 3dgs.ply)")
    parser.add_argument("output", type=Path, help="Output USDZ path")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help=(
            "Python executable to run the 3DGRUT exporter with "
            "(default: current Python)."
        ),
    )
    return parser.parse_args(argv)


def _validate_file(path: Path, desc: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{desc} not found: {resolved}")
    return resolved


def _validate_python(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Python executable not found: {resolved}")
    return resolved


def _check_threedgrut_import(python_exe: Path) -> bool:
    proc = subprocess.run(
        [str(python_exe), "-c", "import threedgrut; print(threedgrut.__file__)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0


def convert(ply: Path, output: Path, python_exe: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        "-m",
        "threedgrut.export.scripts.ply_to_usd",
        str(ply),
        "--output_file",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        ply = _validate_file(args.ply, "PLY")
        python_exe = _validate_python(args.python)
        if not _check_threedgrut_import(python_exe):
            raise RuntimeError(
                "3DGRUT is not installed for the selected Python. "
                "Install it from https://github.com/nv-tlabs/3dgrut "
                "and re-run with --python /path/to/3dgrut/python."
            )
        convert(ply, args.output, python_exe)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] 3DGRUT exporter failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Wrote USDZ to {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

