#!/usr/bin/env python3
"""Thin wrapper around the vendored gsplat training script.

This keeps the user-facing entrypoint stable:
  python train_3dgs.py --help
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    module_root = Path(__file__).resolve().parent
    script = module_root / "gsplat" / "examples" / "simple_trainer.py"
    if not script.is_file():
        raise SystemExit(f"Missing training script: {script}")
    gsplat_root = str(module_root / "gsplat")
    if gsplat_root not in sys.path:
        sys.path.insert(0, gsplat_root)
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
