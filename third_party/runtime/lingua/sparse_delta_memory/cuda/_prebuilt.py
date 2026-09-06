"""Optional loader for checksum-packaged SDM CUDA extensions."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def load_prebuilt(name: str):
    """Load one explicitly configured extension, or request upstream JIT."""

    configured = os.environ.get("SDM2_CUDA_EXTENSION_DIR")
    if configured is None:
        return None
    root = Path(configured).resolve()
    directory = root / name
    if not directory.is_dir():
        raise ImportError(f"missing prebuilt SDM extension directory: {directory}")
    candidates = sorted(directory.glob("*.so"))
    if len(candidates) != 1:
        raise ImportError(
            f"expected one prebuilt SDM extension for {name}, found {candidates}"
        )
    path = candidates[0].resolve()
    if root not in path.parents:
        raise ImportError(f"prebuilt SDM extension escapes configured root: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load prebuilt SDM extension: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module
