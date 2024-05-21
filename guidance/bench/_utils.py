"""Shared utility functions for module."""

import os
from pathlib import Path

def lib_bench_dir() -> Path:
    """Library directory to store configurations and cached assets for benchmarking.

    If the library directory does not exist, it is created as a side effect.

    The library bench directory path can also be set via env var `GUIDANCE_BENCH_DIR`.

    Returns:
        Path: Library's directory path for benchmarking.
    """

    env_lib_path = os.environ.get("GUIDANCE_BENCH_DIR", None)
    if env_lib_path is None:
        lib_path = Path.home() / ".guidance-bench"
    else:
        lib_path = Path(env_lib_path)
    Path.mkdir(lib_path, parents=True, exist_ok=True)

    return lib_path
