from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_shared_on_path() -> Path:
    """
    Make recsys-shared importable.

    Priority:
    1) env RECSYS_SHARED_PATH
    2) ../recsys-shared (sibling dir, e.g. /Users/Shared/.../HK/LU/recsys-shared)
    """
    env_path = os.getenv("RECSYS_SHARED_PATH")
    if env_path:
        shared_path = Path(env_path).expanduser().resolve()
    else:
        # shared/ 就在 back/ 目录下
        shared_path = (Path(__file__).resolve().parent / "shared").resolve()

    if not shared_path.exists():
        raise FileNotFoundError(
            f"shared not found: {shared_path}. "
            "Set RECSYS_SHARED_PATH to your shared directory."
        )

    shared_path_str = str(shared_path)
    if shared_path_str not in sys.path:
        sys.path.insert(0, shared_path_str)

    return shared_path
