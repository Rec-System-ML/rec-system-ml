from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_shared_on_path() -> Path:
    """
    Ensure the project root is on sys.path so ``shared`` and ``interest_graph`` import.

    Returns the bundled ``shared/`` directory (for default KuaiRand data paths).

    Optional: set ``RECSYS_SHARED_PATH`` to override the **data root** only — the
    directory that contains ``data/KuaiRand-1K/data``. Imports always use this
    project's ``shared`` package.
    """
    project_root = Path(__file__).resolve().parent
    bundled_shared = project_root / "shared"
    if not bundled_shared.is_dir():
        raise FileNotFoundError(
            f"Bundled shared package not found: {bundled_shared}",
        )

    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    env_data_root = os.getenv("RECSYS_SHARED_PATH")
    if env_data_root:
        return Path(env_data_root).expanduser().resolve()

    return bundled_shared
