"""Plugin foundations.

A plugin *is* a :class:`~e2am.trainer.callbacks.Callback` — same lifecycle,
same guarantees. This module adds the two conventions every integration
follows:

* **Fail fast on missing dependencies**: the constructor imports the backing
  package via :func:`require` and raises a helpful error immediately —
  not silently mid-training, where the callback runner would swallow it.
* **Never crash training afterwards**: network and backend failures inside
  hooks are logged and dropped.
"""

from __future__ import annotations

import importlib
from typing import Any

from e2am.exceptions import E2AMError
from e2am.trainer.callbacks import Callback

__all__ = ["Callback", "PluginError", "require"]


class PluginError(E2AMError):
    """Raised when a plugin cannot be constructed (e.g. missing package)."""


def require(package: str, pip_name: str | None = None) -> Any:
    """Import a plugin's backing package or fail with install instructions.

    Args:
        package: Importable module name (e.g. ``"wandb"``).
        pip_name: Name to show in the pip command when it differs.

    Returns:
        The imported module.

    Raises:
        PluginError: If the package is not installed.
    """
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise PluginError(
            f"The {package!r} package is required for this plugin. "
            f"Install it with: pip install {pip_name or package}"
        ) from exc
