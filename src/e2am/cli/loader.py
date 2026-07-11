"""Loading user model scripts for the CLI.

``e2am train`` and ``e2am benchmark`` operate on a plain Python file that
exports well-known factory functions — no subclassing, no registration:

* ``get_model() -> nn.Module`` (required)
* ``get_loaders() -> (train_loader, val_loader | None)`` (train only)
* ``get_optimizer(model) -> torch.optim.Optimizer`` (optional; Adam 1e-3)
* ``get_loss() -> callable`` (optional; CrossEntropyLoss)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from e2am.exceptions import ConfigError


def load_user_module(path: str | Path) -> ModuleType:
    """Import a user's Python file as a module.

    Raises:
        ConfigError: If the file is missing or fails to import.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"e2am_user_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Cannot import {path} as a Python module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"Error while importing {path}: {exc}") from exc
    return module


def get_factory(module: ModuleType, name: str, required: bool = True) -> Any:
    """Fetch a factory function from the user module.

    Raises:
        ConfigError: If a required factory is missing or not callable.
    """
    factory = getattr(module, name, None)
    if factory is None:
        if required:
            raise ConfigError(
                f"{module.__name__} must define `{name}()` "
                f"(see the docs for the e2am script convention)."
            )
        return None
    if not callable(factory):
        raise ConfigError(f"`{name}` in the script is not callable.")
    return factory
