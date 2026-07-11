"""Centralized, rich-formatted logging for E2AM.

All E2AM modules obtain loggers via :func:`get_logger`, which returns children
of a single ``e2am`` root logger. The root logger is configured exactly once
with a :class:`rich.logging.RichHandler` and does not propagate, so E2AM never
pollutes the user's own logging configuration.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_ROOT_NAME = "e2am"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        handler = RichHandler(
            console=Console(stderr=True),
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an E2AM logger.

    Args:
        name: Optional dotted suffix (e.g. ``"monitor.gpu"``). ``None`` returns
            the E2AM root logger.

    Returns:
        A configured :class:`logging.Logger` under the ``e2am`` namespace.
    """
    _configure_root()
    if name is None:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def set_verbosity(level: int | str) -> None:
    """Set the verbosity of all E2AM loggers.

    Args:
        level: A :mod:`logging` level name (``"DEBUG"``, ``"INFO"``, ...) or
            numeric level.
    """
    _configure_root()
    logging.getLogger(_ROOT_NAME).setLevel(level)
