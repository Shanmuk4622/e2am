"""Framework integrations (Hugging Face, ...).

Imported lazily so ``e2am`` never requires the integrated frameworks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from e2am.integrations.huggingface import E2AMCallback

__all__ = ["E2AMCallback"]


def __getattr__(name: str) -> Any:
    """Lazily import integrations that need optional frameworks."""
    if name == "E2AMCallback":
        from e2am.integrations.huggingface import E2AMCallback

        return E2AMCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
