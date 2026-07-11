"""Drop-in training loop with automatic energy monitoring.

The full :class:`Trainer` implementation is under active development. This
placeholder keeps ``from e2am import Trainer`` importable (and typed) so the
package is releasable at every commit; instantiating it points users to the
working ``monitor()`` API in the meantime.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Trainer"]


class Trainer:
    """Placeholder for the upcoming E2AM Trainer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "e2am.Trainer is not available yet in this version. "
            "Wrap your existing training loop with `with e2am.monitor(...):` "
            "to get automatic energy, carbon, and utilization tracking today."
        )
