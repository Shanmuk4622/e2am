"""Slack and Discord notification plugins (stdlib-only, no dependencies).

Both post a compact run summary to an incoming-webhook URL when training
finishes — and an alert when it fails. Network errors are logged and
swallowed: a dead webhook must never kill a training run.
"""

from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING

from e2am.plugins.base import Callback, PluginError
from e2am.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from e2am.trainer.trainer import Trainer

logger = get_logger("plugins.webhooks")


def _summary_message(trainer: Trainer) -> str:
    result = trainer.result
    if result is None:
        return f"E2AM run {trainer.run_name}: finished (no result available)."
    lines = [
        f"E2AM · {result.project} / {result.run_name} — {result.status}",
        f"epochs: {result.epochs_completed}/{result.epochs_requested}",
    ]
    if result.final_val_accuracy is not None:
        lines.append(f"val accuracy: {result.final_val_accuracy:.4f}")
    if result.monitor is not None:
        lines.append(
            f"energy: {result.monitor.total_energy_wh:.4g} Wh · "
            f"carbon: {result.monitor.carbon.emissions_g:.4g} g CO2eq"
        )
    if result.green is not None and result.green.green_score is not None:
        lines.append(f"green score: {result.green.green_score:.1f}/100")
    return "\n".join(lines)


class _WebhookPlugin(Callback):
    """Shared webhook mechanics; subclasses define the payload shape."""

    #: JSON key the service expects the message text under.
    text_key = "text"

    def __init__(self, webhook_url: str, timeout_s: float = 10.0) -> None:
        if not webhook_url.startswith("https://"):
            raise PluginError("webhook_url must be an https:// URL.")
        self.webhook_url = webhook_url
        self.timeout_s = timeout_s

    def _post(self, message: str) -> None:
        payload = json.dumps({self.text_key: message}).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "e2am"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s):
                pass
        except Exception as exc:
            logger.warning("%s notification failed: %s", type(self).__name__, exc)

    def on_fit_end(self, trainer: Trainer) -> None:
        self._post(_summary_message(trainer))

    def on_exception(self, trainer: Trainer, exc: BaseException) -> None:
        self._post(
            f"E2AM · {trainer.config.project} / {trainer.run_name} FAILED: "
            f"{type(exc).__name__}: {exc}"
        )


class SlackPlugin(_WebhookPlugin):
    """Post the run summary to a Slack incoming webhook.

    Args:
        webhook_url: Slack incoming-webhook URL (``https://hooks.slack.com/...``).
        timeout_s: HTTP timeout for the notification.
    """

    text_key = "text"


class DiscordPlugin(_WebhookPlugin):
    """Post the run summary to a Discord channel webhook.

    Args:
        webhook_url: Discord webhook URL (``https://discord.com/api/webhooks/...``).
        timeout_s: HTTP timeout for the notification.
    """

    text_key = "content"
