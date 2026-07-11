"""Integration plugins — every plugin is a Trainer :class:`Callback`.

Slack/Discord are dependency-free; the others import their backing package
at construction time and raise a helpful :class:`PluginError` if missing.
"""

from __future__ import annotations

from e2am.plugins.base import PluginError, require
from e2am.plugins.mlflow import MLflowPlugin
from e2am.plugins.tensorboard import TensorBoardPlugin
from e2am.plugins.wandb import WandbPlugin
from e2am.plugins.webhooks import DiscordPlugin, SlackPlugin

__all__ = [
    "DiscordPlugin",
    "MLflowPlugin",
    "PluginError",
    "SlackPlugin",
    "TensorBoardPlugin",
    "WandbPlugin",
    "require",
]
