"""Plugin tests: fake backends injected into sys.modules, real Trainer runs."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from e2am.plugins.base import PluginError, require  # noqa: E402
from e2am.trainer import Trainer  # noqa: E402

pytestmark = pytest.mark.torch


def _tiny_trainer(callbacks: list, epochs: int = 2) -> Trainer:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(32, 4, generator=g)
    y = (x.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = nn.Linear(4, 2)
    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=loader,
        val_loader=loader,
        epochs=epochs,
        device="cpu",
        run_name="plugin-run",
        progress=False,
        monitor_enabled=False,
        profile_enabled=False,
        save_artifacts=False,
        callbacks=callbacks,
    )


class _Bomb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise ValueError("boom")


def _failing_trainer(callbacks: list) -> Trainer:
    model = _Bomb()
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)))
    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=loader,
        epochs=1,
        device="cpu",
        run_name="plugin-fail",
        progress=False,
        monitor_enabled=False,
        profile_enabled=False,
        save_artifacts=False,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


def test_require_missing_package_raises_with_pip_hint() -> None:
    with pytest.raises(PluginError, match="pip install nonexistent-e2am-pkg"):
        require("nonexistent_e2am_pkg", pip_name="nonexistent-e2am-pkg")


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------


def test_wandb_plugin_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_wandb = MagicMock()
    fake_run = MagicMock()
    fake_run.summary = {}
    fake_wandb.init.return_value = fake_run
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    from e2am.plugins.wandb import WandbPlugin

    plugin = WandbPlugin(entity="my-team")
    trainer = _tiny_trainer([plugin], epochs=2)
    trainer.fit()

    fake_wandb.init.assert_called_once()
    init_kwargs = fake_wandb.init.call_args.kwargs
    assert init_kwargs["project"] == "e2am"
    assert init_kwargs["name"] == "plugin-run"
    assert init_kwargs["entity"] == "my-team"
    assert fake_run.log.call_count == 2  # one per epoch
    logged = fake_run.log.call_args_list[0].args[0]
    assert "train_loss" in logged
    assert fake_run.summary.get("epochs_completed") == 2
    fake_run.finish.assert_called_once()


def test_wandb_plugin_marks_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_wandb = MagicMock()
    fake_run = MagicMock()
    fake_run.summary = {}
    fake_wandb.init.return_value = fake_run
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    from e2am.plugins.wandb import WandbPlugin

    trainer = _failing_trainer([WandbPlugin()])
    with pytest.raises(ValueError, match="boom"):
        trainer.fit()
    fake_run.finish.assert_called_once_with(exit_code=1)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


def test_mlflow_plugin_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    from e2am.plugins.mlflow import MLflowPlugin

    plugin = MLflowPlugin(tracking_uri="file:///tmp/mlruns", experiment_name="exp")
    trainer = _tiny_trainer([plugin], epochs=2)
    trainer.fit()

    fake_mlflow.set_tracking_uri.assert_called_once_with("file:///tmp/mlruns")
    fake_mlflow.set_experiment.assert_called_once_with("exp")
    fake_mlflow.start_run.assert_called_once_with(run_name="plugin-run")
    params = fake_mlflow.log_params.call_args.args[0]
    assert params["trainer.epochs"] == "2"  # nested config flattened to strings
    # 2 per-epoch calls + 1 final summary call
    assert fake_mlflow.log_metrics.call_count == 3
    fake_mlflow.end_run.assert_called_once_with()


def test_mlflow_plugin_marks_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    from e2am.plugins.mlflow import MLflowPlugin

    trainer = _failing_trainer([MLflowPlugin()])
    with pytest.raises(ValueError, match="boom"):
        trainer.fit()
    fake_mlflow.end_run.assert_called_once_with(status="FAILED")


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------


def test_tensorboard_plugin_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    writer = MagicMock()
    fake_module = types.SimpleNamespace(SummaryWriter=MagicMock(return_value=writer))
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_module)
    from e2am.plugins.tensorboard import TensorBoardPlugin

    plugin = TensorBoardPlugin(log_dir=tmp_path / "tb")
    trainer = _tiny_trainer([plugin], epochs=2)
    trainer.fit()

    fake_module.SummaryWriter.assert_called_once_with(log_dir=str(tmp_path / "tb"))
    scalar_names = {call.args[0] for call in writer.add_scalar.call_args_list}
    assert "train_loss" in scalar_names
    assert any(name.startswith("final/") for name in scalar_names)
    writer.close.assert_called_once()


# ---------------------------------------------------------------------------
# Slack / Discord webhooks
# ---------------------------------------------------------------------------


class _UrlopenRecorder:
    def __init__(self) -> None:
        self.requests: list = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Response()


@pytest.mark.parametrize(
    ("plugin_name", "text_key"), [("SlackPlugin", "text"), ("DiscordPlugin", "content")]
)
def test_webhook_plugins_post_summary(
    monkeypatch: pytest.MonkeyPatch, plugin_name: str, text_key: str
) -> None:
    import e2am.plugins.webhooks as webhooks

    recorder = _UrlopenRecorder()
    monkeypatch.setattr(webhooks.urllib.request, "urlopen", recorder)
    plugin_cls = getattr(webhooks, plugin_name)
    plugin = plugin_cls("https://hooks.example.com/services/T/B/x")
    trainer = _tiny_trainer([plugin], epochs=1)
    trainer.fit()

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.full_url == "https://hooks.example.com/services/T/B/x"
    body = json.loads(request.data.decode("utf-8"))
    assert "plugin-run" in body[text_key]
    assert "completed" in body[text_key]


def test_webhook_failure_message_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import e2am.plugins.webhooks as webhooks

    recorder = _UrlopenRecorder()
    monkeypatch.setattr(webhooks.urllib.request, "urlopen", recorder)
    plugin = webhooks.SlackPlugin("https://hooks.example.com/x")
    trainer = _failing_trainer([plugin])
    with pytest.raises(ValueError, match="boom"):
        trainer.fit()
    bodies = [json.loads(r.data.decode("utf-8"))["text"] for r in recorder.requests]
    assert any("FAILED" in body for body in bodies)


def test_webhook_network_error_does_not_crash_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import e2am.plugins.webhooks as webhooks

    def _explode(request, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(webhooks.urllib.request, "urlopen", _explode)
    plugin = webhooks.SlackPlugin("https://hooks.example.com/x")
    trainer = _tiny_trainer([plugin], epochs=1)
    result = trainer.fit()  # must not raise
    assert result.status == "completed"


def test_webhook_rejects_non_https() -> None:
    from e2am.plugins.webhooks import SlackPlugin

    with pytest.raises(PluginError, match="https"):
        SlackPlugin("http://insecure.example.com/hook")


def test_plugins_package_importable_without_backends() -> None:
    import e2am.plugins as plugins

    assert hasattr(plugins, "WandbPlugin")
    assert hasattr(plugins, "SlackPlugin")
