"""MetricsTracker tests."""

import pytest

from e2am.exceptions import E2AMError
from e2am.metrics import MetricsTracker


def test_log_and_series() -> None:
    t = MetricsTracker()
    t.log({"loss": 1.0, "acc": 0.3}, step=0)
    t.log({"loss": 0.5}, step=1)
    steps, values = t.series("loss")
    assert steps == [0, 1]
    assert values == [1.0, 0.5]
    assert t.latest("acc") == 0.3
    assert t.names == ["loss", "acc"]
    assert "loss" in t and "nope" not in t
    assert len(t) == 2


def test_auto_step_increments() -> None:
    t = MetricsTracker()
    t.log({"x": 1.0})
    t.log({"x": 2.0})
    t.log({"x": 3.0}, step=10)
    t.log({"x": 4.0})
    assert t.series("x")[0] == [0, 1, 10, 11]


def test_unknown_metric_raises() -> None:
    with pytest.raises(E2AMError, match="Unknown metric"):
        MetricsTracker().series("ghost")


def test_non_numeric_value_raises() -> None:
    with pytest.raises(E2AMError, match="not a real number"):
        MetricsTracker().log({"bad": "hello"})  # type: ignore[dict-item]


def test_latest_of_unlogged_is_none() -> None:
    assert MetricsTracker().latest("never") is None


def test_to_dict() -> None:
    t = MetricsTracker()
    t.log({"loss": 1.0}, step=0)
    t.log({"loss": 0.5}, step=1)
    d = t.to_dict()
    assert d["loss"]["values"] == [1.0, 0.5]
    assert d["loss"]["steps"] == [0, 1]


def test_to_dataframe_in_clean_subprocess() -> None:
    """to_dataframe needs pandas; run it in a fresh interpreter.

    In-process, pandas' pyarrow-backed strings can hit Windows DLL conflicts
    when torch/matplotlib DLLs are already loaded by other tests — an
    environment quirk, not E2AM behavior — so the pandas path is exercised in
    a clean subprocess where only e2am and pandas are imported.
    """
    import subprocess
    import sys

    script = (
        "from e2am.metrics import MetricsTracker\n"
        "t = MetricsTracker()\n"
        "t.log({'loss': 1.0}, step=0)\n"
        "t.log({'loss': 0.5}, step=1)\n"
        "df = t.to_dataframe()\n"
        "assert list(df.columns) == ['step', 'metric', 'value']\n"
        "assert df['value'].tolist() == [1.0, 0.5]\n"
        "print('DATAFRAME_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0 and "ModuleNotFoundError" in proc.stderr:
        import pytest

        pytest.skip("pandas not installed")
    assert proc.returncode == 0, proc.stderr
    assert "DATAFRAME_OK" in proc.stdout
