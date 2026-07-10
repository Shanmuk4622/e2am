"""Package-level import and metadata tests."""

import e2am


def test_version_is_semver() -> None:
    parts = e2am.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_public_api_declared() -> None:
    assert "monitor" in e2am.__all__
    assert "Trainer" in e2am.__all__


def test_unknown_attribute_raises() -> None:
    import pytest

    with pytest.raises(AttributeError):
        _ = e2am.does_not_exist
