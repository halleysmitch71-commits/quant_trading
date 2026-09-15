"""
Smoke tests — verify the package can be imported and the basic structure
is sound.  These tests should always pass as a baseline sanity check.
"""

import importlib


def test_package_import():
    """The top-level package should be importable."""
    mod = importlib.import_module("quant_research")
    assert hasattr(mod, "__version__")


def test_version_string():
    """Version should be a non-empty string."""
    from quant_research import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_subpackages_importable():
    """All subpackages should be importable without error."""
    subpackages = [
        "quant_research.data",
        "quant_research.signals",
        "quant_research.strategies",
        "quant_research.backtest",
        "quant_research.risk",
        "quant_research.metrics",
        "quant_research.portfolio",
        "quant_research.core",
        "quant_research.execution",
        "quant_research.live",
    ]
    for pkg in subpackages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_base_signal_is_abstract():
    """BaseSignal should not be instantiable directly."""
    from quant_research.signals.base import BaseSignal

    import pytest

    with pytest.raises(TypeError):
        BaseSignal()  # type: ignore[abstract]


def test_config_file_exists():
    """The configuration file should exist at the expected path."""
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    assert config_path.exists(), f"Config not found at {config_path}"


def test_config_loads():
    """The YAML config should parse without errors."""
    from pathlib import Path

    import yaml

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Verify expected top-level keys
    expected_keys = {"data", "splits", "backtest", "costs", "risk", "reporting"}
    assert expected_keys.issubset(cfg.keys()), (
        f"Missing config keys: {expected_keys - cfg.keys()}"
    )

    # No-leverage constraint should be encoded
    assert cfg["backtest"]["max_gross_exposure"] <= 1.0
