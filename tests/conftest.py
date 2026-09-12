"""Shared pytest fixtures and markers (M0.4 skeleton).

The ``base_env`` fixture builds the deterministic single-agent base environment
(no grid coupling) from the canonical config; full RNG-isolation fixtures are
reused from marl/tests/conftest.py when the physics/contract tests grow.
"""

from __future__ import annotations

import pytest

from configs.config_ultimate import (
    DATA_CONFIG,
    ENV_CONFIG,
    IDC_SCALE_CONFIG,
    REWARD_CONFIG,
)
from data_io.data_loader import build_external_series_from_config
from envs.idc_price_env import IDCPriceEnv20D


@pytest.fixture
def seed() -> int:
    """Deterministic default seed for tests that need reproducible randomness."""
    return 2026


@pytest.fixture
def base_env() -> IDCPriceEnv20D:
    """Base single-agent env built from canonical config with deterministic seeds."""
    return IDCPriceEnv20D(
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **IDC_SCALE_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        server_seed=2026,
        task_seed=2026,
        forecast_seed=2026 + ENV_CONFIG["task_forecast_seed_offset"],
    )
