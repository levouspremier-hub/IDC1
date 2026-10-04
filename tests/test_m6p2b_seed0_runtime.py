"""Runtime probe must not mutate original solver options or budget."""
import pytest

from scripts.m6p2b_seed0_runtime import options_for_arm


def test_options_control_budget_and_aliasing():
    base = {"parallel": False, "random_seed": 0, "time_limit": .013}
    assert options_for_arm(base, "published") == base
    one = options_for_arm(base, "single_thread")
    assert one == {**base, "threads": 1}
    assert "threads" not in base
    with pytest.raises(ValueError):
        options_for_arm(base, "unknown")
