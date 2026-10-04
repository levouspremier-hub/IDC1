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


def test_observer_cpu_timer_preserves_model_caller(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import scipy.optimize as optimize

    from scripts.m6p2b_seed0_diagnosis import Observer

    monkeypatch.setattr(optimize, "milp", lambda **kwargs: SimpleNamespace(status=0, message="ok"))
    observer = Observer(tmp_path)
    observer.current = {"solver_calls": []}
    with observer.installed():
        def model_caller():
            c_off = np.array([1.])
            deadline = 0.
            n_vars = 1
            rows = []
            return optimize.milp(c=c_off, options={"time_limit": .1}), deadline, n_vars, rows

        model_caller()
        assert observer.current["solver_calls"][0]["stage"] == "A"
        assert observer.current["solver_calls"][0]["model_variables"] == 1
        assert observer.current["solver_calls"][0]["cpu_s"] >= 0
        assert observer.current["solver_calls"][0]["thread_cpu_s"] >= 0
        with observer.unobserved():
            model_caller()
        assert len(observer.current["solver_calls"]) == 1
