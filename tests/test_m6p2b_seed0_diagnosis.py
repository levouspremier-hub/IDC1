"""Regression requirements for the read-only final-policy diagnostic."""

from types import SimpleNamespace

import pytest
import torch

from scripts.m6p2b_seed0_diagnosis import (
    ORIGINS,
    assert_unchanged,
    case_grid,
    failure_stage,
    parameter_hash,
    validate_case,
)


def test_preregistered_cases_and_reject_unregistered_dates():
    assert len(case_grid()) == 24
    assert {x[0] for x in case_grid()} == set(ORIGINS)
    assert len(set(case_grid())) == 24
    for origin, mode in case_grid():
        validate_case(origin, mode)
    with pytest.raises(ValueError):
        validate_case(100000, "deterministic")
    with pytest.raises(ValueError):
        validate_case(ORIGINS[0], "sample_42")


def test_r_failure_must_not_be_called_stage_a():
    result = SimpleNamespace(failure_class="timeout", stage_a_status="time_limit")
    assert failure_stage(result, [{"stage": "R", "status": 1}]) == "R"
    assert failure_stage(result, []) == "before_first_solver_or_between_stages"
    assert failure_stage(result, [{"stage": "A", "status": 0}]) == "after_A"
    assert failure_stage(result, [{"stage": "B", "status": 4}]) == "B"


def test_parameter_mutation_detected():
    policy = torch.nn.Linear(2, 1)
    before = parameter_hash(policy)
    assert_unchanged(before, parameter_hash(policy), "policy")
    with torch.no_grad():
        policy.weight.add_(1)
    with pytest.raises(ValueError, match="policy"):
        assert_unchanged(before, parameter_hash(policy), "policy")


def test_failed_preflight_writes_complete_artifacts(tmp_path, monkeypatch):
    import json

    import scripts.m6p2b_seed0_diagnosis as diag

    (tmp_path / "uv.lock").write_text("test dependency lock")
    monkeypatch.setattr(diag, "ROOT", tmp_path)

    def reject():
        raise ValueError("release binding mismatch")

    monkeypatch.setattr(diag, "verify_release", reject)
    with pytest.raises(ValueError, match="binding mismatch"):
        diag.run("failed_test")
    folder = tmp_path / "runs/failed_test"
    assert all((folder / name).exists() for name in (
        "config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json"))
    assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        diag.run("failed_test")


def test_observer_preserves_solver_options_and_separates_stages(tmp_path, monkeypatch):
    import numpy as np
    import scipy.optimize as optimize

    import planning.corrector as corrector
    from scripts.m6p2b_seed0_diagnosis import Observer

    calls = []

    def solver(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=0, message="optimal")

    result = SimpleNamespace(failure_class="none", n_variables=1, n_constraints=1,
                             n_integer_variables=0, inventory_audit={},
                             stage_a_solve_time_s=0., stage_b_solve_time_s=0.)
    first, second = {"time_limit": .19}, {"time_limit": .11}

    def planner(snapshot, proposal, **kwargs):
        c_off = np.ones(1)
        c_econ = np.zeros(1)
        optimize.milp(c=c_off, options=first)
        optimize.milp(c=c_econ, options=second)
        return result

    monkeypatch.setattr(optimize, "milp", solver)
    monkeypatch.setattr(corrector, "solve_time_indexed_mip_raw_projection", planner)
    observer = Observer(tmp_path)
    with observer.installed():
        actual = corrector.solve_time_indexed_mip_raw_projection(None, None, time_limit_s=.25)
    assert actual is result
    assert calls[0]["options"] is first and calls[1]["options"] is second
    assert [e["stage"] for e in observer.current["solver_calls"]] == ["A", "B"]
    assert [e["passed_time_limit_s"] for e in observer.current["solver_calls"]] == [.19, .11]
    assert optimize.milp is solver


def test_checkpoint_hash_is_locked():
    from scripts.m6p2b_seed0_diagnosis import CHECKPOINT_SHA

    with pytest.raises(ValueError, match="checkpoint"):
        assert_unchanged(CHECKPOINT_SHA, "0" * 64, "checkpoint")


@pytest.mark.leakage
def test_forecast_evidence_uses_signed_bundle_and_visible_window():
    from scripts.m6p2b_seed0_diagnosis import forecast_evidence

    bundle = SimpleNamespace(split="train", generated_at="2024-04-15T00:00:00+08:00",
                             forecast_provenance={"price": {"source_kind": "seasonal_naive"}},
                             start="5040", forecast_cutoff=48)
    snapshot = SimpleNamespace(forecast=bundle, planning_forecast=SimpleNamespace(
        visible_mask=[True, False], assumed_mask=[False, True], extension_policy="frozen"))
    result = forecast_evidence(snapshot)
    assert result["sources"] == bundle.forecast_provenance
    assert result["generated_at"] == bundle.generated_at
    assert result["visible_mask"] == [True, False]
    bundle.split = "test"
    with pytest.raises(ValueError, match="train"):
        forecast_evidence(snapshot)
