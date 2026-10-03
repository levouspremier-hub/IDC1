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
