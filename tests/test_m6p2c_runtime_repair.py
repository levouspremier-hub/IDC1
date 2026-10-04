"""Real A solve; controlled B outcomes test semantics, not timing performance."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

from contracts.models import DispatchProposal
from planning import model
from planning.corrector import correct
from tests.test_m44_corrector import _snapshot


def _proposal():
    return DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.3)


def _outcomes(monkeypatch, *, b_status=1, corrupt=False):
    original = scipy.optimize.milp
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            return SimpleNamespace(status=b_status, x=None)
        result = original(**kwargs)
        assert result.status == 0
        if corrupt:
            result.x[0] = np.nan
        return result

    monkeypatch.setattr(scipy.optimize, 'milp', solve)
    return calls


def test_b_timeout_retains_validated_a(monkeypatch):
    calls = _outcomes(monkeypatch)
    result = model.solve_time_indexed_mip_raw_projection(_snapshot(), _proposal())
    assert len(calls) == 2
    assert result.stage_b_status == 'time_limit'
    assert result.execution_source == 'stage_a'
    assert result.candidate_check['passed'] is True
    assert result.failure_class == 'stage_b_timeout_feasible'
    assert sum(result.exec_compute_actions) > 0
    assert result.max_constraint_residual <= 1e-6


def test_corrector_exposes_timeout_separately_from_executable_action(monkeypatch):
    _outcomes(monkeypatch)
    result = correct(_snapshot(), _proposal(), time_limit_s=2.0)
    assert str(result.failure) == 'stage_b_timeout_feasible'
    assert result.execution_source == 'stage_a'
    assert result.executable is True
    assert sum(result.exec_compute_actions) > 0


def test_invalid_a_is_not_executed(monkeypatch):
    _outcomes(monkeypatch, corrupt=True)
    result = model.solve_time_indexed_mip_raw_projection(_snapshot(), _proposal())
    assert result.failure_class == 'solver_failure'
    assert result.execution_source == 'none'
    assert not result.candidate_check['passed']


def test_b_solver_error_does_not_use_a(monkeypatch):
    _outcomes(monkeypatch, b_status=4)
    result = model.solve_time_indexed_mip_raw_projection(_snapshot(), _proposal())
    assert result.failure_class == 'solver_failure'
    assert result.execution_source == 'none'


def test_budget_exhausted_after_a_retains_a_without_calling_b(monkeypatch):
    original = scipy.optimize.milp
    clock = [0.0]
    calls = []
    monkeypatch.setattr(model, '_monotonic', lambda: clock[0])

    def solve(**kwargs):
        result = original(**kwargs)
        calls.append(result)
        assert result.status == 0
        clock[0] = 2.0
        return result

    monkeypatch.setattr(scipy.optimize, 'milp', solve)
    result = model.solve_time_indexed_mip_raw_projection(
        _snapshot(), _proposal(), time_limit_s=1.0)
    assert len(calls) == 1
    assert result.stage_b_status == 'not_run'
    assert result.execution_source == 'stage_a'
    assert result.failure_class == 'stage_b_timeout_feasible'


@pytest.mark.parametrize('kind', ['bound', 'integer', 'row', 'nan'])
def test_candidate_checker_rejects_invalid_vectors(kind):
    from scipy.sparse import csr_matrix
    x = {'bound': [2., 0.], 'integer': [.5, .5],
         'row': [1., 1.], 'nan': [float('nan'), 0.]}[kind]
    check = model.check_projection_candidate(
        np.array(x), np.zeros(2), np.ones(2), np.array([1, 0]),
        csr_matrix([[1., 1.]]), np.array([1.]), np.array([1.]))
    assert not check['passed']
