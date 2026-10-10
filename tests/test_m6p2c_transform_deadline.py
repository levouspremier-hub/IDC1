"""B preprocessing is part of the shared deadline, not permission for a late solve."""
import scipy.optimize

from contracts.models import DispatchProposal
from planning import model, numeric_contract
from planning.snapshot_adapter import build_snapshot
from tests.test_m6p2b_schedule_consistency import fixture_env


def test_no_stage_b_solver_is_started_after_transform_consumes_budget(monkeypatch):
    clock = [0.]
    calls_after_transform = []
    real_scale = numeric_contract.scale_constraint_rows
    real_milp = scipy.optimize.milp
    def scale(*args, **kwargs):
        result = real_scale(*args, **kwargs)
        clock[0] = 1.
        return result
    def milp(**kwargs):
        if clock[0] > 0:
            calls_after_transform.append(kwargs['options']['time_limit'])
        return real_milp(**kwargs)
    monkeypatch.setattr(model, '_monotonic', lambda: clock[0])
    monkeypatch.setattr(numeric_contract, 'scale_constraint_rows', scale)
    monkeypatch.setattr(scipy.optimize, 'milp', milp)
    result = model.solve_time_indexed_mip_raw_projection(
        build_snapshot(fixture_env()),
        DispatchProposal(compute_actions=[0.] * 20, storage_action=0.), time_limit_s=.50)
    assert not calls_after_transform
    assert result.stage_b_status == 'not_run'
    assert result.execution_source == 'stage_a'
    assert result.inventory_audit['stage_b_budget_exhausted_after_transform'] is True
    assert result.inventory_audit['execution_witness']['passed']
    assert result.candidate_check['passed']
