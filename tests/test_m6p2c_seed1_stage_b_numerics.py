"""Immutable seed1 failure input; no environment or optimizer updates."""
import json

import pytest
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


@pytest.mark.parametrize('case_name', [
    'm6p2c_seed1_origin9120_step18.json',
    'm6p2c_seed1_origin1008_step18.json',
])
def test_recorded_seed1_feasible_projection_finishes_stage_b(case_name):
    path = Path(__file__).parent / 'fixtures' / case_name
    case = json.loads(path.read_text())
    snapshot = InventorySnapshot.model_validate(case['snapshot'])
    proposal = DispatchProposal.model_validate(case['proposal'])
    # Test numerical correctness independently from performance qualification.
    result = correct(snapshot, proposal, time_limit_s=2.0)
    assert result.stage_a_status == 'optimal'
    assert result.stage_b_status == 'optimal'
    assert result.execution_source == 'stage_b'
    assert result.executable is True
    assert result.candidate_check['passed'] is True


def test_candidate_records_stricter_solver_precision_without_science_changes():
    from scenario.inventory_release import load_config
    from scenario.runtime_release import candidate_config
    candidate = candidate_config(.50)
    inherited = load_config()
    assert candidate['training']['corrector']['solver_feasibility_tolerance'] == 1e-9
    for section in ('policy', 'optimizer', 'ppo', 'sampling', 'scale', 'budgets', 'multipliers'):
        assert candidate['training'][section] == inherited['training'][section]
    assert candidate['training']['corrector']['time_limit_s'] == .50
