"""Equivalent power coordinates must retain the original small-flow B witness."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_origin9744_exact_a_remains_feasible_for_b():
    p = Path(__file__).parent / 'fixtures' / 'm6p2c_seed0_origin9744_step44.json'
    case = json.loads(p.read_text())
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.inventory_audit['primary_objective_certificate']['passed']
    assert result.executable and result.stage_b_status == 'optimal'
    assert result.candidate_check['integer_residual'] == 0.
