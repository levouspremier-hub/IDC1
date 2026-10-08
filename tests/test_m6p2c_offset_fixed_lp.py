"""Original seed1 near-integer B must satisfy the unchanged primary offset."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_origin2544_b_has_exact_original_feasible_witness():
    case = json.loads((Path(__file__).parent / 'fixtures' / 'm6p2c_seed1_origin2544_step23.json').read_text())
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']
    assert result.executable and result.stage_b_status == 'optimal'
    assert result.candidate_check['integer_residual'] == 0.
