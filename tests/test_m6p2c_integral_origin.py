"""B must retain a feasible certified A under a bijection of integer coordinates."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_origin3072_exact_a_is_feasible_for_stage_b():
    case=json.loads((Path(__file__).parent/'fixtures'/'m6p2c_seed0_origin3072_step18.json').read_text())
    result=correct(InventorySnapshot.model_validate(case['snapshot']),
                   DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']
    assert result.stage_b_status=='optimal' and result.executable
    assert result.candidate_check['integer_residual']==0.
