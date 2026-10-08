"""An exact mode candidate must preserve the original primary witness quality."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_seed1_origin6336_keeps_the_certified_primary_objective():
    case=json.loads((Path(__file__).parent/'fixtures'/'m6p2c_seed1_origin6336_step2.json').read_text())
    result=correct(InventorySnapshot.model_validate(case['snapshot']),
                   DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
    assert result.executable, result.inventory_audit.get('primary_objective_certificate')
    assert result.candidate_check['integer_residual']==0.
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.inventory_audit['execution_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']
