"""An original integer B witness must retain its original economic certificate."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_seed2_origin8496_economic_incumbent_is_certified():
    p = Path(__file__).parent / 'fixtures' / 'm6p2c_seed2_origin8496_step6.json'
    case = json.loads(p.read_text())
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.inventory_audit['primary_objective_certificate']['passed']
    assert result.executable and result.stage_b_status == 'optimal'
    assert result.candidate_check['integer_residual'] == 0.
