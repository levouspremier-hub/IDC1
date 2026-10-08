"""Original v21 coverage failure must retain exact integer primal and B quality."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_coverage_seed1_origin1968_has_certified_execution():
    case = json.loads((Path(__file__).parent / 'fixtures' /
                       'm6p2c_seed1_origin1968_step18.json').read_text())
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.stage_a_status == result.stage_b_status == 'optimal'
    assert result.executable, result.inventory_audit.get('execution_witness')
    assert result.candidate_check['integer_residual'] == 0.
    assert result.inventory_audit['execution_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']
    if 'polished_economic_objective_certificate' in result.inventory_audit:
        assert result.inventory_audit['polished_economic_objective_certificate']['passed']
