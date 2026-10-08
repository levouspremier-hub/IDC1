"""Original v21 coverage failure must retain exact integer primal and B quality."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_zero_flow_fractional_charge_support_selects_one_original_mode(monkeypatch):
    import numpy as np
    import scipy.optimize
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness
    real = scipy.optimize.linprog
    calls = []
    def record(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)
    monkeypatch.setattr(scipy.optimize, 'linprog', record)
    # z+q>=1 is the energy-cover support. q is almost 1; c=d=0 at z.
    # Rounding z=0,q~=1 is infeasible due to q<=1-4e-9 in the full rows.
    matrix = csr_matrix([[1.,0.,0.,1.], [0.,0.,0.,1.]])
    witness, audit = certify_or_polish_witness(
        [8e-11,0.,0.,1.-8e-11], [0.,0.,0.,0.], [1.,1.,1.,1.], [1,0,0,0],
        matrix, [1.,-np.inf], [np.inf,1.-4e-9], objective=[0.,0.,0.,0.],
        remaining=lambda:.5, storage_modes=([0],[1],[2]), storage_requirements=(True,False))
    assert witness is not None and audit['passed']
    assert witness[0] == 1. and len(calls)==1
    assert audit['primal_polish']['storage_mode_changes'][0]['selection_reason'] == (
        'required_inventory_fractional_zero_flow_support')


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


def test_zero_flow_fractional_discharge_support_is_symmetric():
    import numpy as np
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness
    witness, audit = certify_or_polish_witness(
        [1.-8e-11,0.,0.,1.-8e-11], [0.,0.,0.,0.], [1.,1.,1.,1.], [1,0,0,0],
        csr_matrix([[-1.,0.,0.,1.], [0.,0.,0.,1.]]), [0.,-np.inf], [np.inf,1.-4e-9],
        objective=[0.,0.,0.,0.], remaining=lambda:.5, storage_modes=([0],[1],[2]),
        storage_requirements=(False,True))
    assert witness is not None and audit['passed'] and witness[0]==0.


def test_neutral_inventory_does_not_change_zero_flow_fractional_support():
    import numpy as np
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness
    witness, audit = certify_or_polish_witness(
        [8e-11,0.,0.,1.-8e-11], [0.,0.,0.,0.], [1.,1.,1.,1.], [1,0,0,0],
        csr_matrix([[1.,0.,0.,1.], [0.,0.,0.,1.]]), [1.,-np.inf], [np.inf,1.-4e-9],
        objective=[0.,0.,0.,0.], remaining=lambda:.5, storage_modes=([0],[1],[2]),
        storage_requirements=(False,False))
    assert witness is None and not audit['passed']
    assert audit['primal_polish']['storage_mode_changes']==[]


def test_fractional_support_cannot_override_original_mode_bounds(monkeypatch):
    import numpy as np
    import scipy.optimize
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness
    def never(**kwargs):
        raise AssertionError('no LP for an integer candidate outside original bounds')
    monkeypatch.setattr(scipy.optimize,'linprog',never)
    witness, audit = certify_or_polish_witness(
        [8e-11,0.,0.,1.-8e-11], [0.,0.,0.,0.], [0.,1.,1.,1.], [1,0,0,0],
        csr_matrix([[1.,0.,0.,1.], [0.,0.,0.,1.]]), [1.,-np.inf], [np.inf,1.-4e-9],
        objective=[0.,0.,0.,0.], remaining=lambda:.5, storage_modes=([0],[1],[2]),
        storage_requirements=(True,False))
    assert witness is None and not audit['passed'] and 'primal_polish' not in audit
