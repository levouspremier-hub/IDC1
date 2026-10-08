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


def test_projection_domain_uses_encoded_rows_and_preserves_tiny_positive_power():
    from fractions import Fraction as F

    import numpy as np

    from planning.numeric_contract import projection_storage_power_bounds

    primary = float(np.nextafter(.5, np.inf))
    result = projection_storage_power_bounds(
        [1., 1.], [1., .5], [2.], primary, .5, .25, .05, .05)
    exact_charge = (F(primary) - F(.25) - F(.25)) / F(.05)
    assert F(result['charge_upper_kw']) >= exact_charge > 0
    assert result['charge_upper_kw'] < 1e-12
    assert result['minimum_compute_offset'] <= .25
    assert F(result['charge_cut_coefficient']) * exact_charge <= 1
    assert not result['charge_mode_forbidden']


def test_projection_domain_forbids_only_proven_branch_and_is_symmetric():
    from planning.numeric_contract import projection_storage_power_bounds

    charge = projection_storage_power_bounds([1., 1.], [1., .5], [2.], .49, .5,
                                            .25, .05, .05)
    discharge = projection_storage_power_bounds([1., 1.], [1., .5], [2.], .49, .5,
                                               -.25, .05, .05)
    assert charge['charge_mode_forbidden'] and charge['charge_upper_kw'] == 0
    assert discharge['discharge_mode_forbidden'] and discharge['discharge_upper_kw'] == 0
    assert not charge['discharge_mode_forbidden']
    assert not discharge['charge_mode_forbidden']


def test_projection_domain_zero_capacity_has_original_fixed_zero_action():
    from planning.numeric_contract import projection_storage_power_bounds

    result = projection_storage_power_bounds([1., 1.], [0., .5], [2.], .75, .5,
                                            .25, .05, .05)
    assert result['minimum_compute_offset'] <= .5
    assert result['charge_upper_kw'] >= 0
