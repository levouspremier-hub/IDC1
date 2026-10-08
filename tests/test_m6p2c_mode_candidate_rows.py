"""An exact mode candidate must preserve the original primary witness quality."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_aggregate_floor_comes_only_from_original_queue_balance_bounds():
    from planning.numeric_contract import aggregate_service_floor
    for arrivals in [0.,3.,10.]:
        for previous_lower in [0.,2.]:
            for next_upper in [0.,4.,15.]:
                floor=aggregate_service_floor(previous_lower,arrivals,next_upper)
                for previous in [previous_lower,previous_lower+2.]:
                    for following in [0.,next_upper]:
                        work=previous+arrivals-following
                        if work>=0.:
                            assert 0.<=floor<=work


def test_charge_mode_is_essential_when_all_other_capacity_is_insufficient():
    import itertools

    from planning.numeric_contract import inventory_mode_cover
    for upper in [[2.,.1,.2],[.3,.3,.3],[1.,2.,3.]]:
        for required in [.2,.5,1.,3.]:
            cover=inventory_mode_cover(50.,50.+required*.5*.95,50.+required*.5*.95,
                                       .5,.95,.95,upper,upper)
            for modes in itertools.product((0,1),repeat=3):
                if sum(u*z for u,z in zip(upper,modes))>=required:
                    assert all(modes[k]==1 for k in cover['essential_charge_modes'])
                if sum(u*(1-z) for u,z in zip(upper,modes))>=required:
                    discharge=inventory_mode_cover(50.+required*.5/.95,50.,50.,
                                                  .5,.95,.95,upper,upper)
                    assert all(modes[k]==0 for k in discharge['essential_discharge_modes'])


def test_conserved_forecast_service_tightens_future_charge_capacity():
    case=json.loads((Path(__file__).parent/'fixtures'/'m6p2c_seed1_origin6336_step2.json').read_text())
    result=correct(InventorySnapshot.model_validate(case['snapshot']),
                   DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
    cover=result.inventory_audit['inventory_mode_cover']
    assert cover['essential_charge_modes']==[0]
    assert result.inventory_audit['coupled_aggregate_service_lower_work'][1]>0.
    assert result.inventory_audit['primary_objective_certificate']['passed']


def test_recorded_seed1_origin6336_keeps_the_certified_primary_objective():
    case=json.loads((Path(__file__).parent/'fixtures'/'m6p2c_seed1_origin6336_step2.json').read_text())
    result=correct(InventorySnapshot.model_validate(case['snapshot']),
                   DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
    assert result.executable, result.inventory_audit.get('primary_objective_certificate')
    assert result.candidate_check['integer_residual']==0.
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.inventory_audit['execution_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']
