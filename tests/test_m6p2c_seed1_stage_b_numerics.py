"""Immutable seed1 failure input; no environment or optimizer updates."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_seed1_feasible_projection_finishes_stage_b():
    path = Path(__file__).parent / 'fixtures/m6p2c_seed1_origin9120_step18.json'
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
