"""Development preflight must stop before any expensive phase after failure."""
import json

import pytest

from scripts import runtime_development as development


@pytest.mark.parametrize('failed_phase', ['static', 'latest', 'history', 'boundary', 'gate'])
def test_pipeline_stops_at_first_failed_stage_and_records_terminal_state(
        failed_phase, tmp_path, monkeypatch):
    calls = []
    def execute(label, command, folder):
        calls.append(label)
        return 1 if label == failed_phase else 0
    monkeypatch.setattr(development, 'execute', execute)
    assert development.preflight('stopped', base_dir=tmp_path) == 1
    assert calls == list(development.PHASES[:development.PHASES.index(failed_phase)+1])
    report = json.loads((tmp_path / 'stopped/report.json').read_text())
    assert report['state'] == 'failed' and report['exit_code'] == 1
    assert report['phases'][-1]['state'] == 'failed'
    assert report['training_state'] == 'not_started'
    assert report['transfer_state'] == 'not_started'


def test_interruption_has_a_terminal_artifact(tmp_path, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(development, 'execute', interrupt)
    assert development.preflight('interrupted', base_dir=tmp_path) == 130
    report = json.loads((tmp_path / 'interrupted/report.json').read_text())
    assert report['state'] == 'interrupted' and report['phases'][-1]['state'] == 'interrupted'


def test_freeze_requires_committed_source_and_complete_registered_boundary(tmp_path):
    with pytest.raises(ValueError, match='boundary'):
        development.freeze_after_preflight(tmp_path / 'report.json', 'v27')
