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


def test_successful_phase_exit_codes_cannot_bypass_missing_boundary_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(development, 'execute', lambda *args: 0)
    assert development.preflight('missing-boundary', base_dir=tmp_path) == 1
    report = json.loads((tmp_path / 'missing-boundary/report.json').read_text())
    assert report['state'] == 'failed' and 'six source-verified' in report['error']
    assert not report['passed']


def test_interrupt_terminates_only_the_owned_child_process_group(tmp_path, monkeypatch):
    launched, signals = {}, []
    class Process:
        pid = 123456789
        waits = 0
        def wait(self, **kwargs):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -15
        def terminate(self):
            signals.append('single_process_only')
        def kill(self):
            signals.append('single_process_only')
    def spawn(command, **kwargs):
        launched.update(kwargs)
        return Process()
    monkeypatch.setattr(development.subprocess, 'Popen', spawn)
    monkeypatch.setattr(development.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        development.execute('gate', ['make', 'check'], tmp_path)
    assert launched.get('start_new_session') is True
    assert signals == [(123456789, development.signal.SIGTERM)]


def test_preflight_rejects_validation_inputs_changed_during_checks(tmp_path, monkeypatch):
    original = development.scientific_binding()
    calls = 0
    def binding():
        nonlocal calls
        calls += 1
        return {**original, 'validation_sha256': 'before' if calls == 1 else 'after'}
    monkeypatch.setattr(development, 'scientific_binding', binding)
    monkeypatch.setattr(development, 'execute', lambda *args: 0)
    monkeypatch.setattr(development, 'load_registry',
                        lambda: {'successful_anchors': dict.fromkeys(range(6))})
    assert development.preflight('changed-validation', base_dir=tmp_path) == 1
    report = json.loads((tmp_path / 'changed-validation/report.json').read_text())
    assert 'validation' in report['error']
