"""Single-seed authorization cannot impersonate the three-seed gate."""

import pytest

from safe_rl_v2 import inventory_train as train


@pytest.mark.parametrize("seed", [1, 2])
def test_seed0_authorization_rejects_other_seeds(seed):
    with pytest.raises(ValueError, match="seed 0"):
        train.require_seed0_launch_gate({}, seed=seed)


def test_seed0_requires_a_verified_release(monkeypatch):
    from scenario import inventory_release as release

    def reject():
        raise ValueError("release rejected")

    monkeypatch.setattr(release, "verify_release", reject)
    with pytest.raises(ValueError, match="release rejected"):
        train.require_seed0_launch_gate({}, seed=0)


def test_seed0_cli_rejects_short_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(train, "run", lambda args: calls.append(args))
    with pytest.raises(SystemExit):
        train.main(["--short", "--seed0-formal", "--seed", "0", "--run-id", "fixture"])
    assert not calls


def test_seed0_cli_has_explicit_authorization(monkeypatch):
    calls = []
    monkeypatch.setattr(train, "run", lambda args: calls.append(args) or 0)
    assert train.main(["--seed0-formal", "--seed", "0", "--run-id", "fixture"]) == 0
    assert calls[0].seed0_formal is True and calls[0].short is False


@pytest.fixture
def qualified_gate(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json

    from safe_rl_v2 import inventory_diagnostics
    from scenario import inventory_release as release

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    old = {'release_path': 'r6', 'release_sha256': 'old', 'training_config_sha256': 'cfg',
           'semantics_binding': {'planning/model.py': 'physics',
                                 'safe_rl_v2/inventory_train.py': 'old-entry',
                                 'scenario/inventory_release.py': 'old-release'}}
    current = copy.deepcopy(old)
    current.update(release_path='r7', release_sha256='new')
    current['semantics_binding']['safe_rl_v2/inventory_train.py'] = 'new-entry'
    folder = tmp_path / 'runs/short'
    folder.mkdir(parents=True)
    for f in ['report.json', 'manifest.json', 'checkpoint_final.pt']:
        (folder / f).write_text('fixture')
    short = {'seed': 0, 'scope': 'controlled_short_run', 'batches': 8,
             'transitions': 1536, 'adam_steps': 128, 'lagrangian_updates': 8,
             'inventory_episodes': [{} for _ in range(32)],
             'batch_records': [{'zero_action_fallback_steps': 0,
                               'storage_signal': {'head_grad_norm_mean': 1}}]}
    gate_folder = tmp_path / 'runs/gate'
    gate_folder.mkdir()
    source = {'seed': 0, 'run_id': 'short', 'checks': {'fixture': True},
              'report_sha256': digest(folder / 'report.json'),
              'source_manifest_sha256': digest(folder / 'manifest.json'),
              'checkpoint_sha256': digest(folder / 'checkpoint_final.pt')}
    gate = {'passed': True, 'seeds': [0], 'formal_three_seed_gate': False,
            'inventory_binding': old, 'short_runs': [source]}
    (gate_folder / 'report.json').write_text(json.dumps(gate))
    (gate_folder / 'manifest.json').write_text(json.dumps({'status': 'success'}))
    authorization = {'gate_path': 'runs/gate', 'gate_hashes': {
        f: digest(gate_folder / f) for f in ['report.json', 'manifest.json']}}
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    monkeypatch.setattr(release, 'verify_release',
                        lambda: {'seed0_formal_authorization': authorization})

    def verified(path, binding):
        assert path == folder and binding == old
        return short

    monkeypatch.setattr(train, 'verify_written_run', verified)
    monkeypatch.setattr(inventory_diagnostics, 'inventory_episode_acceptance',
                        lambda e: {'qualified': True})
    return current, authorization, folder, gate_folder, short


def test_qualified_single_seed_can_launch_without_three_seed_claim(qualified_gate):
    binding, authorization, _, _, _ = qualified_gate
    assert train.require_seed0_launch_gate(binding, seed=0) == authorization


@pytest.mark.parametrize('mutation', ['planning', 'config', 'missing_source'])
def test_inheritance_rejects_changed_runtime_semantics(qualified_gate, mutation):
    binding, _, _, _, _ = qualified_gate
    if mutation == 'planning':
        binding['semantics_binding']['planning/model.py'] = 'changed'
    elif mutation == 'config':
        binding['training_config_sha256'] = 'changed'
    else:
        del binding['semantics_binding']['planning/model.py']
    with pytest.raises(ValueError, match='binding differs|semantics'):
        train.require_seed0_launch_gate(binding, seed=0)


@pytest.mark.parametrize('artifact', ['gate_report', 'checkpoint'])
def test_single_seed_rejects_tampered_evidence(qualified_gate, artifact):
    binding, _, short, gate, _ = qualified_gate
    p = gate / 'report.json' if artifact == 'gate_report' else short / 'checkpoint_final.pt'
    p.write_text('tampered')
    with pytest.raises(ValueError, match='hash mismatch|no longer qualify'):
        train.require_seed0_launch_gate(binding, seed=0)


def test_single_seed_rejects_failed_live_episode(qualified_gate, monkeypatch):
    from safe_rl_v2 import inventory_diagnostics

    binding, _, _, _, _ = qualified_gate
    monkeypatch.setattr(inventory_diagnostics, 'inventory_episode_acceptance',
                        lambda e: {'inventory': False})
    with pytest.raises(ValueError, match='no longer qualify'):
        train.require_seed0_launch_gate(binding, seed=0)
