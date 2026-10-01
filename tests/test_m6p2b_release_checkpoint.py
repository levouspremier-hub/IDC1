"""V2 semantic binding rejects historical checkpoints and preserves real weights."""

import pytest
import torch

from checkpointing import VersionedCheckpoint
from checkpointing import inventory_eval_input as inv
from safe_rl_v2.formal_train_loop import build_seeded_policy, load_frozen_training_config


@pytest.fixture(params=[520, 523])
def config_and_binding(monkeypatch, request):
    config = load_frozen_training_config()
    config["training"]["policy"]["obs_dim"] = request.param
    binding = {'inventory_version': 'terminal-inventory-v1', 'release_sha256': 'a' * 64}
    monkeypatch.setattr(inv, 'load_config', lambda: config)
    monkeypatch.setattr(inv, 'checkpoint_binding', lambda: binding)
    return config, binding


def test_old_formal_checkpoint_is_rejected_by_new_export(tmp_path, config_and_binding):
    source = tmp_path / 'old.pt'
    VersionedCheckpoint('contract-v9', 21, 520, 'm13gfck-formal-train-resume-v1',
                        'a' * 40, {}).save(source)
    with pytest.raises(ValueError, match='schema|obs_dim'):
        inv.export_policy(source, tmp_path / 'new.pt')


def test_export_preserves_weights_and_enforces_semantic_binding(tmp_path, config_and_binding):
    config, binding = config_and_binding
    obs_dim = config["training"]["policy"]["obs_dim"]
    policy = build_seeded_policy(config, obs_dim=obs_dim, seed=2)
    source, exported = tmp_path / 'source.pt', tmp_path / 'eval.pt'
    VersionedCheckpoint(
        'contract-v9', 21, config['training']['policy']['obs_dim'], inv.FORMAL_SCHEMA, 'a' * 40,
        {'policy': policy.state_dict(), 'frozen_config': config,
         'training_scope': 'formal_training', 'artifact_role': 'formal_training_resume',
         'code_revision': 'a' * 40}, extras={'inventory_binding': binding}).save(source)
    loaded = inv.export_policy(source, exported)
    obs = torch.linspace(-1, 1, obs_dim)
    assert torch.equal(policy.act_mean(obs), loaded.act_mean(obs))
    with pytest.raises(ValueError, match='semantics/role'):
        inv.load_policy(exported, formal=False)
    payload = torch.load(exported, weights_only=False)
    payload['metadata']['inventory_binding']['release_sha256'] = 'b' * 64
    torch.save(payload, exported)
    with pytest.raises(ValueError, match='semantics/role'):
        inv.load_policy(exported)


def test_wrong_scope_cannot_masquerade_as_formal(tmp_path, config_and_binding):
    config, binding = config_and_binding
    policy = build_seeded_policy(config, obs_dim=config["training"]["policy"]["obs_dim"], seed=0)
    source = tmp_path / 'source.pt'
    VersionedCheckpoint(
        'contract-v9', 21, config['training']['policy']['obs_dim'], inv.FORMAL_SCHEMA, 'a' * 40,
        {'policy': policy.state_dict(), 'frozen_config': config,
         'training_scope': 'controlled_short_run', 'artifact_role': 'controlled_training_resume',
         'code_revision': 'a' * 40}, extras={'inventory_binding': binding}).save(source)
    with pytest.raises(ValueError, match='scope/role/revision'):
        inv.export_policy(source, tmp_path / 'eval.pt')


@pytest.mark.resume
@pytest.mark.parametrize("obs_dim", [520, 523])
def test_bound_checkpoint_restores_adam_multipliers_and_both_rngs(tmp_path, obs_dim):
    import numpy as np

    from safe_rl_v2.controlled_formal_train import _generators
    from safe_rl_v2.formal_train_loop import (
        build_lagrangian,
        build_optimizer,
        load_resume_checkpoint,
        optimizer_state_digest,
    )
    from safe_rl_v2.inventory_train import save_bound
    from safe_rl_v2.ppo_update import minibatch_ppo_step

    config = load_frozen_training_config()
    config["training"]["policy"]["obs_dim"] = obs_dim
    def setup():
        policy = build_seeded_policy(config, obs_dim=obs_dim, seed=0)
        return policy, build_optimizer(config, policy), build_lagrangian(config), *_generators(0)
    def update(policy, opt, lag, sampling, shuffle):
        obs = torch.linspace(-1, 1, 4 * obs_dim).reshape(4, obs_dim)
        raw, logp, _ = policy.act(obs, generator=sampling)
        order = torch.randperm(4, generator=shuffle)
        adv = torch.tensor([1., -1., .5, -.5])[order]
        minibatch_ppo_step(
            policy, opt, observation=obs[order], raw_action=raw.detach()[order],
            old_raw_log_prob=logp.detach()[order], adv_reward=adv,
            adv_business=torch.zeros(4), adv_carbon=torch.zeros(4),
            critic_targets={key: torch.zeros(4) for key in ('reward', 'business', 'carbon')},
            lambda_business=0., lambda_carbon=0., clip_epsilon=.2)
        lag.update({'business': np.zeros(4), 'carbon': np.ones(4)})
    live = setup()
    update(*live)
    path = tmp_path / 'resume.pt'
    save_bound(path, binding={'version': 'unit-test'}, policy=live[0], optimizer=live[1],
               lagrangian=live[2], sampling_generator=live[3], shuffle_generator=live[4],
               config=config, origins=[48, 96, 144, 192], next_batch_index=1, obs_dim=obs_dim,
               origin_provenance={48: 'a' * 64}, master_seed=0, code_revision='a' * 40,
               training_scope='controlled_short_run', artifact_role='controlled_training_resume',
               schema=inv.SHORT_SCHEMA)
    update(*live)
    restored = setup()
    load_resume_checkpoint(
        path, policy=restored[0], optimizer=restored[1], lagrangian=restored[2],
        sampling_generator=restored[3], shuffle_generator=restored[4], config=config,
        expected_obs_dim=obs_dim, expected_schema=inv.SHORT_SCHEMA,
        expected_scope='controlled_short_run', expected_role='controlled_training_resume')
    update(*restored)
    assert all(torch.equal(v, restored[0].state_dict()[k])
               for k, v in live[0].state_dict().items())
    assert optimizer_state_digest(live[1]) == optimizer_state_digest(restored[1])
    assert live[2].state_dict() == restored[2].state_dict()
    assert torch.equal(live[3].get_state(), restored[3].get_state())
    assert torch.equal(live[4].get_state(), restored[4].get_state())


def test_matrix_cannot_change_frozen_design_or_asset_hash(tmp_path, monkeypatch):
    import copy
    import json
    from pathlib import Path

    from scenario import inventory_release as release

    original = json.loads((Path(__file__).resolve().parent.parent /
                          'configs/experiments/m9_experiment_matrix_v3.json').read_text())
    old_path = tmp_path / 'configs/experiments/m9_experiment_matrix_v3.json'
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps(original))
    matrix = copy.deepcopy(original)
    matrix['schema'] = 'm9-experiment-matrix-v4'
    matrix['policy_observation'] = release.observation_spec()
    matrix['sources'].pop('training_config_v1')
    matrix['sources']['training_config_v2'] = {
        'logical_path': release.CONFIG_PATH, 'sha256': 'a' * 64}
    hashes = {str(tmp_path / s['logical_path']): s['sha256']
              for s in matrix['sources'].values()}
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr(release, 'sha', lambda path: hashes[str(path)])
    path = tmp_path / release.MATRIX_PATH
    path.write_text(json.dumps(matrix))
    assert release.load_matrix()['training_schedule'] == original['training_schedule']
    matrix['pairing']['preregistered_scenario_seed'] = 7
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match='frozen experimental design'):
        release.load_matrix()
    matrix['pairing'] = original['pairing']
    matrix['sources']['refs_v4']['sha256'] = 'b' * 64
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match='live asset mismatch: refs_v4'):
        release.load_matrix()


def test_formal_gate_rechecks_source_runs_instead_of_trusting_passed_flag(tmp_path, monkeypatch):
    import json

    from safe_rl_v2 import inventory_train as train
    from scenario.inventory_release import RUN_REVISION
    from scripts import m6p2b_inventory_repair as diagnostics

    folder = tmp_path / f'runs/m6p2b_short_gate_{RUN_REVISION}'
    folder.mkdir(parents=True)
    binding = {'version': 'unit-test'}
    (folder / 'manifest.json').write_text(json.dumps({'status': 'success'}))
    (folder / 'report.json').write_text(json.dumps({
        'passed': True, 'inventory_binding': binding, 'short_runs': [],
        'formal_three_seed_gate': True}))
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    monkeypatch.setattr(diagnostics, 'short_gate', lambda: ([], {'passed': False}, {}))
    with pytest.raises(ValueError, match='live short-run artifacts'):
        train.require_short_gate(binding)


def test_frozen_v2_config_accepts_same_ordered_origins_from_json_and_selector():
    from scenario.inventory_release import CONFIG_PATH, ROOT, load_config, load_matrix
    if not (ROOT / CONFIG_PATH).exists():
        pytest.skip('v2 calibration not available on this checkout')
    config = load_config()
    assert config['version'] == 'v2'
    assert len(config['calibration']['origins']) == 24
    assert load_matrix()['schema'] == 'm9-experiment-matrix-v4'


def test_old_520_checkpoint_cannot_enter_versioned_523_path(tmp_path, config_and_binding):
    config, binding = config_and_binding
    config["training"]["policy"]["obs_dim"] = 523
    path = tmp_path / "old-v2.pt"
    VersionedCheckpoint(
        "contract-v9", 21, 520, inv.FORMAL_SCHEMA, "a" * 40, {},
        extras={"inventory_binding": binding}).save(path)
    with pytest.raises(ValueError, match="obs_dim 520"):
        inv.export_policy(path, tmp_path / "new.pt")


def test_calibration_backend_note_is_not_a_runtime_parameter_but_thread_count_is(
        tmp_path, monkeypatch):
    import json

    from scenario import inventory_release as release

    path = release.ROOT / release.CONFIG_PATH
    if not path.exists():
        pytest.skip('v2 calibration not available on this checkout')
    config = json.loads(path.read_text())
    local = tmp_path / 'config.json'
    local.write_text(json.dumps(config))
    monkeypatch.setattr(release, 'CONFIG_PATH', str(local))
    # The measured candidate and frozen config carry different explanatory notes.
    assert release.load_config()['training']['backend']['torch_num_threads'] == 1
    config['training']['backend']['torch_num_threads'] = 2
    local.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='measured calibration: backend'):
        release.load_config()
