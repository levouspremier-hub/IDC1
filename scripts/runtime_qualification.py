"""Pinned train-only runtime profiling and serial qualification; never 512 batches."""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpointing import VersionedCheckpoint
from checkpointing.inventory_eval_input import FORMAL_SCHEMA
from runs.writer import write_run
from safe_rl_v2.controlled_formal_train import apply_frozen_thread_setting
from safe_rl_v2.formal_train_loop import (
    build_seeded_policy,
    build_train_env,
)
from safe_rl_v2.inventory_diagnostics import evaluate_origin, inventory_episode_acceptance
from scenario.inventory_release import ROOT, sha
from scenario.runtime_release import BUDGETS, build_candidate, verify_candidate
from scripts.idc_remote import make_receipt
from scripts.m6p2b_seed0_diagnosis import CHECKPOINT_SHA, SOURCE, dump, parameter_hash

ORIGINS = (5040, 5088, 5136, 5184, 5568, 6576)
MODES = ('deterministic', 'sample0', 'sample1', 'sample2')


def load_diagnostic_policy(config):
    source = SOURCE / 'checkpoint_final.pt'
    if sha(source) != CHECKPOINT_SHA:
        raise ValueError('historical diagnostic checkpoint hash mismatch')
    ckpt = VersionedCheckpoint.load(source, expected_action_dim=21,
                                    expected_obs_dim=523, expected_schema_hash=FORMAL_SCHEMA)
    if (ckpt.state.get('artifact_role') != 'formal_training_resume'
            or ckpt.state.get('training_scope') != 'formal_training'
            or ckpt.extras['inventory_binding']['observation_dimension'] != 523):
        raise ValueError('historical diagnostic checkpoint role mismatch')
    policy = build_seeded_policy(config, obs_dim=523, seed=0)
    policy.load_state_dict(ckpt.state['policy'], strict=True)
    policy.eval()
    return policy


def save(run_id, candidate, rows, report, status, error=None):
    write_run(run_id, config=candidate, metrics=pd.DataFrame([
        {k: v for k, v in row.items() if not isinstance(v, (dict, list, tuple))}
        for row in rows]), report={**report, 'failure': error}, status=status,
        failure_classification=error, command=' '.join(sys.argv),
        dependency_lock_hash=sha(ROOT / 'uv.lock'),
        data_hash=sha(ROOT / 'data/processed/singapore_2024/half_hour.parquet'),
        scenario_hash=sha(ROOT / 'data/manifest/formal_splits_v5/train.json'))


def episode(folder, candidate, origin, mode, policy=None):
    if origin not in ORIGINS or mode not in MODES:
        raise ValueError('unregistered train diagnostic case')
    folder.mkdir(parents=True, exist_ok=False)
    config = candidate['config']
    apply_frozen_thread_setting(config)
    policy = load_diagnostic_policy(config) if policy is None else policy
    before = parameter_hash(policy)
    generator = torch.Generator(device='cpu').manual_seed(
        0 if mode == 'deterministic' else int(mode[-1]))

    def action(obs):
        with torch.inference_mode():
            tensor = torch.as_tensor(obs, dtype=torch.float32)
            if mode == 'deterministic':
                from evaluation.controlled_run import deterministic_action
                raw = torch.as_tensor(deterministic_action(policy, obs))
                logp = policy.evaluate_raw_actions(tensor, raw)
            else:
                raw, logp, _ = policy.act(tensor, generator)
            action_records.append({'observation': np.asarray(obs).tolist(),
                                   'a_raw': raw.tolist(), 'old_raw_log_prob': float(logp)})
            if not torch.isfinite(raw).all() or not torch.isfinite(logp).all():
                raise ValueError('nonfinite diagnostic raw action/log probability')
            return raw.cpu().numpy().astype(np.float32)

    started = time.perf_counter()
    action_records = []
    try:
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter('ignore', RuntimeWarning)
            result, rows, _, _ = evaluate_origin(
                config, origin, 0, action, run_id=folder.name, strict=True)
        with (folder / 'steps.jsonl').open('w') as stream:
            from scripts.m6p2b_seed0_diagnosis import jsonable
            for row in rows:
                stream.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
        dump(folder / 'raw_actions.json', action_records)
        result.update(mode=mode, pid=os.getpid(), elapsed_s=time.perf_counter()-started,
                      parameter_hash_before=before, parameter_hash_after=parameter_hash(policy),
                      parameter_updates=0, budget_s=config['runtime_budget_s'])
        result['acceptance'] = inventory_episode_acceptance(result)
        # Qualification requires the uniform target, not the registered initial-gap exception.
        result['acceptance']['uniform_terminal_target'] = result['target_qualified'] is True
        result['acceptance']['parameters_unchanged'] = before == parameter_hash(policy)
        dump(folder / 'episode.json', result)
        if not all(result['acceptance'].values()):
            raise ValueError('episode failed frozen qualification standards')
        return result
    except Exception as exc:
        dump(folder / 'failure.json', {'reason': str(exc),
             'evidence': getattr(exc, 'evidence', {}), 'origin': origin, 'mode': mode,
             'raw_actions': action_records,
             'budget_s': config['runtime_budget_s'], 'parameter_hash_before': before,
             'parameter_hash_after': parameter_hash(policy)})
        raise


def profile(run_id, path):
    candidate = verify_candidate(path)
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    rows = []
    report = {'train_only': True, 'formal_training_ready': False,
              'candidate_sha256': sha(path),
              'timing_note': 'profile overhead; not qualification speed'}
    save(run_id, candidate, rows, report, 'running')
    try:
        config = candidate['config']
        apply_frozen_thread_setting(config)
        from safe_rl_v2.formal_train_loop import _verified_train_input
        for i, origin in enumerate((5040, 5040, 5088, 5136)):
            profiler = cProfile.Profile()
            started = time.perf_counter()
            profiler.enable()
            env, injection = build_train_env(origin, master_seed=0, config=config)
            profiler.disable()
            rows.append({'index': i, 'origin': origin, 'env_build_s': time.perf_counter()-started,
                         'cache_hits': _verified_train_input.cache_info().hits,
                         'cache_misses': _verified_train_input.cache_info().misses,
                         'provenance': injection.provenance_hash})
            output = io.StringIO()
            pstats.Stats(profiler, stream=output).sort_stats('cumulative').print_stats(60)
            (folder / f'env_profile_{i}.txt').write_text(output.getvalue())
            env.close()
        from contracts.inventory import InventorySnapshot
        from contracts.models import DispatchProposal
        from planning.corrector import correct
        from scripts.m6p2b_seed0_runtime import CAPTURE, CAPTURE_SHA
        if sha(CAPTURE) != CAPTURE_SHA:
            raise ValueError('capture hash mismatch')
        capture = json.loads(CAPTURE.read_text())
        snapshot = InventorySnapshot.model_validate(capture['snapshot'])
        proposal = DispatchProposal.model_validate(capture['proposal'])
        replay = []
        for i in range(8):
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = correct(snapshot, proposal, time_limit_s=config['runtime_budget_s'])
            replay.append({'index': i, 'wall_s': time.perf_counter()-started,
                           'executable': result.executable, 'source': result.execution_source,
                           'failure': str(result.failure), 'a_s': result.stage_a_solve_time_s,
                           'b_s': result.stage_b_solve_time_s,
                           'exec': [*result.exec_compute_actions, result.exec_storage_action]})
        report.update(env_rows=rows, replay=replay, passed=all(r['executable'] for r in replay))
        save(run_id, candidate, rows, report, 'success' if report['passed'] else 'failed')
        return 0 if report['passed'] else 1
    except Exception as exc:
        save(run_id, candidate, rows, report, 'failed', str(exc))
        raise


def diagnose(run_id, path, *, soak=False):
    candidate = verify_candidate(path)
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    rows = []
    report = {'train_only': True, 'candidate_sha256': sha(path), 'formal_training_ready': False,
              'checkpoint_sha256': CHECKPOINT_SHA, 'parameter_updates': 0, 'soak': soak}
    save(run_id, candidate, rows, report, 'running')

    def isolated(index, origin, mode, label):
        output = folder / label / f'{index:06d}_{origin}_{mode}'
        subprocess.run([sys.executable, '-m', 'scripts.runtime_qualification', 'episode',
                        '--candidate', str(path), '--output', str(output),
                        '--origin', str(origin), '--mode', mode], cwd=ROOT, check=True)
        return json.loads((output / 'episode.json').read_text())

    try:
        cases = [(o, m) for o in ORIGINS for m in MODES]
        for i, (origin, mode) in enumerate(cases):
            rows.append(isolated(i, origin, mode, 'before' if soak else 'isolated'))
        policy = load_diagnostic_policy(candidate['config'])
        shared_started = time.perf_counter()
        index = 0
        while True:
            origin, mode = cases[index % len(cases)]
            row = episode(folder / 'shared' / f'{index:06d}_{origin}_{mode}',
                          candidate, origin, mode, policy)
            rows.append(row)
            index += 1
            print(json.dumps({'phase': 'shared', 'episodes': index,
                              'elapsed_s': time.perf_counter()-shared_started}), flush=True)
            if (soak and time.perf_counter()-shared_started >= 14400) or (
                    not soak and index == 24):
                break
        report.update(shared_elapsed_s=time.perf_counter()-shared_started, shared_episodes=index)
        if soak:
            for i, (origin, mode) in enumerate(cases):
                rows.append(isolated(i, origin, mode, 'after'))
        verify_candidate(path)
        report.update(passed=True, episodes=rows, completed_episodes=len(rows))
        save(run_id, candidate, rows, report, 'success')
        dump(folder / 'artifact_verification.json', {'hashes': make_receipt(folder)})
        return 0
    except Exception as exc:
        report.update(passed=False, episodes=rows, completed_episodes=len(rows))
        save(run_id, candidate, rows, report, 'failed', str(exc))
        return 1


def gate(run_id, path):
    candidate = verify_candidate(path)
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    report = {'train_only': True, 'candidate_sha256': sha(path), 'passed': False,
              'formal_training_ready': False}
    save(run_id, candidate, [], report, 'running')
    with (folder / 'make_check.log').open('w') as log:
        result = subprocess.run(['make', 'check'], cwd=ROOT, stdout=log, stderr=log,
                                env={**os.environ, 'PYTEST_ADDOPTS':
                                     '--junitxml=' + str(folder / 'pytest.xml')})
    verify_candidate(path)
    report.update(passed=result.returncode == 0, exit_code=result.returncode)
    save(run_id, candidate, [], report, 'success' if report['passed'] else 'failed',
         None if report['passed'] else 'make check failed')
    return result.returncode


def _same_state(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(
            _same_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(
            _same_state(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def resume_audit(run_id, path):
    # Controlled boundary failures; real unchanged 4x48 collection/PPO between them.
    from types import SimpleNamespace
    from unittest.mock import patch

    from checkpointing.inventory_eval_input import SHORT_SCHEMA
    from checkpointing.versioned import read_checkpoint_payload
    from safe_rl_v2 import inventory_train as entry
    from safe_rl_v2.controlled_formal_train import _generators
    from safe_rl_v2.formal_train_loop import (
        build_lagrangian,
        build_optimizer,
        load_resume_checkpoint,
    )
    candidate = verify_candidate(path)
    config = candidate['config']
    apply_frozen_thread_setting(config)
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    rows = []
    report = {'candidate_sha256': sha(path), 'train_only': True, 'passed': False,
              'formal_training_ready': False, 'failures': 'controlled boundary stop',
              'workload': 'two real batches, each 4 episodes x 48 steps; frozen PPO'}
    save(run_id, candidate, rows, report, 'running')
    real_batch = entry.run_training_batch
    previous = None
    try:
        for boundary in (0, 1, 2):
            child_id = run_id + f'_boundary_{boundary}'
            def batch(*args, audit_boundary=boundary, **kwargs):
                if kwargs['batch_index'] == audit_boundary:
                    raise RuntimeError('controlled recovery audit boundary stop')
                return real_batch(*args, **kwargs)
            args = SimpleNamespace(short=True, seed=0, seed0_formal=False, run_id=child_id,
                                   resume_from=None if previous is None else str(previous),
                                   runtime_candidate=str(path))
            with patch.object(entry, 'run_training_batch', batch):
                if entry.run(args) != 1:
                    raise ValueError('controlled failure did not stop')
            child = ROOT / 'runs' / child_id
            failed = json.loads((child / 'failed_batch.json').read_text())
            if failed['reason'] != 'controlled recovery audit boundary stop':
                raise ValueError('unexpected real failure during resume audit')
            checkpoint = child / 'checkpoint_before_batch.pt'
            state = read_checkpoint_payload(checkpoint)['state']
            policy = build_seeded_policy(config, obs_dim=523, seed=0)
            optimizer = build_optimizer(config, policy)
            lag = build_lagrangian(config)
            sampling, shuffle = _generators(0)
            restored = load_resume_checkpoint(
                checkpoint, policy=policy, optimizer=optimizer, lagrangian=lag,
                sampling_generator=sampling, shuffle_generator=shuffle, config=config,
                expected_obs_dim=523, expected_schema=SHORT_SCHEMA,
                expected_scope='controlled_short_run', expected_role='controlled_training_resume')
            checks = {'cursor': restored['next_batch_index'] == boundary,
                      'policy': _same_state(policy.state_dict(), state['policy']),
                      'adam': _same_state(optimizer.state_dict(), state['optimizer']),
                      'multipliers': _same_state(lag.state_dict(), state['lagrangian']),
                      'sampling_rng': torch.equal(
                          sampling.get_state(), state['sampling_generator']),
                      'shuffle_rng': torch.equal(shuffle.get_state(), state['shuffle_generator']),
                      'dates': restored['origins'] == state['origins'],
                      'ledger': restored['origin_provenance'] == {
                          int(k): v for k, v in state['origin_provenance'].items()}}
            if not all(checks.values()):
                raise ValueError('restored batch boundary differs')
            rows.append({'boundary': boundary, 'run_id': child_id, 'checks': checks,
                         'checkpoint_sha256': sha(checkpoint),
                         'adam_steps': failed['adam_steps_at_failure'],
                         'multiplier_updates': failed['multiplier_updates_at_failure']})
            previous = checkpoint
        if [r['adam_steps'] for r in rows] != [0, 16, 32] or [
                r['multiplier_updates'] for r in rows] != [0, 1, 2]:
            raise ValueError('updates jumped across explicit recovery')
        verify_candidate(path)
        report.update(passed=True, boundaries=rows, explicit_new_run_ids=True)
        save(run_id, candidate, rows, report, 'success')
        return 0
    except Exception as exc:
        report['boundaries'] = rows
        save(run_id, candidate, rows, report, 'failed', str(exc))
        return 1


def short_run_qualified(folder, path):
    from safe_rl_v2.inventory_train import verify_written_run
    from scenario.runtime_release import candidate_binding
    report = verify_written_run(folder, candidate_binding(path))
    return (report['batches'] == 8 and report['transitions'] == 1536
            and report['adam_steps'] == 128 and report['lagrangian_updates'] == 8
            and len(report['inventory_episodes']) == 32
            and all(all(inventory_episode_acceptance(e).values())
                    and e['target_qualified'] is True for e in report['inventory_episodes'])
            and all(b['zero_action_fallback_steps'] == 0
                    for b in report['batch_records']))


def coverage_run_qualified(folder, path):
    from scenario.runtime_release import validate_learning_coverage
    validate_learning_coverage(folder, path)
    return True


def reused_short_qualified(folder, path, seed):
    from scenario.runtime_release import validate_short_evidence
    validate_short_evidence(folder, path, seed)
    return True


def numerics(run_id, path):
    """Fixed failure inputs plus registered action variants, without env or PPO."""
    import hashlib

    import scipy.optimize

    from contracts.inventory import InventorySnapshot
    from contracts.models import DispatchProposal
    from planning.corrector import correct
    from scenario.runtime_release import (
        NUMERIC_FIXTURES,
        NUMERIC_VARIANTS,
        validate_numeric_evidence,
    )
    candidate = verify_candidate(path)
    apply_frozen_thread_setting(candidate['config'])
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    rows = []
    report = {'passed': False, 'candidate_sha256': sha(path),
              'numeric_contract': candidate['config']['numeric_contract'],
              'environment_steps': 0, 'parameter_updates': 0, 'observations': rows,
              'formal_training_ready': False, 'train_only': True}
    save(run_id, candidate, rows, report, 'running')
    real_milp = scipy.optimize.milp
    real_linprog = scipy.optimize.linprog
    try:
        for name in NUMERIC_FIXTURES:
            source = ROOT / 'tests/fixtures' / name
            fixture = json.loads(source.read_text())
            snapshot = InventorySnapshot.model_validate(fixture['snapshot'])
            for variant in NUMERIC_VARIANTS:
                working_snapshot = snapshot
                if variant == 'recorded':
                    proposal = DispatchProposal.model_validate(fixture['proposal'])
                elif variant.startswith('soc_'):
                    delta = float(np.spacing(snapshot.soc_kwh)) if variant.endswith('ulp') else 1e-9
                    if '_minus_' in variant:
                        delta = -delta
                    working_snapshot = snapshot.model_copy(
                        update={'soc_kwh': snapshot.soc_kwh + delta})
                    proposal = DispatchProposal.model_validate(fixture['proposal'])
                elif variant in ('zero_charge', 'full_discharge'):
                    full = variant == 'full_discharge'
                    proposal = DispatchProposal(
                        compute_actions=[float(full)] * len(snapshot.group_work_capacity),
                        storage_action=1. if full else -1.)
                else:
                    generator = np.random.default_rng(int(variant[-1]))
                    proposal = DispatchProposal(
                        compute_actions=generator.random(len(snapshot.group_work_capacity)).tolist(),
                        storage_action=float(generator.uniform(-1., 1.)))
                calls = []
                polish_calls = []
                def observe(_calls=calls, **kwargs):
                    con = kwargs['constraints'][0]
                    matrix = con.A.tocsr()
                    arrays = {'data': matrix.data, 'indices': matrix.indices,
                              'indptr': matrix.indptr, 'lb': kwargs['bounds'].lb,
                              'ub': kwargs['bounds'].ub, 'clb': con.lb, 'cub': con.ub,
                              'objective': kwargs['c'], 'integrality': kwargs['integrality']}
                    _calls.append({'options': dict(kwargs['options']),
                                  'matrix_shape': list(matrix.shape),
                                  'mathematical_input_hashes': {
                                      key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                                      for key, value in arrays.items()}})
                    return real_milp(**kwargs)
                def observe_polish(_calls=polish_calls, **kwargs):
                    arrays = {'objective': kwargs['c'], 'bounds': kwargs['bounds'],
                              'equality_rhs': kwargs['b_eq'],
                              'inequality_rhs': kwargs['b_ub']}
                    for label in ('A_eq', 'A_ub'):
                        sparse = kwargs[label].tocsr()
                        arrays.update({label+'_'+key: getattr(sparse, key)
                                       for key in ('data', 'indices', 'indptr')})
                    _calls.append({'method': kwargs['method'],
                                   'options': dict(kwargs['options']),
                                   'mathematical_input_hashes': {
                                       key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                                       for key, value in arrays.items()}})
                    return real_linprog(**kwargs)
                scipy.optimize.milp = observe
                scipy.optimize.linprog = observe_polish
                started = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        result = correct(working_snapshot, proposal,
                            time_limit_s=candidate['config']['runtime_budget_s'])
                finally:
                    scipy.optimize.milp = real_milp
                    scipy.optimize.linprog = real_linprog
                row = {'fixture': name, 'fixture_sha256': sha(source), 'variant': variant,
                       'source_failure_sha256': fixture['source_failure_sha256'],
                       'snapshot_soc_kwh': working_snapshot.soc_kwh,
                       'snapshot_perturbation': variant if variant.startswith('soc_') else None,
                       'proposal': proposal.model_dump(), 'wall_s': time.perf_counter() - started,
                       'passed': result.executable and result.candidate_check.get('passed', False),
                       'stage_a_status': result.stage_a_status,
                       'stage_b_status': result.stage_b_status,
                       'execution_source': result.execution_source,
                       'candidate_check': result.candidate_check,
                       'inventory_audit': result.inventory_audit, 'solver_calls': calls,
                       'primal_polish_calls': polish_calls}
                rows.append(row)
                dump(folder / 'observations.json', rows)
                save(run_id, candidate, rows, report, 'running')
                if not row['passed']:
                    raise ValueError(f'numeric qualification stopped at {name}/{variant}')
        report['passed'] = True
        validate_numeric_evidence(report, path)
        save(run_id, candidate, rows, report, 'success')
        return 0
    except Exception as exc:
        report['passed'] = False
        save(run_id, candidate, rows, report, 'failed', str(exc))
        return 1
    finally:
        scipy.optimize.milp = real_milp
        scipy.optimize.linprog = real_linprog


def qualify(run_id, paths, *, budget=None, reuse_short_prefix=None):
    if reuse_short_prefix is not None:
        import re
        if budget is None or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}',
                                             reuse_short_prefix):
            raise ValueError('reuse short evidence requires a safe prefix and fixed budget')
    candidates = [verify_candidate(path) for path in paths]
    if [c['config']['runtime_budget_s'] for c in candidates] != list(BUDGETS):
        raise ValueError('qualification must preregister exactly .25/.50/1.00 in order')
    if budget is not None and (type(budget) not in (int, float) or budget not in BUDGETS):
        raise ValueError('fixed qualification requires a registered budget')
    selected_pairs = [(p, c) for p, c in zip(paths, candidates, strict=True)
                      if budget is None or c['config']['runtime_budget_s'] == budget]
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    rows = []
    report = {'passed': False, 'train_only': True, 'formal_training_ready': False,
              'attempts': rows, 'formal_512_started': False}
    if budget is not None:
        report.update(qualification_scope='fixed_approved_budget', registered_budgets=[budget])
    save(run_id, {'candidates': candidates}, [], report, 'running')
    selected = None
    try:
        report['preflight'] = []
        for action in ('gate', 'resume-audit'):
            child = run_id + '_' + action.replace('-', '_')
            status = subprocess.run([sys.executable, '-m', 'scripts.runtime_qualification',
                action, '--candidate', str(selected_pairs[0][0]), '--run-id', child],
                cwd=ROOT).returncode
            report['preflight'].append({'action': action, 'run_id': child, 'exit_code': status})
            save(run_id, {'candidates': candidates}, [], report, 'running')
            if status != 0:
                raise ValueError('qualification preflight failed: ' + action)
        for path, candidate in selected_pairs:
            budget = candidate['config']['runtime_budget_s']
            label = str(round(budget*100)).zfill(3)
            attempt = {'budget_s': budget, 'candidate_sha256': sha(path), 'passed': False,
                       'phases': []}
            rows.append(attempt)
            def phase(action, suffix, phase_label=label, phase_path=path, phase_attempt=attempt):
                child = run_id + '_' + phase_label + '_' + suffix
                status = subprocess.run([sys.executable, '-m', 'scripts.runtime_qualification',
                    action, '--candidate', str(phase_path), '--run-id', child], cwd=ROOT).returncode
                phase_attempt['phases'].append(
                    {'action': action, 'run_id': child, 'exit_code': status})
                save(run_id, {'candidates': candidates}, [], report, 'running')
                return status == 0
            numerical = 'numeric_contract' in candidate['config']
            if numerical and not phase('numerics', 'numerics'):
                continue
            if not phase('diagnose', 'diagnose'):
                continue
            shorts_passed = True
            for seed in (0, 1, 2):
                if reuse_short_prefix is not None:
                    child = reuse_short_prefix + f'_seed{seed}'
                    accepted = reused_short_qualified(ROOT/'runs'/child, path, seed)
                    attempt['phases'].append({'action': 'short', 'run_id': child, 'seed': seed,
                        'exit_code': 0 if accepted else 1, 'quality_passed': accepted,
                        'reused_same_candidate': True,
                        'original_report_sha256': sha(ROOT/'runs'/child/'report.json')})
                    save(run_id, {'candidates': candidates}, [], report, 'running')
                    if not accepted:
                        shorts_passed = False
                        break
                    continue
                child = run_id + '_' + label + f'_short_seed{seed}'
                status = subprocess.run([sys.executable, '-m', 'safe_rl_v2.inventory_train',
                    '--short', '--seed', str(seed), '--runtime-candidate', str(path),
                    '--run-id', child], cwd=ROOT).returncode
                accepted = status == 0 and short_run_qualified(ROOT/'runs'/child, path)
                attempt['phases'].append({'action': 'short', 'run_id': child, 'seed': seed,
                                          'exit_code': status, 'quality_passed': accepted})
                save(run_id, {'candidates': candidates}, [], report, 'running')
                if not accepted:
                    shorts_passed = False
                    break
            if not shorts_passed:
                continue
            coverage_passed = True
            if numerical:
                for seed in (0, 1, 2):
                    child = run_id + '_' + label + f'_coverage_seed{seed}'
                    status = subprocess.run([sys.executable, '-m', 'safe_rl_v2.inventory_train',
                        '--coverage', '--seed', str(seed), '--runtime-candidate', str(path),
                        '--run-id', child], cwd=ROOT).returncode
                    accepted = status == 0 and coverage_run_qualified(ROOT/'runs'/child, path)
                    attempt['phases'].append({'action': 'coverage', 'run_id': child, 'seed': seed,
                                             'exit_code': status, 'quality_passed': accepted})
                    save(run_id, {'candidates': candidates}, [], report, 'running')
                    if not accepted:
                        coverage_passed = False
                        break
            if not coverage_passed or not phase('soak', 'soak'):
                continue
            attempt['passed'] = True
            selected = budget
            break
        for path in paths:
            verify_candidate(path)
        report.update(passed=selected is not None, selected_budget_s=selected,
                      qualification_complete=selected is not None,
                      release_freeze_pending=selected is not None)
        save(run_id, {'candidates': candidates}, [], report,
             'success' if report['passed'] else 'failed',
             None if report['passed'] else 'all registered budgets failed qualification')
        return 0 if report['passed'] else 1
    except Exception as exc:
        save(run_id, {'candidates': candidates}, [], report, 'failed', str(exc))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=(
        'freeze', 'profile', 'gate', 'resume-audit', 'qualify', 'diagnose', 'soak',
        'numerics', 'episode'))
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidates', nargs=3, type=Path)
    parser.add_argument('--budget', type=float, choices=BUDGETS)
    parser.add_argument('--run-id')
    parser.add_argument('--reuse-short-prefix', help='reverify same-candidate fresh short evidence')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--origin', type=int, choices=ORIGINS)
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if args.action == 'freeze':
        if args.output.exists():
            raise FileExistsError('candidate exists; use a new version')
        dump(args.output, build_candidate(args.budget))
        verify_candidate(args.output)
        return 0
    if args.action == 'qualify':
        return qualify(args.run_id, args.candidates, budget=args.budget,
                       reuse_short_prefix=args.reuse_short_prefix)
    if args.action == 'gate':
        return gate(args.run_id, args.candidate)
    if args.action == 'resume-audit':
        return resume_audit(args.run_id, args.candidate)
    if args.action == 'profile':
        return profile(args.run_id, args.candidate)
    if args.action == 'numerics':
        return numerics(args.run_id, args.candidate)
    if args.action == 'episode':
        episode(args.output, verify_candidate(args.candidate), args.origin, args.mode)
        return 0
    return diagnose(args.run_id, args.candidate, soak=args.action == 'soak')


if __name__ == '__main__':
    raise SystemExit(main())
