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
        scenario_hash=sha(ROOT / 'configs/training/idc_training_config_v2_r5.json'))


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
                return deterministic_action(policy, obs)
            raw, _, _ = policy.act(tensor, generator)
            return raw.cpu().numpy().astype(np.float32)

    started = time.perf_counter()
    try:
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter('ignore', RuntimeWarning)
            result, rows, _, _ = evaluate_origin(config, origin, 0, action, run_id=folder.name)
        with (folder / 'steps.jsonl').open('w') as stream:
            from scripts.m6p2b_seed0_diagnosis import jsonable
            for row in rows:
                stream.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'profile', 'diagnose', 'soak', 'episode'))
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--budget', type=float, choices=BUDGETS)
    parser.add_argument('--run-id')
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
    if args.action == 'profile':
        return profile(args.run_id, args.candidate)
    if args.action == 'episode':
        episode(args.output, verify_candidate(args.candidate), args.origin, args.mode)
        return 0
    return diagnose(args.run_id, args.candidate, soak=args.action == 'soak')


if __name__ == '__main__':
    raise SystemExit(main())
