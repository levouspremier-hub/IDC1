"""Run one fixed v26 numeric/three-seed short gate; never start formal here."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from runs.writer import write_run
from scenario.inventory_release import sha
from scenario.runtime_release import validate_short_evidence, verify_candidate

ROOT = Path.cwd()
CANDIDATE = ROOT / 'configs/release/idc_runtime_candidate_v26_050.json'
PREFIX = 'runtime_v26_fixed_short'
PARENT = PREFIX + '_gate'


def main():
    candidate = verify_candidate(CANDIDATE)
    states = []
    def persist(status, error=None):
        write_run(PARENT, config={'seeds': [0,1,2], 'batches_per_seed': 8,
            'candidate_sha256': sha(CANDIDATE), 'fresh': True},
            metrics=pd.DataFrame([{'seed': s['seed'], 'passed': s.get('passed', False)}
                                  for s in states]),
            report={'passed': status == 'success', 'error': error, 'seeds': states,
                    'candidate_sha256': sha(CANDIDATE), 'formal_started': False,
                    'parameter_updates_performed_by_launcher': 0}, status=status,
            failure_classification=error, dependency_lock_hash=sha(ROOT/'uv.lock'),
            data_hash=sha(CANDIDATE), scenario_hash=sha(CANDIDATE),
            command='one numeric gate and one fresh 8-batch short per seed; no resume')
    persist('running')
    try:
        subprocess.run([sys.executable, '-m', 'scripts.runtime_qualification', 'numerics',
            '--candidate', str(CANDIDATE), '--run-id', PREFIX+'_numerics'], check=True)
        for seed in (0,1,2):
            rid = PREFIX + f'_seed{seed}'
            states.append({'seed': seed, 'run_id': rid, 'started_at': time.time()})
            persist('running')
            print(json.dumps({'phase': 'fixed_short_started', 'seed': seed, 'run_id': rid}),
                  flush=True)
            subprocess.run([sys.executable, '-m', 'safe_rl_v2.inventory_train', '--short',
                '--seed', str(seed), '--runtime-candidate', str(CANDIDATE),
                '--run-id', rid], check=True)
            result = validate_short_evidence(ROOT/'runs'/rid, CANDIDATE, seed)
            states[-1].update(passed=True, batches=result['batches'],
                transitions=result['transitions'], adam_steps=result['adam_steps'],
                multiplier_updates=result['lagrangian_updates'], finished_at=time.time(),
                report_sha256=sha(ROOT/'runs'/rid/'report.json'),
                checkpoint_sha256=sha(ROOT/'runs'/rid/'checkpoint_final.pt'))
            persist('running')
            print(json.dumps({'phase': 'fixed_short_passed', **states[-1]}), flush=True)
        persist('success')
    except BaseException as exc:
        persist('failed', f'{type(exc).__name__}: {exc}')
        raise


main()
