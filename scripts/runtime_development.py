"""Measured local preflight before freezing; never launches remote or formal training."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from runs.writer import write_run
from scenario.inventory_release import ROOT, sha
from scenario.runtime_release import BUDGETS, build_candidate, candidate_config, verify_candidate
from scripts.runtime_numeric_diagnostic import load_registry, scientific_binding

PHASES = ('static', 'latest', 'history', 'boundary', 'gate')


def execute(label, command, folder):
    with (folder / (label + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def freeze_after_preflight(report_path, version):
    # Six success anchors are preregistered on Z; unavailable inputs cannot be
    # quietly replaced with a smaller passing cohort.
    if len(load_registry().get('successful_anchors', {})) != 6:
        raise ValueError('registered successful boundary anchors are incomplete')
    report = json.loads(Path(report_path).read_text())
    if report.get('passed') is not True or report.get('exit_code') != 0:
        raise ValueError('local preflight has not passed')
    current = scientific_binding()
    for key in ('source_sha256', 'validation_sha256', 'dependency_lock_hash'):
        if report['scientific_binding'].get(key) != current[key]:
            raise ValueError('scientific source or validation inputs changed after local preflight')
    if not re.fullmatch(r'v[1-9][0-9]*', version):
        raise ValueError('invalid new candidate version')
    candidates = [build_candidate(budget) for budget in BUDGETS]
    paths = [ROOT / 'configs/release' /
             f'idc_runtime_candidate_{version}_{round(budget*100):03d}.json' for budget in BUDGETS]
    if any(path.exists() for path in paths):
        raise FileExistsError('candidate exists; preserve old evidence and use a new version')
    for path, candidate in zip(paths, candidates, strict=True):
        path.write_text(json.dumps(candidate, indent=2) + '\n')
        verify_candidate(path)
    return [str(path.relative_to(ROOT)) for path in paths]


def preflight(run_id, *, base_dir=None, freeze_version=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}', run_id):
        raise ValueError('invalid preflight run id')
    root = ROOT / 'runs' if base_dir is None else Path(base_dir)
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    binding = scientific_binding()
    config = candidate_config(.50)
    rows = []
    report = {'state': 'running', 'passed': False, 'exit_code': None, 'phases': rows,
              'scientific_binding': binding, 'train_only': True, 'formal_training_ready': False,
              'training_state': 'not_started', 'transfer_state': 'not_started',
              'training_elapsed_s': None, 'transfer_elapsed_s': None,
              'checkpoint_verification_elapsed_s': None, 'formal_512_started': False,
              'remote_required_order': ['unique_live_job_and_execution_lock',
                                        'host_fast_diagnostic',
                                        'fresh_seed0_8', 'fresh_seed1_8', 'fresh_seed2_8',
                                        'same_source_qualify_and_full_sha_checkpoint_verification']}
    def save():
        write_run(run_id, base_dir=str(root), config={'phases': PHASES, 'candidate':
                  config}, metrics=pd.DataFrame(rows), report=report,
                  status='success' if report['state'] == 'succeeded' else report['state'],
                  failure_classification=report.get('error'), command=' '.join(sys.argv),
                  dependency_lock_hash=sha(ROOT / 'uv.lock'), data_hash=sha(
                      ROOT / 'data/processed/singapore_2024/half_hour.parquet'),
                  scenario_hash=sha(ROOT / 'data/manifest/formal_splits_v5/train.json'),
                  manifest_metadata={'scientific_binding': binding})
    started = time.perf_counter()
    save()
    try:
        for label in PHASES:
            child = run_id + '_' + label
            if label in ('static', 'gate'):
                command = ['make', 'static' if label == 'static' else 'check']
            else:
                command = [sys.executable, '-m', 'scripts.runtime_numeric_diagnostic',
                           '--suite', label, '--run-id', child]
            row = {'phase': label, 'state': 'running', 'exit_code': None,
                   'command': ' '.join(command), 'child_run': child}
            rows.append(row)
            save()
            phase_started = time.perf_counter()
            code = execute(label, command, folder)
            row.update(exit_code=code, elapsed_s=time.perf_counter()-phase_started,
                       state='succeeded' if code == 0 else 'failed')
            save()
            if code != 0:
                raise ValueError(f'preflight stopped at {label}: exit {code}')
        current = scientific_binding()
        if any(current[k] != binding[k] for k in (
                'source_sha256', 'validation_sha256', 'dependency_lock_hash')):
            raise ValueError('scientific source or validation inputs changed during preflight')
        if len(load_registry().get('successful_anchors', {})) != 6:
            raise ValueError('boundary expansion pending: six source-verified success anchors')
        report.update(passed=True, state='succeeded', exit_code=0)
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        if rows and rows[-1]['state'] == 'running':
            rows[-1].update(state='interrupted' if interrupted else 'failed',
                            exit_code=130 if interrupted else 1,
                            elapsed_s=time.perf_counter()-phase_started)
        report.update(state='interrupted' if interrupted else 'failed', passed=False,
                      exit_code=130 if interrupted else 1, error=f'{type(exc).__name__}: {exc}')
    report['elapsed_s'] = time.perf_counter()-started
    if freeze_version is not None and report['passed']:
        # Keep the manifest running until the requested freeze has a real result.
        (folder / 'report.json').write_text(json.dumps(report, indent=2))
        try:
            report['frozen_candidates'] = freeze_after_preflight(
                folder / 'report.json', freeze_version)
        except Exception as exc:
            report.update(state='failed', passed=False, exit_code=1,
                          error=f'freeze: {type(exc).__name__}: {exc}')
    save()
    print(json.dumps({k: report.get(k) for k in ('state', 'exit_code', 'elapsed_s', 'error')}))
    return report['exit_code']


def main():
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--freeze-version', help='new vN; all preflight evidence must pass first')
    args = parser.parse_args()
    return preflight(args.run_id, freeze_version=args.freeze_version)


if __name__ == '__main__':
    raise SystemExit(main())
