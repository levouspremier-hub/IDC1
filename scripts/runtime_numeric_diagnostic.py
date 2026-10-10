"""Train-only production replay; collect failures without authorizing any training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from contracts.inventory import InventorySnapshot, validate_inventory_snapshot
from contracts.models import DispatchProposal
from runs.writer import write_run
from scenario.inventory_release import ROOT, sha
from scenario.runtime_release import NUMERIC_VARIANTS, candidate_config, source_paths
from scripts.runtime_qualification import replay_numeric_case

REGISTRY = ROOT / 'tests/fixtures/runtime_numeric_registry.json'
BOUNDARY_VARIANTS = ('soc_min', 'soc_min_inward_ulp', 'soc_max', 'soc_max_inward_ulp',
                     'target_minus_ulp', 'target_plus_ulp', 'target_minus_nano',
                     'target_plus_nano', 'tiny_charge', 'tiny_discharge',
                     'joint_headroom_charge', 'joint_headroom_discharge')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def load_registry():
    registry = json.loads(REGISTRY.read_text())
    if registry['version'] != 'runtime-numeric-diagnostic-v1' or not registry['fixtures']:
        raise ValueError('invalid numeric registry')
    return registry


def validate_fixture(fixture):
    for key in ('snapshot', 'proposal', 'source_failure_sha256', 'origin'):
        if key not in fixture:
            raise ValueError('missing fixture field: ' + key)
    snapshot = fixture['snapshot']
    for key in InventorySnapshot.model_fields:
        if key not in snapshot:
            raise ValueError('missing snapshot field: ' + key)
    for key in ('compute_actions', 'storage_action', 'schema_version'):
        if key not in fixture['proposal']:
            raise ValueError('missing proposal field: ' + key)
    forecast = snapshot['forecast']
    for key in ('split', 'generated_at', 'forecast_provenance', 'forecast_cutoff'):
        if key not in forecast:
            raise ValueError('missing forecast field: ' + key)
    if forecast['split'] != 'train' or not forecast['forecast_provenance']:
        raise ValueError('only sourced train forecasts may be replayed')
    for provenance in forecast['forecast_provenance'].values():
        for key in ('generated_at', 'information_cutoff_exclusive', 'sources'):
            if key not in provenance or not provenance[key]:
                raise ValueError('missing forecast provenance: ' + key)
        for record in provenance['sources']:
            path = (ROOT / record['logical_path']).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != record['sha256']:
                raise ValueError('forecast source hash mismatch: ' + record['logical_path'])
    parsed = InventorySnapshot.model_validate(snapshot)
    validate_inventory_snapshot(parsed)
    DispatchProposal.model_validate(fixture['proposal'])


def load_fixture(name, entry, *, fixture_dir=None):
    path = (ROOT / 'tests/fixtures' if fixture_dir is None else fixture_dir) / name
    if sha(path) != entry['sha256']:
        raise ValueError('fixture hash mismatch: ' + name)
    fixture = json.loads(path.read_text())
    if fixture['source_failure_sha256'] != entry['source_failure_sha256']:
        raise ValueError('failure source hash mismatch: ' + name)
    source = (ROOT / entry['source_path']).resolve()
    if not source.is_relative_to(ROOT) or sha(source) != entry['source_failure_sha256']:
        raise ValueError('failure file hash mismatch: ' + name)
    historical = json.loads(source.read_text())['failure_step']
    if (historical['snapshot'] != fixture['snapshot']
            or historical['proposal'] != fixture['proposal']):
        raise ValueError('fixture differs from bound historical failure: ' + name)
    if len(historical['raw_action']) != 21 or not np.all(np.isfinite(historical['raw_action'])):
        raise ValueError('historical raw action is incomplete: ' + name)
    fixture['historical_raw_action'] = historical['raw_action']
    fixture['historical_old_raw_log_prob'] = historical['old_raw_log_prob']
    validate_fixture(fixture)
    return fixture


def cases_for(suite, registry):
    names = [registry['latest']] + [n for n in registry['fixtures'] if n != registry['latest']]
    cases = []
    if suite in ('latest', 'history', 'all'):
        for name in names[:1] if suite == 'latest' else names:
            for variant in ('recorded',) if suite == 'latest' else NUMERIC_VARIANTS:
                cases.append({'case_id': name[:-5] + '/' + variant, 'fixture': name,
                              'variant': variant, 'expectation': 'must_pass', 'suite': 'history'})
    if suite in ('boundary', 'all'):
        # All original horizons are retained: H=1,3,4,12,25,30,42,46,47.
        for name in names:
            for variant in BOUNDARY_VARIANTS:
                cases.append({'case_id': name[:-5] + '/boundary_' + variant,
                              'fixture': name, 'variant': variant,
                              'expectation': 'exploratory', 'suite': 'boundary'})
    return cases


def case_input(fixture, variant):
    data = copy.deepcopy(fixture)
    s, p = data['snapshot'], data['proposal']
    if variant.startswith('soc_') and variant in NUMERIC_VARIANTS:
        delta = float(np.spacing(s['soc_kwh'])) if variant.endswith('ulp') else 1e-9
        s['soc_kwh'] += -delta if '_minus_' in variant else delta
    elif variant.startswith('soc_min'):
        s['soc_kwh'] = s['soc_min_kwh']
        if variant.endswith('ulp'):
            s['soc_kwh'] = float(np.nextafter(s['soc_kwh'], np.inf))
    elif variant.startswith('soc_max'):
        s['soc_kwh'] = s['soc_max_kwh']
        if variant.endswith('ulp'):
            s['soc_kwh'] = float(np.nextafter(s['soc_kwh'], -np.inf))
    elif variant.startswith('target_'):
        target = s['terminal_inventory']['target_kwh']
        delta = float(np.spacing(target)) if variant.endswith('ulp') else 1e-9
        s['soc_kwh'] = target + (-delta if '_minus_' in variant else delta)
    elif variant in ('zero_charge', 'full_discharge'):
        full = variant == 'full_discharge'
        p['compute_actions'] = [float(full)] * len(s['group_work_capacity'])
        p['storage_action'] = 1. if full else -1.
    elif variant.startswith('sample'):
        generator = np.random.default_rng(int(variant[-1]))
        p['compute_actions'] = generator.random(len(s['group_work_capacity'])).tolist()
        p['storage_action'] = float(generator.uniform(-1., 1.))
    elif variant.startswith(('tiny_', 'joint_')):
        p['storage_action'] = -1e-12 if variant.endswith('charge') and not variant.endswith(
            'discharge') else 1e-12
        if variant.startswith('joint_'):
            # Conservative network perturbation preserves service schedules/demands;
            # infeasibility is possible, so these remain exploratory.
            guard = s['service_guard']
            powers = [base + sum(c * sum(row[g] for row in allocation)
                                for g, c in enumerate(coefficients))
                      + max(coefficients) * served
                      for base, coefficients, allocation, served in zip(
                          guard['reserve_base_power_kw'],
                          guard['reserve_power_coefficients_kw_per_work'],
                          guard['known_service_allocation'], guard['aggregate_service_work'],
                          strict=True)]
            s['access_limit_kw'] = max(powers) + 1e-10
            guard['charge_limits_kw'] = [min(s['bess_charge_power_max_kw'],
                                            max(s['access_limit_kw'] - power, 0.))
                                        for power in powers]
            guard['charge_limit_kw'] = guard['charge_limits_kw'][0]
            s['soc_kwh'] = s['terminal_inventory']['target_kwh'] + (
                -1e-10 if p['storage_action'] < 0 else 1e-10)
    elif variant != 'recorded':
        raise ValueError('unregistered numeric perturbation: ' + variant)
    return data


def classify(row):
    audit = row['inventory_audit']
    if audit.get('primary_objective_certificate', {}).get('passed') is False:
        return 'a_objective_bound_certificate_failure'
    if audit.get('polished_economic_objective_certificate', {}).get('passed') is False:
        return 'b_objective_certificate_failure'
    if any(audit.get(k, {}).get('passed') is False for k in (
            'stage_a_witness', 'execution_witness', 'reachability_witness')):
        return 'original_matrix_certificate_failure'
    if row.get('candidate_check', {}).get('passed') is False:
        return 'original_matrix_certificate_failure'
    if audit.get('timeout_phase') or 'time_limit' in (
            row['stage_a_status'], row['stage_b_status']):
        return 'timeout'
    if row['passed'] and all(audit.get(k, {}).get('passed') is True for k in (
            'stage_a_witness', 'execution_witness', 'primary_objective_certificate')):
        return 'passed'
    if row.get('diagnostic_original_b_anchor_certificate', {}).get('passed') is True:
        return 'solver_missed_certified_feasible_witness'
    # No inference of genuine infeasibility or solver misclassification from status alone.
    return 'unattributed'


def evaluate_case(case, registry, config):
    name = case['fixture']
    fixture = load_fixture(name, registry['fixtures'][name])
    data = case_input(fixture, case['variant'])
    validate_fixture(data)
    row = replay_numeric_case(InventorySnapshot.model_validate(data['snapshot']),
                              DispatchProposal.model_validate(data['proposal']),
                              budget=config['runtime_budget_s'], name=name,
                              variant=case['variant'], fixture=fixture,
                              source=ROOT / 'tests/fixtures' / name, diagnostic=True)
    from scipy.optimize._highspy import _core
    highs_version = '.'.join(str(getattr(_core, 'HIGHS_VERSION_' + part))
                             for part in ('MAJOR', 'MINOR', 'PATCH'))
    row.update(case, input_sha256=digest(data), reconstruction_input=data,
               seed=int(case['variant'][-1]) if case['variant'].startswith('sample') else 0,
               solver_version={'scipy': scipy.__version__, 'backend': 'scipy-highs',
                               'highs': highs_version})
    row['classification'] = classify(row)
    row['passed'] = row['classification'] == 'passed'
    row['failure_stage'] = None if row['passed'] else (
        'A' if row['stage_b_status'] == 'not_run' else 'B_or_execution_certificate')
    solve_s = sum(c['elapsed_s'] for c in row['solver_calls'] + row['primal_polish_calls'])
    audit = row['inventory_audit']
    row['timings'] = {'solver_s': solve_s,
                      'model_build_s': audit.get('model_build_time_s'),
                      'b_transform_s': audit.get('stage_b_transform_time_s'),
                      'a_certificate_s': audit.get('stage_a_certificate_time_s'),
                      'b_certificate_s': audit.get('stage_b_certificate_time_s'),
                      'non_solver_s': max(row['wall_s'] - solve_s, 0.)}
    return row


def scientific_binding():
    paths = sorted(set(source_paths()) | {'scripts/runtime_numeric_diagnostic.py',
                                         'scripts/runtime_numeric_anchor_extract.py'})
    hashes = {p: sha(ROOT / p) for p in paths}
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=ROOT)
    return {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                           text=True).strip(),
            'file_hashes': hashes, 'source_sha256': digest(hashes),
            'diff_sha256': hashlib.sha256(diff).hexdigest(), 'dirty': bool(diff),
            'dependency_lock_hash': sha(ROOT / 'uv.lock')}


def run(suite, run_id, *, base_dir=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}', run_id):
        raise ValueError('invalid run id')
    root = ROOT / 'runs' if base_dir is None else Path(base_dir)
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    config = candidate_config(.50)
    binding = scientific_binding()
    registry = load_registry()
    rows = []
    report = {'state': 'running', 'passed': False, 'observations': rows, 'suite': suite,
              'scientific_binding': binding, 'registry_sha256': sha(REGISTRY),
              'parameter_updates': 0, 'environment_steps': 0, 'train_only': True,
              'formal_training_ready': False, 'diagnostic_only': True,
              'classification_note': 'infeasible status is unattributed without a proof',
              'exit_code': None}
    def save():
        write_run(run_id, base_dir=str(root), config={'config': config, 'registry': registry},
                  metrics=pd.DataFrame([{k: v for k, v in row.items()
                                         if not isinstance(v, (dict, list))} for row in rows]),
                  report=report, status='success' if report['state'] == 'succeeded' else
                  report['state'], command=' '.join(sys.argv),
                  dependency_lock_hash=binding['dependency_lock_hash'],
                  data_hash=sha(ROOT / 'data/processed/singapore_2024/half_hour.parquet'),
                  scenario_hash=sha(ROOT / 'data/manifest/formal_splits_v5/train.json'),
                  failure_classification=None if report['state'] == 'succeeded' else
                  report['state'], manifest_metadata={'scientific_binding': binding,
                                                       'diagnostic_only': True})
    started = time.perf_counter()
    save()
    try:
        for case in cases_for(suite, registry):
            case_started = time.perf_counter()
            try:
                row = evaluate_case(case, registry, config)
            except Exception as exc:
                category = 'input_source_contract_failure' if isinstance(
                    exc, (ValueError, KeyError, FileNotFoundError)) else 'unattributed'
                row = {**case, 'passed': False, 'classification': category,
                       'failure_stage': 'input_or_runtime_exception',
                       'error': f'{type(exc).__name__}: {exc}'}
            row.update(total_wall_s=time.perf_counter()-case_started,
                       scientific_source_sha256=binding['source_sha256'],
                       dependency_lock_hash=binding['dependency_lock_hash'])
            case_folder = folder / 'cases'
            case_folder.mkdir(exist_ok=True)
            case_path = case_folder / (str(len(rows)).zfill(4) + '.json')
            case_path.write_text(json.dumps(row, indent=2))
            summary = copy.deepcopy(row)
            summary.pop('reconstruction_input', None)
            for call in summary.get('solver_calls', []) + summary.get('primal_polish_calls', []):
                call.pop('solution', None)
            summary.get('inventory_audit', {}).pop('service_guard', None)
            summary.update(case_record=str(case_path.relative_to(folder)),
                           case_record_sha256=sha(case_path))
            rows.append(summary)
            save()
        final_binding = scientific_binding()
        if any(final_binding[k] != binding[k] for k in (
                'source_sha256', 'diff_sha256', 'dependency_lock_hash')):
            raise ValueError('scientific source changed during diagnostic')
        report.update(state='succeeded' if all(r['passed'] for r in rows) else 'failed',
                      passed=bool(rows) and all(r['passed'] for r in rows),
                      elapsed_s=time.perf_counter()-started, case_count=len(rows),
                      passed_count=sum(r['passed'] for r in rows))
        report['exit_code'] = 0 if report['passed'] else 1
    except BaseException as exc:
        report.update(state='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit))
                      else 'failed', error=f'{type(exc).__name__}: {exc}', exit_code=1,
                      elapsed_s=time.perf_counter()-started)
    save()
    print(json.dumps({k: report.get(k) for k in (
        'state', 'exit_code', 'case_count', 'passed_count', 'elapsed_s')}), flush=True)
    return report['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=('latest', 'history', 'boundary', 'all'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    return run(args.suite, args.run_id)


if __name__ == '__main__':
    raise SystemExit(main())
