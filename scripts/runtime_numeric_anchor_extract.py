"""Extract fixed train success anchors from historical executed actions, without solving."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from planning.snapshot_adapter import build_snapshot
from runs.writer import write_run
from safe_rl_v2.formal_train_loop import build_train_env
from scenario.inventory_release import ROOT, sha
from scripts.runtime_numeric_diagnostic import (
    digest,
    load_fixture,
    load_registry,
    scientific_binding,
    validate_fixture,
)

# Second completed origin: H=4,3,1; failure origin: H=48,30,5.
SELECTED = {1: (44, 45, 47), 2: (0, 18, 43)}


def validate_observation(observed, recorded):
    actual = np.asarray(observed, dtype=np.float32)
    expected = np.asarray(recorded, dtype=np.float32)
    if expected.shape != (len(actual) + 3,) or not np.array_equal(actual, expected[:-3]):
        raise ValueError('reconstructed train observation differs from historical trace')


def extract(run_id):
    registry = load_registry()
    entry = registry['fixtures'][registry['latest']]
    fixture = load_fixture(registry['latest'], entry)
    source = ROOT / entry['source_path']
    historical = json.loads(source.read_text())
    config_path = source.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    transitions = historical['partial_transitions']
    origins = list(map(int, historical['origin_provenance']))
    if len(origins) != 3 or len(transitions) != 140:
        raise ValueError('unregistered historical anchor trace length/origins')
    folder = ROOT / 'runs' / run_id
    folder.mkdir(exist_ok=False)
    output = folder / 'fixtures'
    output.mkdir()
    rows = []
    binding = scientific_binding()
    report = {'passed': False, 'state': 'running', 'parameter_updates': 0,
              'solver_calls': 0, 'source_sha256': sha(source),
              'source_config_sha256': sha(config_path), 'scientific_binding': binding,
              'selection': SELECTED, 'observations': rows, 'train_only': True,
              'formal_training_ready': False, 'executed_historical_actions': 0}
    def save():
        write_run(run_id, config={'historical_config': config, 'selection': SELECTED},
                  metrics=pd.DataFrame(rows), report=report, status='success' if report['passed']
                  else report['state'], command='python -m scripts.runtime_numeric_anchor_extract',
                  dependency_lock_hash=sha(ROOT / 'uv.lock'), data_hash=sha(
                      ROOT / 'data/processed/singapore_2024/half_hour.parquet'),
                  scenario_hash=sha(ROOT / 'data/manifest/formal_splits_v5/train.json'),
                  manifest_metadata={'scientific_binding': binding})
    started = time.perf_counter()
    save()
    pending = []
    try:
        cursor = 0
        final_snapshot = None
        for episode_index, origin in enumerate(origins):
            env, injection = build_train_env(origin, master_seed=0, config=config)
            if injection.provenance_hash != historical['origin_provenance'][str(origin)]:
                raise ValueError('train injection source mismatch')
            observed, _ = env.reset(seed=0)
            for step in range(48 if episode_index < 2 else 44):
                trace = transitions[cursor]
                validate_observation(observed, trace['observation'])
                if not trace['correction_info']['correction_executable']:
                    raise ValueError('anchor trace was not successfully executed')
                if step in SELECTED.get(episode_index, ()):
                    raw = np.asarray(trace['raw_action'], dtype=np.float32)
                    item = {'source_failure_sha256': sha(source), 'origin': origin,
                            'source_transition_index': cursor,
                            'snapshot': build_snapshot(env).model_dump(mode='json'),
                            'proposal': {'schema_version': fixture['proposal']['schema_version'],
                                         'compute_actions': raw[:20].astype(float).tolist(),
                                         'storage_action': float(raw[-1])}}
                    validate_fixture(item)
                    name = f'm6p2c_train_success_origin{origin}_step{step}.json'
                    pending.append((name, item))
                observed, _, terminated, truncated, _ = env.step(
                    np.asarray(trace['exec_action'], dtype=np.float32))
                validate_observation(observed, trace['next_observation'])
                if bool(terminated) != trace['terminated'] or bool(truncated) != trace['truncated']:
                    raise ValueError('historical termination mismatch')
                cursor += 1
            if episode_index == 2:
                final_snapshot = build_snapshot(env).model_dump(mode='json')
            env.close()
        if final_snapshot != fixture['snapshot']:
            raise ValueError('final historical failure snapshot reconstruction mismatch')
        for name, item in pending:
            path = output / name
            path.write_text(json.dumps(item, indent=2) + '\n')
            rows.append({'fixture': name, 'sha256': sha(path), 'input_sha256': digest(item),
                         'source_transition_index': item['source_transition_index'],
                         'horizon': item['snapshot']['planning_horizon_steps'],
                         'soc_kwh': item['snapshot']['soc_kwh']})
        report.update(passed=True, state='succeeded', executed_historical_actions=cursor,
                      reconstructed_failure_snapshot_equal=True,
                      elapsed_s=time.perf_counter()-started,
                      exit_code=0)
    except Exception as exc:
        report.update(state='failed', error=f'{type(exc).__name__}: {exc}',
                      elapsed_s=time.perf_counter()-started, exit_code=1)
    save()
    print(json.dumps({k: report.get(k) for k in ('state', 'error', 'elapsed_s', 'exit_code')}))
    return report['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    return extract(parser.parse_args().run_id)


if __name__ == '__main__':
    raise SystemExit(main())
