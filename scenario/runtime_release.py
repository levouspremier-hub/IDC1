"""Separate signed qualification candidates; historical releases stay immutable."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scenario.inventory_release import ROOT, _git, load_config, load_matrix, sha

VERSION = 'verified-a-stop-batch-v1'
BUDGETS = (.25, .50, 1.00)


def require_candidate_scope(*, short):
    if not short:
        raise ValueError('candidate cannot authorize formal training')


def candidate_config(budget):
    if budget not in BUDGETS:
        raise ValueError('unregistered runtime budget')
    config = copy.deepcopy(load_config())
    config.update(status='qualification_candidate', runtime_execution_version=VERSION,
                  runtime_budget_s=budget)
    config['training']['corrector']['time_limit_s'] = budget
    config['inherited_calibration_scope'] = {
        'source': 'r5', 'scales_and_research_parameters_unchanged': True,
        'measured_solver_budget_s': .25, 'candidate_budget_calibrated': False,
        'qualification_required': ['48_episodes', 'three_seed_8_batches', 'shared_4h']}
    return config


def source_paths():
    folders = ('scenario', 'contracts', 'checkpointing', 'planning', 'safe_rl_v2',
               'evaluation', 'viz', 'scripts', 'runs', 'envs', 'idc_model')
    return sorted({str(p.relative_to(ROOT)) for folder in folders
                   for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts}
                  | {'safe_rl/corrector_wrapper.py', 'pyproject.toml', 'uv.lock', 'Makefile'})


def asset_binding():
    # Bind the complete existing data set and recursively referenced evidence.
    from scenario.inventory_release import ASSET_PATHS, RELEASE_PATH
    paths = {ROOT / p for p in ASSET_PATHS.values()}
    paths.add(ROOT / RELEASE_PATH)
    paths.update(p for p in (ROOT / 'data').rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.name != '.DS_Store')
    config = load_config()
    paths.update(ROOT / config['calibration'][f'run_{kind}_path']
                 for kind in ('manifest', 'report'))
    paths.update(ROOT / config[k]['evidence_path'] for k in
                 ('reward_repair', 'service_temperature_reserve'))
    paths.update(ROOT / e['path'] for e in config['reward_shaping']['train_evidence'])
    visited = {}

    def references(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and (key == 'path' or key.endswith('_path')):
                    path = ROOT / child
                    if path.is_file() and path.resolve().is_relative_to(ROOT):
                        paths.add(path)
                else:
                    references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)

    while paths:
        path = paths.pop()
        logical = str(path.relative_to(ROOT))
        if logical in visited:
            continue
        visited[logical] = sha(path)
        if path.suffix == '.json':
            references(json.loads(path.read_text()))
    return dict(sorted(visited.items()))


def build_candidate(budget):
    config = candidate_config(budget)
    load_matrix()
    sources = source_paths()
    if _git('status', '--porcelain', '--', *sources):
        raise ValueError('candidate execution closure must be committed')
    return {'schema': 'idc-runtime-candidate-v1', 'execution_version': VERSION,
            'formal_training_ready': False, 'train_only': True,
            'config': config, 'sources': {p: sha(ROOT / p) for p in sources},
            'source_revision': _git('log', '-1', '--format=%H', '--', *sources),
            'assets': asset_binding(), 'lock_sha256': sha(ROOT / 'uv.lock')}


def verify_candidate(path):
    path = Path(path)
    recorded = json.loads(path.read_text())
    expected = build_candidate(recorded['config']['runtime_budget_s'])
    if recorded != expected:
        raise ValueError('runtime candidate code/config/asset binding mismatch')
    return recorded


def candidate_binding(path):
    candidate = verify_candidate(path)
    return {'runtime_candidate_sha256': sha(path), 'execution_version': VERSION,
            'sources': candidate['sources'], 'assets': candidate['assets'],
            'config': candidate['config'], 'formal_training_ready': False}
