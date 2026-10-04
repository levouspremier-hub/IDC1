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
               'evaluation', 'viz', 'scripts', 'envs', 'idc_model')
    tracked = _git('ls-files', '--', *folders).splitlines()
    return sorted({p for p in tracked if p.endswith('.py')}
                  | {'runs/writer.py', 'safe_rl/corrector_wrapper.py',
                     'pyproject.toml', 'uv.lock', 'Makefile'})


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


def qualification_evidence(folder, candidate_path):
    """Reopen all mandatory phases, not a caller-supplied ready Boolean."""
    from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
    from safe_rl_v2.inventory_train import verify_written_run
    candidate = verify_candidate(candidate_path)
    folder = Path(folder)
    report = json.loads((folder / 'report.json').read_text())
    if (report.get('passed') is not True or report.get('qualification_complete') is not True
            or report.get('formal_512_started') is not False
            or report.get('selected_budget_s') != candidate['config']['runtime_budget_s']):
        raise ValueError('formal release requires complete matching host qualification')
    evidence = {str((folder / name).relative_to(ROOT)): sha(folder / name)
                for name in ('report.json', 'manifest.json')}
    phases = report['preflight']
    if [(p['action'], p['exit_code']) for p in phases] != [('gate', 0), ('resume-audit', 0)]:
        raise ValueError('formal release requires host gate and real recovery audit')
    attempts = report['attempts']
    if [a['budget_s'] for a in attempts] != list(BUDGETS[:len(attempts)]):
        raise ValueError('budget selection differs from preregistered order')
    passed = [a for a in attempts if a.get('passed') is True]
    if (len(passed) != 1 or passed[0] is not attempts[-1]
            or passed[0]['candidate_sha256'] != sha(candidate_path)):
        raise ValueError('qualification must select first fully passing budget')
    phases = phases + passed[0]['phases']
    expected = ['gate', 'resume-audit', 'diagnose', 'short', 'short', 'short', 'soak']
    if [p['action'] for p in phases] != expected:
        raise ValueError('qualification phase sequence incomplete')
    for phase in phases:
        child = folder.parent / phase['run_id']
        manifest = json.loads((child / 'manifest.json').read_text())
        result = json.loads((child / 'report.json').read_text())
        if phase['exit_code'] != 0 or manifest['status'] != 'success':
            raise ValueError('qualification phase is not successful')
        if phase['action'] == 'short':
            verify_written_run(child, candidate_binding(candidate_path))
            if (result['seed'] != phase['seed'] or result['batches'] != 8
                    or result['transitions'] != 1536 or result['adam_steps'] != 128
                    or result['lagrangian_updates'] != 8
                    or len(result['inventory_episodes']) != 32):
                raise ValueError('short training workload differs')
            episodes = result['inventory_episodes']
        elif phase['action'] in ('diagnose', 'soak'):
            if result.get('candidate_sha256') != sha(candidate_path):
                raise ValueError('diagnostic candidate binding mismatch')
            if result['passed'] is not True or result['parameter_updates'] != 0:
                raise ValueError('fixed-policy diagnosis did not pass')
            episodes = result['episodes']
            if (phase['action'] == 'diagnose' and len(episodes) != 48) or (
                    phase['action'] == 'soak' and result['shared_elapsed_s'] < 14400):
                raise ValueError('diagnostic or shared soak duration incomplete')
            if any(e['parameter_hash_before'] != e['parameter_hash_after'] for e in episodes):
                raise ValueError('diagnostic policy was updated')
        else:
            if result['passed'] is not True:
                raise ValueError('gate/recovery evidence did not pass')
            episodes = []
        if any(not all(inventory_episode_acceptance(e).values())
               or e['target_qualified'] is not True for e in episodes):
            raise ValueError('phase failed frozen service/physics/uniform target standards')
        for path in child.rglob('*'):
            if path.is_file():
                evidence[str(path.relative_to(ROOT))] = sha(path)
    if [p['seed'] for p in phases if p['action'] == 'short'] != [0, 1, 2]:
        raise ValueError('three short seeds required')
    return evidence


def build_runtime_release(candidate_path, qualification_folder):
    evidence = qualification_evidence(qualification_folder, candidate_path)
    return {'schema': 'idc-runtime-formal-release-v1', 'status': 'frozen',
            'execution_version': VERSION, 'formal_training_ready': True,
            'candidate_path': str(Path(candidate_path).relative_to(ROOT)),
            'candidate_sha256': sha(candidate_path),
            'qualification_folder': str(Path(qualification_folder).relative_to(ROOT)),
            'evidence': evidence, 'allowed_seeds': [0], 'fresh_initialization_required': True,
            'formal_512_automatically_started': False}


def verify_runtime_release(path):
    recorded = json.loads(Path(path).read_text())
    expected = build_runtime_release(ROOT / recorded['candidate_path'],
                                     ROOT / recorded['qualification_folder'])
    if recorded != expected:
        raise ValueError('formal runtime release evidence/code/config mismatch')
    return recorded


def runtime_checkpoint_binding(path):
    release = verify_runtime_release(path)
    binding = candidate_binding(ROOT / release['candidate_path'])
    return {**binding, 'runtime_release_sha256': sha(path), 'formal_training_ready': True}
