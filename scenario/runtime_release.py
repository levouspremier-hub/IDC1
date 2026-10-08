"""Separate signed qualification candidates; historical releases stay immutable."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scenario.inventory_release import MATRIX_PATH, ROOT, _git, load_config, load_matrix, sha

VERSION = 'certified-lexicographic-learning-coverage-v2'
BUDGETS = (.25, .50, 1.00)
NUMERIC_FIXTURES = (
    'm6p2c_seed1_origin9120_step18.json',
    'm6p2c_seed1_origin1008_step18.json',
    'm6p2c_seed1_origin7584_step1.json',
    'm6p2c_seed0_origin6192_step47.json',
    'm6p2c_seed0_origin384_step36.json',
    'm6p2c_seed0_origin480_step45.json',
    'm6p2c_seed1_origin576_step44.json',
    'm6p2c_seed1_origin1968_step18.json',
    'm6p2c_seed1_origin6336_step2.json',
    'm6p2c_seed0_origin3072_step18.json',
)
NUMERIC_VARIANTS = ('recorded', 'zero_charge', 'full_discharge',
                    'sample0', 'sample1', 'sample2', 'sample3',
                    'soc_plus_ulp', 'soc_minus_ulp', 'soc_plus_nano', 'soc_minus_nano')


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
    from planning.model import INVENTORY_SOLVER_FEASIBILITY_TOLERANCE, INVENTORY_STAGE_B_PRESOLVE
    tolerance = INVENTORY_SOLVER_FEASIBILITY_TOLERANCE
    config['training']['corrector']['solver_feasibility_tolerance'] = tolerance
    config['training']['corrector']['inventory_stage_b_presolve'] = INVENTORY_STAGE_B_PRESOLVE
    from planning.numeric_contract import (
        COUPLED_CHARGE_DOMAIN_VERSION,
        FIXED_INTEGER_POLISH_VERSION,
        INTEGER_REPRESENTATION_ULPS,
        INVENTORY_MODE_COVER_VERSION,
        NUMERIC_CONTRACT_VERSION,
        ONE_STEP_STORAGE_BOUND_VERSION,
        PRIMAL_WITNESS_TOLERANCE,
        STAGE_B_COORDINATE_VERSION,
    )
    config['training']['corrector']['primary_mip_rel_gap'] = 0.
    config['training']['corrector']['primary_mip_abs_gap'] = 0.
    config['numeric_contract'] = {
        'version': NUMERIC_CONTRACT_VERSION,
        'integer_representation_ulps': INTEGER_REPRESENTATION_ULPS,
        'primal_witness_tolerance': PRIMAL_WITNESS_TOLERANCE,
        'integer_normalization_requires_original_primal_certificate': True,
        'learning_origin_cycles': 2,
        'primary_offset_tolerance': 1e-6,
        'one_step_storage_bounds_version': ONE_STEP_STORAGE_BOUND_VERSION,
        'coupled_charge_domain_version': COUPLED_CHARGE_DOMAIN_VERSION,
        'stage_b_coordinate_version': STAGE_B_COORDINATE_VERSION,
        'fixed_integer_primal_polish_version': FIXED_INTEGER_POLISH_VERSION,
        'inventory_mode_cover_version': INVENTORY_MODE_COVER_VERSION,
    }
    config['training']['backend']['note'] = (
        f'CPU / torch=1; inventory solver feasibility={tolerance:g}; '
        f'inventory B presolve={INVENTORY_STAGE_B_PRESOLVE}; '
        'random_seed=0 / parallel=False')
    config['inherited_calibration_scope'] = {
        'source': 'r5', 'scales_and_research_parameters_unchanged': True,
        'measured_solver_budget_s': .25, 'candidate_budget_calibrated': False,
        'qualification_required': ['numeric_witness_replay', '48_episodes',
                                   'three_seed_8_batches', 'three_seed_two_origin_cycles',
                                   'shared_4h']}
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
    paths.update(ROOT / 'tests/fixtures' / name for name in NUMERIC_FIXTURES)
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


def validate_numeric_evidence(result, candidate_path):
    """Require all registered counterexamples/variants and their actual certificates."""
    candidate = verify_candidate(candidate_path)
    expected = {(name, variant) for name in NUMERIC_FIXTURES for variant in NUMERIC_VARIANTS}
    rows = result.get('observations', [])
    if (result.get('passed') is not True or result.get('candidate_sha256') != sha(candidate_path)
            or result.get('numeric_contract') != candidate['config']['numeric_contract']
            or result.get('environment_steps') != 0 or result.get('parameter_updates') != 0
            or len(rows) != len(expected)
            or {(r['fixture'], r['variant']) for r in rows} != expected):
        raise ValueError('complete bound numeric qualification is required')
    for row in rows:
        path = ROOT / 'tests/fixtures' / row['fixture']
        fixture = json.loads(path.read_text())
        expected_soc = fixture['snapshot']['soc_kwh']
        if row['variant'].startswith('soc_'):
            import numpy as np
            delta = float(np.spacing(expected_soc)) if row['variant'].endswith('ulp') else 1e-9
            expected_soc += -delta if '_minus_' in row['variant'] else delta
        if (row['fixture_sha256'] != sha(path)
                or row['source_failure_sha256'] != fixture['source_failure_sha256']
                or row['snapshot_soc_kwh'] != expected_soc
                or row['passed'] is not True or not row['candidate_check']['passed']
                or row['candidate_check']['integer_residual'] != 0):
            raise ValueError('numeric fixture provenance or executable witness differs')
        audit = row['inventory_audit']
        for key in ('stage_a_witness', 'execution_witness', 'primary_objective_certificate'):
            if audit.get(key, {}).get('passed') is not True:
                raise ValueError('numeric qualification lacks certified lexicographic witnesses')
        if not row['solver_calls'] or any(
                call['options'].get('mip_feasibility_tolerance') != 1e-10
                or call['options'].get('random_seed') != 0
                or call['options'].get('parallel') is not False
                or not 0 < call['options']['time_limit'] <= candidate['config']['runtime_budget_s']
                for call in row['solver_calls']):
            raise ValueError('numeric qualification solver options differ')
        primary = [c for c in row['solver_calls'] if c['options'].get('mip_rel_gap') == 0.]
        if not primary or any(c['options'].get('mip_abs_gap') != 0. for c in primary):
            raise ValueError('numeric qualification lacks the primary optimality contract')
        expected_presolve = candidate['config']['training']['corrector'][
            'inventory_stage_b_presolve']
        if row['stage_b_status'] == 'optimal' and row['solver_calls'][-1]['options'].get(
                'presolve') is not expected_presolve:
            raise ValueError('numeric qualification B presolve differs')
        polishes = [audit[key]['primal_polish']
                    for key in ('reachability_witness', 'stage_a_witness', 'execution_witness')
                    if 'primal_polish' in audit.get(key, {})]
        calls = row.get('primal_polish_calls', [])
        if len(calls) != len(polishes) or any(
                call['method'] != 'highs-ds'
                or call['options'].get('primal_feasibility_tolerance') != 1e-10
                or call['options'].get('dual_feasibility_tolerance') != 1e-10
                or call['options'].get('random_seed') != 0
                or call['options'].get('parallel') is not False
                or not 0 < call['options']['time_limit'] <= candidate['config']['runtime_budget_s']
                or not call['mathematical_input_hashes'] for call in calls):
            raise ValueError('numeric qualification primal polish calls differ')
        if any(p['solver_status'] != 0 or not p['attempted']
               or p['version'] != candidate['config']['numeric_contract'][
                   'fixed_integer_primal_polish_version'] for p in polishes):
            raise ValueError('numeric qualification primal polish did not succeed')
        if (row['execution_source'] == 'stage_b'
                and 'primal_polish' in audit['execution_witness']
                and audit.get('polished_economic_objective_certificate', {}).get('passed')
                is not True):
            raise ValueError('numeric qualification polished economic incumbent differs')
    return result


def validate_short_evidence(folder, candidate_path, seed):
    """Reuse only a fresh, same-source short run with actual optimizer/RNG state."""
    import torch

    from checkpointing.inventory_eval_input import SHORT_SCHEMA
    from checkpointing.versioned import read_checkpoint_payload
    from safe_rl_v2.formal_train import preregistered_batches
    from safe_rl_v2.formal_train_loop import (
        build_lagrangian,
        build_optimizer,
        build_seeded_policy,
        load_resume_checkpoint,
    )
    from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
    from safe_rl_v2.inventory_train import verify_written_run
    folder = Path(folder)
    candidate = verify_candidate(candidate_path)
    config = candidate['config']
    binding = candidate_binding(candidate_path)
    report = verify_written_run(folder, binding)
    expected = preregistered_batches(load_matrix(), seed)[:8]
    if (type(report['seed']) is not int or report['seed'] != seed
            or report['scope'] != 'controlled_short_run' or report['inventory_binding'] != binding
            or report['batches'] != 8 or report['batches_newly_run'] != 8
            or report['resumed_from'] is not None or report['transitions'] != 1536
            or report['adam_steps'] != 128 or report['lagrangian_updates'] != 8
            or [b['origins'] for b in report['batch_records']] != expected
            or len(report['inventory_episodes']) != 32
            or any(not all(inventory_episode_acceptance(e).values())
                   or e['target_qualified'] is not True for e in report['inventory_episodes'])
            or any(b['zero_action_fallback_steps'] != 0 for b in report['batch_records'])
            or report['validation_run'] or report['test_run']):
        raise ValueError('same-candidate short evidence workload, role or quality differs')
    checkpoint = folder / 'checkpoint_final.pt'
    payload = read_checkpoint_payload(checkpoint)
    if payload['metadata'].get('inventory_binding') != binding:
        raise ValueError('short evidence checkpoint binding differs')
    policy = build_seeded_policy(config, obs_dim=523, seed=seed)
    optimizer = build_optimizer(config, policy)
    lagrangian = build_lagrangian(config)
    restored = load_resume_checkpoint(checkpoint, policy=policy, optimizer=optimizer,
        lagrangian=lagrangian, sampling_generator=torch.Generator(),
        shuffle_generator=torch.Generator(), config=config, expected_obs_dim=523,
        expected_schema=SHORT_SCHEMA, expected_scope='controlled_short_run',
        expected_role='controlled_training_resume')
    steps = [int(s['step']) for s in optimizer.state.values()]
    if (restored['next_batch_index'] != 8
            or restored['origins'] != [o for b in expected for o in b]
            or not steps or set(steps) != {128} or lagrangian._updates != 8
            or payload['state']['source_ledger']['master_seed'] != seed
            or any(not torch.isfinite(p).all() for p in policy.state_dict().values())
            or any(not torch.isfinite(s[k]).all() for s in optimizer.state.values()
                   for k in ('exp_avg', 'exp_avg_sq'))):
        raise ValueError('short evidence actual checkpoint state differs')
    return report


def validate_learning_coverage(folder, candidate_path):
    """Read actual learning state, schedule and quality; a duration is insufficient."""
    import torch

    from checkpointing.inventory_eval_input import SHORT_SCHEMA
    from checkpointing.versioned import read_checkpoint_payload
    from safe_rl_v2.formal_train_loop import (
        build_lagrangian,
        build_optimizer,
        build_seeded_policy,
        load_resume_checkpoint,
    )
    from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
    from safe_rl_v2.inventory_train import coverage_batches, verify_written_run
    folder = Path(folder)
    candidate = verify_candidate(candidate_path)
    config = candidate['config']
    binding = candidate_binding(candidate_path)
    report = verify_written_run(folder, binding)
    batches = coverage_batches(load_matrix(), report['seed'])
    count = len(batches)
    sampling_config = config['training']['sampling']
    adam_per_batch = (sampling_config['epochs_per_batch']
                      * sampling_config['minibatches_per_epoch'])
    if (report['scope'] != 'runtime_learning_qualification'
            or type(report['seed']) is not int or report['seed'] not in (0, 1, 2)
            or report['inventory_binding'] != binding
            or report['resumed_from'] is not None or report['batches_newly_run'] != count
            or report['batches'] != count or len(report['batch_records']) != count
            or [b['origins'] for b in report['batch_records']] != batches
            or report['transitions'] != count * sampling_config['transitions_per_batch']
            or report['adam_steps'] != count * adam_per_batch
            or report['lagrangian_updates'] != count
            or len(report['inventory_episodes']) != sum(map(len, batches))
            or report['validation_run'] or report['test_run']
            or any(not all(inventory_episode_acceptance(e).values())
                   or e['target_qualified'] is not True for e in report['inventory_episodes'])
            or any(b['zero_action_fallback_steps'] != 0 for b in report['batch_records'])):
        raise ValueError('learning coverage workload, origins or quality differs')
    checkpoint = folder / 'checkpoint_final.pt'
    payload = read_checkpoint_payload(checkpoint)
    if payload['metadata'].get('inventory_binding') != binding:
        raise ValueError('learning coverage checkpoint binding differs')
    policy = build_seeded_policy(config, obs_dim=523, seed=report['seed'])
    optimizer = build_optimizer(config, policy)
    lagrangian = build_lagrangian(config)
    restored = load_resume_checkpoint(checkpoint, policy=policy, optimizer=optimizer,
        lagrangian=lagrangian, sampling_generator=torch.Generator(),
        shuffle_generator=torch.Generator(), config=config, expected_obs_dim=523,
        expected_schema=SHORT_SCHEMA, expected_scope='runtime_learning_qualification',
        expected_role='runtime_learning_qualification_resume')
    steps = [int(s['step']) for s in optimizer.state.values()]
    if (restored['next_batch_index'] != count
            or restored['origins'] != [o for b in batches for o in b]
            or not steps or set(steps) != {count * adam_per_batch}
            or lagrangian._updates != count
            or any(not torch.isfinite(p).all() for p in policy.state_dict().values())
            or any(not torch.isfinite(s[k]).all() for s in optimizer.state.values()
                   for k in ('exp_avg', 'exp_avg_sq'))):
        raise ValueError('learning coverage actual checkpoint state differs')
    return report


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
    registered = report.get('registered_budgets')
    if registered is not None:
        if (report.get('qualification_scope') != 'fixed_approved_budget'
                or registered not in [[b] for b in BUDGETS]
                or type(registered[0]) not in (float, int)
                or registered != [candidate['config']['runtime_budget_s']]):
            raise ValueError('fixed qualification budget differs from registered campaign')
    else:
        registered = list(BUDGETS)
    if [a['budget_s'] for a in attempts] != registered[:len(attempts)]:
        raise ValueError('budget selection differs from preregistered order')
    passed = [a for a in attempts if a.get('passed') is True]
    if (len(passed) != 1 or passed[0] is not attempts[-1]
            or passed[0]['candidate_sha256'] != sha(candidate_path)):
        raise ValueError('qualification must select first fully passing budget')
    phases = phases + passed[0]['phases']
    numerical = 'numeric_contract' in candidate['config']
    expected = ['gate', 'resume-audit'] + (['numerics'] if numerical else [])
    expected += ['diagnose', 'short', 'short', 'short']
    expected += (['coverage', 'coverage', 'coverage'] if numerical else []) + ['soak']
    if [p['action'] for p in phases] != expected:
        raise ValueError('qualification phase sequence incomplete')
    for phase in phases:
        child = folder.parent / phase['run_id']
        manifest = json.loads((child / 'manifest.json').read_text())
        result = json.loads((child / 'report.json').read_text())
        if phase['exit_code'] != 0 or manifest['status'] != 'success':
            raise ValueError('qualification phase is not successful')
        if phase['action'] == 'numerics':
            validate_numeric_evidence(result, candidate_path)
            episodes = []
        elif phase['action'] == 'coverage':
            result = validate_learning_coverage(child, candidate_path)
            if result['seed'] != phase['seed']:
                raise ValueError('learning coverage seed differs')
            episodes = result['inventory_episodes']
        elif phase['action'] == 'short':
            verify_written_run(child, candidate_binding(candidate_path))
            if phase.get('reused_same_candidate'):
                validate_short_evidence(child, candidate_path, phase['seed'])
                if phase['original_report_sha256'] != sha(child / 'report.json'):
                    raise ValueError('reused short report differs from the accepted source')
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
    if numerical and [p['seed'] for p in phases if p['action'] == 'coverage'] != [0, 1, 2]:
        raise ValueError('three learning coverage seeds required')
    return evidence


def _campaign_authorization(path, budget):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('campaign authorization must be inside the repository')
    authorization = json.loads(path.read_text())
    seeds = authorization.get('approved_seeds')
    base_path = ROOT / authorization.get('base_release_path', '')
    if (authorization.get('schema') != 'idc-runtime-campaign-authorization-v1'
            or seeds != [0, 1, 2] or any(type(s) is not int for s in seeds)
            or authorization.get('runtime_budget_s') != budget
            or authorization.get('fresh_initialization_required') is not True
            or authorization.get('matrix_sha256') != sha(ROOT / MATRIX_PATH)
            or not isinstance(authorization.get('user_request'), str)
            or not authorization['user_request'].strip()
            or not base_path.resolve().is_relative_to(ROOT.resolve())
            or not base_path.is_file()
            or authorization.get('base_release_sha256') != sha(base_path)):
        raise ValueError('campaign authorization seed/budget/matrix/base binding mismatch')
    base = json.loads(base_path.read_text())
    if (base.get('schema') != 'idc-runtime-formal-release-v1'
            or base.get('formal_training_ready') is not True or base.get('allowed_seeds') != [0]):
        raise ValueError('campaign authorization requires the historical seed0 release reference')
    return path


def build_runtime_release(candidate_path, qualification_folder, *,
                          campaign_authorization_path=None):
    evidence = qualification_evidence(qualification_folder, candidate_path)
    release = {'schema': 'idc-runtime-formal-release-v1', 'status': 'frozen',
            'execution_version': VERSION, 'formal_training_ready': True,
            'candidate_path': str(Path(candidate_path).relative_to(ROOT)),
            'candidate_sha256': sha(candidate_path),
            'qualification_folder': str(Path(qualification_folder).relative_to(ROOT)),
            'evidence': evidence, 'allowed_seeds': [0], 'fresh_initialization_required': True,
            'formal_512_automatically_started': False}
    if campaign_authorization_path is not None:
        budget = verify_candidate(candidate_path)['config']['runtime_budget_s']
        authorization = _campaign_authorization(campaign_authorization_path, budget)
        release.update(schema='idc-runtime-formal-release-v2', allowed_seeds=[0, 1, 2],
                       campaign_authorization_path=str(authorization.relative_to(ROOT)),
                       campaign_authorization_sha256=sha(authorization))
    return release


def verify_runtime_release(path):
    recorded = json.loads(Path(path).read_text())
    schema = recorded.get('schema')
    if schema not in ('idc-runtime-formal-release-v1', 'idc-runtime-formal-release-v2'):
        raise ValueError('unregistered runtime formal release schema')
    kwargs = {}
    if schema == 'idc-runtime-formal-release-v2':
        authorization = ROOT / recorded['campaign_authorization_path']
        kwargs['campaign_authorization_path'] = authorization
    expected = build_runtime_release(ROOT / recorded['candidate_path'],
                                     ROOT / recorded['qualification_folder'], **kwargs)
    if recorded != expected:
        raise ValueError('formal runtime release evidence/code/config mismatch')
    return recorded


def runtime_checkpoint_binding(path):
    release = verify_runtime_release(path)
    binding = candidate_binding(ROOT / release['candidate_path'])
    return {**binding, 'runtime_release_sha256': sha(path), 'formal_training_ready': True}
