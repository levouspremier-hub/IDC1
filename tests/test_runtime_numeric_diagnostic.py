"""Diagnostic failures are evidence, never release authorization."""
import copy
import json

import pytest

from scripts import runtime_numeric_diagnostic as diagnostic

LATEST = 'm6p2c_seed0_origin9744_step44.json'


def test_fixture_hash_and_required_fields_are_enforced(tmp_path):
    registry = diagnostic.load_registry()
    entry = registry['fixtures'][LATEST]
    original = diagnostic.load_fixture(LATEST, entry)
    path = tmp_path / LATEST
    broken = copy.deepcopy(original)
    del broken['snapshot']['planning_forecast']
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='hash'):
        diagnostic.load_fixture(LATEST, entry, fixture_dir=tmp_path)
    with pytest.raises(ValueError, match='planning_forecast'):
        diagnostic.validate_fixture(broken)


def test_latest_precedes_all_history_and_boundary_is_fixed():
    registry = diagnostic.load_registry()
    cases = diagnostic.cases_for('all', registry)
    assert cases[0]['fixture'] == LATEST and cases[0]['variant'] == 'recorded'
    history = diagnostic.cases_for('history', registry)
    assert len(history) == 143
    boundary = diagnostic.cases_for('boundary', registry)
    assert boundary == diagnostic.cases_for('boundary', registry)
    assert len({c['case_id'] for c in boundary}) == len(boundary)
    assert sum(c['expectation'] == 'must_pass' for c in boundary) == 6
    assert sum(c['expectation'] == 'exploratory' for c in boundary) == 228


def test_certificate_failure_cannot_become_a_success():
    row = {'passed': True, 'stage_a_status': 'optimal', 'stage_b_status': 'optimal',
           'candidate_check': {'passed': True}, 'inventory_audit': {
               'stage_a_witness': {'passed': True},
               'primary_objective_certificate': {'passed': True},
               'execution_witness': {'passed': False}}}
    assert diagnostic.classify(row) == 'original_matrix_certificate_failure'
    row['inventory_audit']['execution_witness']['passed'] = True
    row['inventory_audit']['primary_objective_certificate']['passed'] = False
    assert diagnostic.classify(row) == 'a_objective_bound_certificate_failure'
    row['inventory_audit']['primary_objective_certificate']['passed'] = True
    row['inventory_audit']['polished_economic_objective_certificate'] = {'passed': False}
    assert diagnostic.classify(row) == 'b_objective_certificate_failure'


def test_original_9744_failure_is_directly_reproducible(monkeypatch):
    from planning import numeric_contract
    from scenario.runtime_release import candidate_config
    original_translate = numeric_contract.translate_continuous_origin
    monkeypatch.setattr(numeric_contract, 'translate_continuous_origin',
                        lambda a, i, b, m, lo, hi, **kw: original_translate(a, i, b, m, lo, hi))
    monkeypatch.setattr(numeric_contract, 'scale_constraint_rows',
                        lambda m, lo, hi, **kw: (m, lo, hi, {'version': 'disabled-for-test'}))
    registry = diagnostic.load_registry()
    case = diagnostic.cases_for('latest', registry)[0]
    row = diagnostic.evaluate_case(case, registry, candidate_config(.50))
    assert not row['passed']
    assert row['classification'] == 'model_or_fixed_integer_infeasible'
    assert row['diagnostic_original_b_anchor_certificate']['passed']
    assert row['diagnostic_original_b_interval_contradictions']
    assert row['stage_b_status'] == 'infeasible'
    assert row['inventory_audit']['stage_a_witness']['passed']
    assert row['input_sha256'] and row['solver_calls']


def test_source_mismatch_is_terminal_and_all_artifacts_exist(tmp_path, monkeypatch):
    registry = diagnostic.load_registry()
    registry['fixtures'][LATEST]['sha256'] = '0' * 64
    monkeypatch.setattr(diagnostic, 'load_registry', lambda: registry)
    assert diagnostic.run('latest', 'bad-source', base_dir=tmp_path) == 1
    folder = tmp_path / 'bad-source'
    assert all((folder / name).exists() for name in (
        'config.yaml', 'metrics.parquet', 'report.json', 'figures', 'manifest.json'))
    report = json.loads((folder / 'report.json').read_text())
    assert report['state'] == 'failed' and report['exit_code'] == 1
    assert report['observations'][0]['classification'] == 'input_source_contract_failure'
    assert report['scientific_binding']['file_hashes']['planning/model.py']
    assert report['parameter_updates'] == 0


def test_successful_anchor_extraction_rejects_divergent_reconstruction():
    from scripts.runtime_numeric_anchor_extract import validate_observation
    validate_observation([1., 2.], [1., 2., .5, .5, 1.])
    with pytest.raises(ValueError, match='observation'):
        validate_observation([1., 2.], [1., 3., .5, .5, 1.])


def test_preregistered_unavailable_boundary_inputs_are_counted_and_fail():
    registry = diagnostic.load_registry()
    # Exercise the missing-source contract even after real anchors are recovered.
    registry['successful_anchors'] = {}
    assert len(diagnostic.cases_for('boundary', registry)) == 234
    assert len(diagnostic.cases_for('all', registry)) == 377
    unavailable = next(c for c in diagnostic.cases_for('boundary', registry)
                       if c['fixture'].startswith('m6p2c_train_success_'))
    with pytest.raises(ValueError, match='source-verified.*missing'):
        diagnostic.evaluate_case(unavailable, registry, {'runtime_budget_s': .50})


def test_recovered_success_anchors_match_original_train_trace():
    registry = diagnostic.load_registry()
    assert len(registry['successful_anchors']) == 6
    for name, entry in registry['successful_anchors'].items():
        fixture = diagnostic.load_fixture(name, entry)
        assert fixture['source_transition_index'] == entry['source_transition_index']
        assert diagnostic.digest(fixture['snapshot']) == entry['snapshot_sha256']


def test_invalid_registry_bootstrap_still_has_a_failed_five_artifact_run(tmp_path, monkeypatch):
    def invalid_registry():
        raise ValueError('registry schema is invalid')
    monkeypatch.setattr(diagnostic, 'load_registry', invalid_registry)
    assert diagnostic.run('latest', 'invalid-registry', base_dir=tmp_path) == 1
    folder = tmp_path / 'invalid-registry'
    assert all((folder / name).exists() for name in (
        'config.yaml', 'metrics.parquet', 'report.json', 'manifest.json', 'figures'))
    report = json.loads((folder / 'report.json').read_text())
    assert report['state'] == 'failed' and report['exit_code'] == 1
    assert report['case_count'] == 0 and report['parameter_updates'] == 0


def test_joint_boundary_perturbation_never_relaxes_the_original_access_capacity():
    registry = diagnostic.load_registry()
    fixture = diagnostic.load_fixture(LATEST, registry['fixtures'][LATEST])
    for variant in ('joint_headroom_charge', 'joint_headroom_discharge'):
        perturbed = diagnostic.case_input(fixture, variant)
        assert perturbed['snapshot']['access_limit_kw'] <= fixture['snapshot']['access_limit_kw']
        diagnostic.validate_fixture(perturbed)


def test_exploratory_infeasibility_requires_original_tolerance_aware_proof():
    from scripts.runtime_qualification import certify_interval_contradiction
    proof = certify_interval_contradiction([1.], [0.], 'a' * 64)
    row = {'expectation': 'exploratory', 'passed': False,
           'classification': 'model_or_fixed_integer_infeasible',
           'diagnostic_original_b_sha256': 'a' * 64,
           'diagnostic_infeasibility_certificate': proof}
    assert diagnostic.accept_case(row)['accepted']
    assert row['passed'] is False
    assert not diagnostic.accept_case({**row, 'expectation': 'must_pass'})['accepted']
    assert not diagnostic.accept_case({**row, 'classification': 'timeout'})['accepted']
    assert not diagnostic.accept_case({**row, 'diagnostic_original_b_sha256': 'b' * 64})[
        'accepted']
    tiny = certify_interval_contradiction([1e-15], [0.], 'a' * 64)
    assert not tiny['passed']
    assert not diagnostic.accept_case({**row, 'diagnostic_infeasibility_certificate': tiny})[
        'accepted']
    forged = {**proof, 'lower': 0., 'upper': 1.}
    assert not diagnostic.accept_case({**row, 'diagnostic_infeasibility_certificate': forged})[
        'accepted']


@pytest.mark.parametrize('category', diagnostic.CLASSIFICATIONS[:-1])
def test_exploratory_status_alone_never_authorizes_acceptance(category):
    assert not diagnostic.accept_case({'expectation': 'exploratory', 'passed': False,
                                       'classification': category})['accepted']


def test_diagnostic_run_reports_acceptance_separately_from_executable_success(
        tmp_path, monkeypatch):
    from scripts.runtime_qualification import certify_interval_contradiction
    case = {'case_id': 'proved/interval', 'fixture': LATEST, 'variant': 'recorded',
            'expectation': 'exploratory', 'suite': 'boundary'}
    row = {**case, 'passed': False, 'classification': 'model_or_fixed_integer_infeasible',
           'diagnostic_original_b_sha256': 'a' * 64,
           'diagnostic_infeasibility_certificate': certify_interval_contradiction(
               [1.], [0.], 'a' * 64)}
    monkeypatch.setattr(diagnostic, 'cases_for', lambda *args: [case])
    monkeypatch.setattr(diagnostic, 'evaluate_case', lambda *args: copy.deepcopy(row))
    assert diagnostic.run('boundary', 'proved', base_dir=tmp_path) == 0
    report = json.loads((tmp_path / 'proved/report.json').read_text())
    assert report['passed'] and report['accepted_count'] == 1 and report['passed_count'] == 0
    assert report['observations'][0]['passed'] is False
    assert report['formal_training_ready'] is False


def test_validation_binding_includes_registered_fixtures_tests_and_configs():
    binding = diagnostic.scientific_binding()
    assert binding['validation_file_hashes']['tests/fixtures/runtime_numeric_registry.json']
    assert binding['validation_file_hashes']['tests/test_runtime_numeric_diagnostic.py']
    assert any(p.startswith('configs/') for p in binding['validation_file_hashes'])
