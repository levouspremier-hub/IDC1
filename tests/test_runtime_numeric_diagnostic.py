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
    assert all(c['expectation'] == 'exploratory' for c in boundary)


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
    assert row['classification'] == 'solver_missed_certified_feasible_witness'
    assert row['diagnostic_original_b_anchor_certificate']['passed']
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
