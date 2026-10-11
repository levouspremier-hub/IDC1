"""Frozen candidates must bind and verify every registered numeric input."""
import copy

import pytest

from scenario import runtime_release
from scripts import runtime_numeric_diagnostic as diagnostic


def test_all_registered_fixtures_are_candidate_assets():
    registry = diagnostic.load_registry()
    assets = runtime_release.asset_binding()
    for name, entry in {**registry['fixtures'], **registry['successful_anchors']}.items():
        assert assets['tests/fixtures/' + name] == entry['sha256']
        assert assets[entry['source_path']] == entry['source_failure_sha256']


@pytest.mark.parametrize('kind', ['fixtures', 'successful_anchors'])
@pytest.mark.parametrize('defect', ['missing', 'wrong_hash'])
def test_asset_freeze_rejects_invalid_registered_inputs(monkeypatch, kind, defect):
    registry = copy.deepcopy(diagnostic.load_registry())
    name = next(iter(registry[kind]))
    if defect == 'missing':
        registry[kind]['missing-registered-fixture.json'] = registry[kind].pop(name)
    else:
        registry[kind][name]['sha256'] = '0' * 64
    monkeypatch.setattr(diagnostic, 'load_registry', lambda: registry)
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime_release.asset_binding()
