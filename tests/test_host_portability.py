"""Portable verification keeps immutable asset identity and generator provenance."""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenario import exogenous_drivers_b6 as b6
from scenario.portable_numeric import (
    FROZEN_RECIPE_REVISION,
    assert_frozen_parquet,
    assert_power_roundoff,
    recipe_ast,
    verify_frozen_recipe,
)

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / 'data/processed/singapore_2024/exogenous_drivers_v3.parquet'


def test_frozen_recipe_requires_registered_revision_and_unchanged_generator():
    verify_frozen_recipe(ROOT, FROZEN_RECIPE_REVISION, PARQUET)
    with pytest.raises(ValueError):
        verify_frozen_recipe(ROOT, '0' * 40, PARQUET)
    source = subprocess.check_output(
        ['git', 'show', FROZEN_RECIPE_REVISION + ':scenario/exogenous_drivers_b6.py'],
        cwd=ROOT, text=True)
    changed = source.replace('B5_ARRIVAL_SCALE_INHERITED = False',
                             'B5_ARRIVAL_SCALE_INHERITED = True')
    assert recipe_ast(source) != recipe_ast(changed)


def test_even_one_ulp_coordinated_parquet_change_is_rejected(tmp_path):
    frame = pd.read_parquet(PARQUET)
    i = frame.index[frame.local_pv_kw > 0][0]
    frame.loc[i, 'local_pv_kw'] = np.nextafter(frame.loc[i, 'local_pv_kw'], np.inf)
    altered = tmp_path / 'altered.parquet'
    frame.to_parquet(altered, index=False)
    with pytest.raises(ValueError):
        assert_frozen_parquet(altered)


@pytest.mark.parametrize('scale', [500., 800.])
def test_power_roundoff_rejects_material_changes_zero_changes_and_nonfinite(scale):
    base = np.array([0., 10.])
    assert_power_roundoff(np.array([0., np.nextafter(10., np.inf)]), base, scale)
    for actual in [np.array([0., 10. + 1e-9]), np.array([1e-15, 10.]),
                   np.array([0., np.nan])]:
        with pytest.raises(ValueError):
            assert_power_roundoff(actual, base, scale)


def test_verified_loader_returns_exact_frozen_values_after_roundoff(monkeypatch):
    original = b6.build_v3_frame

    def drift(*args, **kwargs):
        frame = original(*args, **kwargs)
        i = frame.index[frame.local_pv_kw > 0][0]
        frame.loc[i, 'local_pv_kw'] = np.nextafter(frame.loc[i, 'local_pv_kw'], np.inf)
        return frame

    monkeypatch.setattr(b6, 'build_v3_frame', drift)
    bundle = b6.load_verified_v3_bundle()
    pd.testing.assert_frame_equal(bundle['frame'], pd.read_parquet(PARQUET), check_exact=True)


def test_historical_bridge_rejects_changed_generator(tmp_path):
    from scenario.portable_numeric import RECIPE_SOURCES

    for name in RECIPE_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(
            ['git', 'show', FROZEN_RECIPE_REVISION + ':' + name], cwd=ROOT))
    # Git lookup must remain the actual repository; only candidate source reads change.
    script = tmp_path / 'scripts/materialize_singapore_exogenous_b6.py'
    script.write_text(script.read_text() + '\nraise RuntimeError("changed generator")\n')
    # Use the real repo for history via a gitdir pointer, without copying its contents.
    git_dir = subprocess.check_output(
        ['git', 'rev-parse', '--absolute-git-dir'], cwd=ROOT, text=True).strip()
    (tmp_path / '.git').write_text('gitdir: ' + git_dir + '\n')
    with pytest.raises(ValueError, match='recipe changed'):
        verify_frozen_recipe(tmp_path, FROZEN_RECIPE_REVISION, PARQUET)
