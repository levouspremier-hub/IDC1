"""Authenticated verification of the original frozen B6 recipe across platforms.

A historical recipe is accepted only with its registered bytes and unchanged
materialization implementation. This does not reissue or rewrite frozen assets.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

FROZEN_RECIPE_REVISION = '34f2de03d1b86a98a59c983b0783bd602e357f6f'
FROZEN_PARQUET_SHA256 = '07b648f0a15db1d8c39838e3e501dafa2f9956155489702e3379cdb775858612'
RECIPE_SOURCES = {
    'scenario/exogenous_drivers_b6.py':
        '45ce9a63e0ab1eaa1b7713ca6c268542d4562a30bc3edc8e6e61be631b45e36c',
    'scripts/materialize_singapore_exogenous_b6.py':
        'f4259ed85550590edf5d4e9bd99f141fa4b7c8bd4f8afa7484e0d3d2cb32d07c',
}
# These two consumers verify an existing artifact; all generator code, imports,
# constants, declarations and the complete materializer script remain anchored.
VERIFIER_FUNCTIONS = {'_assert_frames_identical', 'load_verified_v3_bundle'}
REGISTERED_RECIPES = {
    'refs': '577f1db4d5ca78553aa1be866f643ebeb7e8cb13',
    'splits': '577f1db4d5ca78553aa1be866f643ebeb7e8cb13',
    'forecast': '626f97f3cbd13082e14197028dd62d356b8e54e4',
}
REVISION_CONSUMERS = {
    'scenario/exogenous_drivers_b6.py': VERIFIER_FUNCTIONS,
    'scenario/b6_refs.py': {'refs_code_revision'},
    'scenario/b6_split_manifests.py': {'resolve_materializer_revision'},
    'scenario/formal_scenario_b6.py': {'b6_formal_code_revision'},
}


@lru_cache(maxsize=128)
def recipe_ast(source, excluded=('_assert_frames_identical', 'load_verified_v3_bundle')):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name in excluded)]
    return ast.dump(tree, include_attributes=False)


@lru_cache(maxsize=128)
def historical_source(root, revision, name):
    return subprocess.check_output(['git', 'show', revision + ':' + name], cwd=root)


def verified_recipe_revision(root, paths, live_revision, kind):
    """Preserve a registered generator stamp only for byte/AST-proven equivalent recipes."""
    registered = REGISTERED_RECIPES[kind]
    actual = subprocess.check_output(
        ['git', 'log', '-1', '--format=%H', '--', *paths], cwd=root, text=True).strip()
    # Retain the caller's fail-closed behavior for a forged or simulated live revision.
    if actual != live_revision:
        return live_revision
    for name in paths:
        original = historical_source(str(root), registered, name)
        current = (Path(root) / name).read_bytes()
        if name in REVISION_CONSUMERS:
            excluded = tuple(sorted(REVISION_CONSUMERS[name]))
            equal = (recipe_ast(current.decode(), excluded)
                     == recipe_ast(original.decode(), excluded))
        else:
            equal = current == original
        if not equal:
            return live_revision  # Changed generator must use new assets and a new recipe stamp.
    return registered


def assert_frozen_parquet(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != FROZEN_PARQUET_SHA256:
        raise ValueError('Portable verification requires the registered frozen parquet SHA')


def verify_frozen_recipe(root, declared_revision, parquet):
    if declared_revision != FROZEN_RECIPE_REVISION:
        raise ValueError('Unregistered historical materializer revision')
    assert_frozen_parquet(parquet)
    for name, digest in RECIPE_SOURCES.items():
        original = subprocess.check_output(
            ['git', 'show', declared_revision + ':' + name], cwd=root)
        if hashlib.sha256(original).hexdigest() != digest:
            raise ValueError('Registered materializer source bytes differ')
        current = (Path(root) / name).read_bytes()
        equal = (recipe_ast(current.decode()) == recipe_ast(original.decode())
                 if name.startswith('scenario/') else current == original)
        if not equal:
            raise ValueError('Materialization recipe changed; requires a new asset version')


def assert_power_roundoff(actual, expected, physical_scale_kw):
    if actual.shape != expected.shape:
        raise ValueError('Power shape differs')
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError('Nonfinite power')
    if not np.array_equal(actual == 0., expected == 0.):
        raise ValueError('Power zero locations differ')
    if not (np.abs(actual - expected) <=
            8 * np.finfo(np.float64).eps * physical_scale_kw).all():
        raise ValueError('Power differs beyond physical-scale float64 roundoff')
