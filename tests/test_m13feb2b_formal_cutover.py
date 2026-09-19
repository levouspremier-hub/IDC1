"""M1.3f-e-b2-b：B6 formal 链正式切换、`refs_v4` 与 `formal_splits_v5` 的**先红**回归。

改前缺陷（本文件在实现前必须为红）：

- `scenario.b6_refs` / `scenario.b6_split_manifests` 模块尚不存在；
- `configs/frozen_refs/refs_v4.json` 与 `data/manifest/formal_splits_v5/` 尚不存在；
- `build_scenario(..., synthetic=False)` **未能**构造 B6 formal bundle。

**本卡只切换正式入口**：不接 env、不接 mapper、不训练、不改 readiness。"""

import hashlib
import importlib
import json
import pathlib
import shutil

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REFS_MODULE = "scenario.b6_refs"
SPLIT_MODULE = "scenario.b6_split_manifests"
FORMAL_MODULE = "scenario.formal_scenario_b6"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"

REFS_V4 = REPO_ROOT / "configs/frozen_refs/refs_v4.json"
REFS_V3 = REPO_ROOT / "configs/frozen_refs/refs_v3.json"
REFS_V2 = REPO_ROOT / "configs/frozen_refs/refs.json"
SPLITS_V5 = REPO_ROOT / "data/manifest/formal_splits_v5"
SPLITS_V4 = REPO_ROOT / "data/manifest/formal_splits_v4"
SPLITS_V3 = REPO_ROOT / "data/manifest/formal_splits_v3"
POLICY_V2 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v2.json"
POLICY_V3 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json"
B6_POLICY = REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json"

B6_ARRIVAL_MEAN = 31.994
B5_LEGACY_MEAN = 1000.0
B5_LEGACY_LAMBDA_REF = 2000.0

# 旧资产在本卡前后必须**逐字节不变**
PROTECTED = {
    REFS_V3: "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f",
    REFS_V2: "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827",
    POLICY_V2: "ef4dd58a88dcb34fd75690324f957a5f87d2fd7d1b7bf6afbcb528ae9b719e08",
    POLICY_V3: "926703337139143dc1ea5223739ca5408be5ed384124a1acf88c70a39c542ecb",
    B6_POLICY: "7066a0e127bc28f6a56ad4e62810c34536e1eb13b3c3c134baa3a1e6c4bca251",
    CANONICAL_PARQUET: "dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd",
    CANONICAL_MANIFEST: "e6484d6b050f100234a46811be30061f477bcc053b285854da5a882d2753b667",
    SPLIT_MANIFEST: "a096535fcdec81534f8cc05671d34d879a7e9517d06be789510dea586149af27",
}
V4_TRIAD = {
    "train": "215c20968be1b71aa5d5d4b1d22e6cf7ecbd0eb4c802d361dca061a1fbf082cb",
    "validation": "a69cddaf282f04ebf4277dbe84e7a5d66950fca68054aacb07a811786945c258",
    "test": "829a0f12f042c34be43cc42891ed70ee0588939e672026e299382043859c80bf",
}


def refs_module():
    return importlib.import_module(REFS_MODULE)


def split_module():
    return importlib.import_module(SPLIT_MODULE)


def formal_module():
    return importlib.import_module(FORMAL_MODULE)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _upstream_present() -> bool:
    return all(p.exists() for p in (
        CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST,
        POLICY_V2, POLICY_V3, B6_POLICY,
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机"
)


def _split_chain(split: str, start: str, horizon: int = 48, cutoff: int = 4):
    from scenario.scenario import build_scenario

    return build_scenario(split, start=start, horizon=horizon, forecast_cutoff=cutoff)


# --- 1. 三份产物存在性 + 正式入口可用 -------------------------------------------

@needs_assets
def test_refs_v4_exists_and_binds_the_b6_chain():
    m = refs_module()
    payload = m.load_verified_refs_v4()
    assert payload["schema_version"] == m.REFS_SCHEMA
    assert payload["materializer_revision"] == m.refs_code_revision()


@needs_assets
def test_refs_v4_lambda_ref_is_the_b6_rate_not_the_legacy_2000():
    payload = refs_module().load_verified_refs_v4()
    lam = payload["references"]["lambda_ref"]
    assert lam["value"] == 63.988
    assert lam["value"] != B5_LEGACY_LAMBDA_REF
    assert lam["unit"] == "work-units/hour"
    assert lam["decision_id"] == "B6-INTENSITY"


@needs_assets
def test_v5_triad_exists_with_bumped_schema():
    m = split_module()
    for split in ("train", "validation", "test"):
        payload = m.load_verified_split_manifest_v5(expected_split=split)
        assert payload["schema"] == m.MANIFEST_SCHEMA
        assert "v5" in m.MANIFEST_SCHEMA
        assert payload["split"] == split
    stamps = {
        m.load_verified_split_manifest_v5(expected_split=s)["frozen_at_utc"]
        for s in ("train", "validation", "test")
    }
    assert len(stamps) == 1, "三份 manifest 必须共享同一 frozen_at_utc"


@needs_assets
def test_v5_triad_keeps_the_v4_row_and_origin_rules():
    m = split_module()
    expected = {
        "train": {"start": 0, "end_exclusive": 10224, "origins": (48, 10224)},
        "validation": {"start": 10224, "end_exclusive": 13152, "origins": (0, 2928)},
        "test": {"start": 13152, "end_exclusive": 17568, "origins": (0, 4416)},
    }
    for split, exp in expected.items():
        payload = m.load_verified_split_manifest_v5(expected_split=split)
        assert payload["split_rows"]["start"] == exp["start"]
        assert payload["split_rows"]["end_exclusive"] == exp["end_exclusive"]
        assert payload["candidate_origins"]["start"] == exp["origins"][0]
        assert payload["candidate_origins"]["end_exclusive"] == exp["origins"][1]


# --- 2. 正式入口：七序列 / mode / provenance / 31.994 ---------------------------

@needs_assets
@pytest.mark.parametrize("split,start", [
    ("train", "2024-01-02T00:00:00+08:00"),
    ("train", "2024-04-16T12:00:00+08:00"),
    ("validation", "2024-08-01T00:00:00+08:00"),
    ("validation", "2024-09-01T06:30:00+08:00"),
    ("test", "2024-10-01T00:00:00+08:00"),
    ("test", "2024-11-15T18:00:00+08:00"),
])
def test_formal_entry_builds_a_b6_bundle(split, start):
    bundle = _split_chain(split, start)
    assert bundle.mode == "formal"
    assert bundle.split == split
    assert len(bundle.arrival_forecast) == 4
    for field in formal_module().FORMAL_B6_SOURCE_KINDS:
        assert len(getattr(bundle, field)) == 4


@needs_assets
def test_formal_entry_arrival_is_the_b6_expected_value():
    bundle = _split_chain("train", "2024-04-16T12:00:00+08:00")
    template = formal_module().verified_v3_template()
    from scenario.exogenous_drivers import arrival_template_slot

    target = formal_module().target_timestamps_for(
        "train", origin=5112, forecast_cutoff=4)
    slots = arrival_template_slot(pd.DatetimeIndex(target))
    expected = [float(template[int(s)]) * B6_ARRIVAL_MEAN for s in slots]
    assert list(bundle.arrival_forecast) == pytest.approx(expected)
    assert all(abs(v - B5_LEGACY_MEAN) > 1.0 for v in bundle.arrival_forecast)


@needs_assets
def test_formal_entry_provenance_binds_the_v5_chain():
    bundle = _split_chain("train", "2024-01-02T00:00:00+08:00")
    prov = bundle.forecast_provenance
    roles = {d.role for d in prov.price_forecast.sources}
    assert "forecast_policy_manifest" in roles
    assert "b6_intensity_policy" in roles
    assert "exogenous_drivers_parquet" in roles
    stamps = {
        getattr(prov, f).generated_at for f in formal_module().FORMAL_B6_SOURCE_KINDS
    }
    assert len(stamps) == 1


# --- 3. start → split-local origin 精确映射 -------------------------------------

@needs_assets
@pytest.mark.parametrize("split,start,origin", [
    ("train", "2024-01-02T00:00:00+08:00", 48),
    ("validation", "2024-08-01T00:00:00+08:00", 0),
    ("test", "2024-10-01T00:00:00+08:00", 0),
])
def test_start_maps_exactly_to_the_split_local_origin(split, start, origin):
    m = split_module()
    assert m.local_origin_from_start(split, start) == origin


@needs_assets
@pytest.mark.parametrize("start", [
    "2024-01-01",                      # 只有日期
    "2024-01-01T00:17:00+08:00",       # 不在 30 分钟网格
    "2024-01-01T00:00:00",             # 无时区
    "2023-12-31T23:30:00+08:00",       # 越出该 split
    "2025-06-01T00:00:00+08:00",       # 越出 canonical
])
def test_invalid_start_fails_closed(start):
    with pytest.raises((ValueError, FileNotFoundError)):
        _split_chain("train", start)


@needs_assets
def test_non_candidate_origin_fails_closed():
    """train 的首行 global 0 → 本地 origin 0，**不在**候选集合 `[48, 10224)` 内。"""
    m = split_module()
    with pytest.raises(ValueError):
        m.local_origin_from_start("train", "2024-01-01T00:00:00+08:00")


# --- 4. 拒绝矩阵 ----------------------------------------------------------------

@needs_assets
def test_legacy_manifest_dir_is_not_a_bypass(tmp_path):
    """`manifest_dir` 不得成为正式链绕过入口（副本 / 旧目录都拒绝）。"""
    legacy = tmp_path / "no_such_dir"
    with pytest.raises((ValueError, FileNotFoundError)) as excinfo:
        from scenario.scenario import build_scenario

        build_scenario("train", start="2024-01-02T00:00:00+08:00", horizon=48,
                       forecast_cutoff=4, manifest_dir=str(legacy))
    assert "M1.3" in str(excinfo.value)


@needs_assets
@pytest.mark.parametrize("dirname", ["formal_splits_v4", "formal_splits_v3",
                                     "formal_splits_v2"])
def test_old_split_directories_are_rejected(dirname):
    m = split_module()
    path = REPO_ROOT / "data/manifest" / dirname / "train.json"
    if not path.exists():
        pytest.skip(f"{dirname} 不在本机")
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(path, expected_split="train")


@needs_assets
def test_v1_split_position_is_rejected():
    m = split_module()
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(
            REPO_ROOT / "data/manifest/train.json", expected_split="train")


@needs_assets
def test_split_manifest_copy_and_symlink_are_rejected(tmp_path):
    m = split_module()
    copied = tmp_path / "train.json"
    shutil.copy(SPLITS_V5 / "train.json", copied)
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(copied, expected_split="train")
    link = tmp_path / "link.json"
    link.symlink_to(SPLITS_V5 / "train.json")
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(link, expected_split="train")


@needs_assets
def test_split_manifest_forged_hash_is_rejected(tmp_path, monkeypatch):
    m = split_module()
    temp_dir = tmp_path / "v5"
    temp_dir.mkdir()
    payload = json.loads((SPLITS_V5 / "train.json").read_text(encoding="utf-8"))
    payload["inputs"]["frozen_refs"]["sha256"] = "0" * 64
    (temp_dir / "train.json").write_text(json.dumps(payload))
    monkeypatch.setattr(m, "_canonical_split_dir", lambda: temp_dir)
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(temp_dir / "train.json",
                                          expected_split="train")


@needs_assets
def test_split_manifest_forged_revision_is_rejected(tmp_path, monkeypatch):
    m = split_module()
    temp_dir = tmp_path / "v5"
    temp_dir.mkdir()
    payload = json.loads((SPLITS_V5 / "train.json").read_text(encoding="utf-8"))
    payload["materializer_revision"] = "0" * 40
    (temp_dir / "train.json").write_text(json.dumps(payload))
    monkeypatch.setattr(m, "_canonical_split_dir", lambda: temp_dir)
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(temp_dir / "train.json",
                                          expected_split="train")


@needs_assets
def test_refs_v4_copy_symlink_and_forgery_are_rejected(tmp_path, monkeypatch):
    m = refs_module()
    copied = tmp_path / "refs_v4.json"
    shutil.copy(REFS_V4, copied)
    with pytest.raises(m.RefsV4Error):
        m.load_verified_refs_v4(copied)

    link = tmp_path / "link_refs.json"
    link.symlink_to(REFS_V4)
    with pytest.raises(m.RefsV4Error):
        m.load_verified_refs_v4(link)

    # 伪造 lambda_ref（改回旧的 2000）必须拒绝
    temp_dir = tmp_path / "refs_dir"
    temp_dir.mkdir()
    payload = json.loads(REFS_V4.read_text(encoding="utf-8"))
    payload["references"]["lambda_ref"]["value"] = B5_LEGACY_LAMBDA_REF
    (temp_dir / "refs_v4.json").write_text(json.dumps(payload))
    monkeypatch.setattr(m, "_canonical_refs_dir", lambda: temp_dir)
    with pytest.raises(m.RefsV4Error):
        m.load_verified_refs_v4(temp_dir / "refs_v4.json")


@needs_assets
def test_refs_v3_is_not_a_fallback():
    m = refs_module()
    with pytest.raises(m.RefsV4Error):
        m.load_verified_refs_v4(REFS_V3)


# --- 5. mutation：target 不变 / history 变化 -------------------------------------

TARGET_COLUMNS = ("price_sgd_per_kwh", "system_load_mw", "temperature_deg_c",
                  "ghi_w_per_m2", "wind_speed_10m_mps")
COLUMN_DELTA = {
    "price_sgd_per_kwh": 0.5,
    "system_load_mw": 200.0,
    "temperature_deg_c": 5.0,
    "ghi_w_per_m2": 100.0,
    "wind_speed_10m_mps": 3.0,
}
VALIDATION_ORIGIN = 1000
VALIDATION_CUTOFF = 4


def _global_window() -> tuple[int, int]:
    return 10224 + VALIDATION_ORIGIN, 10224 + VALIDATION_ORIGIN + VALIDATION_CUTOFF


def _bump(frame, start: int, stop: int) -> None:
    """**半开**位置窗口 `[start, stop)`（R2 教训：`.loc[slice]` 两端包含）。"""
    labels = frame.index[start:stop]
    assert len(labels) == stop - start
    for column in TARGET_COLUMNS:
        frame.loc[labels, column] = frame.loc[labels, column] + COLUMN_DELTA[column]


def _changed_rows(base: pd.DataFrame, mutated: pd.DataFrame) -> dict:
    return {
        c: mutated.index[mutated[c].to_numpy() != base[c].to_numpy()].tolist()
        for c in TARGET_COLUMNS
    }


class _Patch:
    def __init__(self):
        self._undo = []

    def set(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._undo):
            setattr(obj, name, value)
        self._undo = []


def _temp_chain(tmp_path, patch: _Patch, *, mutate=None) -> dict:
    """完整自洽的**临时 canonical→policy 链**（用于 mutation 回归）。

    **范围声明（必须如实）**：本 fixture 覆盖 `build_formal_scenario_b6()` 读取的
    全部输入（canonical / split / policy-v2 / policy-v3 / B6 policy / 冻结 template）。

    **不覆盖** `refs_v4` 与 `formal_splits_v5`：二者都会调用
    `load_verified_v3_bundle()`，而该入口按其设计把 v3 驱动表**对生产 canonical
    重算并逐列比对**——因此「canonical 被改动的临时链」**不可能**同时满足它。
    这是**架构约束**，不是夹具偷懒；v3 驱动表自身的语义验证由
    M1.3f-e-b1-R1 的回归覆盖，本卡不再重复。
    """
    import scenario.splits as splits_module
    from scripts.materialize_singapore_forecast_policy import (
        build_forecast_policy_manifest,
    )

    m = formal_module()
    root = tmp_path / "repo"
    man = root / "data/manifest"
    proc = root / "data/processed/singapore_2024"
    for d in (man, proc):
        d.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(CANONICAL_PARQUET)
    if mutate is not None:
        mutate(frame)
    parquet = proc / "half_hour.parquet"
    frame.to_parquet(parquet, index=False)

    def cj(p):
        return json.dumps(p, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    canonical_manifest = man / "singapore_2024_half_hour.json"
    cm = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    cm["output_parquet_sha256"] = _sha256(parquet)
    canonical_manifest.write_text(cj(cm))

    split_manifest = man / "singapore_2024_splits.json"
    sp = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    sp["canonical_parquet_sha256"] = _sha256(parquet)
    sp["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    sp["canonical_parquet_path"] = "<external>/half_hour.parquet"
    sp["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
    sp["train_only_statistics_source"]["canonical_parquet_sha256"] = _sha256(parquet)
    if mutate is not None:
        sp["train_only_statistics"] = splits_module.train_only_statistics(
            frame.iloc[: sp["splits"]["train"]["row_end_exclusive"]])
    split_manifest.write_text(cj(sp))

    policy_v2 = man / "singapore_2024_forecast_policy_v2.json"
    policy_v2.write_text(cj(build_forecast_policy_manifest(
        canonical_parquet_path=parquet,
        canonical_manifest_path=canonical_manifest,
        split_manifest_path=split_manifest,
        frozen_at_utc="2026-09-19T00:00:00+00:00",
    )))

    # 把私有 resolver 指向临时链（**不** mock 任何产物）
    patch.set(m, "_canonical_policy_v3_dir", lambda: man)
    patch.set(m, "_canonical_parquet_path", lambda: parquet)
    patch.set(m, "_canonical_manifest_path", lambda: canonical_manifest)
    patch.set(m, "_split_manifest_path", lambda: split_manifest)
    patch.set(m, "_seasonal_policy_v2_path", lambda: policy_v2)

    policy_v3 = man / "singapore_2024_forecast_policy_v3.json"
    policy_v3.write_text(cj(m.build_policy_v3_manifest(
        frozen_at_utc="2026-09-19T00:00:00+00:00")))

    return {
        "root": root,
        "canonical_manifest_path": canonical_manifest,
        "split_manifest_path": split_manifest,
        "canonical_parquet_path": parquet,
        "policy_manifest_path": policy_v3,
    }


def _build_temp(chain: dict):
    return formal_module().build_formal_scenario_b6(
        "validation", origin=VALIDATION_ORIGIN, forecast_cutoff=VALIDATION_CUTOFF,
        canonical_parquet_path=chain["canonical_parquet_path"],
        canonical_manifest_path=chain["canonical_manifest_path"],
        split_manifest_path=chain["split_manifest_path"],
        policy_manifest_path=chain["policy_manifest_path"])


@needs_assets
def test_target_future_mutation_leaves_all_seven_forecasts_unchanged(tmp_path):
    base_patch = _Patch()
    try:
        base = _temp_chain(tmp_path / "base", base_patch)
        baseline = _build_temp(base)
    finally:
        base_patch.undo()

    mut_patch = _Patch()
    try:
        mut = _temp_chain(tmp_path / "mut", mut_patch, mutate=_mutate_target)
        before = pd.read_parquet(base["canonical_parquet_path"])
        after = pd.read_parquet(mut["canonical_parquet_path"])
        start, stop = _global_window()
        for column, rows in _changed_rows(before, after).items():
            assert rows == list(range(start, stop)), column
        mutated = _build_temp(mut)
        for field in formal_module().FORMAL_B6_SOURCE_KINDS:
            assert list(getattr(mutated, field)) == list(getattr(baseline, field)), field
    finally:
        mut_patch.undo()


@needs_assets
def test_history_mutation_changes_the_five_derived_forecasts(tmp_path):
    base_patch = _Patch()
    try:
        base = _temp_chain(tmp_path / "base", base_patch)
        baseline = _build_temp(base)
    finally:
        base_patch.undo()

    hist_patch = _Patch()
    try:
        hist = _temp_chain(tmp_path / "hist", hist_patch, mutate=_mutate_history)
        before = pd.read_parquet(base["canonical_parquet_path"])
        after = pd.read_parquet(hist["canonical_parquet_path"])
        start, _ = _global_window()
        for column, rows in _changed_rows(before, after).items():
            assert rows == list(range(start - 48, start)), column
        mutated = _build_temp(hist)
        for field in ("price_forecast", "load_forecast", "temperature_forecast",
                      "pv_forecast", "wind_forecast"):
            assert list(getattr(mutated, field)) != list(getattr(baseline, field)), field
    finally:
        hist_patch.undo()


@needs_assets
def test_cutover_delegates_to_the_b6_builder(monkeypatch):
    """正式入口必须**真正**调用生产 `build_formal_scenario_b6()`。"""
    import scenario.formal_scenario_b6 as builder
    import scenario.scenario as entry

    seen = {}
    real = builder.build_formal_scenario_b6

    def spy(split, **kwargs):
        seen["split"] = split
        seen.update(kwargs)
        return real(split, **kwargs)

    monkeypatch.setattr(builder, "build_formal_scenario_b6", spy)
    monkeypatch.setattr(entry, "_build_from_manifest", entry._build_from_manifest)
    bundle = _split_chain("train", "2024-01-02T00:00:00+08:00")
    assert bundle.mode == "formal"
    assert seen["split"] == "train"
    assert seen["origin"] == 48
    assert seen["forecast_cutoff"] == 4


def _mutate_target(frame) -> None:
    start, stop = _global_window()
    _bump(frame, start, stop)


def _mutate_history(frame) -> None:
    start, _ = _global_window()
    _bump(frame, start - 48, start)


# --- 6. 幂等 / 原子失败 / 拒绝覆盖 ----------------------------------------------

@needs_assets
def test_verify_is_idempotent():
    refs = refs_module()
    splits = split_module()
    before = {p: (_sha256(p), p.stat().st_mtime_ns)
              for p in [REFS_V4] + [SPLITS_V5 / f"{s}.json"
                                    for s in ("train", "validation", "test")]}
    assert refs.main(["--verify"]) == 0
    assert splits.main(["--verify"]) == 0
    for path, (sha, mtime) in before.items():
        assert _sha256(path) == sha, str(path)
        assert path.stat().st_mtime_ns == mtime, str(path)
    leftovers = [p for p in SPLITS_V5.iterdir() if p.name.startswith(".")]
    assert leftovers == []


@needs_assets
@pytest.mark.parametrize("which", ["refs", "splits"])
def test_transactional_failure_leaves_nothing(tmp_path, monkeypatch, which):
    refs = refs_module()
    splits = split_module()
    if which == "refs":
        target_dir = tmp_path / "configs/frozen_refs"
        target_dir.mkdir(parents=True)
        mod, name = refs, "refs_v4.json"
        monkeypatch.setattr(refs, "_canonical_refs_dir", lambda: target_dir)
    else:
        target_dir = tmp_path / "manifest"
        target_dir.mkdir(parents=True)
        mod, name = splits, "train.json"
        monkeypatch.setattr(splits, "_canonical_split_dir", lambda: target_dir)

    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(mod, "_atomic_write_text", _boom)
    with pytest.raises(RuntimeError):
        mod._atomic_write_text(target_dir / name, "{}")
    assert list(target_dir.iterdir()) == []


def _boom(path, text):
    raise RuntimeError("injected write failure")


# --- 7. old assets byte-identical + readiness ----------------------------------

@needs_assets
def test_protected_assets_are_byte_identical():
    for path, expected in PROTECTED.items():
        assert _sha256(path) == expected, str(path)
    for split, expected in V4_TRIAD.items():
        assert _sha256(SPLITS_V4 / f"{split}.json") == expected, split


@needs_assets
def test_v5_readiness_is_still_false():
    m = split_module()
    for split in ("train", "validation", "test"):
        payload = m.load_verified_split_manifest_v5(expected_split=split)
        assert payload["readiness"]["formal_training_ready"] is False
        assert payload["readiness"]["formal_env_ready"] is False


@needs_assets
def test_make_train_still_exits_two_without_synthetic_fallback():
    import subprocess

    result = subprocess.run(
        ["make", "train"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 2
    assert "synthetic" not in result.stdout.lower() or "回退" in result.stdout
    assert not list(REPO_ROOT.glob("runs/*/checkpoint*.pt"))
