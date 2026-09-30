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
from pathlib import Path

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
    # CHAIN-REFRESH（2026-09-24）：B6 正式资产链前向刷新的新冻结值；
    # 断言仍是**逐字节**相等，未放宽。
PROTECTED = {
    REFS_V3: "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f",
    REFS_V2: "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827",
    POLICY_V2: "fa1019f0fad905a9edcf663d947a46f741cecc91ace71bbac85f0d0874e4e72e",
    POLICY_V3: "23863ea44b3a882449f8930370b1e02182f5467e4c4d52acbfadaf675acfe754",
    B6_POLICY: "7066a0e127bc28f6a56ad4e62810c34536e1eb13b3c3c134baa3a1e6c4bca251",
    CANONICAL_PARQUET: "dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd",
    CANONICAL_MANIFEST: "e6484d6b050f100234a46811be30061f477bcc053b285854da5a882d2753b667",
    SPLIT_MANIFEST: "a096535fcdec81534f8cc05671d34d879a7e9517d06be789510dea586149af27",
}
V4_TRIAD = {
    "train": "e3bb8adb686cd567869eb19499c94d30425570a64c8fc8c9db63d925631c9e12",
    "validation": "e79c1987c69aeb0244416ad79ebd351b8cd9a16ea4175960ba228b3fca265984",
    "test": "67ada463a1a45b4052f5c399dc2a18b1f41f996d006debe1f365a60dbd99e197",
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
def test_real_refs_materializer_transaction_leaves_nothing(tmp_path, monkeypatch):
    """**真实** `materialize_refs_v4()`：注入原子写失败 → 无产物、零临时文件。"""
    refs = refs_module()
    target_dir = tmp_path / "frozen_refs"
    target_dir.mkdir(parents=True)
    monkeypatch.setattr(refs, "_canonical_refs_dir", lambda: target_dir)
    monkeypatch.setattr(refs, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(refs, "_atomic_write_text", _boom)

    with pytest.raises(RuntimeError):
        refs.materialize_refs_v4(frozen_at_utc="2026-09-20T00:00:00+00:00")

    assert not (target_dir / "refs_v4.json").exists()
    assert list(target_dir.iterdir()) == []


@needs_assets
@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_real_v5_triad_transaction_rolls_back_every_stage(
        tmp_path, monkeypatch, fail_at):
    """**真实** triad 事务：在**第 1/2/3 次**原子写入失败 → 三文件全不存在。"""
    splits = split_module()
    target_dir = tmp_path / "formal_splits_v5"
    target_dir.mkdir(parents=True)
    monkeypatch.setattr(splits, "_canonical_split_dir", lambda: target_dir)
    monkeypatch.setattr(splits, "_generator_is_dirty", lambda: False)

    calls = {"n": 0}
    real_write = splits._atomic_write_text

    def failing_write(path, text):
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise RuntimeError(f"injected failure at write #{fail_at}")
        real_write(path, text)

    monkeypatch.setattr(splits, "_atomic_write_text", failing_write)
    # 冻结时刻由 refs_v4 锚定（不得自由填写）
    with pytest.raises(RuntimeError):
        splits.materialize_split_manifest_triad_v5()

    for split in ("train", "validation", "test"):
        assert not (target_dir / f"{split}.json").exists(), split
    assert list(target_dir.iterdir()) == []


@needs_assets
def test_real_materializers_still_refuse_to_overwrite(tmp_path, monkeypatch):
    """「已存在且不同」仍拒绝覆盖，且**没有**新增公开 overwrite 参数。"""
    refs = refs_module()
    splits = split_module()
    refs_dir = tmp_path / "frozen_refs"
    refs_dir.mkdir()
    (refs_dir / "refs_v4.json").write_text('{"schema_version": "forged"}')
    monkeypatch.setattr(refs, "_canonical_refs_dir", lambda: refs_dir)
    monkeypatch.setattr(refs, "_generator_is_dirty", lambda: False)
    with pytest.raises(refs.RefsV4Error):
        refs.materialize_refs_v4(frozen_at_utc="2026-09-20T00:00:00+00:00")

    v5_dir = tmp_path / "formal_splits_v5"
    v5_dir.mkdir()
    (v5_dir / "train.json").write_text('{"schema": "forged"}')
    monkeypatch.setattr(splits, "_canonical_split_dir", lambda: v5_dir)
    monkeypatch.setattr(splits, "_generator_is_dirty", lambda: False)
    with pytest.raises(splits.SplitManifestV5Error):
        splits.materialize_split_manifest_triad_v5()

    import inspect

    for fn in (refs.materialize_refs_v4, splits.materialize_split_manifest_triad_v5):
        params = set(inspect.signature(fn).parameters)
        assert not ({"out_path", "out_dir", "overwrite", "replace",
                     "force", "manifest_path"} & params), (fn.__name__, params)


# --- R1-1. v5 语义伪造（改前 ACCEPTED，改后必须 REJECTED） ----------------------

def _forged_v5(tmp_path, monkeypatch, mutate) -> Path:
    m = split_module()
    temp_dir = tmp_path / "canonical_v5"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / "train.json"
    payload = json.loads((SPLITS_V5 / "train.json").read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                 ensure_ascii=False) + "\n")
    monkeypatch.setattr(m, "_canonical_split_dir", lambda: temp_dir)
    monkeypatch.setattr(m, "_generator_is_dirty", lambda: False)
    return target


@needs_assets
def test_untampered_v5_copy_is_accepted(tmp_path, monkeypatch):
    """**接受性对照**：未篡改的副本（经私有 resolver 视为 canonical）必须通过。"""
    m = split_module()
    assert m.load_verified_split_manifest_v5(
        _forged_v5(tmp_path, monkeypatch, lambda _p: None),
        expected_split="train")["split"] == "train"


@needs_assets
@pytest.mark.parametrize("mutate", [
    pytest.param(lambda p: p["time_range"].update(
        start="2024-01-02T00:00:00+08:00",
        end_exclusive="2024-08-02T00:00:00+08:00"), id="time_range"),
    pytest.param(lambda p: p.update(frozen_at_utc="2026-01-01T00:00:00+00:00"),
                 id="frozen_at_utc"),
    pytest.param(lambda p: p["split_rows"].update(count=10223), id="split_rows"),
    pytest.param(lambda p: p["candidate_origins"].update(start=49),
                 id="candidate_origins"),
    pytest.param(lambda p: p["readiness"].update(formal_env_ready=True),
                 id="readiness"),
    pytest.param(lambda p: p["inputs"]["canonical_parquet"].update(
        sha256="0" * 64), id="input_path_sha"),
    pytest.param(lambda p: p["inputs"]["frozen_refs"].update(
        path="configs/frozen_refs/refs_v3.json"), id="input_role_path"),
    pytest.param(lambda p: p.update(materializer_revision="0" * 40),
                 id="materializer_revision"),
    pytest.param(lambda p: p.update(extra=1), id="undeclared_field"),
    pytest.param(lambda p: p.update(history_steps=47), id="history_steps"),
    pytest.param(lambda p: p.update(contract_version="contract-v8"), id="contract"),
])
def test_forged_v5_semantics_are_rejected(tmp_path, monkeypatch, mutate):
    m = split_module()
    target = _forged_v5(tmp_path, monkeypatch, mutate)
    with pytest.raises(m.SplitManifestV5Error):
        m.load_verified_split_manifest_v5(target, expected_split="train")


@needs_assets
def test_v5_rebuild_mismatch_names_the_differing_field(tmp_path, monkeypatch):
    m = split_module()
    target = _forged_v5(tmp_path, monkeypatch, lambda p: p["time_range"].update(
        start="2024-01-02T00:00:00+08:00", end_exclusive="2024-08-02T00:00:00+08:00"))
    with pytest.raises(m.SplitManifestV5Error) as excinfo:
        m.load_verified_split_manifest_v5(target, expected_split="train")
    assert "time_range" in str(excinfo.value)


# --- R1-2. revision 覆盖 --------------------------------------------------------

REQUIRED_B6_PATHS = {
    "scenario/b6_refs.py",
    "scenario/b6_split_manifests.py",
    "scenario/scenario.py",
    "scenario/formal_scenario_b6.py",
    "scenario/arrival_intensity_policy.py",
    "scenario/exogenous_drivers_b6.py",
    "scenario/splits.py",
    "scripts/materialize_b6_refs.py",
    "scripts/materialize_b6_split_manifests.py",
}


@needs_assets
def test_b6_revision_paths_cover_every_formal_semantics_file():
    refs = refs_module()
    splits = split_module()
    assert REQUIRED_B6_PATHS - set(refs.B6_REFS_SOURCE_PATHS) == set()
    assert REQUIRED_B6_PATHS - set(splits.B6_SPLIT_SOURCE_PATHS) == set()


@needs_assets
@pytest.mark.parametrize("which", ["refs", "splits"])
def test_dirty_and_revision_use_identical_path_tuples(monkeypatch, which):
    mod = refs_module() if which == "refs" else split_module()
    seen: list[tuple] = []
    real_git = mod._git

    def spy(*args):
        seen.append(args)
        return real_git(*args)

    monkeypatch.setattr(mod, "_git", spy)
    mod.resolve_materializer_revision() if which == "splits" else mod.refs_code_revision()
    mod._generator_is_dirty()
    paths = tuple(getattr(mod, "B6_SPLIT_SOURCE_PATHS" if which == "splits"
                         else "B6_REFS_SOURCE_PATHS"))
    assert tuple(seen[0])[-len(paths):] == paths
    assert tuple(seen[1])[-len(paths):] == paths


@needs_assets
@pytest.mark.parametrize("which", ["refs", "splits"])
def test_materializer_or_cutover_change_invalidates_the_old_revision(
        tmp_path, monkeypatch, which):
    """改动 public cutover / materializer → 旧 revision **必须**被拒绝。"""
    mod = refs_module() if which == "refs" else split_module()
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(mod, "_git", lambda *a: "0" * 40)
    if which == "refs":
        with pytest.raises(mod.RefsV4Error):
            mod.load_verified_refs_v4()
    else:
        with pytest.raises(mod.SplitManifestV5Error):
            mod.load_verified_split_manifest_v5(expected_split="train")




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
def test_make_train_fails_closed_without_a_synthetic_fallback():
    """`make train` 必须 **fail closed**，且**不**回退 synthetic。

    **M1.3g-f-c-k 更新**：训练已由 `train release v1` 正式放行，`make train` 现进入
    **正式**入口 `safe_rl_v2.formal_train`；无参数时以 **usage 错误** exit 2
    （显式要求 `--run-id` / `--seed`），**仍不**跑任何合成 dry run。

    原第三条断言「`runs/` 下不得存在 `checkpoint*.pt`」在本卡后**不再成立**：
    正式入口的批次边界与最终 checkpoint 是**预期**产物。故改以「失败发生在做任何工作
    **之前**」为判据（stderr 报出缺失的必需参数），意图不变：fail closed、不回退合成。
    """
    import subprocess

    result = subprocess.run(
        ["make", "train"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 2
    assert "synthetic" not in result.stdout.lower() or "回退" in result.stdout
    combined = result.stdout + result.stderr
    assert "--run-id" in combined and "--seed" in combined, combined[-400:]


def _boom(path, text):
    """注入的原子写失败（真实事务测试用）。"""
    raise RuntimeError("injected write failure")
