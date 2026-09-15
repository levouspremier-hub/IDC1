"""M1.3d 测试：连续 truth split、origin 边界门禁与 train-only 统计。

本卡**只**冻结 truth 三段切分与 train-only 描述统计；**不生成 forecast**、
**不创建正式 ScenarioBundle**、**不创建 `train.json`**、**不开始训练**。

改前缺陷（本文件在实现前必须为红）：`scenario/splits.py` 与
`scripts/materialize_singapore_splits.py` 不存在，没有任何冻结 split。
"""

import hashlib
import importlib
import json
import pathlib
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIMEZONE = "Asia/Singapore"
TOTAL_ROWS = 17568
CANONICAL_SCHEMA = "m1.3b-singapore-2024-half-hour-v1"
SPLIT_SCHEMA = "m1.3d-singapore-2024-splits-v1"

SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_ROW_COUNTS = {"train": 10224, "validation": 2928, "test": 4416}
EXPECTED_STARTS = {
    "train": "2024-01-01T00:00:00+08:00",
    "validation": "2024-08-01T00:00:00+08:00",
    "test": "2024-10-01T00:00:00+08:00",
}
EXPECTED_ENDS_EXCLUSIVE = {
    "train": "2024-08-01T00:00:00+08:00",
    "validation": "2024-10-01T00:00:00+08:00",
    "test": "2025-01-01T00:00:00+08:00",
}
CANONICAL_COLUMNS = (
    "timestamp", "price_sgd_per_kwh", "system_load_mw",
    "national_igs_mwh_per_half_hour", "temperature_deg_c",
    "wind_speed_10m_mps", "ghi_w_per_m2", "weather_source_timestamp",
    "weather_age_minutes",
)
UNAVAILABLE = ("local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival")
SPLIT_MODULE = "scenario.splits"
MATERIALIZER = "scripts.materialize_singapore_splits"


# --- fixture ---------------------------------------------------------------

def _canonical_frame(rows: int = TOTAL_ROWS, *, igs_offset: float = 0.0) -> pd.DataFrame:
    """构造与 M1.3b canonical 同形的确定性表（含 signed IGS 与负电价）。"""
    start = datetime(2024, 1, 1)
    stamps = [start + timedelta(minutes=30 * i) for i in range(rows)]
    index = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(stamps).tz_localize(TIMEZONE),
        "price_sgd_per_kwh": np.where(index % 7 == 0, -0.02, 0.12 + 0.001 * index),
        "system_load_mw": 6000.0 + index,
        "national_igs_mwh_per_half_hour": np.where(index % 3 == 0, -0.05, 200.0) + igs_offset,
        "temperature_deg_c": 28.0 + 0.01 * index,
        "wind_speed_10m_mps": 2.0 + 0.001 * index,
        "ghi_w_per_m2": np.clip(200.0 - index, 0.0, None),
        "weather_source_timestamp": pd.to_datetime(stamps).tz_localize(TIMEZONE),
        "weather_age_minutes": np.where(index % 2 == 0, 0, 30),
    })


def write_canonical_fixture(root: pathlib.Path, *, igs_offset: float = 0.0) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "half_hour.parquet"
    _canonical_frame(igs_offset=igs_offset).to_parquet(parquet, index=False)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    manifest = root / "singapore_2024_half_hour.json"
    manifest.write_text(json.dumps({
        "schema": CANONICAL_SCHEMA,
        "year": 2024,
        "timezone": TIMEZONE,
        "row_count": TOTAL_ROWS,
        "frequency": "30min",
        "output_parquet_sha256": digest,
        "materializer_revision": "0" * 39 + "1",
    }, indent=2), encoding="utf-8")
    return {"parquet": parquet, "canonical_manifest": manifest, "sha256": digest}


def _materialize(fixture, out: pathlib.Path, **over):
    module = importlib.import_module(MATERIALIZER)
    kwargs = dict(
        canonical_parquet_path=fixture["parquet"],
        canonical_manifest_path=fixture["canonical_manifest"],
        manifest_path=out / "singapore_2024_splits.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    kwargs.update(over)
    return module.materialize_splits(**kwargs)


def _snap(path: pathlib.Path) -> tuple:
    st = path.stat()
    return (st.st_size, hashlib.sha256(path.read_bytes()).hexdigest(), st.st_mtime_ns)


def _load(fixture, split_manifest, split):
    module = importlib.import_module(SPLIT_MODULE)
    return module.load_truth_split(
        split,
        canonical_parquet_path=fixture["parquet"],
        canonical_manifest_path=fixture["canonical_manifest"],
        split_manifest_path=split_manifest,
    )


# --- 1. split 冻结形状 ------------------------------------------------------

def test_module_is_importable():
    importlib.import_module(SPLIT_MODULE)
    importlib.import_module(MATERIALIZER)


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_frozen_row_counts(tmp_path, split):
    module = importlib.import_module(SPLIT_MODULE)
    assert module.SPLIT_ROW_COUNTS[split] == EXPECTED_ROW_COUNTS[split]
    assert module.SPLIT_ROW_STARTS[split] == sum(
        EXPECTED_ROW_COUNTS[n] for n in SPLIT_NAMES[:SPLIT_NAMES.index(split)]
    )


def test_split_row_counts_sum_to_the_canonical_total():
    module = importlib.import_module(SPLIT_MODULE)
    assert sum(module.SPLIT_ROW_COUNTS.values()) == TOTAL_ROWS


def test_splits_have_no_overlap_and_no_gap():
    module = importlib.import_module(SPLIT_MODULE)
    cursor = 0
    for name in SPLIT_NAMES:
        spec = module.SPLIT_SPECS[name]
        assert spec["row_start"] == cursor, f"{name} 起点不连续"
        cursor = spec["row_end_exclusive"]
        assert spec["row_count"] == spec["row_end_exclusive"] - spec["row_start"]
    assert cursor == TOTAL_ROWS


@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_half_open_boundaries_are_exact(tmp_path, split):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    entry = manifest["splits"][split]
    assert entry["start"] == EXPECTED_STARTS[split]
    assert entry["end_exclusive"] == EXPECTED_ENDS_EXCLUSIVE[split]
    assert entry["row_count"] == EXPECTED_ROW_COUNTS[split]


def test_leap_day_belongs_only_to_train(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["leap_day_split"] == "train"
    for split in SPLIT_NAMES:
        frame = _load(fixture, result["manifest_path"], split)
        has_leap = ((frame["timestamp"].dt.month == 2) & (frame["timestamp"].dt.day == 29)).any()
        assert bool(has_leap) == (split == "train"), \
            f"Feb 29 只应属于 train，实际出现在 {split}"


def test_split_reader_returns_a_copy_with_canonical_columns(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    frame = _load(fixture, result["manifest_path"], "train")
    assert tuple(frame.columns) == CANONICAL_COLUMNS
    assert len(frame) == EXPECTED_ROW_COUNTS["train"]
    frame.loc[frame.index[0], "price_sgd_per_kwh"] = 12345.0
    again = _load(fixture, result["manifest_path"], "train")
    assert again.loc[again.index[0], "price_sgd_per_kwh"] != 12345.0, "必须返回副本"


@pytest.mark.parametrize("bad", ("Train", "valid", "", "test ", "all", 0, None))
def test_invalid_split_names_are_rejected(tmp_path, bad):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    with pytest.raises((ValueError, KeyError, TypeError)):
        _load(fixture, result["manifest_path"], bad)


# --- 2. origin 边界门禁 -----------------------------------------------------

def _episode(split, origin, horizon):
    return importlib.import_module(SPLIT_MODULE).validate_episode_origin(
        split, origin, horizon
    )


def _forecast(split, origin, cutoff):
    return importlib.import_module(SPLIT_MODULE).validate_forecast_origin(
        split, origin, cutoff
    )


def test_episode_origin_last_legal_and_first_illegal():
    end = EXPECTED_ROW_COUNTS["train"]
    assert _episode("train", end - 4, 4) == end - 4
    with pytest.raises(ValueError):
        _episode("train", end - 3, 4)


def test_forecast_origin_last_legal_and_first_illegal():
    end = EXPECTED_ROW_COUNTS["train"]
    assert _forecast("train", end - 4, 4) == end - 4
    with pytest.raises(ValueError):
        _forecast("train", end - 3, 4)


def test_forecast_origin_may_reach_the_boundary_exactly():
    """`origin + C == row_end_exclusive` 合法（半开区间，最后可见点仍在段内）。

    `origin` 是**该 split 内**从 0 开始的 half-hour step；返回值是**全局**行号。
    """
    local_end = EXPECTED_ROW_COUNTS["validation"]
    global_start = EXPECTED_ROW_COUNTS["train"]
    assert _forecast("validation", local_end - 6, 6) == global_start + local_end - 6
    with pytest.raises(ValueError):
        _forecast("validation", local_end - 5, 6)


def test_h_ge_c_does_not_double_purge():
    """`H >= C` 时**不得**额外执行 `H + C` 扣除。

    若实现错误地要求 `origin + H + C <= end`，则本用例的第二行会失败。
    """
    end = EXPECTED_ROW_COUNTS["train"]
    origin = end - 8
    assert _episode("train", origin, 8) == origin           # H=8
    assert _forecast("train", origin, 4) == origin          # C=4 < H，不应再扣 4
    assert _forecast("train", origin, 8) == origin          # H == C
    with pytest.raises(ValueError):
        _episode("train", origin, 9)


@pytest.mark.parametrize("bad", (True, False, 0, -1, 1.5, "4", None, [4]))
def test_horizon_and_cutoff_must_be_strict_positive_integers(bad):
    with pytest.raises((ValueError, TypeError)):
        _episode("train", 100, bad)
    with pytest.raises((ValueError, TypeError)):
        _forecast("train", 100, bad)


def test_origin_must_belong_to_the_named_split():
    """origin 只在该 split 内有效；不得自动换段、不得截断。

    `origin` 是 split-**本地** step：train 的合法本地 step 上界是 10223，
    把它当作 validation 的本地 step 同样越界 —— 两个方向都必须报错，
    绝不能「自动换到另一个 split」。
    """
    train_last_local = EXPECTED_ROW_COUNTS["train"] - 1
    assert _episode("train", train_last_local, 1) == train_last_local
    with pytest.raises(ValueError):
        _episode("train", EXPECTED_ROW_COUNTS["train"], 1)
    with pytest.raises(ValueError):
        _episode("validation", train_last_local, 1)     # 不是 validation 的本地 step
    assert _episode("validation", 0, 1) == EXPECTED_ROW_COUNTS["train"]


def test_case_does_not_change_delta_t():
    """本卡不改变环境的步长定义（接线留给 M1.3g）。"""
    module = importlib.import_module(SPLIT_MODULE)
    assert module.STEP_MINUTES == 30


# --- 3. hash 与 manifest 校验 ----------------------------------------------

def test_canonical_parquet_hash_mismatch_fails(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    fixture["parquet"].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        _load(fixture, result["manifest_path"], "train")


def test_canonical_manifest_hash_mismatch_fails(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    split_manifest = json.loads(
        pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    split_manifest["canonical_manifest_sha256"] = "0" * 64
    pathlib.Path(result["manifest_path"]).write_text(
        json.dumps(split_manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        _load(fixture, result["manifest_path"], "train")


@pytest.mark.parametrize("mutation,label", [
    (lambda m: [], "顶层不是 object"),
    (lambda m: {k: v for k, v in m.items() if k != "splits"}, "缺 splits"),
    (lambda m: {**m, "total_rows": "17568"}, "total_rows 不是整数"),
    (lambda m: {**m, "total_rows": True}, "total_rows 是 bool"),
    (lambda m: {**m, "frequency": "60min"}, "frequency 错误"),
    (lambda m: {**m, "timezone": "UTC"}, "timezone 错误"),
    (lambda m: {k: v for k, v in m.items() if k != "canonical_parquet_sha256"},
     "缺 canonical_parquet_sha256"),
])
def test_malformed_split_manifest_fails_closed(tmp_path, mutation, label):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    path = pathlib.Path(result["manifest_path"])
    good = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(mutation(good)), encoding="utf-8")
    with pytest.raises((ValueError, TypeError, KeyError)):
        _load(fixture, path, "train")


def test_split_manifest_paths_are_repo_relative(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    text = pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8")
    assert str(REPO_ROOT) not in text
    assert "/Users/" not in text
    assert str(pathlib.Path.home()) not in text


# --- 4. train-only 统计 -----------------------------------------------------

def _stats(manifest):
    return manifest["train_only_statistics"]


def test_train_only_statistics_cover_the_available_columns(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    stats = _stats(manifest)
    for column in ("price_sgd_per_kwh", "system_load_mw",
                   "national_igs_mwh_per_half_hour", "temperature_deg_c",
                   "wind_speed_10m_mps", "ghi_w_per_m2"):
        assert column in stats, column
        for key in ("count", "finite_count", "min", "max", "negative_count", "mean"):
            assert key in stats[column], (column, key)
    assert stats["price_sgd_per_kwh"]["count"] == EXPECTED_ROW_COUNTS["train"]
    assert manifest["train_only_statistics_source"]["split"] == "train"


def test_validation_and_test_values_do_not_change_train_statistics(tmp_path):
    base = write_canonical_fixture(tmp_path / "a")
    out_a = tmp_path / "out_a"
    result_a = _materialize(base, out_a)
    stats_a = _stats(json.loads(
        pathlib.Path(result_a["manifest_path"]).read_text(encoding="utf-8")))

    # 只改 validation/test 区间的 IGS（row >= 10224）；**train 行必须逐字节不变**
    shifted = write_canonical_fixture(tmp_path / "b")
    out_b = tmp_path / "out_b"
    module = importlib.import_module(MATERIALIZER)
    frame = pd.read_parquet(shifted["parquet"])
    frame.loc[frame.index >= EXPECTED_ROW_COUNTS["train"],
              "national_igs_mwh_per_half_hour"] += 5000.0
    base_frame = pd.read_parquet(write_canonical_fixture(tmp_path / "a")["parquet"])
    assert frame.iloc[:EXPECTED_ROW_COUNTS["train"]].equals(
        base_frame.iloc[:EXPECTED_ROW_COUNTS["train"]]
    ), "本用例的 train 行必须与基准完全一致"
    frame.to_parquet(shifted["parquet"], index=False)
    shifted["canonical_manifest"].write_text(json.dumps({
        "schema": CANONICAL_SCHEMA, "year": 2024, "timezone": TIMEZONE,
        "row_count": TOTAL_ROWS, "frequency": "30min",
        "output_parquet_sha256": hashlib.sha256(shifted["parquet"].read_bytes()).hexdigest(),
        "materializer_revision": "0" * 39 + "1",
    }, indent=2), encoding="utf-8")
    assert module is not None
    result_b = _materialize(shifted, out_b)
    stats_b = _stats(json.loads(
        pathlib.Path(result_b["manifest_path"]).read_text(encoding="utf-8")))

    assert stats_b["national_igs_mwh_per_half_hour"] == \
        stats_a["national_igs_mwh_per_half_hour"], "validation/test 变更不得影响 train 统计"


def test_train_mutation_does_change_train_statistics(tmp_path):
    base = write_canonical_fixture(tmp_path / "a")
    result_a = _materialize(base, tmp_path / "out_a")
    stats_a = _stats(json.loads(
        pathlib.Path(result_a["manifest_path"]).read_text(encoding="utf-8")))

    shifted = write_canonical_fixture(tmp_path / "b")
    frame = pd.read_parquet(shifted["parquet"])
    frame.loc[frame.index < EXPECTED_ROW_COUNTS["train"],
              "national_igs_mwh_per_half_hour"] += 5000.0
    frame.to_parquet(shifted["parquet"], index=False)
    shifted["canonical_manifest"].write_text(json.dumps({
        "schema": CANONICAL_SCHEMA, "year": 2024, "timezone": TIMEZONE,
        "row_count": TOTAL_ROWS, "frequency": "30min",
        "output_parquet_sha256": hashlib.sha256(shifted["parquet"].read_bytes()).hexdigest(),
        "materializer_revision": "0" * 39 + "1",
    }, indent=2), encoding="utf-8")
    result_b = _materialize(shifted, tmp_path / "out_b")
    stats_b = _stats(json.loads(
        pathlib.Path(result_b["manifest_path"]).read_text(encoding="utf-8")))
    assert stats_b["national_igs_mwh_per_half_hour"] != \
        stats_a["national_igs_mwh_per_half_hour"], "train 变更必须影响 train 统计"


def test_signed_igs_and_negative_price_are_preserved(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    stats = _stats(manifest)
    assert stats["national_igs_mwh_per_half_hour"]["negative_count"] > 0
    assert stats["national_igs_mwh_per_half_hour"]["min"] < 0
    assert stats["price_sgd_per_kwh"]["negative_count"] > 0
    assert stats["price_sgd_per_kwh"]["min"] < 0


def test_unavailable_columns_stay_unavailable_and_are_not_invented(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    unavailable = manifest["unavailable_not_materialized"]
    assert set(unavailable) == set(UNAVAILABLE)
    for column in UNAVAILABLE:
        assert column not in _stats(manifest), f"不得为 {column} 生成统计"
    for split in SPLIT_NAMES:
        frame = _load(fixture, result["manifest_path"], split)
        for column in UNAVAILABLE:
            assert column not in frame.columns, f"不得伪造 {column}"


def test_readiness_flags_are_honest(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))
    readiness = manifest["readiness"]
    assert readiness["truth_splits_ready"] is True
    assert readiness["forecast_ready"] is False
    assert readiness["formal_scenario_bundle_ready"] is False
    assert readiness["formal_training_ready"] is False


def test_no_forecast_scenario_bundle_or_train_manifest_is_created(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    _materialize(fixture, out)
    created = sorted(p.name for p in out.iterdir())
    assert created == ["singapore_2024_splits.json"], created
    for forbidden in ("train.json", "validation.json", "test.json"):
        assert not (out / forbidden).exists()
        assert not (REPO_ROOT / "data/manifest" / forbidden).exists()


# --- 5. 物化的 revision / 幂等 / 原子性 --------------------------------------

@pytest.fixture(autouse=True)
def _assume_clean_generator(monkeypatch, request):
    """默认假定生成实现文件干净（真实工作树在开发期可能带未提交修改）。"""
    if request.node.name == "test_dirty_generator_is_rejected":
        return
    try:
        module = importlib.import_module(MATERIALIZER)
    except ImportError:
        return  # 模块尚不存在：让测试本身按预期失败，而不是让夹具报错
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)


def test_revision_is_git_verified_and_points_at_the_split_generator():
    module = importlib.import_module(MATERIALIZER)
    revision = module.resolve_split_materializer_revision()
    assert isinstance(revision, str) and len(revision) == 40
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *module.SPLIT_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert revision == expected


def test_dirty_generator_is_rejected(tmp_path, monkeypatch):
    module = importlib.import_module(MATERIALIZER)
    fixture = write_canonical_fixture(tmp_path)
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        _materialize(fixture, tmp_path / "out")
    assert not (tmp_path / "out" / "singapore_2024_splits.json").exists()


def test_identical_rematerialization_does_not_rewrite(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    _materialize(fixture, out)
    manifest = out / "singapore_2024_splits.json"
    before = _snap(manifest)
    _materialize(fixture, out)
    assert _snap(manifest) == before, "同输入重跑不得改写 manifest"


def test_first_write_failure_leaves_no_half_state(tmp_path, monkeypatch):
    module = importlib.import_module(MATERIALIZER)
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"

    def boom(path, text):
        raise OSError("injected split manifest install failure")

    monkeypatch.setattr(module, "_atomic_write_text", boom)
    with pytest.raises(OSError):
        _materialize(fixture, out)
    assert not (out / "singapore_2024_splits.json").exists()
    assert [p.name for p in out.iterdir()] == [], "不得留下临时文件或半成品"


def test_existing_different_manifest_is_not_silently_overwritten(tmp_path):
    fixture = write_canonical_fixture(tmp_path)
    out = tmp_path / "out"
    _materialize(fixture, out)
    manifest = out / "singapore_2024_splits.json"
    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["total_rows"] = 999
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    before = _snap(manifest)
    with pytest.raises(ValueError):
        _materialize(fixture, out)
    assert _snap(manifest) == before


# --- 6. 真实 canonical 的 slow 验收 -----------------------------------------

@pytest.mark.slow
def test_real_canonical_splits_are_consistent(tmp_path):
    canonical_parquet = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
    canonical_manifest = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
    if not canonical_parquet.exists() or not canonical_manifest.exists():
        pytest.skip("canonical 产物不在本机")
    out = tmp_path / "out"
    result = _materialize({
        "parquet": canonical_parquet,
        "canonical_manifest": canonical_manifest,
    }, out)
    module = importlib.import_module(SPLIT_MODULE)
    frames = {
        split: module.load_truth_split(
            split,
            canonical_parquet_path=canonical_parquet,
            canonical_manifest_path=canonical_manifest,
            split_manifest_path=result["manifest_path"],
        )
        for split in SPLIT_NAMES
    }
    assert sum(len(f) for f in frames.values()) == TOTAL_ROWS
    for split, frame in frames.items():
        assert len(frame) == EXPECTED_ROW_COUNTS[split]
        assert str(frame["timestamp"].dt.tz) == TIMEZONE
    assert (frames["train"]["timestamp"].max()
            < frames["validation"]["timestamp"].min())
    assert (frames["validation"]["timestamp"].max()
            < frames["test"]["timestamp"].min())
