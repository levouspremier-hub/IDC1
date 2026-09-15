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
STATISTIC_COLUMNS = importlib.import_module(SPLIT_MODULE).STATISTIC_COLUMNS
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


# --- 11. 第一轮返修：manifest 语义严格校验 ---------------------------------

def _materialize_and_get_manifest(fixture, out):
    result = _materialize(fixture, out)
    return pathlib.Path(result["manifest_path"])


def _tampered_manifest(path: pathlib.Path, mutation) -> pathlib.Path:
    good = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(mutation(json.loads(json.dumps(good)))), encoding="utf-8")
    return path


def _assert_clean_rejection(callable_, label):
    """畸形外部输入只允许 ValueError（含 SplitError）/ FileNotFoundError。"""
    with pytest.raises((ValueError, FileNotFoundError)) as excinfo:
        callable_()
    leaked = (KeyError, TypeError, AttributeError, IndexError)
    assert not isinstance(excinfo.value, leaked), (
        f"{label}: 泄漏了 {type(excinfo.value).__name__}: {excinfo.value}"
    )


MANIFEST_TAMPERINGS = {
    "year": lambda m: {**m, "year": 1999},
    "step_minutes": lambda m: {**m, "step_minutes": 60},
    "canonical_parquet_path": lambda m: {**m, "canonical_parquet_path": "data/raw/evil.parquet"},
    "canonical_manifest_path": lambda m: {**m, "canonical_manifest_path": "/etc/passwd"},
    "materializer_revision_type": lambda m: {**m, "materializer_revision": 12345},
    "materializer_revision_format": lambda m: {**m, "materializer_revision": "nope"},
    "no_overlap": lambda m: {**m, "no_overlap": False},
    "no_gap": lambda m: {**m, "no_gap": False},
    "randomized": lambda m: {**m, "randomized": True},
    "leap_day_split": lambda m: {**m, "leap_day_split": "test"},
    "episode_origin_rule": lambda m: {**m, "episode_origin_rule": "任意规则"},
    "forecast_origin_rule": lambda m: {**m, "forecast_origin_rule": "任意规则"},
    "readiness_forecast_ready": lambda m: {
        **m, "readiness": {**m["readiness"], "forecast_ready": True}},
    "readiness_training_ready": lambda m: {
        **m, "readiness": {**m["readiness"], "formal_training_ready": True}},
    "readiness_missing_key": lambda m: {
        **m, "readiness": {k: v for k, v in m["readiness"].items()
                           if k != "truth_splits_ready"}},
    "unavailable_empty": lambda m: {**m, "unavailable_not_materialized": {}},
    "unavailable_fake_available": lambda m: {
        **m, "unavailable_not_materialized": {
            k: {"status": "available"} for k in m["unavailable_not_materialized"]}},
    "unavailable_wrong_type": lambda m: {**m, "unavailable_not_materialized": ["a", "b"]},
    "train_stats_empty": lambda m: {**m, "train_only_statistics": {}},
    "train_stats_missing_column": lambda m: {
        **m, "train_only_statistics": {
            k: v for k, v in m["train_only_statistics"].items()
            if k != "price_sgd_per_kwh"}},
    "train_stats_forged_value": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"],
            "price_sgd_per_kwh": {
                **m["train_only_statistics"]["price_sgd_per_kwh"], "mean": 9999.0}}},
    "train_stats_bool_for_int": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"],
            "price_sgd_per_kwh": {
                **m["train_only_statistics"]["price_sgd_per_kwh"], "count": True}}},
    "train_stats_nan": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"],
            "price_sgd_per_kwh": {
                **m["train_only_statistics"]["price_sgd_per_kwh"], "min": float("nan")}}},
    "train_stats_source_split": lambda m: {
        **m, "train_only_statistics_source": {
            **m["train_only_statistics_source"], "split": "test"}},
    "train_stats_source_columns": lambda m: {
        **m, "train_only_statistics_source": {
            **m["train_only_statistics_source"], "columns": ["price_sgd_per_kwh"]}},
    "train_stats_source_hash": lambda m: {
        **m, "train_only_statistics_source": {
            **m["train_only_statistics_source"],
            "canonical_parquet_sha256": "0" * 64}},
    "train_stats_source_revision": lambda m: {
        **m, "train_only_statistics_source": {
            **m["train_only_statistics_source"],
            "statistics_implementation_revision": "bad"}},
}


@pytest.mark.parametrize("label", sorted(MANIFEST_TAMPERINGS))
def test_tampered_split_manifest_is_rejected_by_the_reader(tmp_path, label):
    """篡改后的冻结声明必须被 reader 拒绝 —— 不能只验证「生成时写对了」。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, MANIFEST_TAMPERINGS[label])
    _assert_clean_rejection(
        lambda: _load(fixture, manifest_path, "train"), label)


@pytest.mark.parametrize("label", sorted(MANIFEST_TAMPERINGS))
def test_tampered_split_manifest_is_rejected_by_the_materializer(tmp_path, label):
    """同一批篡改也必须让重新物化 fail closed，而不是静默接受既有声明。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, MANIFEST_TAMPERINGS[label])
    _assert_clean_rejection(
        lambda: _materialize(fixture, tmp_path / "out"), label)


def test_reader_rejects_a_declared_path_that_is_not_the_caller_supplied_one(tmp_path):
    """路径声明必须与调用者实际提供的 logical repo path 一致。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, lambda m: {
        **m,
        "canonical_parquet_path": "data/processed/other/half_hour.parquet",
        "canonical_manifest_path": "data/manifest/other.json",
    })
    _assert_clean_rejection(lambda: _load(fixture, manifest_path, "train"), "path-mismatch")


# --- 12. 第一轮返修：canonical 连续时间轴严格校验 ---------------------------

def _repack_canonical(fixture, out_root: pathlib.Path, mutate) -> dict:
    """写坏 canonical parquet，并**同步更新** canonical manifest 的 hash（自洽坏数据）。"""
    frame = pd.read_parquet(fixture["parquet"])
    frame = mutate(frame)
    out_root.mkdir(parents=True, exist_ok=True)
    parquet = out_root / "half_hour.parquet"
    frame.to_parquet(parquet, index=False)
    manifest = out_root / "singapore_2024_half_hour.json"
    good = json.loads(fixture["canonical_manifest"].read_text(encoding="utf-8"))
    good["output_parquet_sha256"] = hashlib.sha256(parquet.read_bytes()).hexdigest()
    good["row_count"] = len(frame)
    manifest.write_text(json.dumps(good), encoding="utf-8")
    return {"parquet": parquet, "canonical_manifest": manifest}


def _swap_rows(lo: int, hi: int):
    def mutate(frame):
        a = frame.iloc[lo].copy()
        frame.iloc[lo] = frame.iloc[hi].values
        frame.iloc[hi] = a.values
        return frame
    return mutate


def _shift_timestamp(row: int, minutes: int):
    def mutate(frame):
        frame.iloc[row, frame.columns.get_loc("timestamp")] = (
            frame.iloc[row]["timestamp"] + pd.Timedelta(minutes=minutes))
        return frame
    return mutate


def _copy_timestamp(src: int, dst: int):
    def mutate(frame):
        frame.iloc[dst, frame.columns.get_loc("timestamp")] = frame.iloc[src]["timestamp"]
        return frame
    return mutate


TIMELINE_CORRUPTIONS = {
    "train_internal_swap": _swap_rows(100, 101),
    "validation_internal_swap": _swap_rows(10224 + 10, 10224 + 11),
    "test_internal_swap": _swap_rows(14000, 14001),
    "duplicate_timestamp": _copy_timestamp(100, 101),
    "non_grid_17min": _shift_timestamp(100, 17),
    "non_grid_7min": _shift_timestamp(500, 7),
    "non_monotonic": _copy_timestamp(300, 200),
    "boundary_swap": _swap_rows(10223, 10224),
}


@pytest.mark.parametrize("label", sorted(TIMELINE_CORRUPTIONS))
def test_corrupt_canonical_timeline_is_rejected_by_the_materializer(tmp_path, label):
    """首末行与行数都对、但整表不连续的 canonical 必须被**物化器**拒绝。"""
    fixture = write_canonical_fixture(tmp_path)
    broken = _repack_canonical(fixture, tmp_path / "broken", TIMELINE_CORRUPTIONS[label])
    _assert_clean_rejection(
        lambda: _materialize(broken, tmp_path / "out"), label)


@pytest.mark.parametrize("label", sorted(TIMELINE_CORRUPTIONS))
def test_corrupt_canonical_timeline_is_rejected_by_the_reader(tmp_path, label):
    """reader 在返回切片前同样必须验证整条时间轴，不能只查首末行。"""
    fixture = write_canonical_fixture(tmp_path)
    broken = _repack_canonical(fixture, tmp_path / "broken", TIMELINE_CORRUPTIONS[label])
    manifest_path = _materialize(fixture, tmp_path / "out")["manifest_path"]
    # 把 split manifest 的两个 canonical hash 改成与坏 canonical 自洽
    good = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    good["canonical_parquet_sha256"] = hashlib.sha256(
        broken["parquet"].read_bytes()).hexdigest()
    good["canonical_manifest_sha256"] = hashlib.sha256(
        broken["canonical_manifest"].read_bytes()).hexdigest()
    good["train_only_statistics_source"]["canonical_parquet_sha256"] = \
        good["canonical_parquet_sha256"]
    pathlib.Path(manifest_path).write_text(json.dumps(good), encoding="utf-8")
    _assert_clean_rejection(lambda: _load(broken, manifest_path, "train"), label)


def test_intact_canonical_timeline_is_exactly_the_frozen_half_hour_grid(tmp_path):
    """正例：17,568 行、严格 30min 网格、首末时刻精确。"""
    fixture = write_canonical_fixture(tmp_path)
    frame = pd.read_parquet(fixture["parquet"])
    stamps = frame["timestamp"]
    assert len(stamps) == TOTAL_ROWS
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    deltas = stamps.diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta(minutes=30)
    assert stamps.iloc[0] == pd.Timestamp("2024-01-01T00:00:00+08:00")
    assert stamps.iloc[-1] == pd.Timestamp("2024-12-31T23:30:00+08:00")


# --- 13. 回归：不得回退既有语义 ---------------------------------------------

def test_boundaries_and_origin_rules_are_unchanged():
    module = importlib.import_module(SPLIT_MODULE)
    assert module.SPLIT_ROW_COUNTS == EXPECTED_ROW_COUNTS
    for name in SPLIT_NAMES:
        assert module.SPLIT_SPECS[name]["start"] == EXPECTED_STARTS[name]
        assert module.SPLIT_SPECS[name]["end_exclusive"] == EXPECTED_ENDS_EXCLUSIVE[name]
    end = EXPECTED_ROW_COUNTS["train"]
    assert _episode("train", end - 4, 4) == end - 4
    assert _forecast("train", end - 4, 4) == end - 4
    with pytest.raises(ValueError):
        _episode("train", end - 3, 4)
    with pytest.raises(ValueError):
        _forecast("train", end - 3, 4)


def test_upstream_manifests_and_raw_are_untouched_by_this_card():
    """本卡不得改动上游资产。"""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "data/raw", "data/manifest/singapore_2024.json",
         "data/manifest/singapore_2024_half_hour.json",
         "data/processed", "configs/frozen_refs/refs.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert status == "", f"上游资产被改动：{status}"


# --- 14. M1.3d-R2：外部字段的类型审计（不得泄漏异常） -----------------------

LEAKY_EXCEPTIONS = (KeyError, TypeError, AttributeError, IndexError)


def _assert_no_leak(excinfo, label):
    assert not isinstance(excinfo.value, LEAKY_EXCEPTIONS), (
        f"{label}: 泄漏了 {type(excinfo.value).__name__}: {excinfo.value}"
    )


def _reject(fixture, manifest_path, label, split="train"):
    """断言被拒绝，且**只**抛 ValueError（含 SplitError）/FileNotFoundError。"""
    with pytest.raises((ValueError, FileNotFoundError)) as excinfo:
        _load(fixture, manifest_path, split)
    _assert_no_leak(excinfo, label)


def _source_tamper(value):
    return lambda m: {
        **m,
        "train_only_statistics_source": {
            **m["train_only_statistics_source"], "columns": value},
    }


@pytest.mark.parametrize("label,value", [
    ("int", 5), ("bool_true", True), ("bool_false", False), ("float", 1.5),
    ("string", "abc"), ("dict", {"a": 1}), ("null", None),
    ("nested_list", [["price_sgd_per_kwh"]]),
])
def test_statistics_source_columns_scalar_types_are_rejected_cleanly(tmp_path, label, value):
    """`columns` 为标量/容器时必须是 ValueError 一族，**不得泄漏 TypeError**。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, _source_tamper(value))
    _reject(fixture, manifest_path, f"columns={label}")


@pytest.mark.parametrize("label,value", [
    ("truncated", ["price_sgd_per_kwh"]),
    ("reversed", list(reversed(list(STATISTIC_COLUMNS)))),
    ("extra", [*STATISTIC_COLUMNS, "extra"]),
    ("non_string_elements", [1, 2, 3, 4, 5, 6]),
    ("mixed_elements", [STATISTIC_COLUMNS[0], 2, 3, 4, 5, 6]),
])
def test_statistics_source_columns_must_be_exactly_the_frozen_list(tmp_path, label, value):
    """`columns` 必须是精确 `list[str]`，顺序与内容严格等于 `STATISTIC_COLUMNS`。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, _source_tamper(value))
    _reject(fixture, manifest_path, f"columns={label}")


@pytest.mark.parametrize("value", [
    tuple(STATISTIC_COLUMNS), "price_sgd_per_kwh", 5, True, 1.5, None, {"a": 1},
])
def test_str_list_validator_rejects_anything_but_a_plain_string_list(value):
    """直接测校验器：`str`、标量、`dict`、**`tuple`** 都不是合法的 `list[str]`。

    （`tuple` 无法经由 JSON manifest 送达 —— `json.dumps` 会把它变成 list ——
    故此处直接对校验器断言，而不是伪造一个到不了的输入。）
    """
    module = importlib.import_module(SPLIT_MODULE)
    with pytest.raises(ValueError) as excinfo:
        module._require_str_list(value, field="columns", expected=STATISTIC_COLUMNS)
    assert not isinstance(excinfo.value, LEAKY_EXCEPTIONS), type(excinfo.value).__name__


def test_statistics_source_columns_accepts_the_exact_frozen_list(tmp_path):
    """正例：精确等于冻结列表时必须通过。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    assert len(_load(fixture, manifest_path, "train")) == EXPECTED_ROW_COUNTS["train"]


@pytest.mark.parametrize("label,value", [
    ("list_with_keyword", ["unavailable"]),
    ("dict_with_keyword", {"status": "unavailable"}),
    ("list_of_dicts", [{"status": "unavailable", "reason": "x"}]),
    ("bare_int", 1), ("bare_bool", True), ("bare_null", None), ("bare_float", 1.5),
    ("empty_list", []), ("empty_dict", {}),
    ("string_without_keyword", "frozen and fine"),
])
def test_unavailable_entries_must_be_strings_labelled_unavailable(tmp_path, label, value):
    """每个 unavailable 条目必须是标注为 unavailable 的**字符串**；容器一律拒绝。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, lambda m, v=value: {
        **m, "unavailable_not_materialized": {
            k: v for k in m["unavailable_not_materialized"]}})
    _reject(fixture, manifest_path, f"unavailable={label}")


@pytest.mark.parametrize("label,value", [
    ("empty_string", ""),
    ("no_keyword", "frozen"),
    ("claims_available", "available: present"),
    ("claims_materialized", "materialized and ready"),
    ("dict_in_string", "{'status': 'available'}"),
])
def test_unavailable_strings_must_actually_say_unavailable(tmp_path, label, value):
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, lambda m, v=value: {
        **m, "unavailable_not_materialized": {
            k: v for k in m["unavailable_not_materialized"]}})
    _reject(fixture, manifest_path, f"unavailable-text={label}")


@pytest.mark.parametrize("label,value", [
    ("missing", "MISSING"), ("int", 12345), ("bool", True), ("null", None),
    ("list", ["2026-09-15T00:00:00+00:00"]),
    ("no_timezone", "2026-09-15T00:00:00"),
    ("not_iso", "yesterday"),
    ("date_only", "2026-09-15"),
    ("non_utc_offset", "2026-09-15T08:00:00+08:00"),
])
def test_frozen_at_utc_must_be_a_canonical_utc_timestamp(tmp_path, label, value):
    """`frozen_at_utc` 必须是规范 UTC（带 `+00:00`）；缺失/异构/非规范一律拒绝。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    if value == "MISSING":
        mutation = lambda m: {k: v for k, v in m.items() if k != "frozen_at_utc"}  # noqa: E731
    else:
        mutation = lambda m, v=value: {**m, "frozen_at_utc": v}  # noqa: E731
    _tampered_manifest(manifest_path, mutation)
    _reject(fixture, manifest_path, f"frozen_at_utc={label}")


def test_frozen_at_utc_accepts_the_canonical_form(tmp_path):
    """正例：`2026-09-15T00:00:00+00:00` 必须保持通过。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frozen_at_utc"] == "2026-09-15T00:00:00+00:00"
    assert len(_load(fixture, manifest_path, "train")) == EXPECTED_ROW_COUNTS["train"]


NESTED_FIELD_TAMPERS = {
    "splits_not_dict": lambda m: {**m, "splits": []},
    "split_entry_not_dict": lambda m: {**m, "splits": {**m["splits"], "train": "x"}},
    "split_entry_missing_key": lambda m: {
        **m, "splits": {**m["splits"], "train": {
            k: v for k, v in m["splits"]["train"].items() if k != "start"}}},
    "split_row_count_string": lambda m: {
        **m, "splits": {**m["splits"], "train": {
            **m["splits"]["train"], "row_count": "10224"}}},
    "split_row_count_bool": lambda m: {
        **m, "splits": {**m["splits"], "train": {
            **m["splits"]["train"], "row_count": True}}},
    "readiness_not_dict": lambda m: {**m, "readiness": "ready"},
    "readiness_extra_key": lambda m: {
        **m, "readiness": {**m["readiness"], "extra": True}},
    "unavailable_not_dict": lambda m: {**m, "unavailable_not_materialized": []},
    "statistics_not_dict": lambda m: {**m, "train_only_statistics": "x"},
    "statistics_entry_not_dict": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"], "price_sgd_per_kwh": 5}},
    "statistics_entry_missing_field": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"], "price_sgd_per_kwh": {
                k: v for k, v in m["train_only_statistics"]["price_sgd_per_kwh"].items()
                if k != "mean"}}},
    "statistics_mean_string": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"], "price_sgd_per_kwh": {
                **m["train_only_statistics"]["price_sgd_per_kwh"], "mean": "0.1"}}},
    "statistics_count_container": lambda m: {
        **m, "train_only_statistics": {
            **m["train_only_statistics"], "price_sgd_per_kwh": {
                **m["train_only_statistics"]["price_sgd_per_kwh"], "count": [1]}}},
    "statistics_source_not_dict": lambda m: {
        **m, "train_only_statistics_source": ["train"]},
    "statistics_source_missing_key": lambda m: {
        **m, "train_only_statistics_source": {
            k: v for k, v in m["train_only_statistics_source"].items()
            if k != "split"}},
    "materializer_revision_null": lambda m: {**m, "materializer_revision": None},
    "materializer_revision_list": lambda m: {**m, "materializer_revision": []},
    "canonical_hash_null": lambda m: {**m, "canonical_parquet_sha256": None},
    "canonical_hash_list": lambda m: {**m, "canonical_manifest_sha256": ["x"]},
    "manifest_top_level_list": lambda m: [],
    "manifest_top_level_string": lambda m: "nope",
    "year_list": lambda m: {**m, "year": [2024]},
    "no_overlap_string": lambda m: {**m, "no_overlap": "true"},
    "leap_day_split_list": lambda m: {**m, "leap_day_split": ["train"]},
    "origin_rule_null": lambda m: {**m, "episode_origin_rule": None},
}


@pytest.mark.parametrize("label", sorted(NESTED_FIELD_TAMPERS))
def test_nested_tampering_never_leaks_a_python_exception(tmp_path, label):
    """所有嵌套外部字段的畸形输入都必须干净 fail closed，不泄漏任何内建异常。"""
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, NESTED_FIELD_TAMPERS[label])
    _reject(fixture, manifest_path, label)


@pytest.mark.parametrize("label", sorted(NESTED_FIELD_TAMPERS))
def test_nested_tampering_is_also_rejected_by_the_materializer(tmp_path, label):
    fixture = write_canonical_fixture(tmp_path)
    manifest_path = _materialize_and_get_manifest(fixture, tmp_path / "out")
    _tampered_manifest(manifest_path, NESTED_FIELD_TAMPERS[label])
    _assert_clean_rejection(lambda: _materialize(fixture, tmp_path / "out"), label)
