"""M1.3g-e-b：B6 arrival-to-Task **确定性 mapper** 的**先红**回归。

改前缺陷（本文件在实现前必须为红）：

- `scenario.arrival_mapper` 模块尚不存在（`ModuleNotFoundError`）；
- `scripts.materialize_b6_arrival_mapper` 物化器尚不存在；
- `data/manifest/m13g_arrival_mapper_v1.json` 尚不存在。

**本卡只实现 mapper**：不接 env、不改 `step()`、不训练、不评估、不改 readiness。"""

import hashlib
import importlib
import json
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAPPER_MODULE = "scenario.arrival_mapper"
MATERIALIZER_MODULE = "scripts.materialize_b6_arrival_mapper"
MANIFEST = REPO_ROOT / "data/manifest/m13g_arrival_mapper_v1.json"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
POLICY_V3 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json"
B6_POLICY = REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json"
V5_DIR = REPO_ROOT / "data/manifest/formal_splits_v5"
V4_DIR = REPO_ROOT / "data/manifest/formal_splits_v4"
V3_DIR = REPO_ROOT / "data/manifest/formal_splits_v3"
REFS_V4 = REPO_ROOT / "configs/frozen_refs/refs_v4.json"
REFS_V3 = REPO_ROOT / "configs/frozen_refs/refs_v3.json"

WORK_UNIT_SCALE = 1_000_000
E_W_MIN = 9.712937937508368
E_W_MAX = 60.705862109427294
TASK_FIELDS = (
    "task_id", "profile_key", "name", "arrival_time", "duration",
    "load_profile", "workload", "deadline", "priority",
    "interruptible", "parallelizable",
)

PROTECTED = {
    B6_POLICY: "7066a0e127bc28f6a56ad4e62810c34536e1eb13b3c3c134baa3a1e6c4bca251",
    POLICY_V3: "926703337139143dc1ea5223739ca5408be5ed384124a1acf88c70a39c542ecb",
    REFS_V4: "b5b64ef28224b53734ef186aff67687ff0db2dbee3d7e83dbed251af864e3ea7",
    CANONICAL_PARQUET: "dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd",
    CANONICAL_MANIFEST: "e6484d6b050f100234a46811be30061f477bcc053b285854da5a882d2753b667",
    SPLIT_MANIFEST: "a096535fcdec81534f8cc05671d34d879a7e9517d06be789510dea586149af27",
}


def mapper():
    return importlib.import_module(MAPPER_MODULE)


def materializer():
    return importlib.import_module(MATERIALIZER_MODULE)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _upstream_present() -> bool:
    return all(p.exists() for p in (
        CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST, POLICY_V3,
        B6_POLICY, REFS_V4, V5_DIR / "train.json",
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机")


# --- 1. manifest 存在性 + E 定义 -------------------------------------------------

@needs_assets
def test_mapper_manifest_exists_and_freezes_the_approved_parameters():
    m = mapper()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == m.MAPPER_MANIFEST_SCHEMA
    assert payload["work_unit_scale"] == WORK_UNIT_SCALE
    assert payload["delta_t_hours"] == 0.5
    assert payload["abs_tol_work"] == 1e-9
    assert payload["rel_tol_work"] == 1e-12
    assert payload["max_tasks_per_slot"] == 4
    assert payload["max_growth_rounds"] == 4
    assert payload["profile_order"] == [
        "E_micro_inference", "A_inference", "B_rl_training",
        "C_dl_training", "D_preprocess",
    ]
    prof = payload["profiles"]["E_micro_inference"]
    assert prof["duration_steps"] == 1
    assert prof["deadline_steps"] == 2
    assert prof["load_range"] == [0.04, 0.25]
    assert prof["priority_range"] == [2.6, 3.4]
    assert prof["interruptible"] is False
    assert prof["parallelizable"] is False
    assert prof["source"] == "modeled_scenario"


@needs_assets
def test_manifest_records_live_source_hashes():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = payload["sources"]
    for role, path in (
        ("b6_intensity_policy", B6_POLICY),
        ("forecast_policy_v3", POLICY_V3),
        ("frozen_refs_v4", REFS_V4),
        ("formal_split_v5_train", V5_DIR / "train.json"),
    ):
        assert sources[role]["sha256"] == _sha256(path), role
    assert len(payload["source_revision"]) == 40


# --- 2. 守恒 / 字段 / 边界 -------------------------------------------------------

def _stream(split="train", start="2024-01-02T00:00:00+08:00", horizon=48, seed=7):
    return mapper().build_arrival_task_stream(
        split, start=start, horizon=horizon, seed=seed)


@needs_assets
@pytest.mark.parametrize("split,start", [
    ("train", "2024-01-02T00:00:00+08:00"),
    ("train", "2024-04-16T12:00:00+08:00"),
    ("validation", "2024-08-01T00:00:00+08:00"),
    ("test", "2024-10-01T00:00:00+08:00"),
])
def test_slot_ledger_is_exactly_one_to_one(split, start):
    stream = _stream(split, start)
    agg = mapper().verified_realized_aggregate()
    total = 0
    for slot in stream.slots:
        micro = sum(slot.ledger)
        assert micro == slot.aggregate_micro, slot.slot_index
        assert micro % WORK_UNIT_SCALE == 0
        assert micro // WORK_UNIT_SCALE == int(agg[slot.slot_index])
        total += micro
    assert total == sum(s.aggregate_micro for s in stream.slots)
    assert stream.content_hash == mapper().canonical_content_hash(stream)
    assert len(stream.content_hash) == 64


@needs_assets
def test_every_task_has_the_eleven_fields_and_is_within_bounds():
    for slot in _stream().slots:
        for task in slot.tasks:
            for field in TASK_FIELDS:
                assert getattr(task, field) is not None, field
            assert task.profile_key == "E_micro_inference"
            assert task.name == "E_micro_inference"
            assert task.workload > 0.0
            assert E_W_MIN <= task.workload <= E_W_MAX, task.workload
            assert task.duration == 1
            assert task.deadline == 2
            assert task.deadline >= task.duration
            assert task.interruptible is False
            assert task.parallelizable is False
            assert len(task.load_profile) == task.duration
            assert len(set(np.round(task.load_profile, 15))) == 1  # 平坦
            assert all(0.04 <= v <= 0.25 for v in task.load_profile)
            assert 2.6 <= task.priority <= 3.4


@needs_assets
def test_flat_load_profile_reproduces_the_workload_within_tolerance():
    m = mapper()
    C = m.c_idc_base_work_per_hour()
    d = 0.5
    for slot in _stream().slots:
        for task in slot.tasks:
            rebuilt = float(np.sum(task.load_profile) * C * d)
            diff = abs(rebuilt - task.workload)
            assert diff <= max(1e-9, 1e-12 * abs(task.workload)), (diff, task.workload)


@needs_assets
def test_priority_is_deterministic_from_the_episode_seed():
    a = _stream(seed=7)
    b = _stream(seed=7)
    c = _stream(seed=8)
    pa = [t.priority for s in a.slots for t in s.tasks]
    pb = [t.priority for s in b.slots for t in s.tasks]
    pc = [t.priority for s in c.slots for t in s.tasks]
    assert pa == pb
    assert pa != pc


@needs_assets
def test_task_ids_are_stable_unique_and_not_python_hash():
    ids = [t.task_id for s in _stream().slots for t in s.tasks]
    assert len(ids) == len(set(ids))
    again = [t.task_id for s in _stream().slots for t in s.tasks]
    assert ids == again
    # 绝不等于 Python 内置 hash（实现禁止使用它）
    assert ids[0] != hash("E_micro_inference")


# --- 3. 覆盖验证：validation/test 只作覆盖验证 ----------------------------------

@needs_assets
@pytest.mark.parametrize("split,start", [
    ("validation", "2024-08-01T00:00:00+08:00"),
    ("test", "2024-10-01T00:00:00+08:00"),
])
def test_validation_and_test_are_only_coverage_checks(split, start):
    """它们只被用来**验证覆盖**，不得参与选参。"""
    stream = _stream(split, start)
    agg = mapper().verified_realized_aggregate()
    for slot in stream.slots:
        assert sum(slot.ledger) == slot.aggregate_micro
        assert slot.aggregate_micro == int(agg[slot.slot_index]) * WORK_UNIT_SCALE


# --- 4. 确定性 / 未来 mutation / prefix ------------------------------------------

@needs_assets
def test_same_inputs_give_the_same_stream_and_hash():
    a = _stream()
    b = _stream()
    assert a.content_hash == b.content_hash
    assert [t.task_id for s in a.slots for t in s.tasks] == \
           [t.task_id for s in b.slots for t in s.tasks]


@needs_assets
def test_future_slot_mutation_does_not_change_the_prefix(monkeypatch):
    m = mapper()
    base = _stream(horizon=8)
    real = m.verified_realized_aggregate

    def mutated():
        arr = np.asarray(real(), dtype=np.int64).copy()
        # 改**未来** slot（最后一个）——此前 slot 的 Task 必须逐位不变
        arr[base.slots[-1].slot_index] += 1000
        return arr

    monkeypatch.setattr(m, "verified_realized_aggregate", mutated)
    after = _stream(horizon=7)
    assert [t.task_id for s in after.slots for t in s.tasks] == \
           [t.task_id for s in base.slots[:-1] for t in s.tasks]
    assert [t.workload for s in after.slots for t in s.tasks] == \
           [t.workload for s in base.slots[:-1] for t in s.tasks]


@needs_assets
def test_forecast_is_never_used_as_task_truth(monkeypatch):
    """把 forecast（期望值）改成与原值不同，Task 必须**完全不变**。"""
    m = mapper()
    base = _stream(horizon=6)
    real = m.expected_arrival_forecast

    def bogus(*a, **k):
        arr = np.asarray(real(*a, **k), dtype=float)
        return arr + 1000.0

    monkeypatch.setattr(m, "expected_arrival_forecast", bogus)
    after = _stream(horizon=6)
    assert after.content_hash == base.content_hash


# --- 5. 分割边界 ----------------------------------------------------------------

@needs_assets
def test_sixty_one_work_splits_into_two_tasks():
    m = mapper()
    slots = m.split_slot_aggregate_micro(61 * WORK_UNIT_SCALE)
    assert len(slots) == 2
    assert sum(slots) == 61 * WORK_UNIT_SCALE


@needs_assets
def test_below_the_lower_bound_fails_closed():
    m = mapper()
    with pytest.raises(m.ArrivalMapperError):
        m.split_slot_aggregate_micro(9 * WORK_UNIT_SCALE)


@needs_assets
def test_beyond_four_task_coverage_fails_closed(monkeypatch):
    m = mapper()
    huge = 5 * 4 * 60_705_862 + 1  # 需要 > 4 个任务
    with pytest.raises(m.ArrivalMapperError):
        m.split_slot_aggregate_micro(huge)
    # 通过公开入口注入超大 aggregate 也必须 fail closed
    real = m.verified_realized_aggregate
    base = np.asarray(real(), dtype=np.int64).copy()
    base[10224] = 10_000  # validation 的首个槽（origin=0 -> 全局 10224）
    monkeypatch.setattr(m, "verified_realized_aggregate", lambda: base)
    with pytest.raises(m.ArrivalMapperError):
        _stream(horizon=8, start="2024-08-01T00:00:00+08:00", split="validation")


@needs_assets
def test_zero_aggregate_slot_is_refused_not_silently_dropped():
    m = mapper()
    with pytest.raises(m.ArrivalMapperError):
        m.split_slot_aggregate_micro(0)


# --- 6. 拒绝矩阵 ----------------------------------------------------------------

@needs_assets
def test_invalid_start_and_split_are_rejected():
    m = mapper()
    for split, start in (
        ("train", "2024-01-01"),                 # 无时区
        ("train", "2024-01-01T00:17:00+08:00"),  # 非网格
        ("train", "2024-01-01T00:00:00+08:00"),  # 非候选 origin
        ("nope", "2024-01-02T00:00:00+08:00"),   # 未知 split
    ):
        with pytest.raises((ValueError, m.ArrivalMapperError)):
            m.build_arrival_task_stream(split, start=start, horizon=4, seed=1)


@needs_assets
def test_old_split_dirs_and_refs_are_rejected():
    m = mapper()
    for legacy in (V4_DIR / "train.json", V3_DIR / "train.json"):
        if legacy.exists():
            with pytest.raises(ValueError):
                m.load_verified_mapper_chain(split_manifest_path=legacy)
    with pytest.raises(ValueError):
        m.load_verified_mapper_chain(refs_path=REFS_V3)


@needs_assets
def test_tampered_source_is_rejected(tmp_path, monkeypatch):
    m = mapper()
    import scenario.b6_split_manifests as splits_mod

    temp_dir = tmp_path / "v5"
    temp_dir.mkdir()
    payload = json.loads((V5_DIR / "train.json").read_text(encoding="utf-8"))
    payload["inputs"]["frozen_refs"]["sha256"] = "0" * 64
    (temp_dir / "train.json").write_text(json.dumps(payload))
    monkeypatch.setattr(splits_mod, "_canonical_split_dir", lambda: temp_dir)
    with pytest.raises(ValueError):
        m.load_verified_mapper_chain(split_manifest_path=temp_dir / "train.json")


@needs_assets
def test_dirty_source_is_rejected(monkeypatch):
    m = mapper()
    monkeypatch.setattr(m, "_generator_is_dirty", lambda: True)
    with pytest.raises(m.ArrivalMapperError):
        _stream(horizon=4)


# --- 7. 不得调用 demo/random generator -------------------------------------------

@needs_assets
def test_demo_and_random_generators_are_never_called(monkeypatch):
    # 动态导入：mypy 不会跟进，避免把含既有类型错误的模块拉进 make check
    tm = importlib.import_module("idc_model.task_model")

    def boom(*a, **k):
        raise AssertionError("mapper 不得调用 demo/random task generator")

    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_demo_tasks", boom)
    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_random_tasks", boom)
    stream = _stream(horizon=8)
    assert stream.slots


# --- 8. materializer CLI / 幂等 / 原子 -------------------------------------------

@needs_assets
def test_materializer_cli_accepts_only_help_and_verify():
    mod = materializer()
    for argv in (["--out-dir", "/tmp/x"], ["--manifest-path", "/tmp/x.json"],
                 ["--revision", "0" * 40], ["--out-path", "/tmp/x.json"]):
        with pytest.raises(SystemExit):
            mod.main(argv)


@needs_assets
def test_verify_is_idempotent():
    mod = materializer()
    before = (_sha256(MANIFEST), MANIFEST.stat().st_mtime_ns)
    for _ in range(3):
        assert mod.main(["--verify"]) == 0
    assert (_sha256(MANIFEST), MANIFEST.stat().st_mtime_ns) == before
    leftovers = [p for p in MANIFEST.parent.iterdir()
                 if p.name.startswith(".m13g_arrival_mapper")]
    assert leftovers == []


@needs_assets
def test_transactional_failure_leaves_nothing(tmp_path, monkeypatch):
    """本物化器**只写一个**产物（单文件 manifest）⇒ 只有一个原子写阶段。

    （多产物物化器——refs / triad——才有「第 1/2/3 阶段」之分；此处如实只测
    它**真正存在**的那一个阶段，不虚构不存在的阶段。）
    """
    mod = materializer()
    m = mapper()
    target_dir = tmp_path / "manifest"
    target_dir.mkdir()
    monkeypatch.setattr(m, "_canonical_manifest_dir", lambda: target_dir)
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)

    def failing(path, body):
        raise RuntimeError("injected failure at the single atomic write")

    monkeypatch.setattr(mod, "_atomic_write_bytes", failing)
    with pytest.raises(RuntimeError):
        mod.materialize_arrival_mapper_manifest()
    assert not (target_dir / "m13g_arrival_mapper_v1.json").exists()
    assert list(target_dir.iterdir()) == []


@needs_assets
def test_existing_different_manifest_is_refused(tmp_path, monkeypatch):
    mod = materializer()
    m = mapper()
    target_dir = tmp_path / "manifest"
    target_dir.mkdir()
    (target_dir / "m13g_arrival_mapper_v1.json").write_text('{"schema":"forged"}')
    monkeypatch.setattr(m, "_canonical_manifest_dir", lambda: target_dir)
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    with pytest.raises(ValueError):
        mod.materialize_arrival_mapper_manifest()


# --- 9. 既有资产逐字节未变 ------------------------------------------------------

@needs_assets
def test_protected_assets_are_byte_identical():
    for path, expected in PROTECTED.items():
        assert _sha256(path) == expected, str(path)
    for split in ("train", "validation", "test"):
        assert (V5_DIR / f"{split}.json").is_file()


# =============================================================================
# M1.3g-e-b-R1：语义信任边界、内容哈希与账本
# =============================================================================

def _forged_manifest(tmp_path, monkeypatch, mutate):
    m = mapper()
    d = tmp_path / "manifest"
    d.mkdir(parents=True, exist_ok=True)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(payload)
    (d / "m13g_arrival_mapper_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    monkeypatch.setattr(m, "_canonical_manifest_dir", lambda: d)
    return d


@needs_assets
def test_untampered_manifest_is_accepted(tmp_path, monkeypatch):
    """**接受性对照**：未篡改的副本（经私有 resolver 视为 canonical）必须通过。"""
    _forged_manifest(tmp_path, monkeypatch, lambda p: None)
    stream = _stream(horizon=2)
    assert len(stream.tasks) == 2


@needs_assets
@pytest.mark.parametrize("mutate", [
    pytest.param(lambda p: p["sources"]["frozen_refs_v4"].update(sha256="0" * 64),
                 id="source_sha"),
    pytest.param(lambda p: p.update(source_revision="0" * 40), id="revision"),
    pytest.param(lambda p: p["sources"]["b6_intensity_policy"].update(sha256="0" * 64),
                 id="b6_source_sha"),
    pytest.param(lambda p: p.update(c_idc_base_work_per_hour=1.0), id="c_idc"),
    pytest.param(lambda p: p["profiles"]["E_micro_inference"].update(load_range=[0.01, 0.99]),
                 id="load_range"),
    pytest.param(lambda p: p.update(max_tasks_per_slot=9), id="max_tasks"),
    pytest.param(lambda p: p.update(work_unit_scale=10), id="scale"),
    pytest.param(lambda p: p.update(frozen_at_utc="2026-01-01T00:00:00+00:00"),
                 id="frozen_at_utc"),
    pytest.param(lambda p: p.update(extra_field=1), id="extra_field"),
    pytest.param(lambda p: p.pop("e_work_bounds"), id="missing_field"),
    pytest.param(lambda p: p.update(profile_order=["A_inference"]), id="profile_order"),
    pytest.param(lambda p: p["approved_parameters"].update(priority_range=[0.0, 1.0]),
                 id="approved_priority"),
    pytest.param(lambda p: p.update(sources={}), id="missing_sources"),
])
def test_forged_manifest_is_rejected_by_the_public_entry(tmp_path, monkeypatch, mutate):
    """**公开入口**（不是 helper）必须拒绝被篡改的 manifest。"""
    _forged_manifest(tmp_path, monkeypatch, mutate)
    with pytest.raises(mapper().ArrivalMapperError):
        _stream(horizon=2)


@needs_assets
def test_content_hash_covers_every_task_field():
    m = mapper()
    stream = _stream(horizon=2)
    base = m.canonical_content_hash(stream)
    task = stream.slots[0].tasks[0]

    for field, value in (
        ("priority", 3.399999),
        ("duration", 3),
        ("deadline", 9),
        ("profile_key", "A_inference"),
        ("name", "A_inference"),
        ("arrival_time", 99),
        ("interruptible", True),
        ("parallelizable", True),
        ("task_id", task.task_id + 1),
        ("workload", task.workload + 1.0),
        ("load_profile", np.full(len(task.load_profile), 0.123456, dtype=np.float64)),
    ):
        original = getattr(task, field)
        setattr(task, field, value)
        try:
            assert m.canonical_content_hash(stream) != base, field
        finally:
            setattr(task, field, original)
    assert m.canonical_content_hash(stream) == base


@needs_assets
def test_content_hash_covers_slot_and_ledger_metadata():
    m = mapper()
    stream = _stream(horizon=2)
    base = m.canonical_content_hash(stream)
    slot = stream.slots[0]
    original = slot.ledger
    object.__setattr__(slot, "ledger", (original[0] + 1, *original[1:]))
    try:
        assert m.canonical_content_hash(stream) != base
    finally:
        object.__setattr__(slot, "ledger", original)


@needs_assets
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_tampered_split_chain_is_rejected_by_the_public_entry(
        tmp_path, monkeypatch, split):
    """经**私有 canonical resolver** 篡改 v5 链 → **公开入口**必须 REJECT。"""
    import scenario.b6_split_manifests as splits_mod

    m = mapper()
    temp_dir = tmp_path / "v5"
    temp_dir.mkdir()
    for name in ("train", "validation", "test"):
        payload = json.loads((V5_DIR / f"{name}.json").read_text(encoding="utf-8"))
        if name == split:
            payload["inputs"]["frozen_refs"]["sha256"] = "0" * 64
        (temp_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    monkeypatch.setattr(splits_mod, "_canonical_split_dir", lambda: temp_dir)
    # 该 split 的 start 仍合法；篡改必须由**公开入口**的链验证捕获
    starts = {"train": "2024-01-02T00:00:00+08:00",
              "validation": "2024-08-01T00:00:00+08:00",
              "test": "2024-10-01T00:00:00+08:00"}
    with pytest.raises(ValueError):
        m.build_arrival_task_stream(split, start=starts[split], horizon=2, seed=1)


@needs_assets
def test_tampered_refs_chain_is_rejected_by_the_public_entry(tmp_path, monkeypatch):
    """篡改 `refs_v4`（`lambda_ref` 改回旧的 2000）→ 公开入口必须 REJECT。"""
    import scenario.b6_refs as refs_mod

    refs_dir = tmp_path / "frozen_refs"
    refs_dir.mkdir()
    payload = json.loads(REFS_V4.read_text(encoding="utf-8"))
    payload["references"]["lambda_ref"]["value"] = 2000.0
    (refs_dir / "refs_v4.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    monkeypatch.setattr(refs_mod, "_canonical_refs_dir", lambda: refs_dir)
    with pytest.raises(ValueError):
        _stream(horizon=2)
