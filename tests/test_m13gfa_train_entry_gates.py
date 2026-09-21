"""M1.3g-f-a：B6 正式训练入口预检与失败归因。

覆盖四件事：

1. **origin 来源** —— real 入口从 verified `formal_splits_v5` train 的
   `candidate_origins` 推导规范带时区 start；`time_range.start` 对应**本地行 0**，
   **不是**合法 episode start（第一个合法 origin 是本地行 48）。
2. **purpose gate** —— 入口显式调用
   `validate_forecast_purpose(bundle, purpose="training")`；
   synthetic / oracle_debug / 非正式 provenance **不得**通过该门。
3. **完整性链** —— 复用现有 verified public 链（B6/v5、`refs_v4`、policy-v3、
   exogenous v3、mapper）；上游失败**明确失败并透出原始错误**，不回退 v1–v4 /
   默认曲线 / synthetic；`train.py` **不重写**自己的 hash 校验器。
4. **readiness 归因** —— `formal_training_ready=false` 必须归因为
   **M1.3 发布门禁未放行**，默认 real 入口非零退出，**不再**写 M1.2。

⚠️ `build_formal_scenario_b6` / v5 loader 带 **dirty gate**（工作树有未提交修改时
fail closed），故先红观测必须在**干净工作树**上取得。

**改前缺陷（本文件对应先红）**：`_require_frozen_real_scenario()` 硬编码
`start="2023-01-01"`（M1.2 时代，无时区、非 30 分钟网格），从不读 v5
`candidate_origins` / `readiness`，也**从不**调用 training purpose gate。
"""

import importlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
V5_TRAIN = REPO_ROOT / "data/manifest/formal_splits_v5" / "train.json"

# v5 manifest 实测：time_range.start == 本地行 0；第一个合法 candidate origin == 本地行 48。
V5_TIME_RANGE_START = "2024-01-01T00:00:00+08:00"
FIRST_CANDIDATE_ORIGIN = 48
CANONICAL_FIRST_EPISODE_START = "2024-01-02T00:00:00+08:00"
LEGACY_HARDCODED_START = "2023-01-01"


def run_cli(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "safe_rl_v2.train", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def train_module():
    return importlib.import_module("safe_rl_v2.train")


def _v5_payload() -> dict:
    return json.loads(V5_TRAIN.read_text(encoding="utf-8"))


def _failed_report(base_dir: pathlib.Path, run_id: str) -> tuple[dict, dict]:
    manifest = json.loads((base_dir / run_id / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((base_dir / run_id / "report.json").read_text(encoding="utf-8"))
    return manifest, report


# =============================================================================
# 0. v5 事实（回归守卫：这些是后续断言的基准，改前已绿）
# =============================================================================

def test_v5_train_manifest_declares_the_candidate_origin_window():
    payload = _v5_payload()
    assert payload["split"] == "train"
    assert payload["candidate_origins"] == {
        "start": FIRST_CANDIDATE_ORIGIN, "end_exclusive": 10224}
    # time_range.start 是**本地行 0**，不是 episode start
    assert payload["time_range"]["start"] == V5_TIME_RANGE_START
    assert payload["readiness"] == {
        "formal_env_ready": False, "formal_training_ready": False}


def test_time_range_start_is_not_a_legal_episode_start():
    """**防退化**：把 `time_range.start` 直接当 episode start 必须被拒绝。"""
    from scenario.b6_split_manifests import local_origin_from_start

    with pytest.raises(Exception) as excinfo:
        local_origin_from_start("train", V5_TIME_RANGE_START)
    assert "候选 origin" in str(excinfo.value) or "origin" in str(excinfo.value).lower()


# =============================================================================
# 1. origin 来源
# =============================================================================

def test_episode_start_derives_from_the_verified_candidate_origin():
    """real 入口必须由 **candidate origin** 推导 start，而不是硬编码 M1.2 起点。"""
    train = train_module()
    derive = getattr(train, "formal_training_episode_start", None)
    assert derive is not None, (
        "real 入口必须提供一个从 verified v5 candidate origin 推导 episode start 的"
        "入口；改前只硬编码 start=\"2023-01-01\"")

    start = derive(_v5_payload())
    assert start == CANONICAL_FIRST_EPISODE_START
    assert start != V5_TIME_RANGE_START, "不得直接把 time_range.start 当 episode start"

    # 往返：推导出的 start 必须精确映射回 candidate origin
    from scenario.b6_split_manifests import local_origin_from_start

    assert local_origin_from_start("train", start) == FIRST_CANDIDATE_ORIGIN


def test_default_real_entry_does_not_use_the_hardcoded_m12_start(tmp_path):
    result = run_cli("--base-dir", str(tmp_path), "--run-id", "blocked")
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert LEGACY_HARDCODED_START not in combined, (
        "默认 real 入口不得再使用硬编码的 M1.2 起点 '2023-01-01'")


# =============================================================================
# 2. purpose gate
# =============================================================================

def test_training_purpose_gate_rejects_non_formal_bundles():
    """**回归守卫**：gate 本身拒绝 synthetic（改前已绿）。"""
    from contracts.validators import validate_forecast_purpose
    from scenario.scenario import build_scenario

    synthetic = build_scenario(
        "train", start=CANONICAL_FIRST_EPISODE_START, horizon=8, forecast_cutoff=4,
        synthetic=True)
    with pytest.raises(ValueError):
        validate_forecast_purpose(synthetic, purpose="training")
    # 反向控制：debug 仍接受它
    validate_forecast_purpose(synthetic, purpose="debug")


def test_real_preflight_calls_the_training_purpose_gate(monkeypatch, tmp_path):
    """real preflight 必须**显式**过 training purpose gate。"""
    import contracts.validators as validators

    train = train_module()
    calls: list[str] = []
    real = validators.validate_forecast_purpose

    def spy(scenario, *, purpose):
        calls.append(purpose)
        return real(scenario, purpose=purpose)

    monkeypatch.setattr(validators, "validate_forecast_purpose", spy)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "gate"])
    assert rc != 0, "当前 readiness=false，real 入口必须非零退出"
    # `build_formal_scenario_b6` 内部也会过门，故计数 ≥ 1；关键是 real preflight
    # **自己**也必须显式调用（改前为 **0 次**，因为 build_scenario 提前失败）。
    assert calls.count("training") >= 1, (
        f"real preflight 必须显式以 purpose='training' 调用 gate，实际 {calls}")
    assert set(calls) == {"training"}, f"real 入口不得使用任何非 training purpose：{calls}"


# =============================================================================
# 3. 完整性链（复用 verified public 链，不重写弱校验器）
# =============================================================================

def test_real_preflight_consults_the_verified_v5_loader(monkeypatch, tmp_path):
    """preflight 必须经**既有 verified public 链**读取 v5 manifest。

    **回归守卫（改前已绿）**：改前 `build_scenario(synthetic=False)` 已路由到
    v5 loader，故该性质部分先已存在；本用例锁定「切到 candidate-origin 接线后
    仍必须走 verified 链」，**不冒充先红**。
    """
    import scenario.b6_split_manifests as v5mod

    train = train_module()
    seen: list[str] = []
    real = v5mod.load_verified_split_manifest_v5

    def spy(*args, **kwargs):
        seen.append(kwargs.get("expected_split"))
        return real(*args, **kwargs)

    monkeypatch.setattr(v5mod, "load_verified_split_manifest_v5", spy)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "chain"])
    assert rc != 0
    # `load_verified_mapper_chain` 内部也会读 v5，故计数 ≥ 1。
    assert seen.count("train") >= 1, f"必须经 verified v5 loader 读取 train split，实际 {seen}"
    assert set(seen) == {"train"}, f"只允许读取 train split，实际 {seen}"


def test_real_preflight_fails_closed_when_the_verified_chain_rejects(monkeypatch, tmp_path):
    """上游校验失败必须**明确失败并透出原始错误**，不得回退。

    **回归守卫（改前已绿）**：改前失败的原始错误已被透出；本用例锁定接线后
    仍然如此，**不冒充先红**。
    """
    import scenario.b6_split_manifests as v5mod

    train = train_module()

    def boom(*args, **kwargs):
        raise v5mod.SplitManifestV5Error("sentinel-tamper")

    monkeypatch.setattr(v5mod, "load_verified_split_manifest_v5", boom)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "tamper"])
    assert rc != 0
    manifest, report = _failed_report(tmp_path, "tamper")
    assert manifest["status"] == "failed"
    assert report["synthetic"] is False
    assert "sentinel-tamper" in report["failure"], (
        f"必须透出上游原始错误，实际 {report['failure']!r}")


def test_real_preflight_verifies_the_frozen_mapper_manifest(monkeypatch, tmp_path):
    """**M1.3g-f-a-R1**：preflight 必须调用公开的
    `load_verified_mapper_manifest()`（冻结参数 manifest 的唯一验证入口）。

    改前 `load_verified_mapper_chain()` **不**调用它，故 mapper 参数 manifest
    被篡改 / revision 陈旧时 preflight 不会发现。
    """
    import scenario.arrival_mapper as mapper_mod

    train = train_module()
    calls: list[int] = []
    real = mapper_mod.load_verified_mapper_manifest

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mapper_mod, "load_verified_mapper_manifest", spy)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "mmok"])
    assert rc != 0, "当前 readiness=false，real 入口必须非零退出"
    assert calls, (
        "real preflight 必须验证冻结 mapper 参数 manifest"
        "（load_verified_mapper_manifest 一次都没被调用）")


def test_mapper_manifest_rejection_surfaces_before_the_readiness_gate(monkeypatch, tmp_path):
    """mapper manifest 校验失败必须在 **readiness 门之前**明确失败并透出原始原因。"""
    import scenario.arrival_mapper as mapper_mod

    train = train_module()

    def boom(*args, **kwargs):
        raise mapper_mod.ArrivalMapperError("mapper-manifest-rejected")

    monkeypatch.setattr(mapper_mod, "load_verified_mapper_manifest", boom)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "mmrej"])
    assert rc != 0
    manifest, report = _failed_report(tmp_path, "mmrej")
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"]
    assert report["synthetic"] is False
    assert "mapper-manifest-rejected" in report["failure"], (
        f"必须透出 mapper loader 的原始错误：{report['failure']!r}")
    assert "ArrivalMapperError" in report["failure"], report["failure"]
    # **绝不能**先走到 readiness=false
    assert "发布门禁" not in report["failure"], (
        f"mapper manifest 校验失败必须早于 readiness 门：{report['failure']!r}")
    assert "readiness" not in report["failure"], report["failure"]
    # 无成功 run / checkpoint
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))


def test_valid_mapper_manifest_still_blocks_on_the_m13_readiness_gate(tmp_path):
    """**未篡改对照**：合法 mapper manifest 通过后，入口**仍**因 v5
    `readiness=false` 明确失败（说明新增门禁没有短路后面的 readiness 门）。"""
    train = train_module()
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "okchain"])
    assert rc != 0
    _manifest, report = _failed_report(tmp_path, "okchain")
    assert "M1.3" in report["failure"], report["failure"]
    assert "readiness" in report["failure"], report["failure"]
    assert "mapper-manifest-rejected" not in report["failure"]
    assert report["synthetic"] is False
    assert report["claims"]["trained"] is False


def test_real_preflight_fails_closed_when_readiness_fields_are_missing(monkeypatch, tmp_path):
    """readiness 字段**缺失**必须明确失败，不得用 `.get()` 静默兜底。

    `safe_rl_v2/train.py` 的红线之一是「不使用 `.get(...)` 默认值」——
    由 `tests/test_m51c_train_buffer_integration.py` 结构性守卫。
    """
    import scenario.b6_split_manifests as v5mod

    train = train_module()
    real = v5mod.load_verified_split_manifest_v5

    def drop_readiness(*args, **kwargs):
        payload = real(*args, **kwargs)
        payload.pop("readiness", None)
        return payload

    monkeypatch.setattr(v5mod, "load_verified_split_manifest_v5", drop_readiness)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "noready"])
    assert rc != 0
    _manifest, report = _failed_report(tmp_path, "noready")
    # 必须是**入口自己**的明确 fail-closed 判定，而不是撞上 KeyError 后被动透出
    assert "缺少" in report["failure"] and "readiness" in report["failure"], (
        f"readiness 缺失必须由入口明确判定并指明字段，实际 {report['failure']!r}")
    assert "KeyError" not in report["failure"], (
        f"不得靠 KeyError 被动失败（等价于 .get() 兜底的反面）：{report['failure']!r}")
    assert report["synthetic"] is False


def test_train_entry_does_not_rewrite_its_own_frozen_asset_verifier():
    """**结构性守卫**：`train.py` 不得自建一套**冻结资产**的 hash 校验器。

    注意 `_dependency_lock_hash()` 对 `uv.lock` 的 sha256 是**既有且正当**的
    依赖锁记录，不属于冻结资产校验，故只禁止**复刻校验链**的写法。
    """
    source = (REPO_ROOT / "safe_rl_v2" / "train.py").read_text(encoding="utf-8")
    for reimplemented in ("def _sha256_file", "def _verify_frozen_chain",
                          "def load_verified_", "def _require_live_binding"):
        assert reimplemented not in source, (
            f"完整性校验必须复用 verified public 链，不得自建 {reimplemented!r}")
    # 不得把冻结资产路径直接读进来自己做校验
    for frozen in ("formal_splits_v5", "refs_v4.json", "singapore_2024_forecast_policy_v3"):
        assert frozen not in source, (
            f"不得在 train.py 直接拼接冻结资产路径 {frozen!r}（须走 verified public 链）")
    # 也不得回退到旧世代资产
    for stale in ("refs_v3", "refs.json", "formal_splits_v1", "formal_splits_v2",
                  "formal_splits_v3", "formal_splits_v4"):
        assert stale not in source, f"不得引用旧世代资产 {stale}"


# =============================================================================
# 4. readiness 归因
# =============================================================================

def test_default_real_entry_attributes_the_blocker_to_m13_readiness(tmp_path):
    result = run_cli("--base-dir", str(tmp_path), "--run-id", "blocked")
    combined = result.stdout + result.stderr
    assert result.returncode != 0, "默认 real 路径不得成功"
    assert "M1.3" in combined, f"归因必须写 M1.3：{combined}"
    assert "M1.2" not in combined, f"归因不得再写 M1.2：{combined}"
    assert LEGACY_HARDCODED_START not in combined
    assert ("readiness" in combined) or ("发布" in combined), (
        f"必须说明 M1.3 发布门禁未放行：{combined}")


def test_blocked_real_entry_writes_a_failed_manifest_only(tmp_path):
    run_cli("--base-dir", str(tmp_path), "--run-id", "blocked")
    manifest, report = _failed_report(tmp_path, "blocked")
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"]
    assert report["synthetic"] is False
    assert report["claims"]["trained"] is False
    # 无 success run、无 checkpoint
    assert "run 产物" not in manifest
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))
    assert not (tmp_path / "blocked" / "checkpoint").exists()


# =============================================================================
# 5. synthetic 路径保持原样（回归守卫）
# =============================================================================

def test_synthetic_smoke_path_is_unchanged(tmp_path):
    result = run_cli(
        "--synthetic-smoke", "--steps", "3", "--seed", "0",
        "--corrector", "off", "--base-dir", str(tmp_path), "--run-id", "syn")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    report = json.loads((tmp_path / "syn" / "report.json").read_text(encoding="utf-8"))
    assert report["synthetic"] is True
    assert report["claims"]["trained"] is False
