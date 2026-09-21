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
    assert calls == ["training"], (
        f"real preflight 必须且只能以 purpose='training' 调用 gate，实际 {calls}")


# =============================================================================
# 3. 完整性链（复用 verified public 链，不重写弱校验器）
# =============================================================================

def test_real_preflight_consults_the_verified_v5_loader(monkeypatch, tmp_path):
    """preflight 必须经**既有 verified public 链**读取 v5 manifest。"""
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
    assert seen == ["train"], f"必须经 verified v5 loader 读取 train split，实际 {seen}"


def test_real_preflight_fails_closed_when_the_verified_chain_rejects(monkeypatch, tmp_path):
    """上游校验失败必须**明确失败并透出原始错误**，不得回退。"""
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


def test_train_entry_does_not_rewrite_its_own_hash_verifier():
    """**结构性守卫**：`train.py` 不得自建一套 hash 校验器。"""
    source = (REPO_ROOT / "safe_rl_v2" / "train.py").read_text(encoding="utf-8")
    assert "hashlib.sha256" not in source, "完整性校验必须复用 verified public 链"
    assert "def _sha256_file" not in source and "def _verify_frozen_chain" not in source
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
