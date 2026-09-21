"""M1.3g-f-b-b：formal env 独立发布产物（方案 B）。

发布产物与**不可变** v5 共存，**不**覆盖 v5 的任何字节：

- `configs/release/idc_formal_env_release_v1.json` —— 唯一 canonical 发布产物；
- `scenario/env_release.py` —— 唯一严格 loader；
- 产物的 `binds` 以 path+sha256 **逐字节绑定** v5 三份、`refs_v4`、mapper manifest；
- 产物**不设**自报时间戳字段（避免把未经锚定的自报时间当验签依据）。

**改前缺陷（本文件对应先红）**：`scenario/env_release.py` 不存在、
canonical 产物不存在，`train.py` 也不验证任何发布产物。
"""

import copy
import importlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RELEASE_MODULE = "scenario.env_release"
CANONICAL_RELEASE = REPO_ROOT / "configs/release/idc_formal_env_release_v1.json"

BOUND_ROLES = (
    "formal_split_v5_train",
    "formal_split_v5_validation",
    "formal_split_v5_test",
    "frozen_refs_v4",
    "m13g_arrival_mapper_v1",
)


def release_module():
    return importlib.import_module(RELEASE_MODULE)


def run_cli(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "safe_rl_v2.train", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def _failed_report(base_dir: pathlib.Path, run_id: str) -> tuple[dict, dict]:
    manifest = json.loads((base_dir / run_id / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((base_dir / run_id / "report.json").read_text(encoding="utf-8"))
    return manifest, report


# =============================================================================
# 1. 合法产物被接受（需要 canonical 产物已物化）
# =============================================================================

def test_canonical_env_release_is_accepted():
    m = release_module()
    payload = m.load_verified_env_release()
    assert payload["schema"] == m.RELEASE_SCHEMA
    assert payload["readiness"] == {
        "formal_env_ready": True, "formal_training_ready": False}
    assert set(payload["binds"]) == set(BOUND_ROLES)
    # 逐字节绑定：每个角色都必须与 live 文件 hash 一致
    for role, entry in payload["binds"].items():
        live = m._sha256_file(REPO_ROOT / entry["path"])
        assert entry["sha256"] == live, role
    assert CANONICAL_RELEASE.is_file()
    # 非空洞性：产物必须真的存在且非空
    assert CANONICAL_RELEASE.stat().st_size > 0


def test_env_release_binds_the_immutable_v5_and_mapper_bytes():
    """产物的 binds 必须与 **v5 三份 + refs_v4 + mapper manifest** 的实际字节一致。"""
    m = release_module()
    payload = m.load_verified_env_release()
    expected = {
        "formal_split_v5_train": "data/manifest/formal_splits_v5/train.json",
        "formal_split_v5_validation": "data/manifest/formal_splits_v5/validation.json",
        "formal_split_v5_test": "data/manifest/formal_splits_v5/test.json",
        "frozen_refs_v4": "configs/frozen_refs/refs_v4.json",
        "m13g_arrival_mapper_v1": "data/manifest/m13g_arrival_mapper_v1.json",
    }
    assert {r: e["path"] for r, e in payload["binds"].items()} == expected


# =============================================================================
# 2. 缺失 / 篡改一律 fail closed（无 fallback）
# =============================================================================

def test_env_release_rejects_a_missing_artifact(monkeypatch):
    m = release_module()
    monkeypatch.setattr(m, "_canonical_release_path",
                        lambda: REPO_ROOT / "configs/release" / "does_not_exist.json")
    with pytest.raises(m.EnvReleaseError):
        m.load_verified_env_release()


@pytest.mark.parametrize("role", BOUND_ROLES)
def test_env_release_rejects_tampered_binding(role):
    """篡改任一 binds 的 hash → 重建比对必须拒绝。"""
    m = release_module()
    payload = copy.deepcopy(m.load_verified_env_release())
    original = payload["binds"][role]["sha256"]
    payload["binds"][role] = {**payload["binds"][role], "sha256": "0" * 64}
    assert payload["binds"][role]["sha256"] != original, "mutation 未生效"
    with pytest.raises(m.EnvReleaseError):
        m.validate_env_release(payload)


def test_env_release_rejects_a_wrong_path_binding():
    m = release_module()
    payload = copy.deepcopy(m.load_verified_env_release())
    payload["binds"]["frozen_refs_v4"] = {
        **payload["binds"]["frozen_refs_v4"], "path": "configs/frozen_refs/refs_v3.json"}
    with pytest.raises(m.EnvReleaseError):
        m.validate_env_release(payload)


@pytest.mark.parametrize("tamper", [
    {"formal_env_ready": False, "formal_training_ready": False},
    {"formal_env_ready": True, "formal_training_ready": True},
    {"formal_env_ready": False, "formal_training_ready": True},
])
def test_env_release_rejects_tampered_readiness(tamper):
    """readiness 必须**精确等于** env=true / training=false。"""
    m = release_module()
    payload = copy.deepcopy(m.load_verified_env_release())
    assert payload["readiness"] != tamper, "mutation 未生效"
    payload["readiness"] = tamper
    with pytest.raises(m.EnvReleaseError):
        m.validate_env_release(payload)


def test_env_release_rejects_a_stale_revision():
    """`release_revision` 必须等于**当前**实现 revision。"""
    m = release_module()
    payload = copy.deepcopy(m.load_verified_env_release())
    payload["release_revision"] = "0" * 40
    with pytest.raises(m.EnvReleaseError):
        m.validate_env_release(payload)


def test_env_release_revision_covers_the_required_implementation_files():
    """revision / dirty 检查必须**至少**覆盖这五个实现文件。"""
    m = release_module()
    covered = set(m.ENV_RELEASE_SOURCE_PATHS)
    for required in ("envs/idc_price_env.py", "scenario/env_injection.py",
                     "safe_rl_v2/train.py", "scenario/env_release.py",
                     "scripts/materialize_env_release.py"):
        assert required in covered, f"revision 覆盖缺失 {required}"


def test_env_release_has_no_self_reported_timestamp():
    """**不得**把未经锚定的自报时间戳当验签依据 ⇒ 产物无时间戳字段。"""
    m = release_module()
    payload = m.load_verified_env_release()
    for forbidden in ("frozen_at_utc", "generated_at", "created_at", "timestamp"):
        assert forbidden not in payload, f"发布产物不得自报时间戳 {forbidden}"


# =============================================================================
# 3. train.py 的合取门
# =============================================================================

def test_train_entry_requires_the_env_release(monkeypatch, tmp_path):
    """无发布产物必须明确失败，且**绝不**回退到「忽略 readiness」。"""
    import safe_rl_v2.train as train
    import scenario.env_release as rel

    def boom(*args, **kwargs):
        raise rel.EnvReleaseError("env-release-missing")

    monkeypatch.setattr(rel, "load_verified_env_release", boom)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "norel"])
    assert rc != 0
    manifest, report = _failed_report(tmp_path, "norel")
    assert manifest["status"] == "failed"
    assert report["synthetic"] is False
    assert "env-release-missing" in report["failure"], report["failure"]
    assert "run 产物" not in report["failure"]


def test_released_env_still_blocks_training(tmp_path):
    """**合法发布产物**存在时，入口仍因 **training 未放行** 明确失败。

    本卡**不**把训练成功当作验收条件。
    """
    result = run_cli("--base-dir", str(tmp_path), "--run-id", "released")
    assert result.returncode != 0, "env 发布不得让训练成功"
    combined = result.stdout + result.stderr
    assert "M1.3" in combined, combined
    # 必须是 **training 未放行** 的归因，而不是「env 未放行」：
    # 改前只会出现 `['formal_env_ready', 'formal_training_ready']` 这样的列表，
    # 不含 `formal_training_ready=false` 这一训练门专属判定。
    assert "formal_training_ready=false" in combined, combined
    assert "synthetic=true" not in combined

    manifest, report = _failed_report(tmp_path, "released")
    assert manifest["status"] == "failed"
    assert report["synthetic"] is False
    assert report["claims"]["trained"] is False
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))


def test_v5_readiness_is_still_required_to_be_both_false(monkeypatch, tmp_path):
    """**合取门**：v5 被改成 env=true 仍必须被拒（不得被发布产物掩盖）。"""
    import scenario.b6_split_manifests as v5mod

    train = importlib.import_module("safe_rl_v2.train")
    real = v5mod.load_verified_split_manifest_v5

    def forged(*args, **kwargs):
        payload = real(*args, **kwargs)
        payload = copy.deepcopy(payload)
        payload["readiness"] = {"formal_env_ready": True, "formal_training_ready": False}
        return payload

    monkeypatch.setattr(v5mod, "load_verified_split_manifest_v5", forged)
    rc = train.main(["--base-dir", str(tmp_path), "--run-id", "v5forged"])
    assert rc != 0
    _manifest, report = _failed_report(tmp_path, "v5forged")
    # 必须显式判定「v5 readiness 必须严格保持 both-false」，而不是泛泛的「未放行」
    assert "both-false" in report["failure"], report["failure"]


# =============================================================================
# 4. 物化脚本：幂等 + 拒绝覆盖
# =============================================================================

def test_materializer_is_idempotent_and_refuses_to_overwrite(tmp_path, monkeypatch):
    """`--verify` 幂等；物化拒绝覆盖**不同**的既存文件。"""
    import scripts.materialize_env_release as mat

    # --verify 幂等（只读，可重复）
    assert mat.verify()["sha256"] == mat.verify()["sha256"]

    # 拒绝覆盖：在临时 canonical 位置上放一个**不同**的既存文件
    tmp_canonical = tmp_path / "release.json"
    tmp_canonical.write_text('{"schema":"bogus"}\n', encoding="utf-8")
    monkeypatch.setattr(mat, "_canonical_release_path", lambda: tmp_canonical)
    with pytest.raises(Exception):
        mat.materialize()
    # 既存文件**未被**覆盖
    assert json.loads(tmp_canonical.read_text(encoding="utf-8")) == {"schema": "bogus"}


def test_materializer_rejects_the_wrong_canonical_path(monkeypatch):
    m = release_module()
    with pytest.raises(m.EnvReleaseError):
        m.load_verified_env_release(REPO_ROOT / "configs/release" / "copy.json")
