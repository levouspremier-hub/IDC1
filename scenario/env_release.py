"""M1.3g-f-b-b：formal env 的**独立发布产物**（方案 B，见任务卡 §bn/§bp）。

**定位**：与**不可变** v5 共存。本模块**不**修改 v5 的任何字节，也**不**覆盖其
`readiness`——v5 的 `readiness` 必须继续**严格等于**冻结常量 both-false
（由 v5 自己的 `_require(readiness == READINESS)` 强制，任何改成 true 的尝试
仍被 fail closed 拒绝）。

本产物声明**发布批准**：

```text
formal_env_ready     = true     # formal env 链已发布
formal_training_ready = false   # 训练**仍未**放行（train.py 无训练循环）
```

并以其 `binds` 以 **path + sha256** 逐字节绑定五个冻结上游：v5 三份、
`refs_v4`、`m13g_arrival_mapper_v1`。

**不设自报时间戳**：产物中**没有** `frozen_at_utc` / `generated_at` 之类的字段，
因此不存在「把未经锚定的自报时间当验签依据」的风险——验签输入**全部**来自
live 文件 hash 与 live git revision。

**没有**任何信任边界参数：`load_verified_env_release()` 只接受唯一 canonical 路径，
副本 / 别名 / symlink 一律拒绝。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from contracts import CONTRACT_VERSION_ID

REPO_ROOT = Path(__file__).resolve().parent.parent

RELEASE_SCHEMA = "idc-formal-env-release-v1"
CANONICAL_RELEASE_LOGICAL = "configs/release/idc_formal_env_release_v1.json"
# 采纳「方案 B」的审计裁决（docs/task_cards/M1.3g.md §bn.3 / §bo.2）。
APPROVED_DECISION_ID = "M1.3g-f-b-a"
SUPERSEDES = None

# 发布代码 revision / dirty 检查**至少**覆盖这五个实现文件（卡 §bp.5 第 2 条）。
ENV_RELEASE_SOURCE_PATHS: tuple[str, ...] = (
    "envs/idc_price_env.py",
    "scenario/env_injection.py",
    "safe_rl_v2/train.py",
    "scenario/env_release.py",
    "scripts/materialize_env_release.py",
)

# 产物逐字节绑定的冻结上游（角色 → 仓库相对逻辑路径）。
RELEASE_BINDING_LOGICAL: dict[str, str] = {
    "formal_split_v5_train": "data/manifest/formal_splits_v5/train.json",
    "formal_split_v5_validation": "data/manifest/formal_splits_v5/validation.json",
    "formal_split_v5_test": "data/manifest/formal_splits_v5/test.json",
    "frozen_refs_v4": "configs/frozen_refs/refs_v4.json",
    "m13g_arrival_mapper_v1": "data/manifest/m13g_arrival_mapper_v1.json",
}

READINESS_KEYS: tuple[str, ...] = ("formal_env_ready", "formal_training_ready")
RELEASE_READINESS: dict[str, bool] = {
    "formal_env_ready": True,
    "formal_training_ready": False,
}

MANIFEST_TOP_KEYS: tuple[str, ...] = (
    "schema", "contract_version", "approved_decision_id", "supersedes",
    "readiness", "binds", "release_revision", "note",
)
BIND_ENTRY_KEYS: tuple[str, ...] = ("path", "sha256")

NOTE = (
    "formal env 的独立发布批准（方案 B）。v5 的 readiness 保持 both-false 且字节不变；"
    "本产物不覆盖 v5 的任何字节。formal_training_ready=false 表示训练仍未放行。"
)


class EnvReleaseError(ValueError):
    """发布产物的位置 / 结构 / 绑定 / revision 校验失败（一律 fail closed）。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnvReleaseError(message)


def _canonical_release_path() -> Path:
    return REPO_ROOT / CANONICAL_RELEASE_LOGICAL


def canonical_release_path() -> Path:
    """唯一 canonical 发布产物路径（公开只读访问器）。"""
    return _canonical_release_path()


def _sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise EnvReleaseError(f"冻结上游不可读：{path}：{error}") from error


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", *ENV_RELEASE_SOURCE_PATHS).strip())


def env_release_code_revision() -> str:
    """本发布实现的 revision（由 Git 解析；**不**用漂移的 HEAD）。"""
    revision = _git("log", "-1", "--format=%H", "--", *ENV_RELEASE_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise EnvReleaseError(f"release revision 无效：{revision!r}")
    return revision


# --- 严格校验 helper ----------------------------------------------------------

def _require_exact_keys(mapping: object, *, field: str,
                        expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise EnvReleaseError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise EnvReleaseError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}")
    return mapping


def _require_hex64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(c not in "0123456789abcdef" for c in value):
        raise EnvReleaseError(f"{field} 必须是 64 位小写十六进制，实际 {value!r}")
    return value


def _require_git_sha40(value: object, *, field: str) -> str:
    """Git revision 是 **40** 位十六进制（**不是** sha256 的 64 位）。"""
    if not isinstance(value, str) or len(value) != 40 \
            or any(c not in "0123456789abcdef" for c in value):
        raise EnvReleaseError(f"{field} 必须是 40 位小写十六进制 git SHA，实际 {value!r}")
    return value


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnvReleaseError(f"{field} 必须是非空字符串，实际 {value!r}")
    return value


# --- 构建 / 校验 --------------------------------------------------------------

def build_env_release() -> dict:
    """由 **live** 文件 hash + **live** revision 构建候选发布产物。

    **确定性**：不含任何时间戳或环境相关字段，故可整体重建比对。
    """
    return {
        "schema": RELEASE_SCHEMA,
        "contract_version": CONTRACT_VERSION_ID,
        "approved_decision_id": APPROVED_DECISION_ID,
        "supersedes": SUPERSEDES,
        "readiness": dict(RELEASE_READINESS),
        "binds": {
            role: {"path": logical, "sha256": _sha256_file(REPO_ROOT / logical)}
            for role, logical in RELEASE_BINDING_LOGICAL.items()
        },
        "release_revision": env_release_code_revision(),
        "note": NOTE,
    }


def validate_env_release(payload: object) -> dict:
    """结构校验 + **整体重建比对**（live hash / live revision / readiness）。"""
    validated = _require_exact_keys(payload, field="release", expected=MANIFEST_TOP_KEYS)

    _require(_require_str(validated["schema"], field="schema") == RELEASE_SCHEMA,
             f"schema 必须等于 {RELEASE_SCHEMA!r}，实际 {validated['schema']!r}")
    _require(_require_str(validated["contract_version"], field="contract_version")
             == CONTRACT_VERSION_ID,
             f"contract_version 必须等于 {CONTRACT_VERSION_ID!r}，"
             f"实际 {validated['contract_version']!r}")
    _require(_require_str(validated["approved_decision_id"], field="approved_decision_id")
             == APPROVED_DECISION_ID,
             f"approved_decision_id 必须等于 {APPROVED_DECISION_ID!r}，"
             f"实际 {validated['approved_decision_id']!r}")
    _require(validated["supersedes"] is None,
             f"supersedes 必须为 null，实际 {validated['supersedes']!r}")

    readiness = _require_exact_keys(validated["readiness"], field="readiness",
                                    expected=READINESS_KEYS)
    _require(readiness == RELEASE_READINESS,
             f"readiness 必须严格等于 {RELEASE_READINESS}（env 已发布、训练未放行）；"
             f"实际 {readiness}")

    binds = _require_exact_keys(validated["binds"], field="binds",
                                expected=tuple(RELEASE_BINDING_LOGICAL))
    for role, entry in binds.items():
        entry = _require_exact_keys(entry, field=f"binds.{role}",
                                    expected=BIND_ENTRY_KEYS)
        declared = _require_str(entry["path"], field=f"binds.{role}.path")
        expected_path = RELEASE_BINDING_LOGICAL[role]
        _require(declared == expected_path,
                 f"binds.{role}.path 必须精确等于冻结逻辑路径 {expected_path!r}；"
                 f"实际 {declared!r}")
        _require_hex64(entry["sha256"], field=f"binds.{role}.sha256")

    _require_git_sha40(validated["release_revision"], field="release_revision")
    _require_str(validated["note"], field="note")

    if _generator_is_dirty():
        raise EnvReleaseError(
            "发布实现有未提交修改：拒绝用旧 revision 为未提交代码背书")

    live = env_release_code_revision()
    _require(validated["release_revision"] == live,
             f"release_revision 必须是**当前**实现的 revision {live}；"
             f"实际 {validated['release_revision']}")

    rebuilt = build_env_release()
    if rebuilt != validated:
        differing = sorted(
            key for key in set(rebuilt) | set(validated)
            if rebuilt.get(key) != validated.get(key))
        raise EnvReleaseError(
            "发布产物与由 live 文件 hash + live revision 重建的候选不符；"
            f"差异顶层字段={differing}")
    return validated


def _require_canonical_location(path: Path | str | None) -> Path:
    """路径必须**精确等于** canonical 发布产物（在读 JSON 之前）；拒绝副本 / symlink。"""
    given = (Path(path).absolute() if path is not None
             else _canonical_release_path().absolute())
    expected = _canonical_release_path().absolute()
    if given == expected:
        if expected.is_symlink():
            raise EnvReleaseError(f"发布产物不得是 symlink：{expected}")
        return expected
    raise EnvReleaseError(
        f"发布产物必须是**唯一 canonical** 文件 {CANONICAL_RELEASE_LOGICAL}；"
        f"实际 {given}（不接受副本、别名或 symlink）")


def load_verified_env_release(path: Path | str | None = None) -> dict:
    """读取并**严格校验** canonical 发布产物（无 fallback、无信任边界参数）。

    除结构 / 重建比对之外，还**复用既有 verified loaders** 证明被绑定的上游
    **语义**有效（不只是字节一致）。
    """
    resolved = _require_canonical_location(path)
    if not resolved.is_file():
        raise EnvReleaseError(
            f"缺少 formal env 发布产物 {CANONICAL_RELEASE_LOGICAL}："
            "env 尚未发布（**不**回退、**不**忽略 v5 readiness）")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnvReleaseError(f"{resolved} 不可读或不是合法 JSON：{error}") from error
    try:
        validated = validate_env_release(payload)
    except EnvReleaseError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise EnvReleaseError(f"{resolved} 结构畸形：{error}") from error

    # 复用既有 verified public 链：被绑定的上游必须**语义**有效。
    from scenario.arrival_mapper import load_verified_mapper_manifest
    from scenario.b6_refs import load_verified_refs_v4
    from scenario.b6_split_manifests import load_verified_split_manifest_v5

    load_verified_refs_v4()
    load_verified_mapper_manifest()
    for split in ("train", "validation", "test"):
        load_verified_split_manifest_v5(expected_split=split)
    return validated


# --- 原子写入 ------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_env_release(path: Path, payload: dict) -> None:
    _atomic_write_text(path, canonical_json(payload))


__all__ = [
    "APPROVED_DECISION_ID",
    "CANONICAL_RELEASE_LOGICAL",
    "ENV_RELEASE_SOURCE_PATHS",
    "EnvReleaseError",
    "RELEASE_BINDING_LOGICAL",
    "RELEASE_READINESS",
    "RELEASE_SCHEMA",
    "build_env_release",
    "canonical_json",
    "canonical_release_path",
    "env_release_code_revision",
    "load_verified_env_release",
    "validate_env_release",
    "write_env_release",
]

if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(load_verified_env_release(), indent=2, ensure_ascii=False))
    raise SystemExit(0)
