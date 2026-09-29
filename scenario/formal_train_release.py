"""M1.3g-f-c-k：**正式训练发布产物**（方案 B 的第二段：放行正式 train-only 训练）。

与 `scenario/env_release.py`（env 发布，`formal_training_ready=false`）**并列**：

```text
env release  v1 : formal_env_ready = true , formal_training_ready = false   （**字节保留**）
train release v1: formal_env_ready = true , formal_training_ready = true    （本模块）
```

本模块**不修改** env release v1 的任何字节，也**不覆盖** v5 split 的 readiness
（v5 的 `readiness` 必须继续严格等于 both-false）。

`binds` 以 **path + sha256** 逐字节绑定**依赖闭包**（先扫描、后绑定）：

- env release v1 已绑定的五个冻结上游（v5 三份 triad、refs_v4、m13g_arrival_mapper_v1）；
- **env release v1 本身**（训练放行以 env 放行为前提）；
- **冻结训练配置 v1**（`configs/training/idc_training_config_v1.json`）；
- **M9.2 矩阵 v3**（预登记批次顺序 / origin 池 / 种子）；
- **正式训练入口与其复用的训练闭环**（`safe_rl_v2/formal_train.py`、
  `safe_rl_v2/formal_train_loop.py`）。

**不设自报时间戳**：验签输入全部来自 live 文件 hash 与 live git revision。
**没有信任边界参数**：loader 只接受唯一 canonical 路径。
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

TRAIN_RELEASE_SCHEMA = "idc-formal-train-release-v1"
CANONICAL_TRAIN_RELEASE_LOGICAL = "configs/release/idc_formal_train_release_v1.json"
APPROVED_DECISION_ID = "M1.3g-f-c-k"

#: 发布代码 revision / dirty 检查覆盖的**实现文件**（在 env release 的集合上加入
#: 正式训练入口与其复用的训练闭环）。
TRAIN_RELEASE_SOURCE_PATHS: tuple[str, ...] = (
    "envs/idc_price_env.py",
    "scenario/env_injection.py",
    "scenario/env_release.py",
    "scenario/formal_train_release.py",
    "scripts/materialize_env_release.py",
    "scripts/materialize_train_release.py",
    "safe_rl_v2/formal_train.py",
    "safe_rl_v2/formal_train_loop.py",
)

#: 逐字节绑定的依赖闭包（角色 → 仓库相对逻辑路径）。
TRAIN_RELEASE_BINDING_LOGICAL: dict[str, str] = {
    "formal_split_v5_train": "data/manifest/formal_splits_v5/train.json",
    "formal_split_v5_validation": "data/manifest/formal_splits_v5/validation.json",
    "formal_split_v5_test": "data/manifest/formal_splits_v5/test.json",
    "frozen_refs_v4": "configs/frozen_refs/refs_v4.json",
    "m13g_arrival_mapper_v1": "data/manifest/m13g_arrival_mapper_v1.json",
    "env_release_v1": "configs/release/idc_formal_env_release_v1.json",
    "training_config_v1": "configs/training/idc_training_config_v1.json",
    "experiment_matrix_v3": "configs/experiments/m9_experiment_matrix_v3.json",
    "formal_train_entry": "safe_rl_v2/formal_train.py",
    "formal_train_loop": "safe_rl_v2/formal_train_loop.py",
}

READINESS_KEYS: tuple[str, ...] = ("formal_env_ready", "formal_training_ready")
TRAIN_RELEASE_READINESS: dict[str, bool] = {
    "formal_env_ready": True,
    "formal_training_ready": True,
}

MANIFEST_TOP_KEYS: tuple[str, ...] = (
    "schema", "contract_version", "approved_decision_id", "supersedes",
    "readiness", "binds", "release_revision", "note",
)
BIND_ENTRY_KEYS: tuple[str, ...] = ("path", "sha256")

NOTE = (
    "formal train-only 训练发布批准（M1.3g-f-c-k）：env release v1 声明 formal_env_ready=true / "
    "formal_training_ready=false，本产物在其之上**单独**放行正式 train-only 训练循环。"
    "本产物不覆盖 v5 split 与 env release v1 的任何字节；不声称收敛或性能。"
)


class TrainReleaseError(ValueError):
    """训练发布产物的位置 / 结构 / 绑定 / revision 校验失败（一律 fail closed）。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainReleaseError(message)


def canonical_train_release_path() -> Path:
    return REPO_ROOT / CANONICAL_TRAIN_RELEASE_LOGICAL


def _sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise TrainReleaseError(f"绑定上游不可读：{path}：{error}") from error


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", *TRAIN_RELEASE_SOURCE_PATHS).strip())


def train_release_code_revision() -> str:
    revision = _git("log", "-1", "--format=%H", "--",
                    *TRAIN_RELEASE_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise TrainReleaseError(f"train release revision 无效：{revision!r}")
    return revision


def build_train_release() -> dict:
    """由 **live** 文件 hash + **live** revision 构建候选发布产物（确定性、无时间戳）。"""
    return {
        "schema": TRAIN_RELEASE_SCHEMA,
        "contract_version": CONTRACT_VERSION_ID,
        "approved_decision_id": APPROVED_DECISION_ID,
        "supersedes": None,
        "readiness": dict(TRAIN_RELEASE_READINESS),
        "binds": {
            role: {"path": logical, "sha256": _sha256_file(REPO_ROOT / logical)}
            for role, logical in TRAIN_RELEASE_BINDING_LOGICAL.items()
        },
        "release_revision": train_release_code_revision(),
        "note": NOTE,
    }


def _require_exact_keys(mapping: object, *, field: str, expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise TrainReleaseError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise TrainReleaseError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}")
    return mapping


def _require_hex64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(c not in "0123456789abcdef" for c in value):
        raise TrainReleaseError(f"{field} 必须是 64 位小写十六进制，实际 {value!r}")
    return value


def validate_train_release(payload: object) -> dict:
    """结构校验 + **整体重建比对**（live hash / live revision / readiness）。"""
    validated = _require_exact_keys(payload, field="release", expected=MANIFEST_TOP_KEYS)
    _require(validated["schema"] == TRAIN_RELEASE_SCHEMA,
             f"schema 必须等于 {TRAIN_RELEASE_SCHEMA!r}，实际 {validated['schema']!r}")
    _require(validated["contract_version"] == CONTRACT_VERSION_ID,
             f"contract_version 必须等于 {CONTRACT_VERSION_ID!r}")
    _require(validated["approved_decision_id"] == APPROVED_DECISION_ID,
             f"approved_decision_id 必须等于 {APPROVED_DECISION_ID!r}")
    _require(validated["supersedes"] is None, "supersedes 必须为 null")

    readiness = _require_exact_keys(validated["readiness"], field="readiness",
                                    expected=READINESS_KEYS)
    _require(readiness == TRAIN_RELEASE_READINESS,
             f"readiness 必须严格等于 {TRAIN_RELEASE_READINESS}；实际 {readiness}")

    binds = _require_exact_keys(validated["binds"], field="binds",
                                expected=tuple(TRAIN_RELEASE_BINDING_LOGICAL))
    for role, entry in binds.items():
        entry = _require_exact_keys(entry, field=f"binds.{role}", expected=BIND_ENTRY_KEYS)
        expected_path = TRAIN_RELEASE_BINDING_LOGICAL[role]
        _require(entry["path"] == expected_path,
                 f"binds.{role}.path 必须精确等于冻结逻辑路径 {expected_path!r}；"
                 f"实际 {entry['path']!r}")
        _require_hex64(entry["sha256"], field=f"binds.{role}.sha256")

    revision = validated["release_revision"]
    _require(isinstance(revision, str) and len(revision) == 40
             and all(c in "0123456789abcdef" for c in revision),
             f"release_revision 必须是 40 位小写十六进制，实际 {revision!r}")
    note = validated["note"]
    _require(isinstance(note, str) and bool(note), "note 必须是非空字符串")

    if _generator_is_dirty():
        raise TrainReleaseError("发布实现有未提交修改：拒绝用旧 revision 为未提交代码背书")
    live = train_release_code_revision()
    _require(validated["release_revision"] == live,
             f"release_revision 必须是**当前**实现的 revision {live}；"
             f"实际 {validated['release_revision']}")

    rebuilt = build_train_release()
    if rebuilt != validated:
        differing = sorted(k for k in set(rebuilt) | set(validated)
                           if rebuilt.get(k) != validated.get(k))
        raise TrainReleaseError(
            f"发布产物与由 live 文件 hash + live revision 重建的候选不符；"
            f"差异顶层字段={differing}")
    return validated


def load_verified_train_release() -> dict:
    """读取并严格校验 canonical 训练发布产物（无 fallback、无信任边界参数）。"""
    from scenario.arrival_mapper import load_verified_mapper_manifest
    from scenario.b6_refs import load_verified_refs_v4
    from scenario.b6_split_manifests import load_verified_split_manifest_v5
    from scenario.env_release import load_verified_env_release

    path = canonical_train_release_path()
    if path.is_symlink():
        raise TrainReleaseError(f"发布产物不得是 symlink：{path}")
    if not path.is_file():
        raise TrainReleaseError(
            f"缺少正式训练发布产物 {CANONICAL_TRAIN_RELEASE_LOGICAL}："
            "训练尚未放行（**不**回退、**不**忽略 env release v1）")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainReleaseError(f"{path} 不可读或不是合法 JSON：{error}") from error
    validated = validate_train_release(payload)

    # 复用既有 verified public 链：被绑定的上游必须**语义**有效
    load_verified_env_release()
    load_verified_refs_v4()
    load_verified_mapper_manifest()
    for split in ("train", "validation", "test"):
        load_verified_split_manifest_v5(expected_split=split)
    return validated


def write_train_release(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True,
                                    ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "APPROVED_DECISION_ID",
    "CANONICAL_TRAIN_RELEASE_LOGICAL",
    "TRAIN_RELEASE_BINDING_LOGICAL",
    "TRAIN_RELEASE_READINESS",
    "TRAIN_RELEASE_SCHEMA",
    "TRAIN_RELEASE_SOURCE_PATHS",
    "TrainReleaseError",
    "build_train_release",
    "canonical_train_release_path",
    "load_verified_train_release",
    "train_release_code_revision",
    "validate_train_release",
    "write_train_release",
]
