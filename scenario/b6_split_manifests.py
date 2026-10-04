"""M1.3f-e-b2-b：**B6 formal split manifest v5** 的唯一 schema / 构造 / 校验实现。

三份 manifest（`data/manifest/formal_splits_v5/{train,validation,test}.json`）
**只声明** B6 链的**冻结信任链**与**合法 origin 集合**：

- **不保存** forecast 数值、未来 truth、默认曲线或 origin 抽样结果；
- **不启动** env 或训练；**不固定** H/C。

## 与 v4 的关系

**行范围、时间范围、history/candidate-origin 规则与 v4 完全一致**
（train `[48,10224)`、validation `[0,2928)`、test `[0,4416)`，**split-local**）；
schema **bump** 到 v5，输入链换成 **B6**：policy-v3、B6 policy、
exogenous **v3** 三项、**refs_v4**。

## 版本隔离

v1（`data/manifest/<split>.json`）、v2、v3、v4 **全部 superseded**，
只接受 v5 的**唯一 canonical 位置**；副本、别名、symlink、伪造 hash/revision、
旧版本 fallback 一律拒绝。

## 单一入口

`load_verified_split_manifest_v5()` 是本模块**唯一**的公开 reader——
物化器与正式入口**必须**复用它；`build_split_manifest_v5()` 在返回前会**真正
调用**既有严格链（refs / policy-v3 / B6 policy / truth split / exogenous v3 bundle）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from scenario.formal_scenario_b6 import (
    B6_POLICY_LOGICAL,
    CANONICAL_MANIFEST_LOGICAL,
    CANONICAL_PARQUET_LOGICAL,
    EXOGENOUS_V3_MANIFEST_LOGICAL,
    EXOGENOUS_V3_PARQUET_LOGICAL,
    EXOGENOUS_V3_SOURCE_LOGICAL,
    SPLIT_MANIFEST_LOGICAL,
    load_verified_policy_v3,
)
from scenario.splits import (
    FREQUENCY,
    SPLIT_NAMES,
    SPLIT_SPECS,
    STEP_MINUTES,
    SplitName,
    load_truth_split,
    logical_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_SCHEMA = "m1.3feb2b-formal-split-manifest-v5"
CONTRACT_VERSION = "contract-v9"
TIMEZONE = "Asia/Singapore"
HISTORY_STEPS = 48

FORMAL_SPLIT_V5_SUBDIR = "formal_splits_v5"
REFS_V4_LOGICAL = "configs/frozen_refs/refs_v4.json"
POLICY_V3_LOGICAL = "data/manifest/singapore_2024_forecast_policy_v3.json"

# 每个 split 的合法候选 origin 起点（与 v4 **完全一致**）
CANDIDATE_ORIGIN_START: dict[str, int] = {
    "train": HISTORY_STEPS, "validation": 0, "test": 0,
}

INPUT_ROLES: tuple[str, ...] = (
    "canonical_parquet",
    "canonical_manifest",
    "split_manifest",
    "forecast_policy_manifest",
    "b6_intensity_policy",
    "exogenous_manifest",
    "exogenous_source_manifest",
    "exogenous_parquet",
    "frozen_refs",
)
TOP_KEYS: tuple[str, ...] = (
    "schema", "contract_version", "split", "split_rows", "time_range", "frequency",
    "history_steps", "candidate_origins", "inputs", "readiness",
    "materializer_revision", "frozen_at_utc",
)
SPLIT_ROWS_KEYS: tuple[str, ...] = ("start", "end_exclusive", "count")
TIME_RANGE_KEYS: tuple[str, ...] = ("timezone", "start", "end_exclusive")
ORIGIN_KEYS: tuple[str, ...] = ("start", "end_exclusive")
READINESS_KEYS: tuple[str, ...] = ("formal_training_ready", "formal_env_ready")
READINESS: dict[str, bool] = {
    "formal_training_ready": False,
    "formal_env_ready": False,
}
INPUT_ENTRY_KEYS: tuple[str, ...] = ("path", "sha256")

FORBIDDEN_FIELD_TOKENS: tuple[str, ...] = (
    "forecast", "origin_index", "future", "truth", "default_curve",
)

# 本模块实现文件（dirty 检查与 revision 使用**同一**集合）。
#
# R1：与 `B6_REFS_SOURCE_PATHS` **同一覆盖面**——本模块、refs 实现、
# **public cutover**、B6 构造器、B6 policy、v3 驱动表、splits、两个 materializer。
B6_SPLIT_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/b6_split_manifests.py",
    "scenario/b6_refs.py",
    "scenario/scenario.py",
    "scenario/formal_scenario_b6.py",
    "scenario/arrival_intensity_policy.py",
    "scenario/exogenous_drivers_b6.py",
    "scenario/splits.py",
    "scripts/materialize_b6_refs.py",
    "scripts/materialize_b6_split_manifests.py",
)


class SplitManifestV5Error(ValueError):
    """v5 split manifest 的**明确失败**（无 fallback、不静默）。"""


# --- 路径 / Git ----------------------------------------------------------------

def _data_root() -> Path:
    """资产根（**私有**；测试只能 monkeypatch 它）。"""
    return REPO_ROOT


def _canonical_split_dir() -> Path:
    return _data_root() / "data" / "manifest" / FORMAL_SPLIT_V5_SUBDIR


def canonical_manifest_path_v5(split: str) -> Path:
    if split not in SPLIT_NAMES:
        raise SplitManifestV5Error(
            f"未知 split：{split!r}，必须属于 {list(SPLIT_NAMES)}")
    return _canonical_split_dir() / f"{split}.json"


def manifest_relative_path_v5(split: SplitName) -> str:
    """正式 v5 triad 的**唯一公开路径来源**。"""
    return logical_repo_path(canonical_manifest_path_v5(split))


def _superseded_dirs() -> tuple[Path, ...]:
    root = _data_root() / "data" / "manifest"
    return (
        root,                      # v1：data/manifest/<split>.json
        root / "formal_splits_v2",
        root / "formal_splits_v3",
        root / "formal_splits_v4",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", *B6_SPLIT_SOURCE_PATHS).strip())


def resolve_materializer_revision() -> str:
    from scenario.portable_numeric import verified_recipe_revision

    revision = _git("log", "-1", "--format=%H", "--",
                    *B6_SPLIT_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise SplitManifestV5Error(f"materializer revision 无效：{revision!r}")
    return verified_recipe_revision(REPO_ROOT, B6_SPLIT_SOURCE_PATHS, revision, 'splits')


def sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise SplitManifestV5Error(f"冻结资产不可读：{path}：{error}") from error


def utc_now() -> str:
    from datetime import UTC

    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --- 严格 helper ---------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitManifestV5Error(message)


def _require_plain_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitManifestV5Error(f"{field} 必须是整数（bool 不算），实际 {value!r}")
    return value


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SplitManifestV5Error(f"{field} 必须是非空字符串，实际 {value!r}")
    return value


def _require_exact_keys(mapping: object, *, field: str,
                        expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise SplitManifestV5Error(
            f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise SplitManifestV5Error(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}"
            "（未来扩展必须 bump schema）"
        )
    return mapping


def _require_canonical_local(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SplitManifestV5Error(f"{field} 不是合法 ISO-8601：{text!r}") from error
    if parsed.tzinfo is None or parsed.isoformat() != text:
        raise SplitManifestV5Error(
            f"{field} 必须是带时区的规范 ISO-8601，实际 {text!r}")
    return text


def _require_canonical_utc(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SplitManifestV5Error(f"{field} 不是合法 ISO-8601：{text!r}") from error
    if parsed.tzinfo is None or parsed.isoformat() != text or not text.endswith("+00:00"):
        raise SplitManifestV5Error(
            f"{field} 必须是规范 UTC ISO-8601，实际 {text!r}")
    return text


def _require_git_sha40(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
        raise SplitManifestV5Error(f"{field} 必须是 40 位小写 Git SHA，实际 {text!r}")
    return text


def _require_hex64(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise SplitManifestV5Error(f"{field} 必须是 64 位小写十六进制，实际 {text!r}")
    return text


def _require_canonical_logical_path(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    if text != text.strip() or any(character.isspace() for character in text):
        raise SplitManifestV5Error(f"{field} 不得含空白：{text!r}")
    if text.startswith("/") or "\\" in text:
        raise SplitManifestV5Error(f"{field} 必须是仓库相对 POSIX 路径：{text!r}")
    for segment in text.split("/"):
        if segment in ("", ".", ".."):
            raise SplitManifestV5Error(f"{field} 不得含空/`.`/`..` 片段：{text!r}")
    return text


# --- 输入解析 -----------------------------------------------------------------

def default_inputs() -> dict[str, Path]:
    """B6 正式链的**唯一**冻结输入（**不可由调用者覆盖**）。"""
    root = _data_root()
    return {
        "canonical_parquet": root / CANONICAL_PARQUET_LOGICAL,
        "canonical_manifest": root / CANONICAL_MANIFEST_LOGICAL,
        "split_manifest": root / SPLIT_MANIFEST_LOGICAL,
        "forecast_policy_manifest": root / POLICY_V3_LOGICAL,
        "b6_intensity_policy": root / B6_POLICY_LOGICAL,
        "exogenous_manifest": root / EXOGENOUS_V3_MANIFEST_LOGICAL,
        "exogenous_source_manifest": root / EXOGENOUS_V3_SOURCE_LOGICAL,
        "exogenous_parquet": root / EXOGENOUS_V3_PARQUET_LOGICAL,
        "frozen_refs": root / REFS_V4_LOGICAL,
    }


def production_logical_paths() -> dict[str, str]:
    return {role: logical_repo_path(p) for role, p in default_inputs().items()}


def _require_live_binding(declared: object) -> None:
    """声明的九个角色必须与**实际生产资产**逐项相符。"""
    from scenario.b6_refs import load_verified_refs_v4

    entries = _require_exact_keys(declared, field="inputs", expected=INPUT_ROLES)
    production = default_inputs()
    paths = production_logical_paths()
    for role in INPUT_ROLES:
        entry = _require_exact_keys(entries[role], field=f"inputs.{role}",
                                    expected=INPUT_ENTRY_KEYS)
        declared_path = _require_canonical_logical_path(
            entry["path"], field=f"inputs.{role}.path")
        if declared_path != paths[role]:
            raise SplitManifestV5Error(
                f"inputs.{role}.path 必须精确等于固定生产 logical path "
                f"{paths[role]!r}，实际 {declared_path!r}"
            )
        declared_sha = _require_hex64(entry["sha256"], field=f"inputs.{role}.sha256")
        measured = sha256_file(production[role])
        if declared_sha != measured:
            raise SplitManifestV5Error(
                f"inputs.{role}.sha256 与实际资产字节不符："
                f"声明={declared_sha} 实测={measured}"
            )
    # refs_v4 必须通过其**正式 loader**（v3/v2 会被明确拒绝）
    load_verified_refs_v4(production["frozen_refs"])


def _verify_frozen_chain(inputs: dict[str, Path], split: str) -> None:
    """**实际调用**既有严格链（不只相信 manifest 自述）。"""
    from scenario.exogenous_drivers_b6 import load_verified_v3_bundle

    load_truth_split(
        split,  # type: ignore[arg-type]
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
    )
    load_verified_policy_v3(inputs["forecast_policy_manifest"])
    from scenario.arrival_intensity_policy import load_verified_b6_policy

    load_verified_b6_policy()
    load_verified_v3_bundle()


# --- 构造 ---------------------------------------------------------------------

def expected_candidate_origins(split: str, split_rows: dict) -> dict[str, int]:
    start = _require_plain_int(split_rows.get("start"), field="split_rows.start")
    end = _require_plain_int(split_rows.get("end_exclusive"),
                             field="split_rows.end_exclusive")
    return {"start": CANDIDATE_ORIGIN_START[split], "end_exclusive": end - start}


def _refs_v4_frozen_at_utc() -> str:
    """v5 triad 的**唯一**冻结时刻锚点：`refs_v4` 的 `frozen_at_utc`。

    **R1**：`frozen_at_utc` 若可自由填写，则「改它」在重建比对下**自洽**，
    无法被发现。这里把它**绑到一个 live 锚点**（已验签的 `refs_v4`），
    使任何与锚点不符的取值都被重建比对拒绝。
    """
    from scenario.b6_refs import load_verified_refs_v4

    return str(load_verified_refs_v4()["frozen_at_utc"])


def build_split_manifest_v5(split: str, *,
                            frozen_at_utc: str | None = None) -> dict:
    """构造候选 v5 manifest（**只读**上游，逐层严格校验）。

    **公开签名只接受 `split` 与冻结时间戳**：输入路径由本模块**固定**，
    不存在 `inputs` mapping、`expected_*` 信任根、DataFrame 注入或 `**kwargs`。
    `frozen_at_utc` 省略时取 **`refs_v4` 的冻结时刻**（唯一锚点）。
    """
    if split not in SPLIT_NAMES:
        raise SplitManifestV5Error(
            f"未知 split：{split!r}，必须属于 {list(SPLIT_NAMES)}")
    resolved = default_inputs()
    _verify_frozen_chain(resolved, split)

    payload = json.loads(resolved["split_manifest"].read_text(encoding="utf-8"))
    entry = payload["splits"][split]
    start, end_exclusive = int(entry["row_start"]), int(entry["row_end_exclusive"])

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "split": split,
        "split_rows": {
            "start": start,
            "end_exclusive": end_exclusive,
            "count": int(entry["row_count"]),
        },
        "time_range": {
            "timezone": TIMEZONE,
            "start": str(entry["start"]),
            "end_exclusive": str(entry["end_exclusive"]),
        },
        "frequency": FREQUENCY,
        "history_steps": HISTORY_STEPS,
        "candidate_origins": expected_candidate_origins(
            split, {"start": start, "end_exclusive": end_exclusive}),
        "inputs": {
            role: {"path": logical_repo_path(path), "sha256": sha256_file(path)}
            for role, path in resolved.items()
        },
        "readiness": dict(READINESS),
        "materializer_revision": resolve_materializer_revision(),
        "frozen_at_utc": frozen_at_utc or _refs_v4_frozen_at_utc(),
    }
    text = json.dumps(manifest, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise SplitManifestV5Error("manifest 含绝对路径")
    return _validate_structure(manifest, expected_split=split)


# --- 结构校验 -----------------------------------------------------------------

def _validate_structure(payload: object, *, expected_split: str) -> dict:
    if expected_split not in SPLIT_NAMES:
        raise SplitManifestV5Error(f"未知 split：{expected_split!r}")
    payload = _require_exact_keys(payload, field="v5 split manifest",
                                  expected=TOP_KEYS)
    _require(payload["schema"] == MANIFEST_SCHEMA,
             f"schema 必须是 {MANIFEST_SCHEMA!r}，实际 {payload['schema']!r}")
    _require(payload["contract_version"] == CONTRACT_VERSION,
             f"contract_version 必须是 {CONTRACT_VERSION!r}")
    _require(payload["split"] == expected_split,
             f"split 必须是 {expected_split!r}，实际 {payload['split']!r}")
    _require(payload["frequency"] == FREQUENCY,
             f"frequency 必须是 {FREQUENCY!r}")
    _require(_require_plain_int(payload["history_steps"], field="history_steps")
             == HISTORY_STEPS, f"history_steps 必须是 {HISTORY_STEPS}")

    split_rows = _require_exact_keys(payload["split_rows"], field="split_rows",
                                     expected=SPLIT_ROWS_KEYS)
    start = _require_plain_int(split_rows["start"], field="split_rows.start")
    end = _require_plain_int(split_rows["end_exclusive"],
                             field="split_rows.end_exclusive")
    count = _require_plain_int(split_rows["count"], field="split_rows.count")
    _require(0 <= start < end, f"split_rows 区间非法：[{start}, {end})")
    _require(count == end - start, f"split_rows.count 必须等于 {end - start}")
    spec = SPLIT_SPECS[expected_split]
    _require(start == spec["row_start"] and end == spec["row_end_exclusive"],
             f"split_rows 必须精确等于冻结的 [{spec['row_start']}, "
             f"{spec['row_end_exclusive']})")

    time_range = _require_exact_keys(payload["time_range"], field="time_range",
                                     expected=TIME_RANGE_KEYS)
    _require(time_range["timezone"] == TIMEZONE,
             f"time_range.timezone 必须是 {TIMEZONE!r}")
    _require_canonical_local(time_range["start"], field="time_range.start")
    _require_canonical_local(time_range["end_exclusive"],
                             field="time_range.end_exclusive")
    _require(datetime.fromisoformat(time_range["start"])
             < datetime.fromisoformat(time_range["end_exclusive"]),
             "time_range.start 必须早于 end_exclusive")

    origins = _require_exact_keys(payload["candidate_origins"],
                                  field="candidate_origins", expected=ORIGIN_KEYS)
    expected_origins = expected_candidate_origins(expected_split, split_rows)
    _require(origins == expected_origins,
             f"candidate_origins 必须是 {expected_origins}，实际 {origins}")

    _require_live_binding(payload["inputs"])

    readiness = _require_exact_keys(payload["readiness"], field="readiness",
                                    expected=READINESS_KEYS)
    _require(readiness == READINESS,
             f"readiness 必须严格等于 {READINESS}（正式 env / 训练尚未就绪）")
    _require_git_sha40(payload["materializer_revision"],
                       field="materializer_revision")
    _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")
    _require_no_forbidden_fields(payload)
    _require_steps_are_30min(payload)
    return payload


def _require_no_forbidden_fields(payload: dict) -> None:
    for key in payload:
        lowered = key.lower()
        for token in FORBIDDEN_FIELD_TOKENS:
            if token in lowered:
                raise SplitManifestV5Error(
                    f"manifest 不得含字段 {key!r}（命中禁用词 {token!r}）")


def _require_steps_are_30min(payload: dict) -> None:
    start = datetime.fromisoformat(payload["time_range"]["start"])
    end = datetime.fromisoformat(payload["time_range"]["end_exclusive"])
    steps = payload["split_rows"]["count"]
    if end - start != timedelta(minutes=STEP_MINUTES * steps):
        raise SplitManifestV5Error(
            f"time_range 跨度与 {steps} 个 {STEP_MINUTES} 分钟步不符")


# --- 唯一加载入口 --------------------------------------------------------------

def _require_canonical_location(path: Path | str, *, expected_split: str) -> Path:
    """路径必须**精确等于** v5 的唯一 canonical manifest（在读 JSON 之前）。"""
    given = Path(path).absolute()
    expected = canonical_manifest_path_v5(expected_split).absolute()
    if given == expected:
        if expected.is_symlink():
            raise SplitManifestV5Error(f"v5 manifest 不得是 symlink：{expected}")
        return expected
    for legacy in _superseded_dirs():
        if given.parent == legacy.absolute():
            raise SplitManifestV5Error(
                f"split manifest 位于已被取代的位置 {logical_repo_path(legacy)}/："
                f"B6 正式链只接受 "
                f"{logical_repo_path(_canonical_split_dir())}/（无 fallback）"
            )
    raise SplitManifestV5Error(
        f"split manifest 路径必须是 {expected_split!r} 的**唯一 canonical** 文件 "
        f"{logical_repo_path(expected)}；实际 {logical_repo_path(given)}"
        "（不接受副本、别名、symlink 或其它生成版本）"
    )


def load_verified_split_manifest_v5(path: Path | str | None = None, *,
                                    expected_split: str) -> dict:
    """**唯一**的「验证并加载」公开入口（物化器与正式入口共用）。

    依次完成：结构校验 → **实时输入绑定**（九个角色 path/hash 对生产资产）→
    **refs_v4 走正式 loader**（v3/v2 拒绝）→ revision 必须等于**当前**实现 →
    **实际调用**既有严格链（truth split / policy-v3 / B6 policy / v3 bundle）。
    """
    if expected_split not in SPLIT_NAMES:
        raise SplitManifestV5Error(f"未知 split：{expected_split!r}")
    if path is None:
        path = canonical_manifest_path_v5(expected_split)
    path = _require_canonical_location(path, expected_split=expected_split)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitManifestV5Error(
            f"{path} 不可读或不是合法 JSON：{error}") from error
    try:
        validated = _validate_structure(payload, expected_split=expected_split)
    except SplitManifestV5Error:
        raise
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise SplitManifestV5Error(f"{path} 结构畸形：{error}") from error

    if _generator_is_dirty():
        raise SplitManifestV5Error(
            "B6 split/materializer 实现有未提交修改：拒绝读取正式 v5 manifest")
    live = resolve_materializer_revision()
    if validated["materializer_revision"] != live:
        raise SplitManifestV5Error(
            f"materializer_revision 必须是**当前**实现的 revision {live}；"
            f"实际 {validated['materializer_revision']}"
        )

    # **R1 修复**：整体**重建比对**。此前只校验 time_range 的格式与持续时间，
    # 因此「时长不变、但起止时刻被改」的伪造（以及 frozen_at_utc、split_rows、
    # candidate_origins、readiness、九角色 path/sha……）可以**静默通过**。
    rebuilt = build_split_manifest_v5(
        expected_split, frozen_at_utc=_refs_v4_frozen_at_utc())
    if rebuilt != validated:
        differing = sorted(
            key for key in set(rebuilt) | set(validated)
            if rebuilt.get(key) != validated.get(key)
        )
        raise SplitManifestV5Error(
            f"v5 manifest 与由 trusted inputs + live hashes + live revision "
            f"重建的 candidate 不符；差异顶层字段={differing}"
        )
    return validated


# --- start → split-local origin 的**精确**映射 --------------------------------

def local_origin_from_start(split: str, start: str) -> int:
    """把带时区的 `start` 精确映射到该 split 的**本地 origin**。

    要求：`start` 必须是**带时区**的规范 ISO、落在 canonical 的**严格 30 分钟网格**上、
    位于该 split 的行范围内、且属于**候选 origin** 集合。任一不符 **fail closed**。
    """
    if split not in SPLIT_NAMES:
        raise SplitManifestV5Error(f"未知 split：{split!r}")
    text = _require_str(start, field="start")
    try:
        stamp = pd.Timestamp(text)
    except (ValueError, TypeError) as error:
        raise SplitManifestV5Error(f"start 不是合法时间戳：{start!r}") from error
    if stamp.tzinfo is None:
        raise SplitManifestV5Error(
            f"start 必须带显式时区偏移（例如 +08:00），实际 {start!r}")
    if pd.Timestamp(text).isoformat() != text:
        raise SplitManifestV5Error(
            f"start 必须是规范带的带时区 ISO-8601，实际 {start!r}")

    from scenario.formal_scenario_b6 import _canonical_parquet_path  # noqa: PLC0415

    stamps = pd.DatetimeIndex(pd.read_parquet(_canonical_parquet_path())["timestamp"])
    matches = stamps[stamps == stamp]
    if len(matches) != 1:
        raise SplitManifestV5Error(
            f"start {start!r} 不在 canonical 的严格 30 分钟网格上（越界或非网格）"
        )
    global_index = int(stamps.get_loc(matches[0]))
    spec = SPLIT_SPECS[split]
    if not (spec["row_start"] <= global_index < spec["row_end_exclusive"]):
        raise SplitManifestV5Error(
            f"start {start!r}（全局行 {global_index}）不属于 {split} 的行范围 "
            f"[{spec['row_start']}, {spec['row_end_exclusive']})"
        )
    local = global_index - spec["row_start"]
    origins = expected_candidate_origins(split, {
        "start": spec["row_start"], "end_exclusive": spec["row_end_exclusive"]})
    if not (origins["start"] <= local < origins["end_exclusive"]):
        raise SplitManifestV5Error(
            f"start {start!r} 映射到 {split} 的本地 origin {local}，"
            f"不在候选 origin 集合 [{origins['start']}, {origins['end_exclusive']}) 内"
        )
    return local


# --- 物化 ---------------------------------------------------------------------

def materialize_split_manifest_triad_v5(*, frozen_at_utc: str | None = None) -> dict:
    """物化 v5 triad：**共享一个** `frozen_at_utc`，原子、幂等、拒绝覆盖。"""
    out_dir = _canonical_split_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if _generator_is_dirty():
        raise SplitManifestV5Error(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书")

    # **R1**：既有的 v5 文件必须先**干净地**失败（不得泄漏 KeyError/TypeError）。
    existing: list[dict] = []
    for split in SPLIT_NAMES:
        path = out_dir / f"{split}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "frozen_at_utc" not in payload:
                raise SplitManifestV5Error(
                    f"既有的 {path} 不是合法的 v5 manifest（顶层或 frozen_at_utc 缺失）："
                    "拒绝覆盖"
                )
            _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")
        except (OSError, json.JSONDecodeError) as error:
            raise SplitManifestV5Error(
                f"既有的 {path} 不可读或不是合法 JSON：{error}") from error
        except SplitManifestV5Error:
            raise
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise SplitManifestV5Error(f"既有的 {path} 结构畸形：{error}") from error
        existing.append(payload)
    if existing and len({p["frozen_at_utc"] for p in existing}) != 1:
        raise SplitManifestV5Error("既有的 v5 manifest 之间 frozen_at_utc 不一致")
    anchor = _refs_v4_frozen_at_utc()
    stamp = frozen_at_utc or (existing[0]["frozen_at_utc"] if existing else anchor)
    if stamp != anchor:
        raise SplitManifestV5Error(
            f"v5 triad 的 frozen_at_utc 必须是 refs_v4 的冻结时刻 {anchor!r}；"
            f"实际 {stamp!r}")

    written: list[Path] = []
    try:
        for split in SPLIT_NAMES:
            target = out_dir / f"{split}.json"
            candidate = build_split_manifest_v5(split, frozen_at_utc=stamp)
            if target.is_file():
                current = json.loads(target.read_text(encoding="utf-8"))
                if current != candidate:
                    raise SplitManifestV5Error(
                        f"{target} 与候选**语义不同**：拒绝覆盖已冻结的 v5 manifest")
                continue
            _atomic_write_text(target, _canonical_json(candidate))
            written.append(target)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {"out_dir": out_dir, "frozen_at_utc": stamp, "written": written}


def _atomic_write_text(path: Path, text: str) -> None:
    import os
    import tempfile
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = _Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物化 B6 formal split manifest v5（M1.3f-e-b2-b）")
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    parser.add_argument("--frozen-at-utc", default=None)
    args = parser.parse_args(argv)

    try:
        if args.verify:
            for split in SPLIT_NAMES:
                load_verified_split_manifest_v5(expected_split=split)
            print(f"verified {_canonical_split_dir()}")
            return 0
        result = materialize_split_manifest_triad_v5(frozen_at_utc=args.frozen_at_utc)
    except (SplitManifestV5Error, OSError, ValueError) as error:
        print(f"materialize_b6_split_manifests: {error}", file=sys.stderr)
        return 1

    print(f"out_dir={result['out_dir']}")
    print(f"frozen_at_utc={result['frozen_at_utc']}")
    print(f"written={[str(p) for p in result['written']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
