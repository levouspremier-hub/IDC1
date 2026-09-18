"""M1.3g-c：**正式 split manifest** 的唯一 schema / 构造 / 校验实现。

三份 manifest（`data/manifest/{train,validation,test}.json`）**只声明**正式
数据 / forecast / ref 的**冻结信任链**与**合法 origin 集合**：

- **不保存** forecast 数值、未来 truth、默认曲线或 origin 抽样结果；
- **不启动** env 或训练；**不固定** H/C（实际边界由后续调用时的
  `validate_episode_origin` / `validate_forecast_origin` 严格执行）。

## 信任链逐字绑定

八个输入角色（canonical parquet / canonical manifest / split manifest /
policy-v2 / exogenous v2 manifest / source v3 manifest / exogenous v2 parquet /
**`refs_v3.json`**）各记 `path` + **实测** SHA-256。
`configs/frozen_refs/refs.json`（v2）**明确拒绝**，不得作为备选或回退。

## 单一「验证并加载」入口

`load_verified_split_manifest()` 是本模块**唯一**的公开 reader——
它做**结构校验 + 实时输入绑定 + 既有严格链调用**，物化器与未来 reader
**都必须**复用它，避免较弱的实现漂移。

## 物化前**实际**调用既有严格链

`build_split_manifest()` 会**真正**调用既有严格链——**不只**相信任何 manifest 的自述：

1. `load_frozen_refs`（refs 的严格校验）**外加**与冻结 SHA-256 的**逐字节**比对；
2. `read_forecast_policy_manifest`（policy-v2 的精确键集合 / 契约版本 / revision）；
3. `load_truth_split`（M1.3d 的完整严格链：canonical manifest → parquet、
   整条时间轴、精确键集合、train-only 统计重算）；
4. `load_verified_exogenous`（外生链逐字节 + 与本次调用对象的**交叉绑定**）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from scenario.forecast import provider_code_revision, read_forecast_policy_manifest
from scenario.formal_scenario import load_verified_exogenous
from scenario.splits import (
    FREQUENCY,
    SPLIT_NAMES,
    STEP_MINUTES,
    SplitName,
    load_truth_split,
    logical_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_SCHEMA = "m1.3g-formal-split-manifest-v4"
CONTRACT_VERSION = "contract-v9"
TIMEZONE = "Asia/Singapore"
HISTORY_STEPS = 48

MANIFEST_DIR = REPO_ROOT / "data/manifest"
# 正式 triad 的**唯一**输出目录与路径来源（R3：未来 g-e/g-f 只能读取这里）
FORMAL_SPLIT_DIR = MANIFEST_DIR / "formal_splits_v4"
# **已被取代**的三个历史位置（仅用于给出清晰的错误信息；
# 真正的硬门禁是下面的 canonical 路径**精确相等**检查）。
SUPERSEDED_SPLIT_DIRS: tuple[Path, ...] = (
    MANIFEST_DIR,                          # v1：superseded_pre_live_input_binding_fix
    MANIFEST_DIR / "formal_splits_v2",     # v2：superseded_pre_canonical_path_fix
    MANIFEST_DIR / "formal_splits_v3",     # v3：superseded_pre_canonical_loader_trust_boundary_fix
)
CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = MANIFEST_DIR / "singapore_2024_half_hour.json"
TRUTH_SPLIT_MANIFEST = MANIFEST_DIR / "singapore_2024_splits.json"
FORECAST_POLICY_MANIFEST = MANIFEST_DIR / "singapore_2024_forecast_policy_v2.json"
EXOGENOUS_MANIFEST = MANIFEST_DIR / "singapore_2024_exogenous_v2.json"
EXOGENOUS_SOURCE_MANIFEST = MANIFEST_DIR / "m13f_materialization_sources_v3.json"
EXOGENOUS_PARQUET = (
    REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
)
REFS_V3 = REPO_ROOT / "configs/frozen_refs/refs_v3.json"
# **被取代**的 v2 refs：明确拒绝，不得作为备选或回退
SUPERSEDED_REFS_V2 = REPO_ROOT / "configs/frozen_refs/refs.json"
# 正式链**唯一**的 refs 形态与它的**冻结** SHA-256（逐字节绑定）
REFS_V3_LOGICAL_PATH = "configs/frozen_refs/refs_v3.json"
EXPECTED_REFS_V3_SHA256 = (
    "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f"
)

INPUT_ROLES: tuple[str, ...] = (
    "canonical_parquet",
    "canonical_manifest",
    "split_manifest",
    "forecast_policy_manifest",
    "exogenous_manifest",
    "exogenous_source_manifest",
    "exogenous_parquet",
    "frozen_refs",
)
TOP_KEYS: tuple[str, ...] = (
    "schema",
    "contract_version",
    "split",
    "split_rows",
    "time_range",
    "frequency",
    "history_steps",
    "candidate_origins",
    "inputs",
    "readiness",
    "materializer_revision",
    "frozen_at_utc",
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

# 每个 split 的**合法候选 origin**（D5：具备 48-step 历史的**完整**集合）。
# train 从 48 起（其 48 步历史必须落在同一 split 内）；validation / test 的
# history 来自**之前已经发生**的 canonical，因此从 0 起。
CANDIDATE_ORIGIN_START: dict[str, int] = {
    "train": HISTORY_STEPS,
    "validation": 0,
    "test": 0,
}

# 物化实现的 Git revision 来源（dirty 检查使用**同一**集合）
MATERIALIZER_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/formal_split_manifests.py",
    "scripts/materialize_singapore_scenario_manifests.py",
    "scenario/formal_scenario.py",
    "scenario/splits.py",
)

# 明确禁止出现在 manifest 中的字段（forecast 数值 / 单点 origin / 未来真值）
FORBIDDEN_FIELD_TOKENS: tuple[str, ...] = (
    "forecast", "origin_index", "future", "truth", "default_curve",
)


class SplitManifestError(ValueError):
    """正式 split manifest 的**明确失败**。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitManifestError(message)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    """物化实现是否有未提交修改（含未跟踪的新文件）。"""
    return bool(_git("status", "--porcelain", "--", *MATERIALIZER_SOURCE_PATHS).strip())


def resolve_materializer_revision() -> str:
    """本物化实现的 Git revision（40 位小写 SHA，不用漂移的 HEAD）。"""
    revision = _git("log", "-1", "--format=%H", "--", *MATERIALIZER_SOURCE_PATHS).strip()
    _require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        f"materializer revision 无效：{revision!r}",
    )
    return revision


def sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise SplitManifestError(f"冻结资产不可读：{path}：{error}") from error


# --- 严格类型 helper（coercion 之前） ----------------------------------------

def _require_plain_int(value: object, *, field: str) -> int:
    """只接受真正的 `int`（**bool 不算**：`True` 是 `int` 的子类）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitManifestError(f"{field} 必须是整数（bool 不算），实际 {value!r}")
    return value


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SplitManifestError(f"{field} 必须是非空字符串，实际 {value!r}")
    return value


def _require_exact_keys(mapping: object, *, field: str, expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise SplitManifestError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise SplitManifestError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}"
            "（未来扩展必须 bump schema）"
        )
    return mapping


def _require_canonical_logical_path(value: object, *, field: str) -> str:
    """仓库相对规范 POSIX 路径：非空、无空白、相对、无反斜杠、无 `.`/`..`/空段。"""
    text = _require_str(value, field=field)
    if text != text.strip() or any(character.isspace() for character in text):
        raise SplitManifestError(f"{field} 不得含空白（含前后缀）：{text!r}")
    if text.startswith("/"):
        raise SplitManifestError(f"{field} 不得是绝对路径：{text!r}")
    if "\\" in text:
        raise SplitManifestError(f"{field} 必须是 POSIX 逻辑路径（不得含反斜杠）：{text!r}")
    for segment in text.split("/"):
        if segment in ("", ".", ".."):
            raise SplitManifestError(f"{field} 不得含空/`.`/`..` 路径片段：{text!r}")
    return text


def _require_hex64(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise SplitManifestError(f"{field} 必须是 64 位小写十六进制，实际 {text!r}")
    return text


def _require_canonical_utc(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SplitManifestError(f"{field} 不是合法 ISO-8601：{text!r}") from error
    if parsed.tzinfo is None or parsed.isoformat() != text or not text.endswith("+00:00"):
        raise SplitManifestError(f"{field} 必须是规范 UTC ISO-8601，实际 {text!r}")
    return text


def _require_canonical_local(value: object, *, field: str) -> str:
    """带时区的规范 ISO-8601（站点本地时区，形如 `+08:00`）。"""
    text = _require_str(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SplitManifestError(f"{field} 不是合法 ISO-8601：{text!r}") from error
    if parsed.tzinfo is None or parsed.isoformat() != text:
        raise SplitManifestError(f"{field} 必须是带时区的规范 ISO-8601，实际 {text!r}")
    return text


def _require_git_sha40(value: object, *, field: str) -> str:
    text = _require_str(value, field=field)
    if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
        raise SplitManifestError(f"{field} 必须是 40 位小写 Git SHA，实际 {text!r}")
    return text


# --- 构造 ---------------------------------------------------------------------

def canonical_split_dir() -> Path:
    """canonical triad 目录的**公开只读**入口（供物化器使用）。"""
    return _canonical_split_dir()


def default_inputs() -> dict[str, Path]:
    """正式链的**唯一**冻结输入（生产路径使用；**不可由调用者覆盖**）。"""
    return {
        "canonical_parquet": CANONICAL_PARQUET,
        "canonical_manifest": CANONICAL_MANIFEST,
        "split_manifest": TRUTH_SPLIT_MANIFEST,
        "forecast_policy_manifest": FORECAST_POLICY_MANIFEST,
        "exogenous_manifest": EXOGENOUS_MANIFEST,
        "exogenous_source_manifest": EXOGENOUS_SOURCE_MANIFEST,
        "exogenous_parquet": EXOGENOUS_PARQUET,
        "frozen_refs": REFS_V3,
    }


def production_logical_paths() -> dict[str, str]:
    """八个角色的**固定生产 logical path**（声明必须精确等于它们）。"""
    return {
        role: logical_repo_path(path) for role, path in default_inputs().items()
    }


def _require_canonical_location(path: Path | str, *, expected_split: str) -> Path:
    """**R3 硬门禁**：路径必须**精确等于**该 split 的唯一 canonical manifest。

    在**读取 JSON 之前**完成；因此：

    - 任意临时目录中的副本、仓库内其它目录、**路径别名**一律拒绝；
    - **symlink / symlink 目录**也不能绕过 —— 比较的是**词法绝对路径**
      （`absolute()` **不做**符号链接解析），指向 canonical 文件的 symlink
      仍然是不等于 canonical 的路径，因此被拒绝；
    - v1 / v2 / v3 三个历史位置给出**明确**的历史版本错误信息；
    - split 与 canonical 文件名不匹配（例如拿 validation 的路径去读 train）拒绝。
    """
    given = Path(path).absolute()
    expected = canonical_manifest_path(expected_split).absolute()
    if given == expected:
        return expected
    for legacy in SUPERSEDED_SPLIT_DIRS:
        if given.parent == legacy.absolute():
            raise SplitManifestError(
                f"split manifest 位于已被取代的位置 {logical_repo_path(legacy)}/："
                f"正式链只接受 {logical_repo_path(FORMAL_SPLIT_DIR)}/（无 fallback）"
            )
    raise SplitManifestError(
        f"split manifest 路径必须是 {expected_split!r} 的**唯一 canonical** 文件 "
        f"{logical_repo_path(expected)}；实际 {logical_repo_path(given)}"
        "（不接受副本、别名、symlink 或其它生成版本）"
    )


def _require_live_binding(declared: object) -> None:
    """**R1 核心**：声明的八个角色必须与**实际生产资产**逐项相符。

    - role 集合精确；
    - 每条 `path` 精确等于**固定生产 logical path**（拒绝外部/被取代路径）；
    - 每条 `sha256` 等于该实际文件的**实测字节** hash。
    """
    entries = _require_exact_keys(
        declared, field="inputs", expected=INPUT_ROLES)
    production = default_inputs()
    production_paths = production_logical_paths()
    for role in INPUT_ROLES:
        entry = _require_exact_keys(
            entries[role], field=f"inputs.{role}", expected=INPUT_ENTRY_KEYS)
        declared_path = _require_canonical_logical_path(
            entry["path"], field=f"inputs.{role}.path")
        if declared_path != production_paths[role]:
            raise SplitManifestError(
                f"inputs.{role}.path 必须精确等于固定生产 logical path "
                f"{production_paths[role]!r}，实际 {declared_path!r}"
            )
        declared_sha = _require_hex64(
            entry["sha256"], field=f"inputs.{role}.sha256")
        measured = sha256_file(production[role])
        if declared_sha != measured:
            raise SplitManifestError(
                f"inputs.{role}.sha256 与实际资产字节不符："
                f"声明={declared_sha} 实测={measured}"
            )


def _verify_frozen_chain(inputs: dict[str, Path], split: str) -> None:
    """**实际**调用既有严格校验（不只相信 manifest 自述）。

    1. `load_frozen_refs`：refs 的严格校验（v2 会被明确拒绝）；
    2. `load_truth_split`：M1.3d 的完整严格链（canonical manifest → parquet、
       整条时间轴、精确键集合、train-only 统计重算）；
    3. `load_verified_exogenous`：外生链逐字节 + 与本次调用对象的**交叉绑定**。
    """
    from scripts.freeze_refs import (  # 局部导入：避免模块级循环
        FreezeRefsError,
        load_frozen_refs,
    )

    refs_path = inputs["frozen_refs"]
    if refs_path.resolve() == SUPERSEDED_REFS_V2.resolve():
        raise SplitManifestError(
            "configs/frozen_refs/refs.json 是 superseded 的 v2："
            "正式链只接受 refs_v3.json"
        )
    try:
        load_frozen_refs(refs_path)
    except FreezeRefsError as error:
        raise SplitManifestError(f"frozen refs 校验失败：{error}") from error
    # **逐字节**绑定：refs 的实测 hash 必须等于冻结登记值
    measured_refs = sha256_file(refs_path)
    if measured_refs != EXPECTED_REFS_V3_SHA256:
        raise SplitManifestError(
            f"frozen refs 的 SHA-256 与冻结登记不符："
            f"期望 {EXPECTED_REFS_V3_SHA256} 实际 {measured_refs}"
        )

    # policy-v2 的严格读取（精确键集合、契约版本、revision 恒等）
    read_forecast_policy_manifest(
        inputs["forecast_policy_manifest"],
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
        expected_revision=provider_code_revision(),
    )

    load_truth_split(
        split,  # type: ignore[arg-type]
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
    )
    load_verified_exogenous(
        inputs["exogenous_manifest"],
        inputs["exogenous_source_manifest"],
        exogenous_parquet_path=inputs["exogenous_parquet"],
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
    )


def build_split_manifest(
    split: str,
    *,
    frozen_at_utc: str | None = None,
) -> dict:
    """构造候选正式 split manifest（**只读**上游，逐层严格校验）。

    **公开签名只接受 `split` 与冻结时间戳**：输入路径由本模块**固定**，
    不存在 `inputs` mapping、`materializer_revision`、`expected_*` 信任根参数、
    DataFrame 注入或 `**kwargs`（未知参数一律 `TypeError`）。
    """
    if split not in SPLIT_NAMES:
        raise SplitManifestError(f"未知 split：{split!r}，必须属于 {list(SPLIT_NAMES)}")
    resolved = default_inputs()
    _verify_frozen_chain(resolved, split)

    split_payload = json.loads(
        resolved["split_manifest"].read_text(encoding="utf-8"))
    entry = split_payload["splits"][split]
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
            split, {"start": start, "end_exclusive": end_exclusive}
        ),
        "inputs": {
            role: {
                "path": logical_repo_path(path),
                "sha256": sha256_file(path),
            }
            for role, path in resolved.items()
        },
        "readiness": dict(READINESS),
        "materializer_revision": resolve_materializer_revision(),
        "frozen_at_utc": frozen_at_utc or _now_utc(),
    }
    return _validate_structure(manifest, expected_split=split)


def utc_now() -> str:
    """规范 UTC 时间戳（**唯一**来源；物化器与构造器共用）。

    物化 triad 时必须在**外部**采样一次并传给三个 `build_split_manifest`，
    否则慢速首冻会写出互不相同的时间戳。
    """
    from datetime import UTC

    return datetime.now(UTC).replace(microsecond=0).isoformat()


# 兼容内部旧名
_now_utc = utc_now


def expected_candidate_origins(split: str, split_rows: dict) -> dict[str, int]:
    """由 row 区间推导**应有**的候选 origin 区间（D5，**split-local** 坐标）。

    `[CANDIDATE_ORIGIN_START[split], split 长度)`：
    train 为 `[48, 10224)`，validation 为 `[0, 2928)`，test 为 `[0, 4416)`。
    """
    start = _require_plain_int(split_rows.get("start"), field="split_rows.start")
    end = _require_plain_int(
        split_rows.get("end_exclusive"), field="split_rows.end_exclusive")
    return {
        "start": CANDIDATE_ORIGIN_START[split],
        "end_exclusive": end - start,
    }


# --- 唯一严格校验入口 ---------------------------------------------------------

def _validate_structure(payload: object, *, expected_split: str) -> dict:
    """**结构**校验（私有）：schema / type / key-set / 边界 / readiness。

    它**不**解析实际资产——实时绑定由 `_require_live_binding` 与
    `load_verified_split_manifest` 完成。
    """
    if expected_split not in SPLIT_NAMES:
        raise SplitManifestError(f"未知 split：{expected_split!r}")
    payload = _require_exact_keys(
        payload, field="formal split manifest", expected=TOP_KEYS)
    _require(
        payload["schema"] == MANIFEST_SCHEMA,
        f"schema 必须是 {MANIFEST_SCHEMA!r}，实际 {payload['schema']!r}",
    )
    _require(
        payload["contract_version"] == CONTRACT_VERSION,
        f"contract_version 必须是 {CONTRACT_VERSION!r}，"
        f"实际 {payload['contract_version']!r}",
    )
    _require(
        payload["split"] == expected_split,
        f"split 必须是 {expected_split!r}，实际 {payload['split']!r}",
    )
    _require(
        payload["frequency"] == FREQUENCY,
        f"frequency 必须是 {FREQUENCY!r}，实际 {payload['frequency']!r}",
    )
    _require(
        _require_plain_int(payload["history_steps"], field="history_steps")
        == HISTORY_STEPS,
        f"history_steps 必须是 {HISTORY_STEPS}",
    )

    split_rows = _require_exact_keys(
        payload["split_rows"], field="split_rows", expected=SPLIT_ROWS_KEYS)
    start = _require_plain_int(split_rows["start"], field="split_rows.start")
    end = _require_plain_int(
        split_rows["end_exclusive"], field="split_rows.end_exclusive")
    count = _require_plain_int(split_rows["count"], field="split_rows.count")
    _require(0 <= start < end, f"split_rows 区间非法：[{start}, {end})")
    _require(count == end - start, f"split_rows.count 必须等于 {end - start}")

    time_range = _require_exact_keys(
        payload["time_range"], field="time_range", expected=TIME_RANGE_KEYS)
    _require(
        time_range["timezone"] == TIMEZONE,
        f"time_range.timezone 必须是 {TIMEZONE!r}，实际 {time_range['timezone']!r}",
    )
    _require_canonical_local(time_range["start"], field="time_range.start")
    _require_canonical_local(
        time_range["end_exclusive"], field="time_range.end_exclusive")
    _require(
        datetime.fromisoformat(time_range["start"])
        < datetime.fromisoformat(time_range["end_exclusive"]),
        "time_range.start 必须早于 end_exclusive",
    )

    origins = _require_exact_keys(
        payload["candidate_origins"], field="candidate_origins", expected=ORIGIN_KEYS)
    expected_origins = expected_candidate_origins(expected_split, split_rows)
    _require(
        origins == expected_origins,
        f"candidate_origins 必须是 {expected_origins}（D5 的完整候选集合），"
        f"实际 {origins}",
    )

    # **实时绑定**：声明的八条角色必须与实际生产资产逐项相符
    _require_live_binding(payload["inputs"])

    readiness = _require_exact_keys(
        payload["readiness"], field="readiness", expected=READINESS_KEYS)
    _require(
        readiness == READINESS,
        f"readiness 必须严格等于 {READINESS}（正式 env / 训练尚未就绪）",
    )

    _require_git_sha40(
        payload["materializer_revision"], field="materializer_revision")
    _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")

    _require_no_forbidden_fields(payload)
    _require_steps_are_30min(payload)
    return payload


def _require_no_forbidden_fields(payload: dict) -> None:
    """**不得**出现 forecast 数值 / 单点 `origin_index` / 未来真值字段。"""
    for key in payload:
        lowered = key.lower()
        for token in FORBIDDEN_FIELD_TOKENS:
            if token in lowered:
                raise SplitManifestError(
                    f"manifest 不得含字段 {key!r}（命中禁用词 {token!r}："
                    "不保存 forecast 数值 / 未来真值 / 单点 origin）"
                )


def _require_steps_are_30min(payload: dict) -> None:
    from datetime import timedelta

    start = datetime.fromisoformat(payload["time_range"]["start"])
    end = datetime.fromisoformat(payload["time_range"]["end_exclusive"])
    delta = end - start
    steps = payload["split_rows"]["count"]
    expected = timedelta(minutes=STEP_MINUTES * steps)
    if delta != expected:
        raise SplitManifestError(
            f"time_range 跨度 {delta} 与 {steps} 个 {STEP_MINUTES} 分钟步不符"
            f"（应为 {expected}）"
        )


def load_verified_split_manifest(path: Path | str, *, expected_split: str) -> dict:
    """**唯一**的「验证并加载」公开入口（物化器与未来 reader 共用）。

    依次完成：

    1. 读取 + 精确 schema / type / key-set / 边界 / readiness 结构校验；
    2. **实时输入绑定**：八个角色的 `path` 精确等于**固定生产 logical path**，
       每条 `sha256` 等于对应**实际文件字节** hash；
    3. `frozen_refs` **精确**为 `refs_v3.json` 且 hash 等于 `ab7f5b58…`
       （v2 `refs.json` 一律拒绝）；
    4. **实际调用**既有严格链：`load_frozen_refs`、`read_forecast_policy_manifest`
       （含 provider revision）、`load_truth_split`、`load_verified_exogenous`。

    畸形输入**只抛** `SplitManifestError` / `ValueError`，
    不泄漏 `KeyError` / `TypeError` / `IndexError`。**没有**任何信任边界参数。
    """
    if expected_split not in SPLIT_NAMES:
        raise SplitManifestError(f"未知 split：{expected_split!r}")
    path = _require_canonical_location(path, expected_split=expected_split)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitManifestError(f"{path} 不可读或不是合法 JSON：{error}") from error
    try:
        validated = _validate_structure(payload, expected_split=expected_split)
    except SplitManifestError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise SplitManifestError(f"{path} 结构畸形：{error}") from error
    # R3：`materializer_revision` 必须**精确等于**当前实现解析出的 revision；
    # dirty 的 formal source 一律拒绝（不得用旧 revision 为未提交代码背书）。
    if _generator_is_dirty():
        raise SplitManifestError(
            "formal/materializer 实现有未提交修改：拒绝读取正式 split manifest"
            "（未提交）"
        )
    expected_revision = resolve_materializer_revision()
    if validated["materializer_revision"] != expected_revision:
        raise SplitManifestError(
            f"materializer_revision 必须是**当前**实现的 revision "
            f"{expected_revision}；实际 {validated['materializer_revision']}"
            "（伪造、历史或未知 revision 一律拒绝）"
        )
    _verify_frozen_chain(default_inputs(), expected_split)
    return validated


def _canonical_split_dir() -> Path:
    """canonical triad 目录的**私有**解析器（R3：测试只能 monkeypatch 它）。"""
    return FORMAL_SPLIT_DIR


def canonical_manifest_path(split: str) -> Path:
    """某个 split 的**唯一 canonical** manifest 绝对路径。"""
    if split not in SPLIT_NAMES:
        raise SplitManifestError(f"未知 split：{split!r}，必须属于 {list(SPLIT_NAMES)}")
    return _canonical_split_dir() / f"{split}.json"


def manifest_relative_path(split: SplitName) -> str:
    """正式 triad 的**唯一公开路径来源**：`data/manifest/formal_splits_v4/<split>.json`。

    R2：此 helper 曾返回 `data/manifest/<split>.json` —— **superseded 的 v1**。
    任何按文档使用它的代码都必须拿到**唯一候选**（v4）。
    """
    return logical_repo_path(canonical_manifest_path(split))
