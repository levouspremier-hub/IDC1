"""M2.1 六个不可变（frozen）Pydantic 契约（**contract-v8**）。

- 所有模型 frozen（不可变），禁止 extra 字段。
- 功率/能量/货币/碳字段单位在各类 `UNITS` 中声明。
- `TaskAllocation.matrix` 为 n_task × n_group 矩形、非负。
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts import CONTRACT_VERSION_ID


class ContractBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default=CONTRACT_VERSION_ID)

    @field_validator("schema_version")
    @classmethod
    def _lock_schema_version(cls, value: object) -> object:
        """**在基底类统一锁定**契约版本：显式声明旧版本或任意字符串一律拒绝。

        `schema_version` 不是「可覆盖的默认值」——全仓唯一版本源是
        `contracts.CONTRACT_VERSION_ID`（M1.3e-R1）。
        """
        if value != CONTRACT_VERSION_ID:
            raise ValueError(
                f"schema_version 必须是 {CONTRACT_VERSION_ID!r}，实际 {value!r}"
                "（版本只能来自 contracts.CONTRACT_VERSION_ID，不得显式覆盖）"
            )
        return value


# --- M1.3e：contract-v8 结构化 forecast provenance -------------------------

SOURCE_KINDS: tuple[str, ...] = (
    "external_forecast", "seasonal_naive", "persistence", "modeled_scenario",
    "synthetic", "oracle_debug", "unavailable",
)
SCENARIO_MODES: tuple[str, ...] = ("formal", "synthetic", "oracle_debug")
FORECAST_PURPOSES: tuple[str, ...] = ("training", "evaluation", "debug")
# 与 ScenarioBundle 的七个 forecast 字段一一对应
BUNDLE_FORECAST_FIELDS: tuple[str, ...] = (
    "price_forecast", "load_forecast", "pv_forecast", "wind_forecast",
    "temperature_forecast", "carbon_forecast", "arrival_forecast",
)
# `mode=formal` 禁止出现的来源类别
NON_FORMAL_SOURCE_KINDS: tuple[str, ...] = ("synthetic", "oracle_debug", "unavailable")

# 每个 `mode` 允许的来源类别：`mode` 是**精确声明**，不是装饰（M1.3e-R1 收紧）。
# - `formal`：只接受「真实可得」来源；不得携带 synthetic / oracle_debug / unavailable；
# - `synthetic`：**七条序列必须全部** `synthetic`（见 MODE_EXACT_SOURCE_KINDS）；
# - `oracle_debug`：**七条序列必须全部** `oracle_debug`。
MODE_ALLOWED_SOURCE_KINDS: dict[str, tuple[str, ...]] = {
    "formal": (
        "external_forecast", "seasonal_naive", "persistence", "modeled_scenario",
    ),
    "synthetic": ("synthetic",),
    "oracle_debug": ("oracle_debug",),
}

# 这两个 mode 要求**七条逐项**等于唯一的来源类别（不是「允许集合」）
MODE_EXACT_SOURCE_KINDS: dict[str, str] = {
    "synthetic": "synthetic",
    "oracle_debug": "oracle_debug",
}

# **任何** `ScenarioBundle` 都不得携带的来源类别：完整 bundle 的七条序列都有值，
# 声明某条「不可得」是自相矛盾；`unavailable` 只用于尚未 materialize 的字段登记
# （split manifest 的 `unavailable_not_materialized`），不得进入场景契约。
BUNDLE_FORBIDDEN_SOURCE_KINDS: tuple[str, ...] = ("unavailable",)

_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_plain_int(value: object, *, field: str) -> int:
    """**coercion 之前**的严格整数：只接受真正的 `int`。

    pydantic 的宽松模式会在字段类型校验前把 `"200"` 变成 `200`、`200.0` 变成 `200`、
    `True` 变成 `1`，因此必须在 `mode="before"` 的 validator 里先挡下来
    （M1.3e-R3）。刻意**不**对整个 `ContractBase` 打开全局 `strict=True`，
    以免无关契约大面积改变行为。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须是整数（bool 不算），实际 {value!r}")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    """严格数值元素：接受 `int`/`float`（**bool 不算数值**）；拒绝 NaN/±Inf。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是数值（bool 不算），实际 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限数值，实际 {value!r}")
    return result


def _normalise_forecast_series(value: object, *, field: str) -> tuple[float, ...]:
    """预测序列元素规则：容器只接受 `list`/`tuple` 并规范化为 `tuple`。

    `AvailableSeries` 与 `ScenarioBundle` **共用这一份实现**，避免两处规则漂移
    （M1.3e-R3 明确要求「公共校验 helper 应只有一份」）。
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{field} 必须是 list/tuple，实际 {type(value).__name__}"
        )
    return tuple(
        _require_finite_number(element, field=f"{field}[{index}]")
        for index, element in enumerate(value)
    )


def _require_canonical_logical_path(value: object, *, field: str) -> str:
    """**规范 POSIX 逻辑路径**：非空、无空白、相对、无反斜杠、无 `.`/`..`/空片段。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空字符串，实际 {value!r}")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{field} 不得含空白（含前后缀）：{value!r}")
    if value.startswith("/"):
        raise ValueError(f"{field} 不得是绝对路径：{value!r}")
    if "\\" in value:
        raise ValueError(f"{field} 必须是 POSIX 逻辑路径（不得含反斜杠）：{value!r}")
    for segment in value.split("/"):
        if segment == "":
            raise ValueError(
                f"{field} 不得含空路径片段（如 'a//b' 或结尾 '/'）：{value!r}"
            )
        if segment in (".", ".."):
            raise ValueError(f"{field} 不得含 '.' 或 '..' 路径片段：{value!r}")
    return value


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _HEX_DIGITS for character in value)
    )


def _require_canonical_timestamp(value: object, *, field: str) -> str:
    """时间戳必须是**带时区**的规范 ISO-8601（naive 或非规范形式一律拒绝）。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是带时区的规范 ISO-8601 字符串，实际 {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} 不是合法 ISO-8601：{value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} 必须带显式时区偏移（不得使用 naive 时间）：{value!r}")
    if parsed.isoformat() != value:
        raise ValueError(f"{field} 不是规范 ISO-8601 形式：{value!r}")
    return value


def _require_ordering(earlier: str, later: str, *, label: str, strict: bool = False) -> None:
    left = datetime.fromisoformat(earlier)
    right = datetime.fromisoformat(later)
    if (left >= right) if strict else (left > right):
        comparator = "<" if strict else "<="
        raise ValueError(f"{label}（要求 {earlier!r} {comparator} {later!r}）")


class ArtifactDigest(ContractBase):
    """一个**上游制品**的逻辑身份：角色 + 仓库相对路径 + 内容 SHA-256。

    **构造时**自校验（M1.3e-R2）：不能等到嵌套进 provenance 之后才检查，
    否则一个畸形 digest 可以借由「先构造再放置」绕过单点校验。
    """

    role: str
    logical_path: str
    sha256: str

    @field_validator("logical_path", mode="before")
    @classmethod
    def _check_logical_path(cls, value: object) -> object:
        return _require_canonical_logical_path(value, field="ArtifactDigest.logical_path")

    @model_validator(mode="after")
    def _validate_digest(self) -> ArtifactDigest:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError(f"ArtifactDigest.role 必须是非空字符串，实际 {self.role!r}")
        if not _is_lower_hex(self.sha256, 64):
            raise ValueError(
                f"ArtifactDigest.sha256 必须是 64 位小写十六进制，实际 {self.sha256!r}"
            )
        return self


def validate_forecast_series_provenance(
    provenance: ForecastSeriesProvenance, *, expected_series_name: str
) -> None:
    """逐序列 provenance 的完整性与时间顺序：**缺任一项即 fail closed**。"""
    if provenance.series_name != expected_series_name:
        raise ValueError(
            f"provenance.series_name {provenance.series_name!r} 与字段 "
            f"{expected_series_name!r} 不一致"
        )
    if provenance.source_kind not in SOURCE_KINDS:
        raise ValueError(f"未知 source_kind：{provenance.source_kind!r}")
    for name in ("method", "model_name", "model_version"):
        if not getattr(provenance, name):
            raise ValueError(f"provenance.{name} 不得为空")
    if not _is_lower_hex(provenance.code_revision, 40):
        raise ValueError(
            "provenance.code_revision 必须是 40 位小写 Git SHA，"
            f"实际 {provenance.code_revision!r}"
        )
    if not provenance.sources:
        raise ValueError("provenance.sources 不得为空（缺少来源 hash）")
    for digest in provenance.sources:
        if not isinstance(digest.role, str) or not digest.role:
            raise ValueError("sources[*].role 不得为空")
        if not isinstance(digest.logical_path, str) or not digest.logical_path:
            raise ValueError("sources[*].logical_path 不得为空")
        if not _is_lower_hex(digest.sha256, 64):
            raise ValueError(
                f"sources[*].sha256 必须是 64 位小写十六进制，实际 {digest.sha256!r}"
            )

    for name in (
        "generated_at", "information_cutoff_exclusive", "target_start",
        "target_end_exclusive",
    ):
        _require_canonical_timestamp(getattr(provenance, name), field=f"provenance.{name}")
    for name in ("lookback_start", "lookback_end_exclusive"):
        value = getattr(provenance, name)
        if value is not None:
            _require_canonical_timestamp(value, field=f"provenance.{name}")

    _require_ordering(
        provenance.generated_at, provenance.target_start,
        label="provenance.generated_at 必须 <= target_start",
    )
    _require_ordering(
        provenance.information_cutoff_exclusive, provenance.target_start,
        label="provenance.information_cutoff_exclusive 必须 <= target_start",
    )
    if provenance.lookback_end_exclusive is not None:
        _require_ordering(
            provenance.lookback_end_exclusive, provenance.information_cutoff_exclusive,
            label=(
                "provenance.lookback_end_exclusive 必须 <= information_cutoff_exclusive"
            ),
        )
    if provenance.lookback_start is not None and provenance.lookback_end_exclusive is not None:
        _require_ordering(
            provenance.lookback_start, provenance.lookback_end_exclusive,
            label="provenance.lookback_start 必须 <= lookback_end_exclusive",
        )
    _require_ordering(
        provenance.target_start, provenance.target_end_exclusive,
        label="provenance.target_start 必须 < target_end_exclusive", strict=True,
    )


class ForecastSeriesProvenance(ContractBase):
    """**逐序列**的结构化来源声明（不得用自由 dict 承载）。

    本模型在**构造时**自校验：字段齐备、时间顺序自洽、SHA/时区规范。
    之所以把校验放在 `models` 而非 `validators`，是因为 `validators` 依赖 `models`，
    反向导入会成环；`contracts.validators` 对外暴露同名语义的入口（委托到本函数）。
    """

    series_name: str
    source_kind: str
    method: str
    generated_at: str
    information_cutoff_exclusive: str
    target_start: str
    target_end_exclusive: str
    lookback_start: str | None
    lookback_end_exclusive: str | None
    model_name: str
    model_version: str
    code_revision: str
    seed: int | None
    # 不可变 tuple：`frozen=True` 只冻结字段赋值，不冻结容器内容。
    # JSON/list 输入会被 pydantic 规范化为 tuple，但**对外不暴露可变容器**。
    sources: tuple[ArtifactDigest, ...]

    @field_validator("seed", mode="before")
    @classmethod
    def _reject_bool_or_non_integer_seed(cls, value: object) -> object:
        """pydantic 的宽松模式会把 `True` 变成 `1`、`"0"` 变成 `0`；此处显式拒绝。"""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"provenance.seed 必须是整数或 null，实际 {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_series(self) -> ForecastSeriesProvenance:
        validate_forecast_series_provenance(self, expected_series_name=self.series_name)
        return self


class ScenarioForecastProvenance(BaseModel):
    """**七个固定字段**的 provenance 容器（不是自由 dict）。

    刻意**不**继承 `ContractBase`：它的字段集合必须**精确**等于七个 forecast 字段
    （`model_fields` 不得混入 `schema_version` 之类的继承字段），
    否则「七个字段与七项 provenance 一一对应」就无法被机器判定。
    版本仍由最外层 `ScenarioBundle.schema_version` 唯一承载。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    price_forecast: ForecastSeriesProvenance
    load_forecast: ForecastSeriesProvenance
    pv_forecast: ForecastSeriesProvenance
    wind_forecast: ForecastSeriesProvenance
    temperature_forecast: ForecastSeriesProvenance
    carbon_forecast: ForecastSeriesProvenance
    arrival_forecast: ForecastSeriesProvenance


def validate_bundle_forecast_provenance(bundle: ScenarioBundle) -> None:
    """七个 forecast 字段与七项 provenance **一一对应**，且 `mode` 与来源自洽。

    「一一对应、不缺不多」由 `ScenarioForecastProvenance` 的固定字段与
    `extra="forbid"` **结构性**保证；此处再逐项校验 `series_name` 与来源类别。
    """
    mode = bundle.mode
    if mode not in SCENARIO_MODES:
        raise ValueError(f"未知 mode：{mode!r}，必须属于 {list(SCENARIO_MODES)}")
    _require_canonical_timestamp(bundle.generated_at, field="generated_at")

    provenance = bundle.forecast_provenance
    if not isinstance(provenance, ScenarioForecastProvenance):
        raise ValueError(
            "forecast_provenance 必须是 ScenarioForecastProvenance，"
            f"实际 {type(provenance).__name__}（不得用自由 dict 承载 provenance）"
        )

    forbidden = set(BUNDLE_FORBIDDEN_SOURCE_KINDS)
    exact_kind = MODE_EXACT_SOURCE_KINDS.get(mode)
    allowed = MODE_ALLOWED_SOURCE_KINDS[mode]
    rejected: list[tuple[str, str]] = []
    mismatched_generated_at: list[tuple[str, str]] = []
    for name in BUNDLE_FORECAST_FIELDS:
        entry = getattr(provenance, name)
        if not isinstance(entry, ForecastSeriesProvenance):
            raise ValueError(f"forecast_provenance.{name} 必须是 ForecastSeriesProvenance")
        validate_forecast_series_provenance(entry, expected_series_name=name)
        if entry.source_kind in forbidden:
            rejected.append((name, entry.source_kind))
            continue
        if exact_kind is not None:
            # synthetic / oracle_debug 要求**七条逐项**等于该 mode 的唯一来源类别
            if entry.source_kind != exact_kind:
                rejected.append((name, entry.source_kind))
        elif entry.source_kind not in allowed:
            rejected.append((name, entry.source_kind))
        if entry.generated_at != bundle.generated_at:
            mismatched_generated_at.append((name, entry.generated_at))
    if rejected:
        expectation = (
            f"该 mode 要求七条逐项为 {exact_kind!r}"
            if exact_kind is not None
            else f"该 mode 允许的来源为 {list(allowed)}"
        )
        raise ValueError(
            f"mode={mode!r} 不接受以下来源声明：{rejected}；"
            f"禁止的来源类别为 {list(BUNDLE_FORBIDDEN_SOURCE_KINDS)}；{expectation}"
        )
    if mismatched_generated_at:
        raise ValueError(
            "forecast_provenance 的 generated_at 必须与 ScenarioBundle.generated_at "
            f"逐项恒等，实际不一致：{mismatched_generated_at}"
        )


class ScenarioBundle(ContractBase):
    """可见预测场景切片（**contract-v8**）。

    M1.3e 起用明确的 `mode` 取代含糊的 `synthetic: bool`，并加入**结构化、
    不可伪造**的 `forecast_provenance`；无 schema 的自由 dict `source_hashes` **已退役**。
    """

    split: str
    start: str
    horizon: int
    forecast_cutoff: int
    # 不可变 tuple（M1.3e-R2）：调用方仍按只读 `Sequence` 语义使用，
    # 但构造后**无法**原地修改，`content_hash()` 因此不会被事后篡改。
    price_forecast: tuple[float, ...]
    load_forecast: tuple[float, ...]
    pv_forecast: tuple[float, ...]
    wind_forecast: tuple[float, ...]
    temperature_forecast: tuple[float, ...]
    carbon_forecast: tuple[float, ...]
    arrival_forecast: tuple[float, ...]
    mode: str
    generated_at: str
    forecast_provenance: ScenarioForecastProvenance

    @field_validator(*BUNDLE_FORECAST_FIELDS, mode="before")
    @classmethod
    def _normalise_forecasts(cls, value: object, info) -> object:
        """七个 forecast 序列与 artifact 共用**同一套**元素规则（M1.3e-R3）。"""
        return _normalise_forecast_series(value, field=info.field_name)

    UNITS: ClassVar[dict[str, str]] = {
        "price_forecast": "SGD/kWh",
        "load_forecast": "MW",
        "pv_forecast": "kW",
        "wind_forecast": "kW",
        "temperature_forecast": "degC",
        "carbon_forecast": "kgCO2/kWh",
        "arrival_forecast": "work-units/step",
    }

    @model_validator(mode="after")
    def _validate_provenance(self) -> ScenarioBundle:
        validate_bundle_forecast_provenance(self)
        return self

    def content_hash(self) -> str:
        """确定性内容 hash（与 dict 键顺序无关）；**覆盖全部 provenance**。"""
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()


# --- M1.3e-R1：available exogenous forecast artifact ------------------------
#
# 这是**五个 canonical truth driver 的因果 forecast artifact**的严格冻结契约。
# 它**不是**完整 `ScenarioBundle`（没有 PV / 风电发电量 / 碳强度 / arrival），
# 因此**不得**被当作完整场景用于训练或评估（由 purpose gate 拒绝）。

AVAILABLE_DRIVER_SERIES: tuple[str, ...] = (
    "price_sgd_per_kwh",
    "system_load_mw",
    "temperature_deg_c",
    "wind_speed_10m_mps",
    "ghi_w_per_m2",
)
DRIVER_UNITS: dict[str, str] = {
    "price_sgd_per_kwh": "SGD/kWh",
    "system_load_mw": "MW",
    "temperature_deg_c": "degC",
    "wind_speed_10m_mps": "m/s",
    "ghi_w_per_m2": "W/m2",
}
# 本 artifact 的**冻结 policy 规则**（M1.3e-R2 锁死）：
# 不是「非空 / 正数」这类弱约束，而是**精确取值**——任何其他合法字符串或正整数
# 都必须被拒绝。
AVAILABLE_FORECAST_SOURCE_KIND = "seasonal_naive"
AVAILABLE_FORECAST_METHOD = "trailing_seasonal_naive"
AVAILABLE_FORECAST_MODEL_NAME = "trailing_seasonal_naive"
AVAILABLE_FORECAST_MODEL_VERSION = "v1"
AVAILABLE_FORECAST_FREQUENCY = "30min"
AVAILABLE_FORECAST_PERIOD_STEPS = 48
AVAILABLE_FORECAST_STEP_MINUTES = 30
SCENARIO_SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

# 四个来源制品的**角色集合与顺序**（缺失、重复、额外、乱序一律拒绝）。
AVAILABLE_SOURCE_ROLES: tuple[str, ...] = (
    "canonical_parquet",
    "canonical_manifest",
    "split_manifest",
    "forecast_policy_manifest",
)
# 角色 → artifact 顶层字段（路径与 SHA-256）：逐序列 digest 必须与顶层**逐项恒等**。
AVAILABLE_SOURCE_PATH_FIELDS: dict[str, str] = {
    "canonical_parquet": "canonical_parquet_path",
    "canonical_manifest": "canonical_manifest_path",
    "split_manifest": "split_manifest_path",
    "forecast_policy_manifest": "policy_manifest_path",
}
AVAILABLE_SOURCE_HASH_FIELDS: dict[str, str] = {
    "canonical_parquet": "canonical_parquet_sha256",
    "canonical_manifest": "canonical_manifest_sha256",
    "split_manifest": "split_manifest_sha256",
    "forecast_policy_manifest": "policy_manifest_sha256",
}


class AvailableSeries(BaseModel):
    """五个 driver 的预测向量：**冻结、定长元组**，不暴露可变内部 list/dict。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price_sgd_per_kwh: tuple[float, ...]
    system_load_mw: tuple[float, ...]
    temperature_deg_c: tuple[float, ...]
    wind_speed_10m_mps: tuple[float, ...]
    ghi_w_per_m2: tuple[float, ...]

    @field_validator(*AVAILABLE_DRIVER_SERIES, mode="before")
    @classmethod
    def _normalise_series(cls, value: object, info) -> object:
        """容器规范化 + 元素严格验型，与 `ScenarioBundle` 共用同一份实现。"""
        return _normalise_forecast_series(value, field=info.field_name)

    def __getitem__(self, driver: str) -> tuple[float, ...]:
        try:
            return getattr(self, driver)
        except AttributeError as error:
            raise KeyError(f"未知 driver：{driver!r}") from error

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        return {driver: self[driver] for driver in AVAILABLE_DRIVER_SERIES}


class AvailableDriverProvenance(BaseModel):
    """五个 driver 的**逐序列**结构化 provenance（无自由 dict）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price_sgd_per_kwh: ForecastSeriesProvenance
    system_load_mw: ForecastSeriesProvenance
    temperature_deg_c: ForecastSeriesProvenance
    wind_speed_10m_mps: ForecastSeriesProvenance
    ghi_w_per_m2: ForecastSeriesProvenance

    def __getitem__(self, driver: str) -> ForecastSeriesProvenance:
        try:
            return getattr(self, driver)
        except AttributeError as error:
            raise KeyError(f"未知 driver：{driver!r}") from error


class AvailableExogenousForecast(ContractBase):
    """**严格冻结**的 driver forecast artifact（contract-v8）。

    与 `ScenarioBundle` 的区别是**结构性**的：这里只有五个 driver，没有
    PV / 风电发电量 / 碳强度 / arrival；因此它带**三条上游 logical path** 与四个
    来源 SHA-256，可被审计到**完整的冻结资产链**（policy → split → canonical →
    parquet），且被 purpose gate 在 `training`/`evaluation` 下拒绝。

    **M1.3e-R2：内部证据闭环。** 顶层四个 path/hash 与逐序列 provenance 的四个
    digest 必须在**构造时**逐项恒等（角色集合与顺序也固定），因此逐序列证据无法
    被局部篡改而不被发现；policy 规则（split/frequency/method/period_steps/
    model_name/model_version/seed）与 target/lookback 时间轴同样在构造时锁死。
    """

    split: str
    origin: int
    global_origin: int
    forecast_cutoff: int
    frequency: str
    method: str
    period_steps: int
    generated_at: str
    target_timestamps: tuple[str, ...]
    series: AvailableSeries
    provenance: AvailableDriverProvenance
    canonical_parquet_path: str
    canonical_manifest_path: str
    split_manifest_path: str
    policy_manifest_path: str
    policy_manifest_sha256: str
    canonical_parquet_sha256: str
    canonical_manifest_sha256: str
    split_manifest_sha256: str
    code_revision: str

    @field_validator(
        "origin", "global_origin", "forecast_cutoff", "period_steps", mode="before"
    )
    @classmethod
    def _require_integers(cls, value: object, info) -> object:
        """**coercion 之前**的严格整数（M1.3e-R3）：bool/float/字符串/None/容器全拒绝。"""
        return _require_plain_int(value, field=info.field_name)

    @field_validator(
        "canonical_parquet_path", "canonical_manifest_path",
        "split_manifest_path", "policy_manifest_path", mode="before",
    )
    @classmethod
    def _require_logical_paths(cls, value: object, info) -> object:
        """四条上游 path 必须规范 POSIX 逻辑路径（首个错误字段名即 `info.field_name`）。"""
        return _require_canonical_logical_path(value, field=info.field_name)

    @model_validator(mode="after")
    def _validate_available_forecast(self) -> AvailableExogenousForecast:
        self._validate_locked_policy()
        self._validate_integer_fields()
        self._validate_paths_and_hashes()
        self._validate_target_timeline()
        self._validate_series_and_provenance_closure()
        return self

    # --- 冻结 policy 规则（精确取值，不是「非空/正数」） ---------------------

    def _validate_locked_policy(self) -> None:
        if self.split not in SCENARIO_SPLIT_NAMES:
            raise ValueError(
                f"split 必须是 {list(SCENARIO_SPLIT_NAMES)} 之一，实际 {self.split!r}"
            )
        for name, expected in (
            ("frequency", AVAILABLE_FORECAST_FREQUENCY),
            ("method", AVAILABLE_FORECAST_METHOD),
        ):
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"{name} 必须精确等于 {expected!r}，实际 {actual!r}")
        if self.period_steps != AVAILABLE_FORECAST_PERIOD_STEPS:
            raise ValueError(
                f"period_steps 必须精确等于 {AVAILABLE_FORECAST_PERIOD_STEPS}，"
                f"实际 {self.period_steps!r}"
            )
        _require_canonical_timestamp(self.generated_at, field="generated_at")

    def _validate_integer_fields(self) -> None:
        for name in ("origin", "global_origin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数，实际 {value!r}")
        if self.global_origin < self.origin:
            raise ValueError("global_origin 不得小于 split 本地的 origin")
        if (
            isinstance(self.forecast_cutoff, bool)
            or not isinstance(self.forecast_cutoff, int)
            or self.forecast_cutoff < 1
        ):
            raise ValueError(f"forecast_cutoff 必须是严格正整数，实际 {self.forecast_cutoff!r}")

    def _validate_paths_and_hashes(self) -> None:
        # 顶层 path 已由 `_require_logical_paths`（mode="before"）在 coercion 前校验；
        # 这里再确认一次，使「经由 model_copy 绕开构造」的对象也被抓住。
        for field in AVAILABLE_SOURCE_PATH_FIELDS.values():
            _require_canonical_logical_path(getattr(self, field), field=field)
        for field in AVAILABLE_SOURCE_HASH_FIELDS.values():
            if not _is_lower_hex(getattr(self, field), 64):
                raise ValueError(
                    f"{field} 必须是 64 位小写十六进制，实际 {getattr(self, field)!r}"
                )
        if not _is_lower_hex(self.code_revision, 40):
            raise ValueError(
                f"code_revision 必须是 40 位小写 Git SHA，实际 {self.code_revision!r}"
            )

    # --- target / lookback 时间轴 -------------------------------------------

    def _step(self) -> timedelta:
        return timedelta(minutes=AVAILABLE_FORECAST_STEP_MINUTES)

    def _target_end_exclusive(self) -> str:
        """target 窗口右端 = 最后一个 target 时间戳 + 一个 30 分钟 step。"""
        return (datetime.fromisoformat(self.target_timestamps[-1]) + self._step()).isoformat()

    def _lookback_start(self) -> str:
        """历史窗口左端 = `generated_at − period_steps × 30 分钟`。"""
        return (
            datetime.fromisoformat(self.generated_at)
            - self._step() * self.period_steps
        ).isoformat()

    def _validate_target_timeline(self) -> None:
        if not self.target_timestamps:
            raise ValueError("target_timestamps 不得为空")
        for stamp in self.target_timestamps:
            _require_canonical_timestamp(stamp, field="target_timestamps[*]")
        if len(self.target_timestamps) != self.forecast_cutoff:
            raise ValueError(
                "target_timestamps 长度 "
                f"{len(self.target_timestamps)} != forecast_cutoff {self.forecast_cutoff}"
            )
        if self.target_timestamps[0] != self.generated_at:
            raise ValueError(
                "target_timestamps[0] 必须等于 generated_at "
                f"{self.generated_at!r}，实际 {self.target_timestamps[0]!r}"
            )
        parsed = [datetime.fromisoformat(stamp) for stamp in self.target_timestamps]
        for index in range(1, len(parsed)):
            if parsed[index] <= parsed[index - 1]:
                raise ValueError(
                    "target_timestamps 必须严格递增且唯一，"
                    f"第 {index} 项 {self.target_timestamps[index]!r} 未严格大于前一项"
                )
            if parsed[index] - parsed[index - 1] != self._step():
                raise ValueError(
                    f"target_timestamps 必须是严格 {AVAILABLE_FORECAST_STEP_MINUTES} "
                    f"分钟网格，第 {index} 项间隔为 {parsed[index] - parsed[index - 1]}"
                )

    # --- 逐序列与顶层的闭环 ------------------------------------------------

    def _validate_series_and_provenance_closure(self) -> None:
        expected_target_end = self._target_end_exclusive()
        expected_lookback_start = self._lookback_start()
        reference_sources: tuple[ArtifactDigest, ...] | None = None
        for driver in AVAILABLE_DRIVER_SERIES:
            values = self.series[driver]
            if len(values) != self.forecast_cutoff:
                raise ValueError(
                    f"series.{driver} 长度 {len(values)} != forecast_cutoff "
                    f"{self.forecast_cutoff}"
                )
            for value in values:
                if not math.isfinite(value):
                    raise ValueError(f"series.{driver} 含非有限值：{value!r}")

            entry = self.provenance[driver]
            if entry.series_name != driver:
                raise ValueError(
                    f"provenance.{driver}.series_name 必须是 {driver!r}，"
                    f"实际 {entry.series_name!r}"
                )
            for field, expected in (
                ("source_kind", AVAILABLE_FORECAST_SOURCE_KIND),
                ("method", AVAILABLE_FORECAST_METHOD),
                ("model_name", AVAILABLE_FORECAST_MODEL_NAME),
                ("model_version", AVAILABLE_FORECAST_MODEL_VERSION),
                ("code_revision", self.code_revision),
                ("generated_at", self.generated_at),
                ("information_cutoff_exclusive", self.generated_at),
                ("lookback_end_exclusive", self.generated_at),
                ("target_start", self.generated_at),
                ("target_end_exclusive", expected_target_end),
                ("lookback_start", expected_lookback_start),
            ):
                actual = getattr(entry, field)
                if actual != expected:
                    raise ValueError(
                        f"provenance.{driver}.{field} 必须等于 {expected!r}，实际 {actual!r}"
                    )
            if entry.seed is not None:
                raise ValueError(
                    f"provenance.{driver}.seed 必须为 None（冻结方法不使用随机数），"
                    f"实际 {entry.seed!r}"
                )
            self._validate_source_closure(driver, entry)
            if reference_sources is None:
                reference_sources = entry.sources
            elif entry.sources != reference_sources:
                raise ValueError(
                    f"provenance.{driver}.sources 必须与其它 driver 的四个 source "
                    "digest 完全相同（path/hash/role 逐项恒等）"
                )

    def _validate_source_closure(
        self, driver: str, entry: ForecastSeriesProvenance
    ) -> None:
        roles = tuple(digest.role for digest in entry.sources)
        if roles != AVAILABLE_SOURCE_ROLES:
            raise ValueError(
                f"provenance.{driver}.sources 的角色必须精确、按序等于 "
                f"{list(AVAILABLE_SOURCE_ROLES)}，实际 {list(roles)}"
            )
        for digest in entry.sources:
            expected_path = getattr(self, AVAILABLE_SOURCE_PATH_FIELDS[digest.role])
            expected_hash = getattr(self, AVAILABLE_SOURCE_HASH_FIELDS[digest.role])
            if digest.logical_path != expected_path:
                raise ValueError(
                    f"provenance.{driver}.sources[{digest.role}].logical_path 必须等于 "
                    f"顶层 {AVAILABLE_SOURCE_PATH_FIELDS[digest.role]} "
                    f"{expected_path!r}，实际 {digest.logical_path!r}"
                )
            if digest.sha256 != expected_hash:
                raise ValueError(
                    f"provenance.{driver}.sources[{digest.role}].sha256 必须等于顶层 "
                    f"{AVAILABLE_SOURCE_HASH_FIELDS[digest.role]} "
                    f"{expected_hash!r}，实际 {digest.sha256!r}"
                )

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def prediction_hash(self) -> str:
        """**只覆盖预测数值、单位与顺序**（用于证明预测不随未来真值变化）。

        刻意**不含**上游 SHA-256 与 revision：那些会随上游文件字节变化，
        而预测值未必变化。
        """
        payload = {
            "units": {driver: DRIVER_UNITS[driver] for driver in AVAILABLE_DRIVER_SERIES},
            "series": {
                driver: list(self.series[driver]) for driver in AVAILABLE_DRIVER_SERIES
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def content_hash(self) -> str:
        """**覆盖完整 artifact**，含 policy/split/canonical provenance 与 revision。

        不得为了通过 leakage 测试而让它忽略审计 provenance（M1.3e-R1）。
        """
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()


def raw_model_payload(model: BaseModel) -> dict:
    """把模型还原成**原始值字典**（递归展开嵌套模型），**不经过序列化器**。

    刻意不用 `model_dump()`：序列化器可能「洗白」值（例如把 `bool` 元素按
    `float` 字段序列化成 `1.0`），用它做复验就会漏掉 R2/R3 要挡住的类型问题
    （M1.3e-R3 §18.5）。
    """
    payload: dict = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            value = raw_model_payload(value)
        payload[name] = value
    return payload


def validate_available_forecast_artifact(artifact: AvailableExogenousForecast) -> None:
    """对 artifact 做**防御性复验**：把原始属性值重新过一遍构造时的**同一套**规则。

    直接复用构造路径（`model_validate` + `mode="before"` validator），
    因此不会像「另写一套更弱的重复规则」那样随时间漂移；对经由
    `model_copy(update=...)` 直接写入 `__dict__` 的对象同样有效。
    """
    type(artifact).model_validate(raw_model_payload(artifact))


class TaskState(ContractBase):
    task_id: str
    remaining_work: float
    deadline: int
    priority: float
    status: str
    max_rate_work_per_step: float

    UNITS: ClassVar[dict[str, str]] = {
        "remaining_work": "work-units",
        "deadline": "step index (latest finish step)",
        "max_rate_work_per_step": "work-units/step",
    }


class PlanningExogenousForecast(ContractBase):
    """规划时域外生量展开（M4.1c）：长度严格等于 `planning_horizon_steps`。

    这不是真实未来预测：`visible_mask` 标记真实可见段（来自 `SystemSnapshot.forecast`），
    `assumed_mask` 标记按 `extension_policy` 生成的规划假设段（窗口外）。
    """

    horizon_steps: int
    price: list[float]
    pv: list[float]
    wind: list[float]
    temperature: list[float]
    carbon: list[float]
    arrival: list[float]
    base_idc_power: list[float]
    visible_mask: list[bool]
    assumed_mask: list[bool]
    extension_policy: str

    UNITS: ClassVar[dict[str, str]] = {
        "horizon_steps": "steps",
        "price": "SGD/kWh",
        "pv": "kW",
        "wind": "kW",
        "temperature": "degC",
        "carbon": "kgCO2/kWh",
        "arrival": "work-units/step",
        "base_idc_power": "kW",
        "visible_mask": "bool",
        "assumed_mask": "bool",
    }


class SystemSnapshot(ContractBase):
    """滚动规划输入契约（M4.1a/M4.1b/M4.1c 扩展）。

    `group_work_capacity` 为逐组 **work capacity**（work-units）；规划用的功率线性近似见
    `group_power_coeff_kw_per_work` / `group_power_upper_kw`，二者须由环境物理链复核。

    `forecast` 是**真实可见预测**（长度 `forecast_cutoff`）；
    `planning_forecast` 是按时域展开的**规划假设**（长度 `planning_horizon_steps`），
    其窗口外取值不是真实预测，不得当作真值使用。
    """

    # 1. 时间
    step: int
    delta_t_hours: float
    planning_horizon_steps: int

    # 2. 储能
    soc_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    soc_capacity_kwh: float
    bess_charge_power_max_kw: float
    bess_discharge_power_max_kw: float
    bess_charge_efficiency: float
    bess_discharge_efficiency: float
    bess_degradation_cost_per_kwh: float

    # 3. 接入与基础负载
    access_limit_kw: float
    base_idc_power_forecast_kw: list[float]

    # 4/5. 任务与预测
    tasks: list[TaskState]
    forecast: ScenarioBundle
    planning_forecast: PlanningExogenousForecast

    # 6. 逐组容量与规划用功率线性近似
    group_work_capacity: list[float]  # 逐组 work capacity（单位见 UNITS）
    group_power_coeff_kw_per_work: list[float]
    group_power_upper_kw: list[float]
    power_approximation_note: str

    budget_remaining_sgd: float

    UNITS: ClassVar[dict[str, str]] = {
        "step": "step index",
        "delta_t_hours": "h",
        "planning_horizon_steps": "steps",
        "soc_kwh": "kWh",
        "soc_min_kwh": "kWh",
        "soc_max_kwh": "kWh",
        "soc_capacity_kwh": "kWh",
        "bess_charge_power_max_kw": "kW",
        "bess_discharge_power_max_kw": "kW",
        "bess_charge_efficiency": "fraction",
        "bess_discharge_efficiency": "fraction",
        "bess_degradation_cost_per_kwh": "SGD/kWh",
        "access_limit_kw": "kW",
        "base_idc_power_forecast_kw": "kW",
        "group_work_capacity": "work-units",
        "group_power_coeff_kw_per_work": "kW/work-unit",
        "group_power_upper_kw": "kW",
        "budget_remaining_sgd": "SGD",
    }


class DispatchProposal(ContractBase):
    """原始动作提案（21 维：20 compute + 1 有符号储能）。"""

    compute_actions: list[float]
    storage_action: float

    UNITS: ClassVar[dict[str, str]] = {
        "compute_actions": "normalized [0,1]",
        "storage_action": "normalized [-1,1]",
    }


class TaskAllocation(ContractBase):
    """任务×组分配矩阵 A[i,g]，矩形、非负。"""

    task_ids: list[str]
    group_ids: list[int]
    matrix: list[list[float]]

    UNITS: ClassVar[dict[str, str]] = {"matrix": "work-units"}

    @model_validator(mode="after")
    def _validate_matrix(self) -> TaskAllocation:
        n_group = len(self.group_ids)
        if len(self.matrix) != len(self.task_ids):
            raise ValueError("matrix 行数 != len(task_ids)")
        for row in self.matrix:
            if len(row) != n_group:
                raise ValueError("matrix 非矩形：每行长度 != len(group_ids)")
            if any(x < 0 for x in row):
                raise ValueError("matrix 必须非负")
        return self


class DispatchResult(ContractBase):
    """单步执行结果：raw 与 exec 分开记录。"""

    raw_compute_actions: list[float]
    raw_storage_action: float
    exec_compute_actions: list[float]
    exec_storage_action: float
    correction_reason: str
    business_gap: float
    solve_time_s: float
    p_grid_kw: float
    soc_next_kwh: float
    cost_sgd: float
    carbon_kg: float

    UNITS: ClassVar[dict[str, str]] = {
        "business_gap": "work-units",
        "solve_time_s": "s",
        "p_grid_kw": "kW",
        "soc_next_kwh": "kWh",
        "cost_sgd": "SGD",
        "carbon_kg": "kgCO2",
    }


class EvaluationRecord(ContractBase):
    """统一评估记录（M6 五方法同 schema）。"""

    method: str
    run_id: str
    service_qualified: bool
    total_cost_sgd: float
    total_carbon_kg: float
    renewable_utilization: float
    peak_kw: float
    reliability: float
    solve_time_avg_s: float
    failure_classification: str | None = None

    UNITS: ClassVar[dict[str, str]] = {
        "total_cost_sgd": "SGD",
        "total_carbon_kg": "kgCO2",
        "renewable_utilization": "fraction [0,1]",
        "peak_kw": "kW",
        "reliability": "fraction [0,1]",
        "solve_time_avg_s": "s",
    }
