"""M1.3 场景提供器（M1.3a 起统一使用 contracts.ScenarioBundle；M1.3e 升到 contract-v8）。

本模块**不再定义**自己的 ScenarioBundle：唯一场景类型是 `contracts.ScenarioBundle`，
因此场景提供、snapshot 与 `contracts.validators` 校验共享同一形状。

**M1.3e 起，本模块只提供两条明确标注的「非正式」路径**：

- `build_scenario(..., synthetic=True)` → `mode="synthetic"`（开发/测试用）；
- `build_oracle_debug_scenario_from_truth(..., oracle_debug=True)` → `mode="oracle_debug"`
  （把传入的**真值**前 `forecast_cutoff` 项当作预测的 oracle 助手，**仅限 debug**）。

两者都**不是**正式 forecast：正式场景必须由 `scenario/forecast.py` 的**因果** provider
（只读 `[origin-48, origin)` 历史）在 **M1.3g** 接线生成。正式路径当前**仍然 blocked**。

数据边界：M1.2 raw freeze 已完成、M1.3b canonical 事实表与 M1.3d truth split 已冻结，
但完整 forecast / 四项缺口尚未就绪，`train.json` / `validation.json` / `test.json`
一律**未创建**。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from contracts.models import (
    ArtifactDigest,
    ForecastSeriesProvenance,
    ScenarioBundle,
    ScenarioForecastProvenance,
)

# 七类预测序列（M1.3a 增补 carbon，与契约字段一一对应）。
SERIES_KEYS = ("price", "load", "pv", "wind", "temperature", "carbon", "arrival")

# 契约字段 ←→ 旧 `SERIES_KEYS` 短名（保持 M1.3a 的调用形状）。
_SERIES_TO_FIELD = {key: f"{key}_forecast" for key in SERIES_KEYS}

_REQUIRED_MANIFEST_FIELDS = ("source", "units", "sha256")

# 非正式（synthetic / oracle_debug）路径**没有**真实时间轴：`start` 在这些调用里只是
# 一个标签（历史测试里甚至是 "s"）。因此这两条路径的 provenance 使用一个显式、固定、
# 非物理的 dev 锚点，绝不冒充 canonical 时间轴 —— 正式时间轴只能来自 split manifest。
DEV_PROVENANCE_ANCHOR = "2024-01-01T00:00:00+08:00"
DEV_PROVENANCE_NOTE = "非物理 dev 锚点：synthetic / oracle_debug 路径没有真实时间轴"

# 生成器自身（用于 provenance 的 code_revision 与其代码 hash）。
GENERATOR_LOGICAL_PATH = "scenario/scenario.py"

__all__ = [
    "DEV_PROVENANCE_ANCHOR",
    "SERIES_KEYS",
    "ScenarioBundle",
    "build_oracle_debug_scenario_from_truth",
    "build_scenario",
]


def build_scenario(
    split: str,
    start: str,
    horizon: int,
    forecast_cutoff: int,
    *,
    synthetic: bool = False,
    manifest_dir: str = "data/manifest",
    seed: int = 0,
) -> ScenarioBundle:
    """构建场景切片。**只有**显式 `synthetic=True` 的合成开发路径可用。

    正式路径（`synthetic=False`）当前**仍然 blocked**，绝不回退到合成数据。
    """
    _validate_cutoff(horizon, forecast_cutoff)
    if synthetic:
        return _build_synthetic(split, start, horizon, forecast_cutoff, seed)
    return _build_from_manifest(split, start, horizon, forecast_cutoff, manifest_dir)


def build_oracle_debug_scenario_from_truth(
    split: str,
    start: str,
    horizon: int,
    forecast_cutoff: int,
    true: dict[str, np.ndarray],
    *,
    oracle_debug: bool,
    seed: int | None = None,
) -> ScenarioBundle:
    """**oracle-debug 助手**：把传入真值的前 `forecast_cutoff` 项当作预测。

    调用者**必须**显式传 `oracle_debug=True`；该参数是 keyword-only 且无默认值，
    因此漏传会直接 `TypeError`，传假值会 `ValueError`。产物的 `mode` 恒为
    `"oracle_debug"`，**不得**用于正式训练或评估（由 `validate_forecast_purpose` 拦截）。

    这**不是**正式 forecast：正式 forecast 只允许来自 `scenario/forecast.py` 的因果
    provider（只读 origin 之前的历史）。
    """
    if oracle_debug is not True:
        raise ValueError(
            "build_oracle_debug_scenario_from_truth 只能用于 oracle-debug；"
            "调用者必须显式传 oracle_debug=True。正式 forecast 请使用 "
            "scenario.forecast 的因果 provider（M1.3e/M1.3g）"
        )
    _validate_cutoff(horizon, forecast_cutoff)
    missing = [k for k in SERIES_KEYS if k not in true]
    if missing:
        raise KeyError(f"缺少预测序列 {missing}（需 {list(SERIES_KEYS)}）")

    visible: dict[str, list[float]] = {
        key: [float(x) for x in np.asarray(true[key])[:forecast_cutoff]]
        for key in SERIES_KEYS
    }
    return _assemble_bundle(
        split=split,
        start=start,
        horizon=horizon,
        forecast_cutoff=forecast_cutoff,
        visible=visible,
        mode="oracle_debug",
        source_kind="oracle_debug",
        method="oracle_debug_truth_copy",
        model_name="oracle_debug_truth_copy",
        seed=seed,
        digest_role="visible_truth_window",
        digest_path="scenario://oracle_debug/visible_truth_window",
    )


def _validate_cutoff(horizon: int, forecast_cutoff: int) -> None:
    if forecast_cutoff < 0 or forecast_cutoff > horizon:
        raise ValueError(f"forecast_cutoff {forecast_cutoff} 必须在 [0, horizon={horizon}] 内")


def _build_synthetic(
    split: str, start: str, horizon: int, forecast_cutoff: int, seed: int
) -> ScenarioBundle:
    """确定性的**合成**开发场景（`mode="synthetic"`），不得冒充真实数据。"""
    rng = np.random.default_rng(seed)
    t = np.arange(horizon, dtype=float)
    true = {
        "price": 0.15 + 0.05 * np.sin(2 * np.pi * t / 24) + 0.02 * rng.normal(size=horizon),
        "load": (
            6000.0 + 800.0 * np.sin(2 * np.pi * (t - 8.0) / 24.0)
            + 100.0 * rng.normal(size=horizon)
        ),
        "pv": np.clip(0.5 * np.sin(np.pi * (t % 24.0 - 6.0) / 12.0), 0.0, None),
        "wind": np.clip(0.3 * np.cos(np.pi * (t % 24.0) / 12.0), 0.0, None),
        "temperature": (
            28.0 + 3.0 * np.sin(2 * np.pi * (t - 14.0) / 24.0)
            + 0.5 * rng.normal(size=horizon)
        ),
        "carbon": 0.5 + 0.1 * np.sin(2 * np.pi * t / 24.0),
        "arrival": np.clip(80.0 + 40.0 * np.sin(2 * np.pi * (t - 8.0) / 24.0), 0.0, None),
    }
    visible = {
        key: [float(x) for x in np.asarray(true[key])[:forecast_cutoff]]
        for key in SERIES_KEYS
    }
    return _assemble_bundle(
        split=split,
        start=start,
        horizon=horizon,
        forecast_cutoff=forecast_cutoff,
        visible=visible,
        mode="synthetic",
        source_kind="synthetic",
        method="deterministic_synthetic_generator",
        model_name="scenario._build_synthetic",
        seed=int(seed),
        digest_role="synthetic_generator_window",
        digest_path="scenario://synthetic/generated_window",
    )


def _assemble_bundle(
    *,
    split: str,
    start: str,
    horizon: int,
    forecast_cutoff: int,
    visible: dict[str, list[float]],
    mode: str,
    source_kind: str,
    method: str,
    model_name: str,
    seed: int | None,
    digest_role: str,
    digest_path: str,
) -> ScenarioBundle:
    """按 contract-v8 组装**非正式**场景：显式 mode + 结构化 provenance。"""
    revision = _generator_revision()
    # `forecast_cutoff=0` 是退化窗口：仍给出**非空** target 区间，
    # 避免把「空窗口」伪装成合法 provenance（`target_start < target_end_exclusive`）。
    target_end = (
        datetime.fromisoformat(DEV_PROVENANCE_ANCHOR)
        + timedelta(hours=max(int(forecast_cutoff), 1))
    ).isoformat()

    window_digest = hashlib.sha256(
        json.dumps(
            {key: visible[key] for key in SERIES_KEYS}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    sources = [
        ArtifactDigest(role=digest_role, logical_path=digest_path, sha256=window_digest)
    ]

    provenance: dict[str, ForecastSeriesProvenance] = {}
    for key in SERIES_KEYS:
        field = _SERIES_TO_FIELD[key]
        provenance[field] = ForecastSeriesProvenance(
            series_name=field,
            source_kind=source_kind,
            method=method,
            generated_at=DEV_PROVENANCE_ANCHOR,
            information_cutoff_exclusive=DEV_PROVENANCE_ANCHOR,
            target_start=DEV_PROVENANCE_ANCHOR,
            target_end_exclusive=target_end,
            lookback_start=None,
            lookback_end_exclusive=None,
            model_name=model_name,
            model_version="v1",
            code_revision=revision,
            seed=seed,
            # M1.3e-R2：`sources` 与七个 forecast 序列均为不可变 tuple
            sources=tuple(sources),
        )

    return ScenarioBundle(
        split=split,
        start=start,
        horizon=horizon,
        forecast_cutoff=forecast_cutoff,
        price_forecast=tuple(visible["price"]),
        load_forecast=tuple(visible["load"]),
        pv_forecast=tuple(visible["pv"]),
        wind_forecast=tuple(visible["wind"]),
        temperature_forecast=tuple(visible["temperature"]),
        carbon_forecast=tuple(visible["carbon"]),
        arrival_forecast=tuple(visible["arrival"]),
        mode=mode,
        generated_at=DEV_PROVENANCE_ANCHOR,
        forecast_provenance=ScenarioForecastProvenance(**provenance),
    )


def _generator_revision() -> str:
    """本生成器实现的 Git revision（40 位小写 SHA）；**不用**漂移的 HEAD。"""
    revision = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", GENERATOR_LOGICAL_PATH],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"生成器 revision 无效：{revision!r}")
    return revision


def _build_from_manifest(
    split: str, start: str, horizon: int, forecast_cutoff: int, manifest_dir: str
) -> ScenarioBundle:
    manifest_path = Path(manifest_dir) / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"正式模式无数据：缺少 manifest {manifest_path}"
            "（M1.2 raw freeze 已完成；M1.3 正式数据集/split manifest 尚未完成）。"
            "正式 causal forecast / ScenarioBundle 接线属 M1.3g，绝不回退到合成数据。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in _REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            raise ValueError(f"manifest 缺少字段 {field!r}（需单位/来源/hash）")
    # 正式 `ScenarioBundle` 接线（canonical split + causal forecast + 四项缺口口径）
    # 属 M1.3f/M1.3g；本卡只建立契约、因果 provider 与 policy manifest。
    raise NotImplementedError(
        "正式 ScenarioBundle 接线属 M1.3g（M1.2 raw freeze 已完成；"
        "M1.3 正式数据集/split/forecast provider 尚未接线）"
    )
