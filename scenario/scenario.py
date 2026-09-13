"""M1.3 场景提供器（M1.3a 起统一使用 contracts.ScenarioBundle）。

本模块**不再定义**自己的 ScenarioBundle：唯一场景类型是 `contracts.ScenarioBundle`（contract-v2），
因此场景提供、snapshot 与 `contracts.validators` 校验共享同一形状。

可见性：预测窗口为 `[t, t + forecast_cutoff)`（与 env / snapshot adapter 同一定义），
不暴露未来真值。

数据边界：M1.2 正式数据接线**仍阻塞**（数据组合未验证）。当前只允许生成显式
`synthetic=True` 的开发/测试场景；正式模式在无 manifest 时抛错，绝不回退到合成数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contracts import CONTRACT_VERSION_ID
from contracts.models import ScenarioBundle

# 六类预测序列（M1.3a 增补 carbon，与契约字段一一对应）。
SERIES_KEYS = ("price", "load", "pv", "wind", "temperature", "carbon")
_REQUIRED_MANIFEST_FIELDS = ("source", "units", "sha256")

__all__ = ["SERIES_KEYS", "ScenarioBundle", "build_scenario", "build_scenario_from_true"]


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
    """构建场景切片。合成模式需显式 `synthetic=True`。"""
    _validate_cutoff(horizon, forecast_cutoff)
    if synthetic:
        return _build_synthetic(split, start, horizon, forecast_cutoff, seed)
    return _build_from_manifest(split, start, horizon, forecast_cutoff, manifest_dir)


def build_scenario_from_true(
    split: str,
    start: str,
    horizon: int,
    forecast_cutoff: int,
    true: dict[str, np.ndarray],
    *,
    synthetic: bool = False,
    source_hashes: dict[str, str] | None = None,
) -> ScenarioBundle:
    """从真值数组构建切片；只物化 `[0, forecast_cutoff)` 的可见预测（拷贝，非视图）。"""
    _validate_cutoff(horizon, forecast_cutoff)
    missing = [k for k in SERIES_KEYS if k not in true]
    if missing:
        raise KeyError(f"缺少预测序列 {missing}（需 {list(SERIES_KEYS)}）")

    def _window(key: str) -> list[float]:
        return [float(x) for x in np.asarray(true[key])[:forecast_cutoff]]

    source_hashes = dict(source_hashes or {})
    return ScenarioBundle(
        split=split,
        start=start,
        horizon=horizon,
        forecast_cutoff=forecast_cutoff,
        price_forecast=_window("price"),
        load_forecast=_window("load"),
        pv_forecast=_window("pv"),
        wind_forecast=_window("wind"),
        temperature_forecast=_window("temperature"),
        carbon_forecast=_window("carbon"),
        source_hashes=source_hashes,
        synthetic=synthetic,
    )


def _validate_cutoff(horizon: int, forecast_cutoff: int) -> None:
    if forecast_cutoff < 0 or forecast_cutoff > horizon:
        raise ValueError(f"forecast_cutoff {forecast_cutoff} 必须在 [0, horizon={horizon}] 内")


def _build_synthetic(
    split: str, start: str, horizon: int, forecast_cutoff: int, seed: int
) -> ScenarioBundle:
    """确定性的**合成**开发场景（`synthetic=True`），不得冒充真实数据。"""
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
    }
    source_hashes = {
        "synthetic_seed": str(seed),
        "generator": "scenario._build_synthetic",
        "contract_version_id": CONTRACT_VERSION_ID,
    }
    return build_scenario_from_true(
        split, start, horizon, forecast_cutoff, true, synthetic=True, source_hashes=source_hashes
    )


def _build_from_manifest(
    split: str, start: str, horizon: int, forecast_cutoff: int, manifest_dir: str
) -> ScenarioBundle:
    manifest_path = Path(manifest_dir) / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"正式模式无数据：缺少 manifest {manifest_path}"
            "（M1.2 阻塞，未验证到满足许可证/匿名访问/全年粒度的数据组合）。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in _REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            raise ValueError(f"manifest 缺少字段 {field!r}（需单位/来源/hash）")
    # 真实数据加载 + 切片 + sha256 校验待 M1.2 解除阻塞后实现；不得回退到合成数据。
    raise NotImplementedError("真实数据加载待 M1.2 解除阻塞后实现")
