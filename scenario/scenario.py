"""M1.3 场景提供器。

`build_scenario` 只返回「当前真值 + 可见预测」，杜绝未来信息泄漏：
- 正式模式：读取 `data/manifest/` 与冻结处理数据；缺单位/来源/hash 或无数据抛异常。
- 合成模式：`synthetic=True` 显式开启，确定性生成并在 bundle 标记 `synthetic=True`。

泄漏不变量：改变 `t+k`（`k >= forecast_cutoff`）的真值，不改变时刻 `t` 的决策输入。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "scenario-v1"
SERIES_KEYS = ("price", "load", "pv", "wind", "temperature")
_REQUIRED_MANIFEST_FIELDS = ("source", "units", "sha256")


@dataclass(frozen=True)
class ScenarioBundle:
    """不可变场景切片：只含当前真值与可见预测（forecast 为不可变 tuple）。"""

    split: str
    start: str
    horizon: int
    forecast_cutoff: int
    current: dict[str, float]
    forecast: dict[str, tuple[float, ...]]
    schema_version: str
    synthetic: bool
    source_hashes: dict[str, str]
    hash: str

    def decision_input(self) -> dict[str, Any]:
        """决策输入 = 当前真值 + 可见预测（不含任何未来真值）。"""
        return {"current": dict(self.current), "forecast": dict(self.forecast)}


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
    """从真值数组构建切片（内部与测试用）；只物化当前值与可见预测（拷贝，非视图）。"""
    _validate_cutoff(horizon, forecast_cutoff)
    current = {k: float(np.asarray(true[k])[0]) for k in SERIES_KEYS}
    forecast = {
        k: tuple(float(x) for x in np.asarray(true[k])[:forecast_cutoff]) for k in SERIES_KEYS
    }
    source_hashes = dict(source_hashes or {})
    h = _hash(split, start, horizon, forecast_cutoff, current, forecast, source_hashes, synthetic)
    return ScenarioBundle(
        split, start, horizon, forecast_cutoff, current, forecast,
        SCHEMA_VERSION, synthetic, source_hashes, h,
    )


def _validate_cutoff(horizon: int, forecast_cutoff: int) -> None:
    if forecast_cutoff < 0 or forecast_cutoff > horizon:
        raise ValueError(f"forecast_cutoff {forecast_cutoff} 必须在 [0, horizon={horizon}] 内")


def _build_synthetic(
    split: str, start: str, horizon: int, forecast_cutoff: int, seed: int
) -> ScenarioBundle:
    rng = np.random.default_rng(seed)
    t = np.arange(horizon, dtype=float)
    true = {
        "price": 0.15 + 0.05 * np.sin(2 * np.pi * t / 24) + 0.02 * rng.normal(size=horizon),
        "load": (
            6000.0 + 800.0 * np.sin(2 * np.pi * (t - 8.0) / 24.0)
            + 100.0 * rng.normal(size=horizon)
        ),
        "pv": np.clip(0.5 * np.sin(np.pi * (t % 24.0 - 6.0) / 12.0), 0.0, None),
        "wind": np.zeros(horizon),
        "temperature": (
            28.0 + 3.0 * np.sin(2 * np.pi * (t - 14.0) / 24.0)
            + 0.5 * rng.normal(size=horizon)
        ),
    }
    source_hashes = {"synthetic_seed": str(seed), "generator": "scenario._build_synthetic"}
    return build_scenario_from_true(
        split, start, horizon, forecast_cutoff, true, synthetic=True, source_hashes=source_hashes
    )


def _build_from_manifest(
    split: str, start: str, horizon: int, forecast_cutoff: int, manifest_dir: str
) -> ScenarioBundle:
    manifest_path = Path(manifest_dir) / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"正式模式无数据：缺少 manifest {manifest_path}（M1.2 阻塞，未下载真实数据）。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in _REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            raise ValueError(f"manifest 缺少字段 {field!r}（需单位/来源/hash）")
    # 真实数据加载 + 切片 + sha256 校验待 M1.2 解除阻塞后实现。
    raise NotImplementedError("真实数据加载待 M1.2 解除阻塞后实现")


def _hash(
    split: str,
    start: str,
    horizon: int,
    forecast_cutoff: int,
    current: dict[str, float],
    forecast: dict[str, tuple[float, ...]],
    source_hashes: dict[str, str],
    synthetic: bool,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "start": start,
        "horizon": horizon,
        "forecast_cutoff": forecast_cutoff,
        "synthetic": synthetic,
        "source_hashes": dict(sorted(source_hashes.items())),
        "current": dict(sorted(current.items())),
        "forecast": {k: [round(x, 6) for x in v] for k, v in sorted(forecast.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
