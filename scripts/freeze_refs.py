#!/usr/bin/env python
"""M6.1 冻结归一化参考值。

只读取声明物理尺度（M1.2 训练数据未冻结前）或训练切分，输出 configs/frozen_refs/refs.json。
测试日与方法均不得重新计算；正式评估缺冻结文件即失败。
"""

from __future__ import annotations

import json
from pathlib import Path

DECLARED_REFS = {
    "price_ref": 1.50,  # SGD/kWh
    "lambda_ref": 2000.0,  # 任务到达
    "queue_ref": 6000.0,  # 工作
    "queue_capacity_ref": 6000.0,
    "cost_ref": 60.0,  # SGD
    "carbon_ref": 15.0,  # kgCO2
    "peak_power_threshold_kW": 18.0,
    "peak_power_ref_kW": 10.0,
    "grid_power_limit_kW": 18.0,
    "sla_penalty_ref": 50.0,
    "wind_ref_kw": 1.0,  # 风电容量尺度（kW），与 pv_ref_kw 同口径声明
    "carbon_factor_ref": 1.0,  # 碳强度尺度（kgCO2/kWh）
}

# 单位与来源（M3.10b）：参考值只用声明物理尺度，禁止按 episode / 测试日重算。
REF_UNITS = {
    "price_ref": "SGD/kWh",
    "lambda_ref": "work-units/hour",
    "queue_ref": "work-units",
    "queue_capacity_ref": "work-units",
    "cost_ref": "SGD",
    "carbon_ref": "kgCO2",
    "peak_power_threshold_kW": "kW",
    "peak_power_ref_kW": "kW",
    "grid_power_limit_kW": "kW",
    "sla_penalty_ref": "SGD",
    "wind_ref_kw": "kW",
    "carbon_factor_ref": "kgCO2/kWh",
}

OUT_PATH = Path("configs/frozen_refs/refs.json")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "frozen-refs-v1",
        "source": "declared physical scales (M1.2 training data not yet frozen)",
        "training_range": None,
        "data_hash": None,
        "references": DECLARED_REFS,
        "units": REF_UNITS,
        "frozen": True,
        "note": "参考值只用声明物理尺度；禁止按 episode / 测试日重算（M3.10b）",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
