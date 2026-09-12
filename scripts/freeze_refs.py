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
    "planned_load_reserve_alpha": 0.40,
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
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
