#!/usr/bin/env python
"""M3.4 旧 α 隔离诊断。

- 记录旧 α 值（0.40）。
- 方法级确立：正式主链 `_loads_from_group_completion` 不使用 α（α=0 与 α=0.40 结果一致）。
- `_actual_loads_from_completed_work`（α 的旧使用者）已无调用点（死代码）。
- 注明：全 episode α 对比受 `reset(seed=0)` 跨 episode 非确定性影响（基线问题，M3.6 处理）。
"""

from __future__ import annotations

import json

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D

OLD_ALPHA = 0.40  # configs/config_ultimate.py planned_load_reserve_alpha 旧值


def main() -> None:
    env = IDCPriceEnv20D()
    cg = np.full(env.model.N, 100.0, dtype=np.float64)

    env.planned_load_reserve_alpha = 0.0
    loads_alpha_0 = env._loads_from_group_completion(cg)

    env.planned_load_reserve_alpha = OLD_ALPHA
    loads_alpha_old = env._loads_from_group_completion(cg)

    print(
        json.dumps(
            {
                "old_alpha_value": OLD_ALPHA,
                "method_alpha_free": bool(np.allclose(loads_alpha_0, loads_alpha_old)),
                "old_alpha_consumer_is_dead_code": True,  # _actual_loads_from_completed_work 无调用点
                "label": "historical sensitivity, not future reserve",
                "note": "全 episode α 对比受 reset 非确定性影响，α 惰性在方法级确立",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
