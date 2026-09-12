"""M4.6 真实性能门禁：环境步耗时、修正器耗时、变量/整数数、内存、rollout 吞吐。

不训练正式策略；结果打印为 JSON。Parquet 产物在 M7.1 统一写入。
"""

from __future__ import annotations

import json
import time
import tracemalloc
from typing import Any

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from planning.model import build_milp
from planning.snapshot_adapter import build_snapshot
from safe_rl.corrector_wrapper import CorrectorWrapper


def main() -> None:
    env = CorrectorWrapper(IDCPriceEnv20D())
    env.reset(seed=0)
    raw: Any = env.env  # 未包装环境
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])

    tracemalloc.start()
    step_times: list[float] = []
    solve_times: list[float] = []
    for _ in range(raw.horizon):
        t0 = time.perf_counter()
        _, _, terminated, truncated, info = env.step(action)
        step_times.append(time.perf_counter() - t0)
        solve_times.append(float(info.get("correction_solve_time_s", 0.0)))
        if terminated or truncated:
            break
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    snapshot = build_snapshot(raw)
    model = build_milp(snapshot)

    report = {
        "n_server_groups": 20,
        "horizon": raw.horizon,
        "n_tasks": len(snapshot.tasks),
        "milp_n_vars": int(model.c.shape[0]),
        "milp_n_integer": int(model.integrality.sum()),
        "env_step_mean_s": float(np.mean(step_times)),
        "env_step_p95_s": float(np.percentile(step_times, 95)),
        "corrector_solve_mean_s": float(np.mean(solve_times)),
        "corrector_solve_p95_s": float(np.percentile(solve_times, 95)),
        "rollout_steps_per_s": float(len(step_times) / max(sum(step_times), 1e-9)),
        "peak_memory_bytes": int(peak_memory),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
