#!/usr/bin/env python
"""M5.1d 主链 smoke 门禁（`make smoke`）。

**非训练**健康检查：用显式种子构造真实 `IDCPriceEnv20D`，取 21 维 raw 动作经**现有**
`CorrectorWrapper` 执行若干步，逐步检查维度、接入上限、SOC 边界与能量平衡字段，
再把整个过程写成 `runs/<run_id>/` 的五类产物。

**本脚本不训练**：不更新任何参数、不实现 PPO、不调用 `optimizer.step()`，
不输出任何性能或收敛结论。`report.json` 的 `claims` 块显式声明三者均为 `false`。

不使用任何外部数据（M1.2 仍阻塞）；`manifest.data_hash` 的依据是**数据来源记录**的
规范化哈希，其原文同时写入 `config.yaml` 的 `data_provenance`，可复现核对。

环境三类种子必须同时显式给出：既有 env 在 `task_seed`/`server_seed`/`forecast_seed`
任一为 `None` 时使用 `np.random.default_rng(None)` 熵源，跨进程不可复现。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# 允许以 `python scripts/smoke_main_chain.py` 直接运行（补仓库根到 sys.path）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from envs.idc_price_env import IDCPriceEnv20D
from runs.writer import write_run
from safe_rl.corrector_wrapper import CorrectorWrapper
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, UNIT_METADATA

REPO_ROOT = Path(__file__).resolve().parent.parent

# 显式场景定义（与 benchmark_corrector 的 normal 场景一致）；scenario_hash 取自它
SCENARIO_KWARGS = {"horizon": 24, "access_limit_kw": 18.0, "bess_soc_init": 0.50}
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}

DEFAULT_SEED = 0
DEFAULT_STEPS = 3
DEFAULT_CORRECTOR_TIME_LIMIT_S = 0.05
ACCESS_LIMIT_TOL_KW = 1e-6

# 充放电互斥 / 能量平衡要求存在且有限的具名字段
ENERGY_BALANCE_FIELDS = (
    "energy_kWh",
    "grid_energy_kWh",
    "bess_charge_kWh",
    "bess_discharge_kWh",
    "bess_charge_power_kW",
    "bess_discharge_power_kW",
    "P_IDC_kW",
)
CHECK_NAMES = ("raw_exec_dims", "access_limit", "soc_bounds", "energy_balance")


class MissingSmokeInfoError(KeyError):
    """环境 info 缺字段：必须报错，不得用默认值伪造。"""


class SmokeCheckError(RuntimeError):
    """具名健康检查失败。"""


def _require(info: dict, key: str):
    if key not in info:
        raise MissingSmokeInfoError(
            f"环境 info 缺少必需字段 {key!r}；禁止用默认值（如 0）伪造缺失约束"
        )
    return info[key]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(obj) -> str:
    return _sha256_bytes(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _dependency_lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return _sha256_bytes(lock.read_bytes()) if lock.exists() else None


def data_provenance() -> dict:
    """数据来源记录（本 smoke 不使用外部数据）。"""
    return {
        "external_dataset": False,
        "source": "synthetic internal env; M1.2 外部数据未接入，本 smoke 不加载任何数据集",
        "env_seed_kwargs": dict(ENV_SEED_KWARGS),
    }


def build_env(seed: int, **over) -> IDCPriceEnv20D:
    kwargs = dict(SCENARIO_KWARGS)
    kwargs.update(ENV_SEED_KWARGS)
    kwargs.update(over)
    return IDCPriceEnv20D(**kwargs)


def sample_raw_action(rng: np.random.Generator) -> np.ndarray:
    """21 维 raw 动作：前 20 维计算强度 ∈ [0,1]，末维有符号储能 ∈ [-1,1]。"""
    compute = rng.random(ACTION_DIM - 1).astype(np.float32)
    storage = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
    return np.concatenate([compute, storage])


def _record(
    checks: dict[str, list[dict]], name: str, step: int, passed: bool | np.bool_, detail: str
) -> None:
    checks.setdefault(name, []).append({"step": step, "passed": bool(passed), "detail": detail})
    if not passed:
        raise SmokeCheckError(f"{name} 在第 {step} 步失败：{detail}")


def run_checks(
    checks: dict[str, list[dict]],
    step: int,
    *,
    raw_action: np.ndarray,
    exec_action: np.ndarray,
    info: dict,
    env: IDCPriceEnv20D,
    action_space,
) -> None:
    # 1) raw / exec 维度与有限性
    dims_ok = (
        raw_action.shape == (ACTION_DIM,)
        and exec_action.shape == (ACTION_DIM,)
        and np.all(np.isfinite(raw_action))
        and np.all(np.isfinite(exec_action))
    )
    _record(
        checks, "raw_exec_dims", step, dims_ok,
        f"raw={raw_action.shape} exec={exec_action.shape}",
    )
    in_space = bool(action_space.contains(exec_action.astype(np.float32)))
    _record(checks, "raw_exec_dims", step, in_space, f"exec 落在动作空间内={in_space}")

    # 2) 接入上限
    access_limit = float(_require(info, "access_limit_kw"))
    grid_kw = float(_require(info, "P_grid_kW"))
    _record(
        checks, "access_limit", step,
        np.isfinite(access_limit) and np.isfinite(grid_kw)
        and grid_kw <= access_limit + ACCESS_LIMIT_TOL_KW,
        f"P_grid_kW={grid_kw:.6f} <= access_limit_kw={access_limit:.6f}",
    )

    # 3) SOC 边界
    soc = float(_require(info, "bess_soc"))
    soc_min, soc_max = float(env.bess_soc_min), float(env.bess_soc_max)
    _record(
        checks, "soc_bounds", step,
        np.isfinite(soc) and soc_min - 1e-9 <= soc <= soc_max + 1e-9,
        f"bess_soc={soc:.6f} ∈ [{soc_min}, {soc_max}]",
    )

    # 4) 能量平衡字段存在且有限 + 充放电互斥
    values = {name: float(_require(info, name)) for name in ENERGY_BALANCE_FIELDS}
    finite = all(np.isfinite(v) for v in values.values())
    _record(checks, "energy_balance", step, finite, f"字段齐全且有限：{sorted(values)}")
    charge, discharge = values["bess_charge_power_kW"], values["bess_discharge_power_kW"]
    exclusive = not (charge > 1e-9 and discharge > 1e-9)
    _record(
        checks, "energy_balance", step, exclusive,
        f"充放电互斥：charge={charge:.6f} discharge={discharge:.6f}",
    )


def _unique_run_id(base_dir: Path, seed: int) -> str:
    """默认 run_id：每次调用唯一，使 `make smoke` 可连续运行且不覆盖既有成功结果。"""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"smoke_main_chain_s{seed}_{stamp}"
    candidate, suffix = base, 1
    while (base_dir / candidate / "manifest.json").exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="主链 smoke 门禁（非训练）")
    parser.add_argument("--base-dir", default="runs", help="run 产物根目录")
    parser.add_argument("--run-id", default=None, help="显式 run id（默认每次唯一）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--corrector-time-limit-s", type=float, default=DEFAULT_CORRECTOR_TIME_LIMIT_S
    )
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps 必须 >= 1")
    if not (args.corrector_time_limit_s > 0.0):
        parser.error("--corrector-time-limit-s 必须为正数")

    base_dir = Path(args.base_dir)
    run_id = args.run_id or _unique_run_id(base_dir, args.seed)
    command = " ".join(
        ["uv run python scripts/smoke_main_chain.py", f"--seed {args.seed}",
         f"--steps {args.steps}", f"--run-id {run_id}"]
    )

    provenance = data_provenance()
    config = {
        "gate": "make_smoke_main_chain",
        "trained": False,
        "run_id": run_id,
        "seed": args.seed,
        "steps": args.steps,
        "scenario_kwargs": dict(SCENARIO_KWARGS),
        "env_seed_kwargs": dict(ENV_SEED_KWARGS),
        "data_provenance": provenance,
        "corrector_time_limit_s": float(args.corrector_time_limit_s),
        "action_dim": ACTION_DIM,
        "action_bounds": {"compute": [0.0, 1.0], "storage": [-1.0, 1.0]},
        "raw_action_source": "seeded numpy default_rng（健康检查用，非策略采样）",
        "contract_version": CONTRACT_VERSION,
        "units": UNIT_METADATA,
    }

    env = build_env(args.seed)
    wrapper = CorrectorWrapper(env, corrector_time_limit_s=float(args.corrector_time_limit_s))
    rng = np.random.default_rng(args.seed)
    wrapper.reset(seed=args.seed)

    checks: dict[str, list[dict]] = {name: [] for name in CHECK_NAMES}
    rows: list[dict] = []
    failure: SmokeCheckError | None = None

    for step in range(args.steps):
        raw_action = sample_raw_action(rng)
        obs, reward, terminated, truncated, info = wrapper.step(raw_action)
        exec_action = np.asarray(_require(info, "exec_action"), dtype=np.float32).reshape(-1)

        try:
            run_checks(
                checks, step,
                raw_action=raw_action, exec_action=exec_action,
                info=info, env=env, action_space=env.action_space,
            )
        except (SmokeCheckError, MissingSmokeInfoError) as exc:
            failure = SmokeCheckError(str(exc))
            break

        rows.append(
            {
                "step": step,
                "hour": int(_require(info, "hour")),
                "raw_compute_mean": float(np.mean(raw_action[: ACTION_DIM - 1])),
                "raw_storage": float(raw_action[ACTION_DIM - 1]),
                "exec_compute_mean": float(np.mean(exec_action[: ACTION_DIM - 1])),
                "exec_storage": float(exec_action[ACTION_DIM - 1]),
                "raw_exec_differs": bool(not np.array_equal(raw_action, exec_action)),
                "P_grid_kW": float(_require(info, "P_grid_kW")),
                "access_limit_kw": float(_require(info, "access_limit_kw")),
                "bess_soc": float(_require(info, "bess_soc")),
                "energy_kWh": float(_require(info, "energy_kWh")),
                "correction_reason": str(_require(info, "correction_reason")),
                "obs_dim": int(np.asarray(obs, dtype=np.float32).shape[0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    status = "failed" if failure is not None else "success"
    report = {
        "gate": "make_smoke_main_chain",
        "trained": False,
        "claims": {
            "trained": False,
            "performance_evaluated": False,
            "convergence_claimed": False,
        },
        "statement": (
            "主链健康检查：仅验证 raw→corrector→env 的字段与物理边界。"
            "本脚本不训练、不评估性能、不宣称收敛。"
        ),
        "status": status,
        "seed": args.seed,
        "steps_executed": len(rows),
        "contract_version": CONTRACT_VERSION,
        "checks": checks,
        "failure": None if failure is None else str(failure),
    }

    metrics = pd.DataFrame(rows)
    try:
        run_dir = write_run(
            run_id,
            config=config,
            metrics=metrics,
            report=report,
            base_dir=str(base_dir),
            seed=args.seed,
            command=command,
            data_hash=_canonical_hash(provenance),
            scenario_hash=_canonical_hash(SCENARIO_KWARGS),
            dependency_lock_hash=_dependency_lock_hash(),
            status=status,
            failure_classification=None if failure is None else "smoke_check_failed",
        )
    except FileExistsError as exc:
        print(f"拒绝覆盖既有成功 run：{exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nrun 产物：{run_dir}")
    if failure is not None:
        print(f"smoke 失败：{failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
