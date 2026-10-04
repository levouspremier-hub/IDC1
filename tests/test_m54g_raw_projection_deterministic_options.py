"""M5.4g 测试：确定性 MIP 选项必须接到 **corrector 实际走**的 raw-projection 两阶段。

M5.4f 把 `deterministic_mip_options` 接到了 `_solve_time_indexed`（另一个函数），
而 `correct()` 走的是 `solve_time_indexed_mip_raw_projection` 的阶段 A/B，
后者的 `_options()` 仍然只返回 `time_limit`。

**本文件用运行时 spy 证明接线**（要求 4：不得只用 AST / 字符串搜索）：
spy 同时挂在 `deterministic_mip_options` 与 `scipy.optimize.milp` 上，
直接观察**真正传给求解器**的 options。
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from planning import model as planning_model
from planning.corrector import correct
from planning.snapshot_adapter import build_snapshot

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
FIXTURE_STEPS_INTO_EPISODE = 6
FIXTURE_PROPOSAL_SEED = 1
FIXTURE_STORAGE_ACTION = 0.3
NORMAL_BUDGET_S = 0.05


def _snapshot_and_proposal():
    env = IDCPriceEnv20D(**ENV_SEED_KWARGS)
    env.reset(seed=0)
    neutral = np.concatenate(
        [np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    for _ in range(FIXTURE_STEPS_INTO_EPISODE):
        env.step(neutral)
    snapshot = build_snapshot(env)
    rng = np.random.default_rng(FIXTURE_PROPOSAL_SEED)
    proposal = DispatchProposal(
        compute_actions=[float(x) for x in rng.random(len(snapshot.group_work_capacity))],
        storage_action=FIXTURE_STORAGE_ACTION,
    )
    return snapshot, proposal


class _Spy:
    """同时观测 `deterministic_mip_options` 与真正传给 `scipy.optimize.milp` 的 options。"""

    def __init__(self, monkeypatch):
        import scipy.optimize as scipy_optimize

        self.builder_calls: list[dict] = []
        self.milp_options: list[dict] = []

        real_builder = planning_model.deterministic_mip_options
        real_milp = scipy_optimize.milp

        def builder(*, time_limit_s):
            out = real_builder(time_limit_s=time_limit_s)
            self.builder_calls.append(out)
            return out

        def milp(*args, **kwargs):
            self.milp_options.append(dict(kwargs.get("options") or {}))
            return real_milp(*args, **kwargs)

        monkeypatch.setattr(planning_model, "deterministic_mip_options", builder)
        monkeypatch.setattr(scipy_optimize, "milp", milp)


# --- 1. 运行时证明：两阶段都走 deterministic_mip_options ---------------------

def test_both_raw_projection_stages_use_deterministic_mip_options(monkeypatch):
    # This wiring assertion requires real A/B solves, rather than performance of
    # a random 20-group instance. Keep its original shared 0.05s budget; the
    # other tests and the cross-process probe retain the original large case.
    from tests.test_m44_corrector import _snapshot

    spy = _Spy(monkeypatch)
    snapshot = _snapshot()
    proposal = DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.3)

    correction = correct(snapshot, proposal, time_limit_s=NORMAL_BUDGET_S)
    assert str(correction.failure) == "none", (
        f"夹具必须真实求解（否则测不到两阶段）：failure={correction.failure}"
    )

    assert len(spy.builder_calls) >= 2, (
        f"raw projection 两阶段都必须调用 deterministic_mip_options，"
        f"实际只调用 {len(spy.builder_calls)} 次"
    )
    assert len(spy.milp_options) >= 2, f"应至少两次 milp 调用，实际 {len(spy.milp_options)}"


def test_every_milp_call_receives_random_seed_and_parallel(monkeypatch):
    spy = _Spy(monkeypatch)
    snapshot, proposal = _snapshot_and_proposal()
    correct(snapshot, proposal, time_limit_s=NORMAL_BUDGET_S)

    assert spy.milp_options, "没有观测到任何 milp 调用"
    for index, options in enumerate(spy.milp_options):
        assert options.get("random_seed") == 0, f"第 {index} 次 milp 缺 random_seed=0：{options}"
        assert options.get("parallel") is False, (
            f"第 {index} 次 milp 缺 parallel=False：{options}"
        )
        assert "time_limit" in options, f"第 {index} 次 milp 缺 time_limit：{options}"


def test_options_are_identical_across_the_two_stages(monkeypatch):
    """两阶段共享同一全局 deadline，但 options 的**确定性部分**必须一致。"""
    spy = _Spy(monkeypatch)
    snapshot, proposal = _snapshot_and_proposal()
    correct(snapshot, proposal, time_limit_s=NORMAL_BUDGET_S)

    deterministic_parts = [
        {k: v for k, v in options.items() if k != "time_limit"}
        for options in spy.milp_options
    ]
    assert deterministic_parts, "没有观测到 milp 调用"
    assert all(part == deterministic_parts[0] for part in deterministic_parts)


def test_remaining_budget_semantics_are_preserved(monkeypatch):
    """只替换 options 构造：time_limit 仍来自共享 deadline 的剩余预算。

    阶段 B 的 time_limit 必然 <= 阶段 A（同一 deadline 继续消耗），
    且两者都 <= 总预算。
    """
    spy = _Spy(monkeypatch)
    snapshot, proposal = _snapshot_and_proposal()
    correct(snapshot, proposal, time_limit_s=NORMAL_BUDGET_S)

    limits = [options["time_limit"] for options in spy.milp_options]
    assert limits, "没有观测到 milp 调用"
    assert all(0.0 <= x <= NORMAL_BUDGET_S for x in limits), limits
    assert limits == sorted(limits, reverse=True), f"剩余预算必须单调不增：{limits}"


def test_no_time_limit_still_gets_the_deterministic_options(monkeypatch):
    """`time_limit_s=None` 时仍必须传 random_seed/parallel，仅不含 time_limit。

    注意：`correct()` 出于 M4.4 的预算纪律**显式拒绝** `None`
    （"time_limit_s 必须为显式正数预算"），故本用例直接驱动
    `solve_time_indexed_mip_raw_projection` —— 要验证的是模型层的 options 构造。
    """
    spy = _Spy(monkeypatch)
    snapshot, proposal = _snapshot_and_proposal()
    planning_model.solve_time_indexed_mip_raw_projection(
        snapshot, proposal, time_limit_s=None
    )

    assert spy.milp_options, "没有观测到 milp 调用"
    for options in spy.milp_options:
        assert options.get("random_seed") == 0
        assert options.get("parallel") is False
        assert "time_limit" not in options, f"无预算时不得出现 time_limit：{options}"


def test_corrector_still_rejects_a_missing_budget():
    """M4.4 的预算纪律不得被本卡放松：correct() 仍必须拒绝 None。"""
    snapshot, proposal = _snapshot_and_proposal()
    with pytest.raises(ValueError, match="time_limit_s"):
        correct(snapshot, proposal, time_limit_s=None)


# --- 2. options 构造函数本身 ------------------------------------------------

def test_builder_is_still_a_pure_function():
    a = planning_model.deterministic_mip_options(time_limit_s=0.05)
    b = planning_model.deterministic_mip_options(time_limit_s=0.05)
    assert a == b
    assert set(a) == {"random_seed", "parallel", "time_limit"}
    assert planning_model.deterministic_mip_options(time_limit_s=None) == {
        "random_seed": 0, "parallel": False
    }


# --- 3. probe 的 solver evidence 必须从同一函数导出 -------------------------

def test_probe_solver_evidence_reports_all_options():
    from scripts import probe_corrector_repro as probe

    evidence = probe.solver_evidence(budget=NORMAL_BUDGET_S)
    options = evidence["options"]
    assert options["random_seed"] == 0, f"evidence 必须记录 random_seed：{options}"
    assert options["parallel"] is False, f"evidence 必须记录 parallel：{options}"
    assert options["time_limit"] == pytest.approx(NORMAL_BUDGET_S)
    # 必须与运行时构造的一致（同一函数导出）
    assert options == planning_model.deterministic_mip_options(time_limit_s=NORMAL_BUDGET_S)


def test_probe_does_not_hand_build_the_solver_options():
    """探针不得自己拼 options —— 必须调用 planning 的构造函数。"""
    import ast

    from scripts import probe_corrector_repro as probe

    source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
    assert "deterministic_mip_options" in source
    tree = ast.parse(source)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "solver_evidence"
    )
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
    }
    assert "deterministic_mip_options" in called


# --- 4. 真实测量（slow）：接线后重新测 --------------------------------------

_CHILD = r'''
import hashlib, json, sys
import numpy as np
from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from planning.corrector import correct
from planning.snapshot_adapter import build_snapshot

steps_into, prop_seed, budget, storage = (
    int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
)
env = IDCPriceEnv20D(task_seed=0, server_seed=0, forecast_seed=300000)
env.reset(seed=0)
neutral = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
for _ in range(steps_into):
    env.step(neutral)
snapshot = build_snapshot(env)
n = len(snapshot.group_work_capacity)
rng = np.random.default_rng(prop_seed)
proposal = DispatchProposal(
    compute_actions=[float(x) for x in rng.random(n)], storage_action=storage
)
c = correct(snapshot, proposal, time_limit_s=budget)
payload = {
    "exec_compute": [float(x) for x in c.exec_compute_actions],
    "exec_storage": float(c.exec_storage_action),
    "failure": str(c.failure), "reason": str(c.reason),
    "business_gap": float(c.business_gap),
    "projection_offset": float(c.projection_offset),
    "stage_a_objective": float(c.stage_a_objective),
    "stage_b_objective": float(c.stage_b_objective),
}
blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(blob).hexdigest())
'''


def _digest(budget: float) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(FIXTURE_STEPS_INTO_EPISODE),
         str(FIXTURE_PROPOSAL_SEED), str(budget), str(FIXTURE_STORAGE_ACTION)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_raw_projection_is_deterministic_across_processes_after_wiring():
    """接线后重新测量：固定夹具、3 个独立进程 × 2 轮。"""
    digests = [_digest(NORMAL_BUDGET_S) for _ in range(6)]
    assert len(set(digests)) == 1, f"接线后仍不一致：{digests}"


# --- 5. 回归：**全部** milp 调用点都必须走共享构造函数 -----------------------

def _spy_all_milp(monkeypatch):
    """spy 全部 milp 调用（无论来自哪个 planning 函数）。"""
    import scipy.optimize as scipy_optimize

    seen: list[dict] = []
    real = scipy_optimize.milp

    def milp(*args, **kwargs):
        seen.append(dict(kwargs.get("options") or {}))
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy_optimize, "milp", milp)
    return seen


def _assert_all_deterministic(seen: list[dict], *, expect_budget: bool) -> None:
    assert seen, "没有观测到任何 milp 调用"
    for index, options in enumerate(seen):
        assert options.get("random_seed") == 0, f"milp #{index} 缺 random_seed：{options}"
        assert options.get("parallel") is False, f"milp #{index} 缺 parallel：{options}"
        if expect_budget:
            assert "time_limit" in options, f"milp #{index} 缺 time_limit：{options}"


def test_raw_projection_never_bypasses_the_shared_builder(monkeypatch):
    """raw projection 的每一次 milp 调用都必须带确定性选项（不是部分调用）。"""
    seen = _spy_all_milp(monkeypatch)
    snapshot, proposal = _snapshot_and_proposal()
    correct(snapshot, proposal, time_limit_s=NORMAL_BUDGET_S)
    _assert_all_deterministic(seen, expect_budget=True)


def test_other_mip_backend_path_also_uses_the_shared_builder(monkeypatch):
    """M5.4f 曾经改到的**另一条**路径（`_solve_time_indexed`）必须仍然正确。

    本用例覆盖「同一模块里存在多个 milp 调用点」这一类错接：
    只改一处、漏掉另一处会让产品以为自己修好了。
    """
    seen = _spy_all_milp(monkeypatch)
    snapshot, _proposal = _snapshot_and_proposal()
    planning_model.solve_time_indexed_mip(snapshot, time_limit_s=NORMAL_BUDGET_S)
    _assert_all_deterministic(seen, expect_budget=True)


def test_all_planning_milp_call_sites_are_covered_by_the_builder():
    """结构性清点：模块内每个 `_milp(` 调用点都必须把 options 交给共享构造。"""
    import ast

    source = pathlib.Path(planning_model.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"_milp", "milp"}
    ]
    assert len(call_sites) >= 3, f"预期至少 3 个 milp 调用点，实际 {len(call_sites)}"

    def _resolves_to_builder(expr, scope: ast.AST) -> bool:
        """options 表达式直接调用构造函数，或其名字在本作用域内由构造函数赋值。"""
        rendered = ast.unparse(expr)
        if "deterministic_mip_options" in rendered:
            return True
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            # 本地 helper（如 `_options()`）：其函数体必须最终调用构造函数
            helper = next(
                (
                    fn for fn in ast.walk(scope)
                    if isinstance(fn, ast.FunctionDef) and fn.name == expr.func.id
                ),
                None,
            )
            if helper is not None and "deterministic_mip_options" in ast.unparse(helper):
                return True
        if isinstance(expr, ast.Name):
            for node in ast.walk(scope):
                targets: list[ast.AST] = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                    value = node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets = [node.target]
                    value = node.value
                else:
                    continue
                if any(isinstance(t, ast.Name) and t.id == expr.id for t in targets):
                    if "deterministic_mip_options" in ast.unparse(value):
                        return True
        return False

    for call in call_sites:
        options_kw = next((kw for kw in call.keywords if kw.arg == "options"), None)
        assert options_kw is not None, f"行 {call.lineno} 的 milp 调用没有 options"
        enclosing = next(
            (
                fn for fn in ast.walk(tree)
                if isinstance(fn, ast.FunctionDef)
                and fn.lineno <= call.lineno <= (fn.end_lineno or fn.lineno)
            ),
            tree,
        )
        assert _resolves_to_builder(options_kw.value, enclosing), (
            f"行 {call.lineno} 的 options 未走共享构造：{ast.unparse(options_kw.value)}"
        )
