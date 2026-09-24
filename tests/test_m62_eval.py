"""M6.2 → M6-P1 测试：统一评估 schema，多方法同 schema，服务资格不默认达标。

**M6-P1 迁移登记**：旧骨架的
`rec.total_cost_sgd` / `rec.total_carbon_kg` 更名为 `purchase_cost_sgd` /
`carbon_kg_co2e`（成本分列，见 `docs/M6_EVALUATION_PROTOCOL.md` §3）；
`service_qualified=False` 的调用方开关**已删除**，改由**显式传入的服务标准**计算
（标准未冻结 ⇒ `None` = 未判定）。**未删除任何测试**。
"""

import pytest

from contracts.models import EvaluationRecord
from envs.idc_price_env import IDCPriceEnv20D
from evaluation.adapter import evaluate, neutral_rule


def _random_rule(obs):
    import numpy as np

    a = np.random.default_rng(0).random(21, dtype=np.float32)
    a[20] = a[20] * 2 - 1
    return a


def _evaluate(env, method, action_fn, **kwargs):
    return evaluate(env, method, action_fn, service_standard=None, **kwargs)


def test_rule_baseline_produces_evaluation_record():
    env = IDCPriceEnv20D()
    rec = _evaluate(env, "rule_baseline", neutral_rule, run_id="r0")
    assert isinstance(rec, EvaluationRecord)
    assert rec.method == "rule_baseline"
    assert rec.purchase_cost_sgd >= 0.0
    assert rec.carbon_kg_co2e >= 0.0


def test_methods_share_schema():
    env1 = IDCPriceEnv20D()
    env2 = IDCPriceEnv20D()
    rec1 = _evaluate(env1, "rule_baseline", neutral_rule, run_id="r1")
    rec2 = _evaluate(env2, "penalty_ppo", _random_rule, run_id="r2")
    assert rec1.model_dump().keys() == rec2.model_dump().keys()


def test_service_qualification_requires_an_explicit_standard():
    """旧口径 `service_qualified=False` 由调用方直接给出；新口径**不得**默认达标。"""
    env = IDCPriceEnv20D()
    rec = _evaluate(env, "rule_baseline", neutral_rule, run_id="r3")
    assert rec.service_qualified is None
    assert rec.service_standard_id is None
    with pytest.raises(TypeError):
        evaluate(env, "rule_baseline", neutral_rule, run_id="r4")
