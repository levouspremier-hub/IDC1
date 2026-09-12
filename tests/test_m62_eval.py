"""M6.2 测试：统一评估 schema，多方法同 schema，服务不达标标记。"""

from contracts.models import EvaluationRecord
from envs.idc_price_env import IDCPriceEnv20D
from evaluation.adapter import evaluate, neutral_rule


def _random_rule(obs):
    import numpy as np

    a = np.random.default_rng(0).random(21, dtype=np.float32)
    a[20] = a[20] * 2 - 1
    return a


def test_rule_baseline_produces_evaluation_record():
    env = IDCPriceEnv20D()
    rec = evaluate(env, "rule", neutral_rule, run_id="r0")
    assert isinstance(rec, EvaluationRecord)
    assert rec.method == "rule"
    assert rec.total_cost_sgd >= 0.0
    assert rec.total_carbon_kg >= 0.0


def test_methods_share_schema():
    env1 = IDCPriceEnv20D()
    env2 = IDCPriceEnv20D()
    rec1 = evaluate(env1, "rule", neutral_rule, run_id="r1")
    rec2 = evaluate(env2, "random", _random_rule, run_id="r2")
    assert rec1.model_dump().keys() == rec2.model_dump().keys()


def test_service_unqualified_flag():
    env = IDCPriceEnv20D()
    rec = evaluate(env, "rule", neutral_rule, run_id="r3", service_qualified=False)
    assert rec.service_qualified is False
