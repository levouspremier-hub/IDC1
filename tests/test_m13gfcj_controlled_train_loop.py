"""M1.3g-f-c-j：冻结配置驱动的训练闭环与批次边界训练恢复。

只覆盖**实现确需**的约定（不为假设性边界堆回归）：

1. 训练闭环只接受**冻结**配置——候选文件必须被拒绝；
2. 训练恢复格式与 `checkpointing/eval_input.py` 的**评估输入**角色互相拒绝；
3. `minibatch_ppo_step` **不**更新 Lagrangian（乘子按整批更新一次）；
4. 策略初值确定性，且**不**扰动调用方的全局 RNG。

真实闭环的端到端证据（3 批连续 vs 2 批 + 恢复）在任务卡 §7 与 `runs/` 产物中，
不在本文件内重跑（3 批 × 4 episode × 48 步 corrector on 约 105 秒）。
"""

import json
import pathlib

import pytest
import torch

from safe_rl_v2 import formal_train_loop as loop

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FROZEN = REPO_ROOT / loop.FROZEN_CONFIG_LOGICAL_PATH
CANDIDATE = REPO_ROOT / "configs/training/idc_training_config_candidate_v1.json"


def test_the_frozen_config_is_the_one_the_loop_accepts():
    config = loop.load_frozen_training_config()
    assert config["schema"] == "idc-training-config-v1"
    assert config["status"] == "frozen"
    assert config["training"]["corrector"]["time_limit_s"] == pytest.approx(0.25)
    summary = loop.config_provenance_summary(config)
    assert summary["calibration_run_id"] == "m13gci_calibration_after_f3r2"
    assert summary["service_standard_id"] == "m6-service-standard-v1"
    assert len(summary["asset_hashes"]) == 6


def test_the_candidate_config_is_rejected_as_a_training_source(tmp_path):
    """候选文件标 `candidate_not_frozen`：**不得**驱动训练。"""
    assert CANDIDATE.exists(), "候选配置应已物化"
    # 候选的真实 schema 与冻结产物不同，故先被 schema 拦下
    with pytest.raises(loop.FormalTrainLoopError, match="schema"):
        loop.load_frozen_training_config(CANDIDATE)
    # 即使把 schema 改成冻结产物的，`status` 门也必须拦下它
    lookalike = json.loads(FROZEN.read_text(encoding="utf-8"))
    lookalike["status"] = "candidate_not_frozen"
    path = tmp_path / "lookalike.json"
    path.write_text(json.dumps(lookalike, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(loop.FormalTrainLoopError, match="status"):
        loop.load_frozen_training_config(path)


def test_resume_schema_and_role_are_distinct_from_the_eval_input_contract():
    from checkpointing.eval_input import EVAL_INPUT_ARTIFACT_ROLES, EVAL_INPUT_SCHEMA

    assert loop.RESUME_SCHEMA != EVAL_INPUT_SCHEMA
    assert loop.RESUME_ARTIFACT_ROLE not in EVAL_INPUT_ARTIFACT_ROLES


def test_minibatch_step_does_not_touch_the_lagrangian(tmp_path):
    """4 次 Adam step 之后乘子与更新计数**不变**；整批更新一次才变。"""
    import numpy as np

    from safe_rl_v2.lagrangian import ConstraintSpec, Lagrangian
    from safe_rl_v2.policy import SafePPOPolicy
    from safe_rl_v2.ppo_update import minibatch_ppo_step

    torch.manual_seed(0)
    policy = SafePPOPolicy(obs_dim=5, action_dim=21, hidden=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    standards = [
        ConstraintSpec(name="business", budget=0.0, unit="violation_task_steps",
                       learning_rate=0.01, max_multiplier=10.0),
        ConstraintSpec(name="carbon", budget=1.0, unit="kgCO2e",
                       learning_rate=0.01, max_multiplier=10.0),
    ]
    lagrangian = Lagrangian(standards)

    n = 8
    obs = torch.randn(n, 5)
    raw = torch.rand(n, 21)
    old_lp = torch.zeros(n)
    zeros = torch.zeros(n)
    targets = {h: torch.zeros(n) for h in ("reward", "business", "carbon")}

    before = lagrangian.multipliers()
    for _ in range(4):
        minibatch_ppo_step(
            policy, optimizer, observation=obs, raw_action=raw,
            old_raw_log_prob=old_lp, adv_reward=torch.ones(n),
            adv_business=zeros, adv_carbon=zeros, critic_targets=targets,
            lambda_business=before["business"], lambda_carbon=before["carbon"],
            clip_epsilon=0.2)
    assert lagrangian.multipliers() == before, "minibatch 更新不得改动乘子"
    assert int(lagrangian._updates) == 0

    lagrangian.update({"business": np.zeros(n), "carbon": np.ones(n) * 2.0})
    assert int(lagrangian._updates) == 1, "整批只更新一次乘子"


def test_seeded_policy_init_is_deterministic_and_leaves_global_rng_alone():
    config = loop.load_frozen_training_config()
    torch.manual_seed(12345)
    before = torch.rand(4)
    first = loop.build_seeded_policy(config, obs_dim=7, seed=0)
    torch.manual_seed(12345)
    second = loop.build_seeded_policy(config, obs_dim=7, seed=0)
    after = torch.rand(4)
    assert torch.equal(before, after), "不得改动调用方的全局 RNG 状态"
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), f"{name} 初值不确定"


def test_resume_checkpoint_rejects_a_foreign_schema(tmp_path):
    """评估输入的 schema 与训练恢复互不相认（角色不同）。"""
    from checkpointing import CURRENT_CONTRACT_VERSION, VersionedCheckpoint

    path = tmp_path / "foreign.pt"
    VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION, action_dim=21, obs_dim=7,
        schema_hash="m6p1-eval-input-v1", code_revision="",
        state={"policy": {}, "optimizer": {}, "lagrangian": {},
               "sampling_generator": torch.Generator().get_state(),
               "shuffle_generator": torch.Generator().get_state(),
               "next_batch_index": 0, "origins": [], "episodes_per_batch": 4,
               "steps_per_episode": 48, "frozen_config": {}, "config_summary": {},
               "training_scope": loop.TRAINING_SCOPE,
               "artifact_role": loop.RESUME_ARTIFACT_ROLE},
    ).save(path)

    config = loop.load_frozen_training_config()
    policy = loop.build_seeded_policy(config, obs_dim=7, seed=0)
    with pytest.raises(Exception, match="schema_hash|schema"):
        loop.load_resume_checkpoint(
            path, policy=policy, optimizer=loop.build_optimizer(config, policy),
            lagrangian=loop.build_lagrangian(config),
            sampling_generator=torch.Generator(), shuffle_generator=torch.Generator(),
            config=config, expected_obs_dim=7)


def test_run_artifact_declares_the_controlled_scope():
    """真实短跑产物必须自述 `controlled_short_run` 且 claims 三项为 false。"""
    report_path = REPO_ROOT / "runs/m13gcj_controlled_3batch/report.json"
    if not report_path.exists():
        pytest.skip("受控短跑产物不在本机（runs/ 未入库）")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["training_scope"] == "controlled_short_run"
    assert report["claims"] == {"trained": False, "performance_evaluated": False,
                                "convergence_claimed": False}
    assert report["adam_steps_total"] == 48
    assert report["lagrangian_updates_total"] == 3
