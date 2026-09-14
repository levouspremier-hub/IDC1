"""M5.4d 测试：corrector 跨进程可复现性探针。

**本文件不预设任何机器相关结论。** 结论逻辑（`decide`）是纯函数，
用合成的 digest 列表独立驱动；真实测量只断言**产物完整性与结论自洽**，
不断言 `distinct` 等于某个具体值。
"""

import json
import pathlib

import pytest
import torch

from envs.idc_price_env import IDCPriceEnv20D
from scripts import probe_corrector_repro as probe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ARTIFACTS = ("config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json")
REQUIRED_DIGEST_FIELDS = (
    "observation", "next_observation", "raw_action", "exec_action",
    "old_raw_log_prob", "reward", "business_cost", "carbon_cost",
    "electricity_cost_sgd", "terminated", "truncated",
    "correction_reason", "planner_backend",
)


# --- 1. 结论逻辑（合成输入驱动，与机器无关） --------------------------------

def test_decide_marks_a_single_distinct_digest_as_reproducible():
    decision = probe.decide("on", ["aaa", "aaa", "aaa"])
    assert decision["distinct"] == 1
    assert decision["reproducible"] is True
    assert decision["blocked"] is False
    assert decision["failure_classification"] is None
    assert decision["conclusion"] == "reproducible"
    assert decision["digests"] == ["aaa", "aaa", "aaa"]


def test_decide_marks_multiple_distinct_digests_as_blocked():
    decision = probe.decide("on", ["aaa", "bbb", "aaa"])
    assert decision["distinct"] == 2
    assert decision["reproducible"] is False
    assert decision["blocked"] is True
    assert decision["failure_classification"] == "corrector_nonreproducible"
    assert decision["conclusion"] == "blocked"
    assert decision["digests"] == ["aaa", "bbb", "aaa"], "必须保留**全部** digest，不挑不删"


def test_decide_off_uses_its_own_failure_classification():
    decision = probe.decide("off", ["aaa", "bbb"])
    assert decision["blocked"] is True
    assert decision["failure_classification"] == "corrector_off_nonreproducible"


def test_decide_never_presets_a_conclusion():
    """同一输入必须给出同一结论；结论只能来自 digest 的多寡。"""
    for mode in ("off", "on"):
        assert probe.decide(mode, ["x"] * 5)["reproducible"] is True
        assert probe.decide(mode, ["x", "y"])["reproducible"] is False


def test_probe_source_contains_no_hardcoded_digest():
    """探针源码不得预设任何 digest 字面量（含 M5.4b 手测留下的那些）。"""
    source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
    for preset in (
        "8468db982fa6", "842b06d3bf97", "28258c1a5eba", "a0d22d1fdcf5",
        "d9bc5cf814ff", "aac1b5c02365", "b422ce8a226b", "b0a030b222c1",
        "7620deab5d68", "99a626f6ad58", "a193f03922f7",
    ):
        assert preset not in source, f"探针不得硬编码 digest：{preset}"
    # 也不得预设布尔结论
    assert '"reproducible": True' not in source
    assert "reproducible=True" not in source


# --- 2. digest 的构造：只排除墙钟字段 ---------------------------------------

def _buffer(mode: str = "on", budget: float = 0.05):
    env = IDCPriceEnv20D(task_seed=0, server_seed=0, forecast_seed=300000)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        policy = probe.make_policy(env)
    gen = torch.Generator()
    gen.manual_seed(0)
    buffer = probe.collect(env, policy, mode=mode, budget=budget, steps=2, generator=gen)
    return buffer


def test_digest_is_stable_under_wall_clock_changes():
    buffer = _buffer("on")
    before = probe.semantic_digest(buffer)
    for t in buffer.transitions:
        for key in probe.WALL_CLOCK_KEYS:
            t.correction_info[key] = 12345.678
    assert probe.semantic_digest(buffer) == before, "墙钟字段必须被排除"


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_action", lambda t: t.raw_action.__setitem__(0, t.raw_action[0] + 1.0)),
        ("exec_action", lambda t: t.exec_action.__setitem__(0, t.exec_action[0] + 1.0)),
        ("old_raw_log_prob", lambda t: setattr(t, "old_raw_log_prob", t.old_raw_log_prob + 1.0)),
        ("terminated", lambda t: setattr(t, "terminated", not t.terminated)),
        ("truncated", lambda t: setattr(t, "truncated", not t.truncated)),
        ("reward", lambda t: setattr(t, "reward", t.reward + 1.0)),
        ("business_cost", lambda t: setattr(t, "business_cost", t.business_cost + 1.0)),
        ("carbon_cost", lambda t: setattr(t, "carbon_cost", t.carbon_cost + 1.0)),
        ("correction_reason", lambda t: t.correction_info.__setitem__("correction_reason", "X")),
        ("planner_backend", lambda t: t.correction_info.__setitem__("planner_backend", "X")),
        ("stage_a_status", lambda t: t.correction_info.__setitem__("stage_a_status", "X")),
        ("projection_offset", lambda t: t.correction_info.__setitem__("projection_offset", 9.9)),
    ],
)
def test_digest_changes_when_a_semantic_field_changes(field, value):
    """每个要求纳入的语义字段都必须**真的**影响 digest（否则该字段等于没测）。"""
    buffer = _buffer("on")
    before = probe.semantic_digest(buffer)
    value(buffer.transitions[0])
    assert probe.semantic_digest(buffer) != before, f"{field} 未进入 digest"


def test_digest_observation_and_next_observation_matter():
    buffer = _buffer("off")
    before = probe.semantic_digest(buffer)
    buffer.transitions[0].observation[0] += 1.0
    assert probe.semantic_digest(buffer) != before
    before2 = probe.semantic_digest(buffer)
    buffer.transitions[0].next_observation[0] += 1.0
    assert probe.semantic_digest(buffer) != before2


def test_digest_covers_every_required_field():
    """固定「必须纳入」的字段清单，防止日后被悄悄删掉。"""
    for name in REQUIRED_DIGEST_FIELDS:
        assert name in probe.DIGEST_FIELDS or name in probe.DIGEST_INFO_FIELDS, name
    assert set(probe.WALL_CLOCK_KEYS) == {
        "correction_solve_time_s", "stage_a_solve_time_s", "stage_b_solve_time_s"
    }
    assert not set(probe.DIGEST_FIELDS) & set(probe.WALL_CLOCK_KEYS)
    assert not set(probe.DIGEST_INFO_FIELDS) & set(probe.WALL_CLOCK_KEYS)


def test_digest_is_deterministic_for_the_same_input():
    a, b = _buffer("on"), _buffer("on")
    assert probe.semantic_digest(a) == probe.semantic_digest(b)


# --- 3. solver 证据 ----------------------------------------------------------

def test_solver_evidence_records_backend_version_and_options():
    evidence = probe.solver_evidence(budget=0.05)
    assert evidence["planner_backend"] == "mip"
    assert "scipy" in evidence["solver"]
    assert evidence["scipy_version"]
    assert evidence["options"] == {"time_limit": 0.05}
    assert evidence["integrality"] == "binary on/off per step"


# --- 4. 真实跨进程测量（slow） ----------------------------------------------

@pytest.mark.slow
def test_probe_off_is_reproducible_and_writes_complete_evidence(tmp_path):
    """off 模式必须跨进程逐位一致，且产物齐全。"""
    exit_code = probe.main([
        "--modes", "off", "--runs", "3", "--base-dir", str(tmp_path), "--run-id", "off_probe",
    ])
    assert exit_code == 0, "corrector 关闭时必须判定为可复现"
    run_dir = tmp_path / "off_probe"
    for name in ARTIFACTS:
        assert (run_dir / name).exists(), f"缺少产物 {name}"

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    off = report["modes"]["off"]
    assert off["runs"] == 3
    assert len(off["digests"]) == 3, "必须保留全部 digest"
    assert off["distinct"] == 1
    assert off["reproducible"] is True and off["blocked"] is False
    assert report["overall"]["blocked"] is False
    assert report["trained"] is False


@pytest.mark.slow
def test_probe_records_the_on_conclusion_without_presetting_it(tmp_path):
    """on 模式：**不断言** distinct 等于几，只断言结论自洽且产物完整。

    - distinct == 1 → reproducible，且**没有** failure_classification
    - distinct > 1  → blocked，failure_classification=corrector_nonreproducible，且保留全部 digest
    """
    exit_code = probe.main([
        "--modes", "on", "--runs", "3", "--base-dir", str(tmp_path), "--run-id", "on_probe",
    ])
    run_dir = tmp_path / "on_probe"
    for name in ARTIFACTS:
        assert (run_dir / name).exists(), f"缺少产物 {name}"

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    on = report["modes"]["on"]
    assert on["runs"] == 3
    assert len(on["digests"]) == 3, "必须保留全部 digest"

    if on["distinct"] == 1:
        assert on["reproducible"] is True and on["blocked"] is False
        assert on["failure_classification"] is None
        assert exit_code == 0
        assert report["overall"]["blocked"] is False
    else:
        assert on["reproducible"] is False and on["blocked"] is True
        assert on["failure_classification"] == "corrector_nonreproducible"
        assert exit_code != 0, "blocked 结论必须反映在退出码上"
        assert report["overall"]["blocked"] is True

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == ("success" if not on["blocked"] else "failed")


@pytest.mark.slow
def test_probe_digests_are_produced_by_independent_processes(tmp_path):
    """每个 digest 都来自一个**独立进程**（不是同进程内重复调用）。"""
    probe.main(["--modes", "off", "--runs", "2", "--base-dir", str(tmp_path),
                "--run-id", "p", "--emit-provenance"])
    provenance = json.loads((tmp_path / "p" / "provenance.json").read_text(encoding="utf-8"))
    assert len(provenance) == 2
    pids = {entry["pid"] for entry in provenance}
    assert len(pids) == 2, f"必须来自不同进程，实际 pid={pids}"
    for entry in provenance:
        assert entry["python"] and entry["started_at"]


@pytest.mark.slow
def test_probe_evidence_records_solver_and_reproducibility_inputs(tmp_path):
    probe.main(["--modes", "off", "--runs", "2", "--base-dir", str(tmp_path), "--run-id", "cfg"])
    import yaml

    config = yaml.safe_load((tmp_path / "cfg" / "config.yaml").read_text(encoding="utf-8"))
    assert config["code_revision"]
    solver = config["solver"]
    assert solver["planner_backend"] == "mip" and solver["scipy_version"]
    assert config["env_seed_kwargs"] == {
        "task_seed": 0, "server_seed": 0, "forecast_seed": 300000
    }
    assert config["horizon"] and config["budget_s"] == probe.MODE_BUDGETS["on"]
    assert config["modes"] == ["off"]
    assert config["runs_per_mode"] == 2
