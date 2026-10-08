"""Historical terminal checkpoint and every native file verification, no rollout."""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT=Path('/Users/levous/Desktop/IDC')
sys.path.insert(0,str(ROOT))
from checkpointing.inventory_eval_input import SHORT_SCHEMA
from checkpointing.versioned import read_checkpoint_payload
from runs.writer import write_run
from safe_rl_v2.formal_train_loop import build_lagrangian,build_optimizer,build_seeded_policy,load_resume_checkpoint
from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
from safe_rl_v2.inventory_train import coverage_batches
from scenario.inventory_release import load_matrix,sha
from scripts.idc_remote import verify_receipt

OUT=Path(__file__).parent
base=ROOT/'runs/remote_runtime-campaign-qualification-v23'
receipt=json.loads((base/'receipt.json').read_text());verify_receipt(base,receipt)
transfer=json.loads((base/'local_transfer_receipt.json').read_text())
assert transfer['all_files_verified'] and transfer['remote_status']['state']=='failed'
candidate_path=ROOT/'configs/release/idc_runtime_candidate_v23_050.json'
assert sha(candidate_path)=='4deb7b4237f3986c8f1abb11596cb67b7940bc17a6f26a1ac3bde16c879dc2b8'
candidate=json.loads(candidate_path.read_text());config=candidate['config']
binding={'runtime_candidate_sha256':sha(candidate_path),'execution_version':candidate['execution_version'],
         'sources':candidate['sources'],'assets':candidate['assets'],'config':config,'formal_training_ready':False}
leaf=base/'runs/runtime_campaign_qualification_v23_050_coverage_seed0'
journal=[json.loads(x) for x in (leaf/'batches.jsonl').read_text().splitlines()]
assert len(journal)==68
episodes=[e for b in journal for e in b['inventory_episodes']]
assert len(episodes)==272 and all(all(inventory_episode_acceptance(e).values())
                                and e['target_qualified'] is True for e in episodes)
assert all(b['zero_action_fallback_steps']==0 for b in journal)
checks=[]
for name in ['checkpoint_before_batch.pt','checkpoint_latest.pt']:
    p=leaf/name;payload=read_checkpoint_payload(p)
    assert payload['metadata']['inventory_binding']==binding
    policy=build_seeded_policy(config,obs_dim=523,seed=0)
    optimizer=build_optimizer(config,policy);lag=build_lagrangian(config)
    sampling,shuffle=torch.Generator(),torch.Generator()
    restored=load_resume_checkpoint(p,policy=policy,optimizer=optimizer,lagrangian=lag,
        sampling_generator=sampling,shuffle_generator=shuffle,config=config,expected_obs_dim=523,
        expected_schema=SHORT_SCHEMA,expected_scope='runtime_learning_qualification',
        expected_role='runtime_learning_qualification_resume')
    assert restored['next_batch_index']==68 and lag._updates==68
    assert len(optimizer.state)==sum(len(g['params']) for g in optimizer.param_groups)
    assert {int(s['step']) for s in optimizer.state.values()}=={1088}
    assert restored['origins']==[o for b in coverage_batches(load_matrix(),0) for o in b]
    assert restored['origins'][:272]==[o for b in journal for o in b['origins']]
    assert all(torch.isfinite(p).all() for p in policy.state_dict().values())
    assert all(torch.isfinite(s[k]).all() for s in optimizer.state.values() for k in ['exp_avg','exp_avg_sq'])
    assert torch.equal(sampling.get_state(),payload['state']['sampling_generator'])
    assert torch.equal(shuffle.get_state(),payload['state']['shuffle_generator'])
    checks.append({'file':name,'sha256':sha(p),'next_batch_index':68,'all_adam_steps':1088,
                   'lagrangian_updates':68,'actual_rng_equal':True,'role_source_finite_origins_verified':True})
failed=json.loads((leaf/'failed_batch.json').read_text())
assert failed['parameter_updates_this_batch']==0 and failed['failure_step']['environment_step_executed'] is False
report={'passed':True,'scope':'historical failed coverage terminal audit; not qualification release',
        'native_files':len(receipt),'all_native_sha_size_verified':True,'checkpoints':checks,
        'qualified_episodes':272,'fallback_steps':0,
        'retained_stage_a_steps':sum(b['stage_a_retained_steps'] for b in journal),
        'failure_origin':failed['origin'],'failure_step':failed['failure_step']['snapshot']['step'],
        'partial_transitions':len(failed['partial_transitions']),'failed_environment_step_executed':False,
        'failed_batch_parameter_updates':0,'failed_sha256':sha(leaf/'failed_batch.json'),
        'environment_steps':0,'parameter_updates':0,'formal_started':False}
write_run(OUT.name,config={'candidate':str(candidate_path),'native_receipt':str(base/'receipt.json')},
    metrics=pd.DataFrame([{'seed':0,'batches':68,'transitions':13056,'adam_steps':1088,'lag_updates':68}]),
    report=report,dependency_lock_hash=candidate['lock_sha256'],data_hash=sha(base/'receipt.json'),
    scenario_hash=sha(candidate_path),command='uv run python runs/'+OUT.name+'/audit.py')
(OUT/'figures/.gitkeep').touch();print(json.dumps(report))
