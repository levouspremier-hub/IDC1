"""Read-only terminal evidence for the historical frozen v21 candidate."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

ROOT=Path('/Users/levous/Desktop/IDC')
sys.path.insert(0,str(ROOT))
import scenario.runtime_release as rr
from checkpointing.inventory_eval_input import SHORT_SCHEMA
from checkpointing.versioned import read_checkpoint_payload
from runs.writer import write_run
from safe_rl_v2.formal_train_loop import build_lagrangian,build_optimizer,build_seeded_policy,load_resume_checkpoint
from safe_rl_v2.inventory_train import coverage_batches

OUT=Path(__file__).parent
base=ROOT/'runs/remote_runtime-campaign-qualification-v21-r2'
receipt=json.loads((base/'receipt.json').read_text())
assert json.loads((base/'local_transfer_receipt.json').read_text())['all_files_verified']
for name,value in receipt.items():
    p=base/name
    assert p.stat().st_size==value['size'] and rr.sha(p)==value['sha256']
candidate_path=ROOT/'configs/release/idc_runtime_candidate_v21_050.json'
candidate=json.loads(candidate_path.read_text())
binding={'runtime_candidate_sha256':rr.sha(candidate_path),'execution_version':candidate['execution_version'],
         'sources':candidate['sources'],'assets':candidate['assets'],'config':candidate['config'],
         'formal_training_ready':False}
# Historical audit of the immutable receipt, NOT live qualification or release.
# The archived source binding is checked directly, never replaced by current HEAD.
seed0=base/'runs/runtime_campaign_qualification_v21_r2_050_coverage_seed0'
assert json.loads((seed0/'report.json').read_text())['inventory_binding']==binding
with patch.object(rr,'verify_candidate',lambda path:candidate),patch.object(rr,'candidate_binding',lambda path:binding):
    successful=rr.validate_learning_coverage(seed0,candidate_path)
seed1=base/'runs/runtime_campaign_qualification_v21_r2_050_coverage_seed1'
journal=[json.loads(x) for x in (seed1/'batches.jsonl').read_text().splitlines()]
assert len(journal)==10
checkpoints=[]
config=candidate['config']
for name in ['checkpoint_before_batch.pt','checkpoint_latest.pt']:
    path=seed1/name
    payload=read_checkpoint_payload(path)
    assert payload['metadata']['inventory_binding']==binding
    policy=build_seeded_policy(config,obs_dim=523,seed=1)
    optimizer=build_optimizer(config,policy);lag=build_lagrangian(config)
    restored=load_resume_checkpoint(path,policy=policy,optimizer=optimizer,lagrangian=lag,
        sampling_generator=torch.Generator(),shuffle_generator=torch.Generator(),config=config,
        expected_obs_dim=523,expected_schema=SHORT_SCHEMA,expected_scope='runtime_learning_qualification',
        expected_role='runtime_learning_qualification_resume')
    steps=[int(s['step']) for s in optimizer.state.values()]
    assert restored['next_batch_index']==10 and set(steps)=={160} and lag._updates==10
    assert restored['origins']==[o for b in coverage_batches(rr.load_matrix(),1) for o in b]
    assert restored['origins'][:40]==[o for b in journal for o in b['origins']]
    assert all(torch.isfinite(p).all() for p in policy.state_dict().values())
    assert all(torch.isfinite(s[k]).all() for s in optimizer.state.values() for k in ['exp_avg','exp_avg_sq'])
    checkpoints.append({'file':name,'sha256':rr.sha(path),'next_batch_index':10,'all_adam_steps':160,
                        'lagrangian_updates':10,'finite':True,'rng_role_source_origins_restored':True})
failed=json.loads((seed1/'failed_batch.json').read_text())
assert failed['parameter_updates_this_batch']==0 and failed['failure_step']['environment_step_executed'] is False
report={'passed':True,'scope':'historical terminal audit; not current release qualification',
    'native_files':len(receipt),'all_sha_size_verified':True,'candidate_sha256':rr.sha(candidate_path),
    'seed0':{'batches':successful['batches'],'transitions':successful['transitions'],
             'adam_steps':successful['adam_steps'],'lagrangian_updates':successful['lagrangian_updates'],
             'qualified_episodes':len(successful['inventory_episodes']),'actual_restore_verified':True},
    'seed1_checkpoints':checkpoints,'failed_sha256':rr.sha(seed1/'failed_batch.json'),
    'failure':{k:failed[k] for k in ['origin','parameter_updates_this_batch','adam_steps_at_failure','multiplier_updates_at_failure']},
    'partial_transitions':len(failed['partial_transitions']),'failed_environment_step_executed':False,
    'environment_steps':0,'parameter_updates':0,'formal_started':False}
write_run(OUT.name,config={'candidate':str(candidate_path),'receipt':str(base/'receipt.json')},
    metrics=pd.DataFrame([{'seed':0,'batches':106},{'seed':1,'batches':10}]),report=report,
    dependency_lock_hash=candidate['lock_sha256'],data_hash=rr.sha(base/'receipt.json'),
    scenario_hash=rr.sha(candidate_path),command='uv run python runs/'+OUT.name+'/audit.py')
(OUT/'figures/.gitkeep').touch()
print(json.dumps(report,ensure_ascii=False))
