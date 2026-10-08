"""Reopen all immutable v23 short artifacts and actual resume/RNG state."""
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
from scenario.runtime_release import candidate_binding,sha,validate_numeric_evidence,validate_short_evidence,verify_candidate
from scripts.idc_remote import verify_receipt

OUT=Path(__file__).parent
base=ROOT/'runs/remote_runtime-fixed-short-v23-v1'
receipt=json.loads((base/'receipt.json').read_text())
transfer=json.loads((base/'local_transfer_receipt.json').read_text())
assert transfer['all_files_verified'] is True
assert transfer['remote_status']['state']=='succeeded' and transfer['remote_status']['exit_code']==0
assert transfer['remote_status']['revision']=='5fd721071cb6256ac00ab30058548463491c9b7e'
verify_receipt(base,receipt)
candidate=ROOT/'configs/release/idc_runtime_candidate_v23_050.json'
frozen=verify_candidate(candidate);binding=candidate_binding(candidate)
prefix='runtime_v23_fixed_short'
for raw in sorted((base/'runs').glob(prefix+'*')):
    canonical=ROOT/'runs'/raw.name
    original={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}
    copied={str(p.relative_to(canonical)) for p in canonical.rglob('*') if p.is_file()}
    assert copied-original <= {'figures/.gitkeep'} and original <= copied
    assert all(sha(raw/p)==sha(canonical/p) for p in original)
numeric=json.loads((base/'runs'/f'{prefix}_numerics/report.json').read_text())
validate_numeric_evidence(numeric,candidate)
seeds=[]
for seed in (0,1,2):
    leaf=base/'runs'/f'{prefix}_seed{seed}'
    report=validate_short_evidence(leaf,candidate,seed)
    checkpoint=leaf/'checkpoint_final.pt';payload=read_checkpoint_payload(checkpoint)
    config=frozen['config'];policy=build_seeded_policy(config,obs_dim=523,seed=seed)
    optimizer=build_optimizer(config,policy);lag=build_lagrangian(config)
    sampling,shuffle=torch.Generator(),torch.Generator()
    state=load_resume_checkpoint(checkpoint,policy=policy,optimizer=optimizer,lagrangian=lag,
        sampling_generator=sampling,shuffle_generator=shuffle,config=config,expected_obs_dim=523,
        expected_schema=SHORT_SCHEMA,expected_scope='controlled_short_run',expected_role='controlled_training_resume')
    assert torch.equal(sampling.get_state(),payload['state']['sampling_generator'])
    assert torch.equal(shuffle.get_state(),payload['state']['shuffle_generator'])
    assert payload['metadata']['inventory_binding']==binding
    assert state['next_batch_index']==8 and lag._updates==8
    assert {int(s['step']) for s in optimizer.state.values()}=={128}
    seeds.append({'seed':seed,'fresh':report['resumed_from'] is None,'batches':8,
        'transitions':1536,'all_adam_steps':128,'lagrangian_updates':8,'qualified_episodes':32,
        'fallback_steps':sum(b['zero_action_fallback_steps'] for b in report['batch_records']),
        'retained_stage_a_steps':sum(b['stage_a_retained_steps'] for b in report['batch_records']),
        'checkpoint_sha256':sha(checkpoint),'report_sha256':sha(leaf/'report.json'),
        'actual_checkpoint_role_source_origins_finite_verified':True,'actual_rng_equal':True})
report={'passed':True,'native_files':len(receipt),'all_native_sha_size_verified':True,
    'canonical_matches_native':True,'host_numeric_observations':len(numeric['observations']),
    'candidate_sha256':sha(candidate),'source_revision':frozen['source_revision'],
    'seeds':seeds,'parameter_updates':0,'environment_steps':0,
    'qualification_job':'runtime-campaign-qualification-v23','full_qualification_passed':False,
    'formal_started':False,'all_three_formal_seeds_complete':False}
write_run(OUT.name,config={'short_job':'runtime-fixed-short-v23-v1','candidate':str(candidate),
    'fixed_pin':transfer['remote_status']['revision']},metrics=pd.DataFrame(seeds),report=report,
    dependency_lock_hash=frozen['lock_sha256'],data_hash=sha(base/'receipt.json'),scenario_hash=sha(candidate),
    command='uv run python runs/'+OUT.name+'/audit.py')
(OUT/'figures/.gitkeep').touch()
print(json.dumps(report,ensure_ascii=False))
