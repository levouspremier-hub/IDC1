"""Operational launcher only: each seed must succeed before the next starts."""
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from runs.writer import write_run
from safe_rl_v2.inventory_train import verify_written_run
from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
from scenario.runtime_release import verify_runtime_release, runtime_checkpoint_binding
from checkpointing import VersionedCheckpoint
from checkpointing.inventory_eval_input import FORMAL_SCHEMA
from scripts.idc_remote import make_receipt, verify_receipt
import torch

ROOT = Path.cwd()
RELEASE = ROOT / 'configs/release/idc_runtime_formal_release_v7.json'
PARENT = 'm6p2c_formal_three_seed_v7_v22'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def storage_guard():
    ps = "$v=Get-Volume -DriveLetter C,D; @($v | ForEach-Object { @{drive=$_.DriveLetter;free=[long]$_.SizeRemaining} }) | ConvertTo-Json -Compress"
    encoded = base64.b64encode(ps.encode('utf-16-le')).decode()
    result = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-EncodedCommand', encoded], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    volumes = json.loads(result.stdout.strip().lstrip('\ufeff'))
    assert {v['drive'] for v in volumes} == {'C','D'}
    assert all(v['free'] >= 30*1024**3 for v in volumes), volumes
    opts = subprocess.check_output(['findmnt','-n','-o','OPTIONS','/'], text=True).strip().split(',')
    assert 'rw' in opts and 'ro' not in opts, opts
    return {'volumes':volumes,'root_mount_options':opts,'time':time.time()}


def main():
    assert not (ROOT/'runs'/PARENT).exists(), 'parent already exists; never repeat'
    release = verify_runtime_release(RELEASE)
    binding = runtime_checkpoint_binding(RELEASE)
    assert release['allowed_seeds']==[0,1,2] and release['fresh_initialization_required']
    states=[]
    def persist(status,error=None):
        write_run(PARENT,config={'seeds':[0,1,2],'serial':True,'release_sha256':sha(RELEASE),'fresh_initialization':True},metrics=pd.DataFrame([{'seed':r['seed'],'completed':r['state']=='success'} for r in states]),report={'seeds':states,'status':status,'error':error,'release_sha256':sha(RELEASE),'pending_seeds':[s for s in (0,1,2) if s not in [r['seed'] for r in states]],'parameter_updates_performed_by_launcher':0},status=status,failure_classification=error,dependency_lock_hash=sha(ROOT/'uv.lock'),data_hash=hashlib.sha256(json.dumps(binding['assets'],sort_keys=True).encode()).hexdigest(),scenario_hash=sha(RELEASE),command='serial fresh formal runtime release v7 seeds 0,1,2')
    try:
        for seed in (0,1,2):
            rid=f'm6p2c_formal_train_seed{seed}_v7_v22'
            assert not (ROOT/'runs'/rid).exists(), 'no overwrite or implicit resume'
            record={'seed':seed,'run_id':rid,'state':'running','started_at':time.time(),'storage_before':storage_guard()}
            states.append(record);persist('running')
            argv=[sys.executable,'-m','safe_rl_v2.inventory_train','--seed',str(seed),'--runtime-release',str(RELEASE),'--run-id',rid]
            print(json.dumps({'phase':'formal_seed_started','seed':seed,'run_id':rid,'argv':argv}),flush=True)
            with (ROOT/'runs'/PARENT/f'seed{seed}.log').open('ab') as log:
                code=subprocess.call(argv,stdout=log,stderr=log)
            record.update(exit_code=code,finished_at=time.time())
            if code:raise RuntimeError(f'seed {seed} exited {code}; next seed blocked')
            report=verify_written_run(ROOT/'runs'/rid,binding)
            assert report['seed']==seed and report['scope']=='formal_training'
            assert report['batches']==512 and report['transitions']==98304
            assert report['adam_steps']==8192 and report['lagrangian_updates']==512
            assert report['episodes']==2048 and len(report['inventory_episodes'])==2048
            assert report['resumed_from'] is None and report['batches_newly_run']==512
            assert all(all(inventory_episode_acceptance(e).values()) and e['target_qualified'] is True for e in report['inventory_episodes'])
            assert all(b['zero_action_fallback_steps']==0 for b in report['batch_records'])
            leaf = ROOT/'runs'/rid
            cp = VersionedCheckpoint.load(leaf/'checkpoint_final.pt', expected_action_dim=21,
                expected_obs_dim=523, expected_schema_hash=FORMAL_SCHEMA)
            state = cp.state
            assert cp.extras == {'inventory_binding': binding}
            assert state['artifact_role']=='formal_training_resume' and state['training_scope']=='formal_training'
            assert state['next_batch_index']==512 and state['lagrangian']['updates']==512
            assert {int(v['step']) for v in state['optimizer']['state'].values()}=={8192}
            assert all(torch.isfinite(t).all() for t in state['policy'].values())
            assert all(torch.isfinite(v[k]).all() for v in state['optimizer']['state'].values() for k in ('exp_avg','exp_avg_sq'))
            assert state['source_ledger']['master_seed']==seed and not (leaf/'failed_batch.json').exists()
            receipt = make_receipt(leaf); verify_receipt(leaf, receipt)
            receipt_path=ROOT/'runs'/PARENT/f'seed{seed}_artifact_receipt.json'
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            record.update(state='success',report_sha256=sha(leaf/'report.json'),checkpoint_sha256=sha(leaf/'checkpoint_final.pt'),artifact_receipt_sha256=sha(receipt_path),all_seed_files_verified=len(receipt))
            persist('running')
        persist('success')
    except BaseException as exc:
        if states and states[-1]['state']=='running':states[-1]['state']='failed'
        persist('failed',f'{type(exc).__name__}: {exc}')
        raise

main()
