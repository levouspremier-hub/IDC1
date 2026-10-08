"""Fixed short -> same-source qualification -> strict release -> fresh formal."""
import ast
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/Users/levous/Desktop/IDC')
sys.path.insert(0, str(ROOT))
import pandas as pd
from runs.writer import write_run
from scenario.runtime_release import (build_runtime_release, verify_runtime_release,
                                     verify_candidate, validate_short_evidence)
from scripts import idc_remote as remote

OUT = Path(__file__).parent
PIN = '728398de2f1704094a7d1432a1a454d79e3f01d6'
SHORT = 'runtime-fixed-short-v25-v1'
PREFIX = 'runtime_v25_fixed_short'
QJOB = 'runtime-campaign-qualification-v25'
QID = 'runtime_campaign_qualification_v25'
FORMAL = 'runtime-formal-three-seed-v7-v25'
PARENT = 'm6p2c_formal_three_seed_v7_v25'
QUAL_ASSET = 'bd21c3a8d561ea52b0c7f66b9f9ccee3f43e60224397f7bbd8649542a43abbaa'
FORMAL_BASE = '789e0a877c1a82b43c50c2a7a79fb1de1a41f19e1f7ff04f6d17fc594bc665a6'
CANDIDATE = ROOT/'configs/release/idc_runtime_candidate_v25_050.json'
RELEASE = ROOT/'configs/release/idc_runtime_formal_release_v7.json'
DOC = ROOT/'docs/dev_reports/RUNTIME_V24_RELEASE_2026-10-09.md'
config = remote.config_load()
events = []
sha = remote.sha

def run(argv, **kw):
    return subprocess.run(argv, check=True, cwd=ROOT, **kw)

def host(code, payload=None, python='python3'):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        config['host'], shlex.join([python, '-c', code])],
        input=None if payload is None else json.dumps(payload).encode(), timeout=90).decode().strip()

def persist(phase, status='running', error=None, **data):
    events.append({'phase': phase, 'time': time.time(), **data})
    write_run(OUT.name, config={'pin': PIN, 'short_job': SHORT, 'qualification_job': QJOB,
        'formal_job': FORMAL, 'seeds': [0,1,2], 'serial': True},
        metrics=pd.DataFrame([{'phase': e['phase'], 'time': e['time']} for e in events]),
        report={'phase': phase, 'events': events, 'error': error,
                'formal_first_batch_verified': phase=='formal_first_batch_verified',
                'all_three_seeds_complete': False}, status=status,
        failure_classification=error, dependency_lock_hash=sha(ROOT/'uv.lock'),
        data_hash=config['asset_set'], scenario_hash=sha(CANDIDATE),
        command='uv run python runs/'+OUT.name+'/controller.py')
    print(json.dumps({'phase': phase, 'status': status, 'error': error, **data}), flush=True)

def storage_guard(input_bytes=0):
    source=(OUT/'host_formal.py').read_text()
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
            and n.name=='storage_guard')
    body='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])
    result=json.loads(host('import base64,json,subprocess,time\n'+body+
                          '\nprint(json.dumps(storage_guard()))'))
    needed=30*1024**3+input_bytes+8*1024**3
    assert next(v['free'] for v in result['volumes'] if v['drive']=='D')>=needed, result
    persist('storage_guard_passed', capacity=result, input_bytes=input_bytes,
            required_D_bytes=needed)
    return result

def ensure_task_relay():
    check="import socket,json; s=socket.socket(); s.settimeout(3); code=s.connect_ex(('127.0.0.1',19065)); s.close(); print(json.dumps({'listening':code==0}))"
    if not json.loads(host(check))['listening']:
        run(['ssh','-fNT','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes',
             '-o','ForwardAgent=no','-o','ServerAliveInterval=30',
             '-o','ServerAliveCountMax=3','-R','127.0.0.1:19065',config['host']])
        assert json.loads(host(check))['listening']
        persist('task_loopback_relay_restored')

def submit(job, revision, argv, asset):
    ensure_task_relay()
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
    config['asset_set']=asset
    remote.dump(remote.CONFIG,config)
    run([sys.executable,'-m','scripts.idc_remote','submit','--job',job,
         '--revision',revision,'--',*argv])
    persist('unique_job_submitted', job=job, revision=revision, asset_set=asset)

def wait_return(job, revision):
    # The original automatic watcher alone collects. Never compete with it.
    folder=ROOT/'runs'/('remote_'+job)
    while True:
        try:
            status=json.loads(remote.ssh(config,['status',job]))
        except (OSError,subprocess.SubprocessError,ValueError) as exc:
            print(json.dumps({'read_error':str(exc),'job':job}),flush=True)
            time.sleep(50)
            continue
        assert status['revision']==revision
        receipt=folder/'local_transfer_receipt.json'
        if status['state'] in remote.TERMINAL and receipt.exists():
            transfer=remote.read(receipt)
            if transfer['all_files_verified'] and transfer['remote_status']['state']==status['state']:
                remote.verify_receipt(folder,remote.read(folder/'receipt.json'))
                persist('terminal_full_return_verified',job=job,remote_status=status)
                assert status['state']=='succeeded' and status['exit_code']==0, status
                return folder
        time.sleep(50)

def canonical_copy(folder, prefix):
    copied=[]
    for src in sorted((folder/'runs').glob(prefix+'*')):
        if not src.is_dir(): continue
        dst=ROOT/'runs'/src.name
        assert not dst.exists(), 'never overwrite canonical evidence: '+str(dst)
        shutil.copytree(src,dst)
        for p in src.rglob('*'):
            if p.is_file(): assert sha(p)==sha(dst/p.relative_to(src))
        # Preserve a transferable marker for the mandatory empty figures directory.
        (dst/'figures').mkdir(exist_ok=True)
        (dst/'figures/.gitkeep').touch(exist_ok=True)
        copied.append(dst)
    assert copied
    return copied

def records_for(base_receipt, prefixes, drop_prefix=None):
    original=remote.read(base_receipt)
    records={n:r for n,r in original.items()
             if drop_prefix is None or not n.startswith(drop_prefix)}
    for prefix in prefixes:
        for folder in sorted((ROOT/'runs').glob(prefix+'*')):
            if not folder.is_dir():continue
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store':
                    assert not p.is_symlink()
                    records[str(p.relative_to(ROOT))]={'sha256':sha(p),'size':p.stat().st_size}
    remote.verify_receipt(ROOT,records)
    return records

LINK_CODE='''import json,sys,hashlib,os
from pathlib import Path
x=json.load(sys.stdin);base=Path(x['base']);dst=base/'assets'/x['digest'];old=base/'assets'/x['old'];prior=json.loads((old/'asset_receipt.json').read_text());missing=[];linked=0
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,r in x['records'].items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 src=old/p if prior.get(name)==r else None
 if src is None:
  for job in x['terminal_jobs']:
   candidate=base/'jobs'/job/'output'/p
   if candidate.is_file():src=candidate;break
 target=dst/p;assert not target.exists()
 if src is not None and src.is_file() and src.stat().st_size==r['size'] and sha(src)==r['sha256']:
  target.parent.mkdir(parents=True,exist_ok=True);os.link(src,target);linked+=1
 else:missing.append(name)
print(json.dumps({'linked_files':linked,'missing_files':missing}))
'''

def deliver(records, old, jobs, label):
    total=sum(r['size'] for r in records.values())
    storage_guard(total)
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    remote.dump(OUT/(label+'_asset_receipt.json'),records)
    remote.ssh(config,['asset-init',digest])
    result=json.loads(host(LINK_CODE,{'base':config['remote_root'],'digest':digest,
        'old':old,'records':records,'terminal_jobs':jobs}))
    if result['missing_files']:
        listing=OUT/(label+'_missing_asset_files.txt')
        listing.write_text('\n'.join(result['missing_files'])+'\n')
        remote.rsync(str(ROOT)+'/',config['host']+':'+config['remote_root']+
            '/assets/'+digest+'/', '--files-from='+str(listing))
    remote.verify_receipt(ROOT,records)
    remote.ssh(config,['asset-finish',digest],records)
    persist('minimal_asset_sealed',label=label,asset_set=digest,files=len(records),bytes=total,
            linked_files=result['linked_files'],transferred_files=len(result['missing_files']))
    storage_guard(total)
    return digest

def main():
    assert not RELEASE.exists(), 'never rewrite strict release'
    verify_candidate(CANDIDATE)
    assert json.loads(subprocess.check_output(['git','show',PIN+':configs/release/idc_runtime_candidate_v25_050.json'],cwd=ROOT,text=True))==verify_candidate(CANDIDATE)
    persist('fixed_source_ready')
    initial=remote.read(ROOT/'runs/m6p2c_v11_qualification_delivery_repair_v1/selected_qualification_asset_receipt.json')
    remote.verify_receipt(ROOT,initial)
    storage_guard(sum(r['size'] for r in initial.values()))
    submit(SHORT,PIN,['python','-c',(OUT/'host_short.py').read_text()],QUAL_ASSET)
    short=wait_return(SHORT,PIN)
    canonical_copy(short,PREFIX)
    for seed in (0,1,2):
        report=validate_short_evidence(ROOT/'runs'/f'{PREFIX}_seed{seed}',CANDIDATE,seed)
        assert report['batches']==8
    persist('three_fixed_shorts_passed',seeds=[0,1,2],batches_per_seed=8)
    qrecords=records_for(ROOT/'runs/m6p2c_v11_qualification_delivery_repair_v1/selected_qualification_asset_receipt.json',[PREFIX])
    qasset=deliver(qrecords,QUAL_ASSET,[SHORT],'qualification')
    submit(QJOB,PIN,['python','-m','scripts.runtime_qualification','qualify','--budget','.50',
        '--candidates',*[f'configs/release/idc_runtime_candidate_v25_{s}.json' for s in ('025','050','100')],
        '--run-id',QID,'--reuse-short-prefix',PREFIX],qasset)
    qfolder=wait_return(QJOB,PIN)
    canonical_copy(qfolder,QID)
    release=build_runtime_release(CANDIDATE,ROOT/'runs'/QID,
        campaign_authorization_path=ROOT/'configs/release/idc_runtime_campaign_authorization_v1.json')
    persist('same_source_full_qualification_passed',candidate_sha256=sha(CANDIDATE))
    assert subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)==''
    RELEASE.write_text(json.dumps(release,indent=2,sort_keys=True)+'\n')
    assert verify_runtime_release(RELEASE)==release
    DOC.write_text('# v25 主机资格与正式启动放行\n\n十一份历史失败与121边界、同源三seed短跑、全门禁/真实恢复/48诊断/三seed106批两周期覆盖/4h共享soak实际通过，全部产物回传SHA通过。统一冻结本版native B presolve=True并在实际求解后严格原矩阵验签；修复加入原约束蕴含的变量界/有效cut、守恒aggregate未来服务下界、必需整数mode的SOC能量cover及B连续坐标平移，必要时共享预算内一次固定整数净能量需求一致且仅nearest原功率容量不足时保留必要分数模式支持的整数候选原连续LP抛光及原目标证书，返回后认证原矩阵；物理/业务/精度/.50预算/PPO/奖励/采样保持。\n\n资格revision '+PIN+'；release SHA '+sha(RELEASE)+'。按用户授权fresh seed0→1→2串行各512批，真实final CP/更新/全部2048回合与文件SHA通过才下seed。不跨版本resume，历史失败与seed0成功保留；validation/test封存。本报告不表示三seed已完成或算法已收敛。\n')
    run(['git','diff','--check'])
    run(['git','add',str(RELEASE.relative_to(ROOT)),str(DOC.relative_to(ROOT))])
    run(['git','commit','-m','release: freeze qualified v25 fresh three-seed campaign'])
    run(['git','push','origin','p5-eval-viz-m6-p2c-root-numerics'])
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    persist('strict_release_frozen',revision=revision,release_sha256=sha(RELEASE))
    frecords=records_for(ROOT/'runs/m6p2c_delivery_capacity_repair_v1/selected_asset_receipt.json',
        [QID,PREFIX],drop_prefix='runs/runtime_campaign_qualification_v10_r4')
    for name, expected in {**verify_candidate(CANDIDATE)['assets'],**release['evidence']}.items():
        if name.startswith(('data/','runs/')):
            assert name in frecords and frecords[name]['sha256']==expected,name
    fasset=deliver(frecords,FORMAL_BASE,[QJOB,SHORT],'formal')
    assert verify_runtime_release(RELEASE)==release
    submit(FORMAL,revision,['python','-c',(OUT/'host_formal.py').read_text()],fasset)
    persist('formal_unique_job_submitted',job=FORMAL,revision=revision)
    code='''import json,torch
from pathlib import Path
from checkpointing import VersionedCheckpoint
from checkpointing.inventory_eval_input import FORMAL_SCHEMA
from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
p=Path('runs/m6p2c_formal_train_seed0_v7_v25');journal=p/'batches.jsonl'
rows=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
result={'completed_batches':len(rows),'failed_batch':(p/'failed_batch.json').exists()}
if rows:
 assert not result['failed_batch']
 assert all(b['zero_action_fallback_steps']==0 and b['adam_steps_this_batch']==16 and b['lagrangian_updates_this_batch']==1 and len(b['inventory_episodes'])==4 and all(all(inventory_episode_acceptance(e).values()) and e['target_qualified'] for e in b['inventory_episodes']) for b in rows)
 cp=VersionedCheckpoint.load(p/'checkpoint_latest.pt',expected_action_dim=21,expected_obs_dim=523,expected_schema_hash=FORMAL_SCHEMA);s=cp.state;n=s['next_batch_index']
 assert n>=1 and s['artifact_role']=='formal_training_resume' and s['training_scope']=='formal_training'
 assert {int(v['step']) for v in s['optimizer']['state'].values()}=={16*n} and s['lagrangian']['updates']==n
 assert all(torch.isfinite(v).all() for v in s['policy'].values())
 assert s['source_ledger']['master_seed']==0
 result.update(checkpoint_next_batch=n,all_adam_steps=16*n,lagrangian_updates=n,checkpoint_release_sha256=cp.extras['inventory_binding']['runtime_release_sha256'])
print(json.dumps(result))
'''
    workspace=Path(config['remote_root'])/'jobs'/FORMAL/'workspace'
    while True:
        try:
            status=json.loads(remote.ssh(config,['status',FORMAL]))
            assert status['state'] not in ('failed','interrupted'),status
            if status['state']=='running':
                live=json.loads(host('import os\nos.chdir('+repr(str(workspace))+')\n'+code,
                    python=str(workspace/'.venv/bin/python')))
                if live['completed_batches']:
                    assert live['checkpoint_release_sha256']==sha(RELEASE)
                    persist('formal_first_batch_verified',status='success',job=FORMAL,
                            first_batch=live,pending_serial_seeds=[1,2])
                    return
        except (OSError,subprocess.SubprocessError,ValueError) as exc:
            print(json.dumps({'formal_read_error':str(exc)}),flush=True)
        time.sleep(50)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        persist('stopped_on_failed_gate',status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
