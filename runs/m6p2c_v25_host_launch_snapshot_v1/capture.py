"""Read real host state; do not collect or alter a scientific job."""
import json,subprocess,shlex,sys
from pathlib import Path
import pandas as pd
ROOT=Path('/Users/levous/Desktop/IDC');sys.path.insert(0,str(ROOT))
from runs.writer import write_run
from scenario.runtime_release import sha,verify_candidate
from scripts import idc_remote as r
OUT=Path(__file__).parent;c=r.config_load()
code='''import json
from pathlib import Path
j=Path('/home/w1877/idc-host/jobs/runtime-fixed-short-v25-v1');s=json.loads((j/'output/remote_status.json').read_text());pid=json.loads((j/'pid.json').read_text())['pid'];numeric=json.loads((j/'workspace/runs/runtime_v25_fixed_short_numerics/report.json').read_text());leaf=j/'workspace/runs/runtime_v25_fixed_short_seed0'
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:argv=(p/'cmdline').read_bytes().decode().split('\\0')
 except OSError:continue
 if 'safe_rl_v2.inventory_train' in argv and '--run-id' in argv and argv[argv.index('--run-id')+1]=='runtime_v25_fixed_short_seed0':active.append(int(p.name))
print(json.dumps({'remote_status':{k:s.get(k) for k in ['job','revision','state','exit_code','error']},'host_pid':pid,'worker_alive':(Path('/proc')/str(pid)).exists(),'host_numeric_passed':numeric.get('passed'),'host_numeric_observations':len(numeric.get('observations',[])),'actual_seed0_processes':active,'seed0_journal_batches':len((leaf/'batches.jsonl').read_text().splitlines()) if (leaf/'batches.jsonl').exists() else 0,'seed0_failed_batch':(leaf/'failed_batch.json').exists(),'qualification_job_exists':(j.parent/'runtime-campaign-qualification-v25').exists(),'formal_job_exists':(j.parent/'runtime-formal-three-seed-v7-v25').exists()}))
'''
live=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',c['host'],shlex.join(['python3','-c',code])],text=True,timeout=40))
assert live['remote_status']['state']=='running' and live['worker_alive']
assert live['host_numeric_passed'] is True and live['host_numeric_observations']==121
assert live['actual_seed0_processes'] and not live['seed0_failed_batch']
assert not live['qualification_job_exists'] and not live['formal_job_exists']
process=json.loads((ROOT/'runs/m6p2c_v25_direct_launch_v1/controller_process.json').read_text());live.update(mac_processes=process,short_passed=False,full_qualification_passed=False,formal_started=False,formal_release_exists=(ROOT/'configs/release/idc_runtime_formal_release_v7.json').exists(),parameter_updates_performed_by_audit=0,environment_steps_performed_by_audit=0)
candidate=ROOT/'configs/release/idc_runtime_candidate_v25_050.json';frozen=verify_candidate(candidate)
write_run(OUT.name,config={'job':'runtime-fixed-short-v25-v1','fixed_pin':process['revision'],'read_only':True},metrics=pd.DataFrame([{'host_numeric_passed':121,'seed0_journal_batches':live['seed0_journal_batches']}]),report=live,dependency_lock_hash=frozen['lock_sha256'],data_hash=sha(candidate),scenario_hash=sha(candidate),command='uv run python runs/'+OUT.name+'/capture.py');(OUT/'figures/.gitkeep').touch();print(json.dumps(live))
