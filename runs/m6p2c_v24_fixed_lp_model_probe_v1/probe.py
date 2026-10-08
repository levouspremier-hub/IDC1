"""Preserve unchanged mathematical inputs; diagnose original fixed B LP only."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import scipy.optimize as so
from scipy.sparse import vstack,csr_matrix
ROOT=Path('/Users/levous/Desktop/IDC');sys.path.insert(0,str(ROOT))
from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct
from planning.numeric_contract import certify_witness
from runs.writer import write_run
from scenario.runtime_release import sha
OUT=Path(__file__).parent
case=json.loads((ROOT/'tests/fixtures/m6p2c_seed1_origin2544_step23.json').read_text());models=[];lps=[]
real_mip=so.milp;real_lp=so.linprog

def mip(**kw):
 r=real_mip(**kw);models.append((kw,r));return r

def lp(**kw):
 r=real_lp(**kw);lps.append((kw,r));return r
with patch.object(so,'milp',mip),patch.object(so,'linprog',lp):
 result=correct(InventorySnapshot.model_validate(case['snapshot']),DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
assert len(models)==2 and len(lps)==1
kw,r=models[-1];m=kw['constraints'][0].A.tocsr();origin=__import__('planning.numeric_contract',fromlist=['translate_continuous_origin'])
# Capture B in its actual translated coordinates, and the original-coordinate LP.
np.savez_compressed(OUT/'actualB.npz',data=m.data,indices=m.indices,indptr=m.indptr,shape=m.shape,lower=kw['bounds'].lb,upper=kw['bounds'].ub,row_lower=kw['constraints'][0].lb,row_upper=kw['constraints'][0].ub,c=kw['c'],integrality=kw['integrality'],x=r.x)
lkw,lr=lps[0]
np.savez_compressed(OUT/'original_fixed_lp.npz',c=lkw['c'],bounds=lkw['bounds'],eq_data=lkw['A_eq'].data,eq_indices=lkw['A_eq'].indices,eq_indptr=lkw['A_eq'].indptr,eq_shape=lkw['A_eq'].shape,b_eq=lkw['b_eq'],ub_data=lkw['A_ub'].data,ub_indices=lkw['A_ub'].indices,ub_indptr=lkw['A_ub'].indptr,ub_shape=lkw['A_ub'].shape,b_ub=lkw['b_ub'])
rows=[]
for name,presolve,shift in [('original_presolve',True,False),('original_no_presolve',False,False),('translated_presolve',True,True),('translated_no_presolve',False,True)]:
 args=dict(lkw);args['options']={**lkw['options'],'presolve':presolve,'time_limit':.50}
 if shift:
  # A single deterministic feasible A continuous origin from the same call.
  anchor=np.array(models[0][1].x);anchor[np.asarray(kw['integrality'])!=0]=0.
  args['bounds']=lkw['bounds']-anchor[:,None]
  args['b_eq']=lkw['b_eq']-lkw['A_eq']@anchor
  args['b_ub']=lkw['b_ub']-lkw['A_ub']@anchor
 rr=real_lp(**args)
 record={'name':name,'status':int(rr.status),'message':rr.message,'objective':None if rr.x is None else float(lkw['c']@(rr.x+anchor if shift else rr.x))}
 if rr.x is not None:
  x=rr.x+anchor if shift else rr.x;record['row_residual']=float(max(np.max(abs(lkw['A_eq']@x-lkw['b_eq']),initial=0),np.max(lkw['A_ub']@x-lkw['b_ub'],initial=0)));record['bound_residual']=float(max(np.max(lkw['bounds'][:,0]-x,initial=0),np.max(x-lkw['bounds'][:,1],initial=0)))
 rows.append(record)
report={'scope':'mathematical diagnostic only, not rollout/qualification','executable':result.executable,'failure':result.failure,'stage_a':result.inventory_audit['stage_a_witness'],'primary':result.inventory_audit['primary_objective_certificate'],'execution':result.inventory_audit['execution_witness'],'fixed_lp_diagnostics':rows,'files':{p.name:sha(p) for p in OUT.glob('*.npz')},'environment_steps':0,'parameter_updates':0}
write_run(OUT.name,config={'fixture':'tests/fixtures/m6p2c_seed1_origin2544_step23.json','budget_s':.50,'production_unchanged':True},metrics=pd.DataFrame(rows),report=report,dependency_lock_hash=sha(ROOT/'uv.lock'),data_hash=sha(ROOT/'tests/fixtures/m6p2c_seed1_origin2544_step23.json'),scenario_hash=sha(ROOT/'configs/release/idc_runtime_candidate_v24_050.json'),command='uv run python runs/'+OUT.name+'/probe.py');(OUT/'figures/.gitkeep').touch()
print(json.dumps({'primary':report['primary'],'diagnostics':rows}))
