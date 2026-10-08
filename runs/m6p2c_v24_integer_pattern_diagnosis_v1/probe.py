"""Original model integer-pattern diagnosis; never used for training."""
import json,sys,inspect
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd,scipy.optimize as so
from scipy.sparse import vstack
ROOT=Path('/Users/levous/Desktop/IDC');sys.path.insert(0,str(ROOT))
from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct
from planning.numeric_contract import certify_witness
from scenario.runtime_release import sha
from runs.writer import write_run
OUT=Path(__file__).parent;calls=[];real=so.milp

def mip(**kw):
 f=inspect.currentframe().f_back;loc=f.f_locals
 r=real(**kw)
 calls.append({'kwargs':kw,'x':None if r.x is None else r.x.copy(),'offsets':{k:int(v) for k,v in loc.items() if k.startswith('off_') and isinstance(v,int)},'names':list(loc.get('names',[])),'shift':None if loc.get('b_shift') is None else loc['b_shift'].copy(),'lb':loc['lb'].copy(),'ub':loc['ub'].copy(),'rl':np.array(loc.get('lbs_b',loc['lbs'])),'ru':np.array(loc.get('ubs_b',loc['ubs']))})
 return r
case=json.loads((ROOT/'tests/fixtures/m6p2c_seed1_origin2544_step23.json').read_text())
with patch.object(so,'milp',mip):result=correct(InventorySnapshot.model_validate(case['snapshot']),DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
a,b=calls;kw=b['kwargs'];m=kw['constraints'][0].A.tocsr();x=b['x']+b['shift'];mask=kw['integrality']!=0;z=b['offsets']['off_z'];h=case['snapshot']['planning_horizon_steps'];c=b['offsets']['off_charge'];d=b['offsets']['off_discharge']
patterns={}
nearest=np.rint(x);flow=nearest.copy();flow[z:z+h]=np.where((x[c:c+h]>0)&(x[d:d+h]<=0),1,flow[z:z+h]);patterns['nearest']=nearest;patterns['flow']=flow;patterns['exactA']=np.rint(a['x']);patterns['AwithCurrentCharge']=patterns['exactA'].copy();patterns['AwithCurrentCharge'][z]=1
rows=[]
rl,ru=b['rl'],b['ru'];eq=np.isfinite(rl)&(rl==ru);up=np.isfinite(ru)&~eq;low=np.isfinite(rl)&~eq;ineq=vstack([m[up],-m[low]],format='csr');rhs=np.r_[ru[up],-rl[low]]
for name,pattern in patterns.items():
 bounds=np.column_stack([b['lb'],b['ub']]);bounds[mask,0]=bounds[mask,1]=pattern[mask]
 r=so.linprog(kw['c'],A_eq=m[eq],b_eq=rl[eq],A_ub=ineq,b_ub=rhs,bounds=bounds,method='highs-ds',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':.50})
 row={'pattern':name,'status':int(r.status),'objective':None if r.x is None else float(kw['c']@r.x)}
 if r.x is not None:_,row['certificate']=certify_witness(r.x,b['lb'],b['ub'],kw['integrality'],m,rl,ru)
 rows.append(row)
diffs=[{'variable':int(i),'A':float(a['x'][i]),'B':float(x[i]),'nearest':float(nearest[i])} for i in np.flatnonzero(mask) if a['x'][i]!=nearest[i]]
np.savez_compressed(OUT/'original_model.npz',data=m.data,indices=m.indices,indptr=m.indptr,shape=m.shape,lb=b['lb'],ub=b['ub'],rl=rl,ru=ru,c=kw['c'],integrality=kw['integrality'],xA=a['x'],xB=x,flow=flow)
report={'scope':'diagnostics only','offsets':b['offsets'],'row_names':b['names']+['projection_offset_bound'],'integer_differences':diffs,'storage':[{'k':k,'zA':float(a['x'][z+k]),'zB':float(x[z+k]),'cB':float(x[c+k]),'dB':float(x[d+k])} for k in range(h)],'patterns':rows,'model_sha256':sha(OUT/'original_model.npz'),'parameter_updates':0,'environment_steps':0}
write_run(OUT.name,config={'fixture':'tests/fixtures/m6p2c_seed1_origin2544_step23.json','original_model_preserved':True},metrics=pd.DataFrame([{'pattern':r['pattern'],'status':r['status']} for r in rows]),report=report,dependency_lock_hash=sha(ROOT/'uv.lock'),data_hash=sha(ROOT/'tests/fixtures/m6p2c_seed1_origin2544_step23.json'),scenario_hash=sha(ROOT/'configs/release/idc_runtime_candidate_v24_050.json'),command='uv run python runs/'+OUT.name+'/probe.py');(OUT/'figures/.gitkeep').touch();print(json.dumps({'diffs':diffs,'storage':report['storage'],'patterns':rows}))
