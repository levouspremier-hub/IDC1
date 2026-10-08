"""Immutable original A model and separate exact-mode LP comparisons, no rollout."""
import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.sparse import vstack

ROOT=Path('/Users/levous/Desktop/IDC')
sys.path.insert(0,str(ROOT))
from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning import numeric_contract as nc
from planning.corrector import correct
from runs.writer import write_run
from scenario.inventory_release import sha

OUT=Path(__file__).parent
fixture=ROOT/'tests/fixtures/m6p2c_seed1_origin6336_step2.json'
case=json.loads(fixture.read_text());calls=[];real=nc.certify_or_polish_witness
def capture(*args,**kw):
    result=real(*args,**kw);calls.append((args,kw,result));return result
with warnings.catch_warnings(),patch.object(nc,'certify_or_polish_witness',capture):
    warnings.simplefilter('ignore')
    result=correct(InventorySnapshot.model_validate(case['snapshot']),
                   DispatchProposal.model_validate(case['proposal']),time_limit_s=.50)
args,kw,polished=calls[-1]
x,lb,ub,integer,matrix,rl,ru=args
np.savez_compressed(OUT/'original_model.npz',x=x,lb=lb,ub=ub,integer=integer,
                   matrix=matrix.toarray(),row_lb=rl,row_ub=ru,objective=kw['objective'])
mask=np.asarray(integer)!=0;nearest=np.rint(x);selected=nearest.copy()
for change in polished[1]['primal_polish']['storage_mode_changes']:
    selected[change['variable']]=change['selected_mode']
rows=[]
for name,assignment in [('nearest',nearest),('flow',selected)]:
    candidate=np.array(x,copy=True);candidate[mask]=assignment[mask]
    before=nc.certify_witness(candidate,lb,ub,integer,matrix,rl,ru)[1]
    lower,upper=np.array(lb,copy=True),np.array(ub,copy=True)
    lower[mask]=upper[mask]=assignment[mask]
    eq=np.isfinite(rl)&(rl==ru);hi=np.isfinite(ru)&~eq;lo=np.isfinite(rl)&~eq
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=opt.linprog(kw['objective'],A_eq=matrix[eq],b_eq=rl[eq],
            A_ub=vstack([matrix[hi],-matrix[lo]],format='csr'),b_ub=np.r_[ru[hi],-rl[lo]],
            bounds=np.c_[lower,upper],method='highs-ds',options={'time_limit':.50,
                'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,
                'random_seed':0,'parallel':False})
    row={'comparison':name,'candidate_original_rows':before,'LP_status':int(r.status)}
    if r.x is not None:
        row.update(certificate=nc.certify_witness(r.x,lb,ub,integer,matrix,rl,ru)[1],
                   objective=float(kw['objective']@r.x))
        np.save(OUT/(name+'_witness.npy'),r.x)
    rows.append(row)
report={'source_failure_sha256':case['source_failure_sha256'],'environment_steps':0,
    'parameter_updates':0,'correct_executable':result.executable,
    'primary_certificate':result.inventory_audit.get('primary_objective_certificate'),
    'comparisons':rows,'original_model_sha256':sha(OUT/'original_model.npz'),
    'scope':'independent diagnostics; no production retry or constraint relaxation'}
write_run(OUT.name,config={'fixture':str(fixture),'budget':.50},report=report,
    metrics=pd.DataFrame([{'comparison':r['comparison'],'status':r['LP_status']} for r in rows]),
    dependency_lock_hash=sha(ROOT/'uv.lock'),scenario_hash=sha(fixture),
    command='uv run python runs/'+OUT.name+'/probe.py')
(OUT/'figures/.gitkeep').touch()
print(json.dumps(report,ensure_ascii=False))
