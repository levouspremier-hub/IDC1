"""One recorded correction and explicitly separate numerical LP comparisons, no rollout."""
import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.optimize as opt

ROOT = Path('/Users/levous/Desktop/IDC')
sys.path.insert(0, str(ROOT))
from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct
from planning import numeric_contract as nc
from runs.writer import write_run
from scenario.runtime_release import sha

OUT = Path(__file__).parent
case = json.loads((ROOT/'tests/fixtures/m6p2c_seed1_origin1968_step18.json').read_text())
captured = []
real = nc.certify_or_polish_witness
def capture(*args, **kwargs):
    result = real(*args, **kwargs)
    captured.append((args, kwargs, result))
    return result
with warnings.catch_warnings(), patch.object(nc, 'certify_or_polish_witness', capture):
    warnings.simplefilter('ignore')
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
args, kw, rejected = captured[-1]
x, lb, ub, integer, matrix, row_lb, row_ub = args
np.savez_compressed(OUT/'original_model.npz', x=x, lb=lb, ub=ub, integer=integer,
                   matrix=matrix.toarray(), row_lb=row_lb, row_ub=row_ub,
                   objective=kw['objective'])
mask=np.asarray(integer)!=0
lower,upper=np.array(lb),np.array(ub)
assignment=np.rint(x)
for mode,c,d in zip(*kw['storage_modes'],strict=True):
    if kw['storage_requirements'][0] and x[c]>0 and x[d]<=0:assignment[mode]=1.
    elif kw['storage_requirements'][1] and x[d]>0 and x[c]<=0:assignment[mode]=0.
lower[mask]=upper[mask]=assignment[mask]
rows=[]
for name,anchor in [('original',np.zeros_like(x)), ('raw_continuous',np.asarray(x).copy()),
                    ('certified_a_continuous',np.asarray(captured[-2][2][0]).copy())]:
    anchor[mask]=0.
    shifted=matrix@anchor
    rl,ru=row_lb-shifted,row_ub-shifted
    eq=np.isfinite(rl)&(rl==ru);hi=np.isfinite(ru)&~eq;lo=np.isfinite(rl)&~eq
    from scipy.sparse import vstack
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=opt.linprog(kw['objective'],A_eq=matrix[eq],b_eq=rl[eq],
            A_ub=vstack([matrix[hi],-matrix[lo]],format='csr'),b_ub=np.r_[ru[hi],-rl[lo]],
            bounds=np.column_stack([lower-anchor,upper-anchor]),method='highs-ds',
            options={'time_limit':.50,'primal_feasibility_tolerance':1e-10,
                     'dual_feasibility_tolerance':1e-10,'random_seed':0,'parallel':False})
    row={'comparison':name,'status':int(r.status),'message':r.message}
    if r.x is not None:
        witness,audit=nc.certify_witness(r.x+anchor,lb,ub,integer,matrix,row_lb,row_ub)
        row.update(certificate=audit,objective=float(kw['objective']@(r.x+anchor)))
        np.save(OUT/(name+'_witness.npy'),r.x+anchor)
    rows.append(row)
print(json.dumps(rows,indent=2))
ghosts=[{'index':int(i),'raw':float(x[i]),'nearest':float(assignment[i])}
        for i in np.flatnonzero(mask) if abs(x[i]-np.rint(x[i]))>1e-15]
report={'comparisons':rows,'ghost_integers':ghosts,'recorded_execution_witness':rejected[1],
        'correct_executable':result.executable,'environment_steps':0,'parameter_updates':0,
        'comparison_scope':'Independent offline LP diagnostics, never a production retry',
        'storage_modes':[int(i) for i in kw['storage_modes'][0]],'source_failure_sha256':case['source_failure_sha256']}
write_run(OUT.name,config={'fixture':'tests/fixtures/m6p2c_seed1_origin1968_step18.json','budget':.50},
    metrics=pd.DataFrame([{'comparison':r['comparison'],'status':r['status']} for r in rows]),
    report=report,command='uv run python runs/'+OUT.name+'/probe.py',
    dependency_lock_hash=sha(ROOT/'uv.lock'),scenario_hash=sha(ROOT/'tests/fixtures/m6p2c_seed1_origin1968_step18.json'))
(OUT/'figures/.gitkeep').touch()
