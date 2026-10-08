"""Exact-rational consequence of original remaining-work and projection rows."""
import json,sys
from pathlib import Path
from fractions import Fraction as F
import numpy as np,pandas as pd,scipy.optimize as so
from scipy.sparse import csr_matrix,vstack
ROOT=Path('/Users/levous/Desktop/IDC');sys.path.insert(0,str(ROOT))
from planning.numeric_contract import certify_witness
from scenario.runtime_release import sha
from runs.writer import write_run
OUT=Path(__file__).parent;base=ROOT/'runs/m6p2c_v25_economic_model_probe_v1'
m=np.load(base/'original_model.npz');d=json.loads((base/'report.json').read_text());offs=d['offsets'];case=json.loads((ROOT/'tests/fixtures/m6p2c_seed2_origin8496_step6.json').read_text());s=case['snapshot'];p=case['proposal'];matrix=csr_matrix((m['data'],m['indices'],m['indptr']),shape=m['shape']);rl=m['rl'];ru=m['ru'];budget=F(float(ru[-1]));work=sum((F(t['remaining_work']) for t in s['tasks']),F(0));raw=[F(u) for u in p['compute_actions']];maximum=F(0)
for inv,u in sorted([(F(float(1./cap)),u) for cap,u in zip(s['group_work_capacity'],raw)],reverse=True):
 take=min(u,work*inv);maximum+=take;work-=take/inv
floor=(sum(raw)-maximum)*F(float(1./len(raw)));charge=(budget-floor-F(p['storage_action']))/F(float(1./s['bess_charge_power_max_kw']));upper=float(np.nextafter(float(charge),np.inf));z=offs['off_z'];c=offs['off_charge'];row=np.zeros(matrix.shape[1]);coefficient=min(float(np.nextafter(1./upper,0.)),1e8);row[c]=coefficient;row[z]=-1
strong=vstack([matrix,csr_matrix(row[None,:])],format='csr');bounds=so.Bounds(m['lb'],m['ub'].copy());bounds.ub[c]=min(bounds.ub[c],upper)
r=so.milp(c=m['c'],integrality=m['integrality'],bounds=bounds,constraints=[so.LinearConstraint(strong,np.r_[rl,-np.inf],np.r_[ru,0.])],options={'time_limit':.50,'presolve':True,'mip_rel_gap':0.,'mip_feasibility_tolerance':1e-10,'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'random_seed':0,'parallel':False})
cert=None
if r.x is not None:_,cert=certify_witness(r.x,m['lb'],m['ub'],m['integrality'],matrix,rl,ru)
report={'scope':'original-matrix diagnostic; not training','minimum_compute_offset':float(floor),'primary_bound':float(budget),'derived_charge_upper':upper,'normalized_cut_coefficient':coefficient,'charge_upper_exact_fraction':str(charge),'status':int(r.status),'message':r.message,'actual_charge':None if r.x is None else float(r.x[c]),'actual_mode':None if r.x is None else float(r.x[z]),'reported_economic':None if r.x is None else float(r.fun),'actual_economic':None if r.x is None else float(m['c']@r.x),'original_matrix_certificate':cert,'parameter_updates':0,'environment_steps':0}
write_run(OUT.name,config={'original_model_sha256':sha(base/'original_model.npz'),'precision':1e-10,'budget_s':.50,'source_unchanged':True},metrics=pd.DataFrame([{'status':r.status,'charge_upper':upper}]),report=report,dependency_lock_hash=sha(ROOT/'uv.lock'),data_hash=sha(ROOT/'tests/fixtures/m6p2c_seed2_origin8496_step6.json'),scenario_hash=sha(ROOT/'configs/release/idc_runtime_candidate_v25_050.json'),command='uv run python runs/'+OUT.name+'/probe.py');(OUT/'figures/.gitkeep').touch();print(json.dumps(report))
