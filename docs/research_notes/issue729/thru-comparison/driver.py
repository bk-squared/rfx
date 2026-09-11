import argparse,os,sys,json,time,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--expected-sha',required=True);args=p.parse_args()
out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'receipt.json').exists(): raise SystemExit('refusing existing output')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
sys.path.insert(0,str(Path.cwd()))
import numpy as np,jax,scipy
from rfx import Simulation
import tests.unit.sparams.test_msl_port_integration as case
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert sha == args.expected_sha, (sha,args.expected_sha)
assert not subprocess.check_output(['git','status','--porcelain'],text=True), 'dirty source checkout'
receipt=dict(source_commit=sha,source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),numpy=np.__version__,scipy=scipy.__version__,jax=jax.__version__,devices=str(jax.devices()),affinity=sorted(os.sched_getaffinity(0)),field_precision='default float32',test='test_msl_thru_line_passive_gate',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
original=Simulation.compute_msl_s_matrix
captured=[]
def capture(self,*a,**kw):
 kw['raw_3probe_dump_path']=str(out/'raw-vi.npz')
 result=original(self,*a,**kw)
 arrays={}
 for key in ['S','S_raw','freqs','Z0','beta','settling_db','reliable','passivity_correction']:
  value=getattr(result,key,None)
  if value is not None: arrays[key]=np.asarray(value)
 np.savez_compressed(out/'result.npz',**arrays)
 mask=(arrays['freqs']>=case.GATE_F_LO)&(arrays['freqs']<=case.GATE_F_HI)
 S=arrays.get('S_raw',arrays['S'])
 measured=dict(mean_s11=float(np.abs(S[0,0,mask]).mean()),mean_s21=float(np.abs(S[1,0,mask]).mean()),mean_z0=float(np.real(arrays['Z0'][0,mask]).mean()),max_column_power=float(np.max(np.sum(np.abs(S[:,:,mask])**2,axis=0))),settling_db=arrays.get('settling_db',np.array([])).tolist(),max_projection_correction=float(np.max(arrays.get('passivity_correction',np.array([0.])))))
 (out/'measured.json').write_text(json.dumps(measured,indent=2)+'\n');print('RAW_MEASUREMENT',measured,flush=True)
 captured.append(measured)
 return result
Simulation.compute_msl_s_matrix=capture
start=time.monotonic()
try:
 case.test_msl_thru_line_passive_gate();outcome={'test_return':'PASS'}
except BaseException as e:
 outcome={'test_return':'FAIL','exception':repr(e)}
 raise
finally:
 outcome['wall_s']=time.monotonic()-start
 (out/'outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
