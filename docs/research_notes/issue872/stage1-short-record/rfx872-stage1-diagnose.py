import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/rfx-codex')
import scripts.stage1_nu_cavity_physics_gate as stage
from rfx.harminv import harminv, _decimation_plan
spec=importlib.util.spec_from_file_location('oldharminv','/tmp/rfx872-harminv-baseline.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
out=Path('/tmp/rfx872-stage1'); out.mkdir(exist_ok=True)
report={}
def capture(y,dt,fmin,fmax,**kwargs):
 np.savez_compressed(out/'input.npz',signal=y,dt=dt,fmin=fmin,fmax=fmax)
 report.update(n=len(y),dt=dt,fmin=fmin,fmax=fmax,plan=_decimation_plan(len(y),dt,fmax,'auto'),rawcycles=(len(y)-1)*dt*(fmin/.7))
 for key,fun,kw in [('baseline_auto',old.harminv,{}),('candidate_auto',harminv,{}),('baseline_raw',old.harminv,{'decimate':False}),('candidate_raw',harminv,{'decimate':False})]:
  modes=fun(y,dt,fmin,fmax,**kw)
  report[key]=[m._asdict() for m in modes]
  print(key,report[key],flush=True)
 return harminv(y,dt,fmin,fmax)
stage.harminv=capture
try: report['gate']=stage.run_gate().__dict__
except Exception as e: report['exception']=repr(e)
(out/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
