import os,sys,runpy,json,importlib,hashlib
from pathlib import Path
root=Path(sys.argv[1]).resolve(); output=Path(sys.argv[2]).resolve()
output.mkdir(exist_ok=False)
os.chdir(root);sys.path.insert(0,str(root))
os.environ['JAX_ENABLE_X64']='1'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
import numpy as np
import scipy,jax,meep,rfx
assert Path(rfx.__file__).resolve().is_relative_to(root)
module=importlib.import_module('rfx.harminv');original=module.harminv;records=[]
def record(signal,dt,fmin,fmax,**kwargs):
 path=output/f'input-{len(records):02d}.npz'
 np.savez_compressed(path,signal=np.asarray(signal))
 result=original(signal,dt,fmin,fmax,**kwargs)
 records.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dt=float(dt),f_min=float(fmin),f_max=float(fmax),kwargs=kwargs,modes=[m._asdict() for m in result]))
 return result
module.harminv=record
try:
 try:
  runpy.run_path(str(root/'validation/crossval/02_ring_resonator.py'),run_name='__main__')
  code=0
 except SystemExit as e:code=e.code
finally:
 module.harminv=original
 (output/'estimator-inputs.json').write_text(json.dumps(dict(numpy=np.__version__,scipy=scipy.__version__,jax=jax.__version__,meep=meep.__version__,records=records),indent=2)+'\n')
print('CASE_EXIT',code)
sys.exit(code)
