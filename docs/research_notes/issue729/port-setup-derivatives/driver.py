import json,sys,warnings
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
import rfx.simulation as core
from tests.unit.autodiff.test_msl_ad_fd_converged import _build_msl_f64_referee
out=Path('/tmp/rfx729-port-setup-derivatives');out.mkdir(exist_ok=True)
original=core.run
captured=[];metadata=[]
class Captured(Exception):pass
def intercept(grid,materials,n_steps,**kwargs):
 sources=kwargs['sources']
 captured.append((materials.eps_r,materials.sigma,jnp.stack([x.waveform for x in sources])))
 metadata.append({'shape':list(grid.shape),'field_dtype':str(kwargs['field_dtype']),'sources':[(int(x.i),int(x.j),int(x.k),x.component) for x in sources]})
 raise Captured
core.run=intercept
def observe(alpha):
 captured.clear()
 sim=_build_msl_f64_referee();grid=sim._build_grid()
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   sim.compute_msl_s_matrix(n_steps=4,n_freqs=2,checkpoint_segments=1,
    eps_override=jnp.ones(grid.shape,dtype=jnp.float64)*alpha)
 except Captured:
  assert len(captured)==1
  return captured[0]
 raise RuntimeError('actual source setup boundary was not reached')
try:
 with jax.experimental.enable_x64():
  primal,tangent=jax.jvp(observe,(jnp.float64(1.),),(jnp.float64(1.),))
  h=1e-4;plus=observe(jnp.float64(1+h));minus=observe(jnp.float64(1-h))
  report={};arrays={}
  for name,p,t,a,b in zip(['eps','sigma','source_waveforms'],primal,tangent,plus,minus):
   p,t,a,b=map(np.asarray,(p,t,a,b));fd=(a-b)/(2*h)
   scale=max(float(np.max(np.abs(fd))),float(np.max(np.abs(t))),1e-300)
   report[name]={'primal_min':float(p.min()),'primal_max':float(p.max()),'jvp_max':float(np.max(np.abs(t))),'fd_max':float(np.max(np.abs(fd))),'relative_max_difference':float(np.max(np.abs(t-fd))/scale)}
   arrays.update({name+'_primal':p,name+'_jvp':t,name+'_fd':fd})
  assert metadata[0]==metadata[1]==metadata[2]
  (out/'comparison.json').write_text(json.dumps({'source_setup':metadata[0],'derivatives':report},indent=2)+'\n')
  np.savez_compressed(out/'arrays.npz',**arrays)
  print(json.dumps(report,indent=2))
finally:core.run=original
