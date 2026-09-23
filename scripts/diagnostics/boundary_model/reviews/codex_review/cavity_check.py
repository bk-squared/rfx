"""Magnetic-sided 24 x 20 x 4 mm cavity, driven by an interior Ez pulse.
CPU-only read-only prototypes compare the declared-face image and deliberate defects.
"""
import time,resource,json
from pathlib import Path
import numpy as np
import jax
import rfx.core.yee as y
import rfx.simulation as low
import rfx.boundaries.pmc as pmc
from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.sources.sources import GaussianPulse
from rfx.harminv import harminv
out=Path(__file__).parent
orig_diff=y._diff_bwd_o; orig_pmc=pmc.apply_pmc_faces; orig_pec=low.apply_pec_faces
records=[]
for label,dx in [('current',.001),('image',.001),('image',.0005),('drop_image_keep_pec_yz',.001),('restore_pec_x_keep_image',.001)]:
 start=time.monotonic()
 y._diff_bwd_o=orig_diff; pmc.apply_pmc_faces=orig_pmc; low.apply_pec_faces=orig_pec
 if label!='current':
  pmc.apply_pmc_faces=lambda st,faces:st
 if label in ('image','restore_pec_x_keep_image'):
  def image_diff(a,axis,periodic,order,bloch=None):
   if axis!=0 or periodic[axis]:return orig_diff(a,axis,periodic,order,bloch)
   ac=a.at[-1].set(-a[-2])
   return orig_diff(ac,axis,periodic,order,bloch).at[0].set(2*a[0])
  y._diff_bwd_o=image_diff
 if label=='restore_pec_x_keep_image':
  low.apply_pec_faces=lambda st,faces:orig_pec(st,set(faces)|{'x_lo','x_hi'})
 jax.clear_caches()
 s=Simulation(freq_max=20e9,domain=(.024,.020,.004),dx=dx,cpml_layers=0,boundary=BoundarySpec(x='pmc',y='pec',z='pec'))
 s.add_source((.007,.006,.002),'ez',waveform=GaussianPulse(f0=8.5e9,bandwidth=.8),amplitude_kind='field')
 s.add_probe((.017,.013,.002),'ez')
 nsteps=round(4096*.001/dx)
 print('CASE',label,dx,flush=True)
 result=s.run(n_steps=nsteps,compute_s_params=False)
 ts=np.asarray(result.time_series).reshape(-1); dt=float(result.grid.dt)
 ring=ts[nsteps//4:]-np.mean(ts[nsteps//4:])
 modes=harminv(ring,dt,5e9,12e9,min_Q=1.,max_modes=24,sv_threshold=1e-3,decimate='auto')
 fs=[float(m.freq) for m in modes]
 f01=[f for f in fs if 6.8e9<=f<=8.2e9]
 f11=[f for f in fs if 9e9<=f<=10.7e9]
 rec={'label':label,'dx_mm':dx*1000,'shape':list(result.grid.shape),'n_steps':nsteps,'dt_s':dt,'frequencies_GHz':[f/1e9 for f in fs],'f01_window_GHz':[f/1e9 for f in f01],'f11_window_GHz':[f/1e9 for f in f11],'wall_s':time.monotonic()-start}
 if f11:rec['derived_separation_mm']=1000/np.sqrt((2*min(f11,key=lambda f:abs(f-9.756058116e9))/299792458)**2-(1/.020)**2)
 records.append(rec)
 np.savez(out/f'cavity_{label}_{dx*1000:g}mm.npz',time_series=ts,dt=dt)
 print(json.dumps(rec),flush=True)
 (out/'cavity.json').write_text(json.dumps({'records':records,'process_cpu_s':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime},indent=2)+'\n')
y._diff_bwd_o=orig_diff;pmc.apply_pmc_faces=orig_pmc;low.apply_pec_faces=orig_pec
