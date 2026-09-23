"""Read-only numerical review: image-plane E updates versus a mirrored periodic domain.
These local one-step diagnostics do not measure cavity frequencies or S parameters.
"""
import os
os.environ.update(JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1')
import json, time, resource, subprocess, hashlib
from pathlib import Path
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import rfx
import rfx.core.yee as y
import rfx.simulation as sim
import rfx.materials.debye as de
import rfx.materials.lorentz as lo
from rfx.grid import Grid
from rfx.nonuniform import _profile_to_inv_arrays
OUT=Path(__file__).parent
start=time.monotonic()
res={'head':subprocess.check_output(['git','-C','/root/workspace/bk-workspace/rfx','rev-parse','HEAD'],text=True).strip(), 'backend':jax.default_backend()}
N=8; dx=0.001; dt=1e-12
# E even, H odd about both integer-node walls, extended with period 2L.
e=np.array([.1,.5,.9,1.2,.7,.6,.4,.2,.3],np.float32)
h=np.array([.002,.003,.001,-.002,.004,-.003,.001,.005],np.float32)
ef=np.r_[e,e[-2:0:-1]]; hf=np.r_[h,-h[::-1]]
eh=e; hh=np.r_[h,np.float32(.037)] # poison stored high ghost on purpose
p_half=(False,True,True); p_full=(True,True,True)
def arr(v): return jnp.broadcast_to(jnp.asarray(v)[:,None,None],(len(v),3,3))
def state(ev,hv):
 s=y.init_state((len(ev),3,3)); return s._replace(ez=arr(ev),hy=arr(hv))
def mats(n,eps,sigma):
 m=y.init_materials((n,3,3)); return m._replace(eps_r=m.eps_r*eps,sigma=m.sigma+sigma)
sh=state(eh,hh); sf=state(ef,hf)
orig_diff=y._diff_bwd_o
# Prototype ONLY the shared uniform backward difference. Other families keep their actual code.
def odd_diff(a,axis,periodic,order,bloch=None):
 if axis!=0 or periodic[axis]: return orig_diff(a,axis,periodic,order,bloch)
 ac=a.at[-1].set(-a[-2])
 out=orig_diff(ac,axis,periodic,order,bloch)
 return out.at[0].set(2*a[0])
y._diff_bwd_o=odd_diff
checks=[]
for eps,sigma in [(1.,0.),(4.,0.),(4.,.2)]:
 mh=mats(N+1,eps,sigma); mf=mats(2*N,eps,sigma)
 # __wrapped__ avoids global cache ambiguity while using the actual source function.
 expected=y.update_e.__wrapped__(sf,mf,dt,dx,p_full).ez[:N+1]
 got=y.update_e.__wrapped__(sh,mh,dt,dx,p_half).ez
 checks.append({'eps_r':eps,'sigma_S_per_m':sigma,'uniform_max_error_V_per_m':float(jnp.max(jnp.abs(got-expected)))})
res['shared_uniform_image']=checks
# Same patch does NOT reach Debye/Lorentz/mixed/fused/NU paths.
family=[]
sh_material=sh._replace(hy=sh.hy.at[-1].set(-sh.hy[-2]))
for name in ['debye','lorentz','mixed']:
 mh=mats(N+1,4.,.2); mf=mats(2*N,4.,.2)
 def setup(m):
  d=de.init_debye([de.DebyePole(2.,1e-11)],m,dt) if name!='lorentz' else None
  l=lo.init_lorentz([lo.lorentz_pole(1.,2*np.pi*1e10,1e9)],m,dt) if name!='debye' else None
  return d,l
 dh,lh=setup(mh); df,lf=setup(mf)
 expected=sim._update_e_with_optional_dispersion(sf,mf,dt,dx,debye=df,lorentz=lf,periodic=p_full)[0].ez[:N+1]
 got=sim._update_e_with_optional_dispersion(sh_material,mh,dt,dx,debye=dh,lorentz=lh,periodic=p_half)[0].ez
 # Demonstrate the design is viable when every material derivative is migrated.
 def bwd_with_image(a,axis):
  prev=y._shift_bwd(a,axis)
  if axis==0:
   prev=prev.at[0].set(-a[0])
   prev=prev.at[-1].set(a[-1]+2*a[-2])
  return prev
 originals=(de._shift_bwd,lo._shift_bwd,sim._shift_bwd)
 de._shift_bwd=lo._shift_bwd=sim._shift_bwd=bwd_with_image
 repaired=sim._update_e_with_optional_dispersion(sh_material,mh,dt,dx,debye=dh,lorentz=lh,periodic=p_half)[0].ez
 de._shift_bwd,lo._shift_bwd,sim._shift_bwd=originals
 family.append({'family':name,'helper_only_error_lo_V_per_m':float(got[0,1,1]-expected[0,1,1]),'helper_only_error_hi_V_per_m':float(got[-1,1,1]-expected[-1,1,1]),'all_consumers_migrated_max_error_V_per_m':float(jnp.max(abs(repaired-expected)))})
res['unmigrated_material_consumers']=family
# Standalone inline odd derivative: static rule, dynamic field, JIT/vmap/grad.
@partial(jax.jit,static_argnames=('rules',))
def rule_diff(a,rules):
 prev=jnp.concatenate([jnp.zeros_like(a[:1]),a[:-1]])
 cur=a
 if rules[0]=='odd': prev=prev.at[0].set(-a[0])
 if rules[1]=='odd': cur=cur.at[-1].set(-a[-2])
 return cur-prev
jitted=rule_diff(jnp.asarray(hh),('odd','odd'))
res['jax_static_rules']={'diff':np.asarray(jitted).tolist(),'vmap_shape':list(jax.vmap(lambda a:rule_diff(a,('odd','odd')))(jnp.stack([jnp.asarray(hh),2*jnp.asarray(hh)])).shape),'gradient':np.asarray(jax.grad(lambda a:jnp.sum(rule_diff(a,('odd','odd'))**2))(jnp.asarray(hh))).tolist()}
# Graded axis: mirrored primal cells give end duals d0 and d_last, not half those.
d=np.array([1.,1.1,1.2,1.3,1.1,.9,.8,.7])*1e-3
stored=np.r_[d,d[-1]]
inv_e,_=_profile_to_inv_arrays(stored)
full_d=np.r_[d,d[::-1]]; full_dual=(full_d+np.roll(full_d,1))/2
oracle=(hf-np.roll(hf,1))/full_dual
actual=np.asarray(rule_diff(jnp.asarray(hh),('odd','odd')))*np.asarray(inv_e)
res['graded_image']={'max_curl_error_A_per_m2':float(np.max(abs(actual-oracle[:N+1]))),'lo_derivative_A_per_m2':float(actual[0]),'hi_derivative_A_per_m2':float(actual[-1]),'half_dual_would_double_endpoint_curl':True}
# (2,4): boundary ribbon must use image-corrected 2nd-order differences.
profile=np.arange(10,dtype=np.float32)/10
profile[-1]=37
jv=arr(profile)
corrected=odd_diff(jv,0,p_half,4)[:,1,1]
res['fourth_order_ribbon']={'width':2,'lo_diff':float(corrected[0]),'hi_diff':float(corrected[-1]),'expected_hi_diff':float(-2*profile[-2])}
y._diff_bwd_o=orig_diff
# Real grid periods and noncommensurate counterexample; no production resizing.
grids=[]
for L in [.024,.0243]:
 g=Grid(freq_max=10e9,domain=(L,.004,.004),dx=.001,cpml_layers=0)
 n_fixed=g.nx-1
 grids.append({'declared_L_mm':L*1000,'stored_nodes':g.nx,'old_roll_period_mm':g.nx*g.dx*1000,'remove_fencepost_nodes':n_fixed,'period_after_just_removing_fencepost_mm':n_fixed*g.dx*1000,'target_c_over_L_GHz':299792458/L/1e9,'c_over_proposed_quantized_period_GHz':299792458/(n_fixed*g.dx)/1e9,'current_index_at_L':g.position_to_index((L,0.,0.))[0],'would_be_out_of_range':g.position_to_index((L,0.,0.))[0]>=n_fixed})
res['period_and_index']=grids
# Actual JAX API's Bloch phase is currently static, so a traced angle is rejected.
try:
 def one(theta):
  phases=(jnp.exp(-1j*theta),1.+0j,1.+0j)
  s=y.init_state((4,3,3),field_dtype=jnp.complex64)
  return y.update_h(s,mats(4,1.,0.),dt,dx,(True,True,True),2,phases).hy
 jax.vmap(one)(jnp.array([0.,.2]))
 res['current_bloch_vmap']='accepted'
except Exception as ex:
 res['current_bloch_vmap']=type(ex).__name__+': '+str(ex)[:500]
# The RIS result fallback peak-normalizes any nonzero trace, even without S parameters.
from types import SimpleNamespace
from rfx.ris import _extract_reflection
ts=np.cos(2*np.pi*5e9*np.arange(128)*1e-12)
synthetic=SimpleNamespace(s_params=None,freqs=None,time_series=ts,dt=1e-12)
ris_s,ris_f=_extract_reflection(synthetic,(1e9,10e9),19)
synthetic.time_series=ts*0.01
ris_s_scaled,_=_extract_reflection(synthetic,(1e9,10e9),19)
res['ris_fallback_synthetic']={'native_s_params':None,'peak_reported_magnitude':float(np.max(abs(ris_s))),'100x_input_scaling_max_output_difference':float(np.max(abs(ris_s-ris_s_scaled))),'scope':'extractor diagnostic, not a scattering experiment'}
res['wall_seconds']=time.monotonic()-start
res['process_cpu_seconds']=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime
(OUT/'checks.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
