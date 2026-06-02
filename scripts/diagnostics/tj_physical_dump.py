"""Physical root-cause dump for the rfx T-junction mesh-divergence.

(A) STRUCTURAL ISOLATION: does a rfx STRAIGHT guide (no junction corners)
    converge with mesh (passivity->1, |S11|->0, |S21|->1) while the T-JUNCTION
    diverges? If yes -> the junction CORNERS (Yee-staircase) are the cause.
(B) FIELD DUMP: transverse Ez profile at each port plane (device, drive 0) vs the
    analytic TE10 sin profile -> higher-order content fraction, coarse vs fine.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux,
    modal_voltage, _plane_field)
from rfx.simulation import run as run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--dx_mm", type=float, required=True); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc=10; BAND=np.linspace(5.0e9,7.0e9,11)
grid = Grid(freq_max=10e9, domain=(0.12,0.12,0.02), dx=dx, cpml_layers=nc, cpml_axes="xy")
def pec(boxes):
    m=init_materials(grid.shape)
    for b in boxes: m=m._replace(sigma=jnp.where(b.mask(grid),1e10,m.sigma))
    return m
freqs=jnp.asarray(BAND); n_steps=grid.num_timesteps(num_periods=30)
px,py,pz=grid.axis_pads
ys=(py+int(round(0.04/dx)), py+int(round(0.08/dx))+1); zs=(pz,grid.nz-pz)
po=max(8,int(round(0.030/dx))); ro=max(3,int(round(0.006/dx)))

# (A) STRAIGHT horizontal 2-port guide (no junction)
mat_h=pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.12,0.12,0.02))])
L=WaveguidePort(x_index=nc+5,y_slice=ys,z_slice=zs,a=0.04,b=0.02,mode=(1,0),mode_type="TE",direction="+x",normal_axis="x",u_slice=ys,v_slice=zs)
R=WaveguidePort(x_index=grid.nx-nc-6,y_slice=ys,z_slice=zs,a=0.04,b=0.02,mode=(1,0),mode_type="TE",direction="-x",normal_axis="x",u_slice=ys,v_slice=zs)
cf=[init_waveguide_port(p,dx,freqs,f0=6e9,ref_offset=ro,probe_offset=po,dft_total_steps=n_steps) for p in (L,R)]
Ss=np.abs(np.asarray(extract_waveguide_s_matrix_flux(grid,mat_h,init_materials(grid.shape),cf,n_steps,
    boundary="cpml",cpml_axes="xy",pec_axes="z",ref_materials_per_port=[mat_h,mat_h])))
cps=np.sum(Ss**2,axis=0)
print(f"[STRAIGHT dx={A.dx_mm}] passivity max={cps.max():.3f} |S11|bm={np.mean(Ss[0,0]):.3f} "
      f"|S21|bm={np.mean(Ss[1,0]):.3f} (matched thru: ->0/->1, passivity->1 if converging)")

# (B) T-junction device, drive port 0, dump transverse Ez profile at port planes
dev=pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.04,0.12,0.02)), Box((0.08,0.08,0),(0.12,0.12,0.02))])
xs=(px+int(round(0.04/dx)), px+int(round(0.08/dx))+1)
top=WaveguidePort(x_index=grid.ny-nc-6,y_slice=None,z_slice=None,a=0.04,b=0.02,mode=(1,0),mode_type="TE",direction="-y",normal_axis="y",u_slice=xs,v_slice=zs)
cfgs=[init_waveguide_port(p,dx,freqs,f0=6e9,ref_offset=ro,probe_offset=po,dft_total_steps=n_steps) for p in (L,R,top)]
cfgs=[c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]
res=run_sim(grid,dev,n_steps,boundary="cpml",cpml_axes="xy",pec_axes="z",waveguide_ports=cfgs,return_state=True)
st=res.state
for i,c in enumerate(cfgs):
    # transverse Ez profile at this port's reference plane (final-state snapshot)
    prof=np.asarray(_plane_field(getattr(st,c.e_v_component), c, c.ref_x)).ravel()
    if prof.size<3 or np.linalg.norm(prof)<1e-30:
        print(f"  port{i}: field ~0 (decayed snapshot)"); continue
    mode=np.asarray(c.ez_profile).ravel()[:prof.size]
    proj=np.vdot(mode,prof)/ (np.vdot(mode,mode)+1e-30)
    resid=prof-proj*mode
    ho=float(np.linalg.norm(resid)/(np.linalg.norm(prof)+1e-30))
    print(f"  port{i} ({c.direction}): higher-order(non-TE10) fraction = {ho:.3f}")
