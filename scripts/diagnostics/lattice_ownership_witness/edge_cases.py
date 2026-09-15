import warnings, numpy as np
warnings.simplefilter("ignore")
from rfx import Simulation, Box
from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes

C0=299792458.0; DX=2e-3; AX,AY,AZ = 40e-3,20e-3,30e-3
def f_te101(a,d): return 0.5*C0*np.sqrt((1/a)**2+(1/d)**2)

def run(build, src, probe, n=6000):
    sim = Simulation(freq_max=12e9, domain=(AX,AY,AZ), dx=DX, boundary="pec")
    build(sim); sim.add_source(src,"ey",amplitude_kind="field"); sim.add_probe(probe,"ey")
    r = sim.run(n_steps=n, skip_preflight=True, compute_s_params=False)
    ts=np.asarray(r.time_series)[:,0].astype(np.float64)
    dt=float(sim._build_grid().dt); N=1<<16
    sp=np.abs(np.fft.rfft(ts*np.hanning(ts.size),n=N)); fr=np.fft.rfftfreq(N,dt)
    b=(fr>3e9)&(fr<14e9); k=np.argmax(sp[b]); return float(fr[b][k])

print("=== one-cell-thick body: does it close BOTH sub-cavities? ===")
# 1-cell PEC slab at z 4->6mm. Upper cavity [6,30]=24mm, lower [0,4]=4mm.
b = lambda s: s.add(Box((0,0,4e-3),(AX,AY,6e-3)), material="pec")
f_up = run(b, (9e-3,10e-3,16e-3), (13e-3,10e-3,20e-3))
print(f"  1-cell slab, probe ABOVE : {f_up/1e9:.4f} GHz  analytic(d=24mm) {f_te101(AX,24e-3)/1e9:.4f}  err {100*(f_up-f_te101(AX,24e-3))/f_te101(AX,24e-3):+.3f}%")
# lower cavity 4mm -> TE101 f = (c/2)sqrt(1/0.04^2+1/0.004^2) = very high, out of band.
# instead check the lower region is ISOLATED: excite below, probe above -> no coupling
sim = Simulation(freq_max=12e9, domain=(AX,AY,AZ), dx=DX, boundary="pec")
b(sim); sim.add_source((9e-3,10e-3,2e-3),"ey",amplitude_kind="field")
sim.add_probe((13e-3,10e-3,20e-3),"ey")
r=sim.run(n_steps=3000, skip_preflight=True, compute_s_params=False)
leak=float(np.max(np.abs(np.asarray(r.time_series)[:,0])))
sim2 = Simulation(freq_max=12e9, domain=(AX,AY,AZ), dx=DX, boundary="pec")
sim2.add_source((9e-3,10e-3,2e-3),"ey",amplitude_kind="field"); sim2.add_probe((13e-3,10e-3,20e-3),"ey")
r2=sim2.run(n_steps=3000, skip_preflight=True, compute_s_params=False)
ref=float(np.max(np.abs(np.asarray(r2.time_series)[:,0])))
print(f"  leak through the 1-cell slab: {leak:.3e}  (no-slab control {ref:.3e})  ratio {leak/ref:.2e}")

print()
print("=== periodic seam ===")
for per, drawn in [("x", "wrapping cells 0 and n-1")]:
    pass
def seam_case(periodic):
    from rfx.boundaries.pec import realized_pec_edge_masks
    import rfx.boundaries.pec as P
    import jax.numpy as jnp
    shape=(8,6,6); C=np.zeros(shape,dtype=bool)
    C[7,2:4,2:4]=True          # a body occupying ONLY the last cell along x
    e=P.realized_pec_edge_masks(jnp.asarray(C), periodic=periodic)
    ey=np.asarray(e[1])
    return sorted(set(np.flatnonzero(ey.any(axis=(1,2))).tolist()))
print("  body in cell x=7 only, non-periodic x -> Ey wall planes:", seam_case((False,False,False)))
print("  body in cell x=7 only, PERIODIC x     -> Ey wall planes:", seam_case((True,False,False)))
print("  (periodic must add plane 0: the far face wraps to node 0)")

print()
print("=== 2-D lane: sheet normal to the length-1 axis vs the same rectangle as a volume ===")
import jax.numpy as jnp
import rfx.boundaries.pec as P
shape=(8,8,1)
F=np.zeros(shape,dtype=bool); F[2:6,2:6,0]=True
sheet=P.SheetSpec(normal_axis=2, plane=0, footprint=jnp.asarray(F))
e_sheet=P.realized_pec_edge_masks(None, sheets=(sheet,))
C=np.zeros(shape,dtype=bool); C[2:5,2:5,0]=True      # cells whose closed node rect is 2..5
e_vol=P.realized_pec_edge_masks(jnp.asarray(C))
for c,n in enumerate("xyz"):
    same=np.array_equal(np.asarray(e_sheet[c]),np.asarray(e_vol[c]))
    print(f"  E{n}: sheet==volume ? {same}   sheet {int(np.asarray(e_sheet[c]).sum())} edges, volume {int(np.asarray(e_vol[c]).sum())}")
