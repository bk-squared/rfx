"""Two-plane forward/backward wave separation for the far-port T-junction
reflection. rfx records modal voltage at TWO planes (v_ref_t at ref_x, v_probe_t
at probe_x). For a single TE10 mode: V(x) = a e^{-jβx} + b e^{+jβx}. Solving the
two equations gives a, b directly -> |S11| = |b/a|, with NO Z_mode (so no
directivity floor) and inherent immunity to source-match (b/a is set by the
junction). Analytic β = sqrt((2πf/c)^2 - (π/W)^2). Pure post-processing.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port, _rect_dft
from rfx.simulation import run as run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--dx_mm", type=float, default=2.0); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc = 10; C0 = 299792458.0; W = 0.04
BAND = np.linspace(5.0e9, 7.0e9, 11)
LX, LY, LZ = 0.30, 0.24, 0.02
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
dev = pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(0.13,LY,LZ)), Box((0.17,0.14,0),(LX,LY,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); xs = (xi(0.13), xi(0.17)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top   = WaveguidePort(x_index=yi(0.21), y_slice=None, z_slice=None, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left, right, top)]
cfgs = [c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]  # drive port 0
res = run_sim(grid, dev, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z", waveguide_ports=cfgs, return_state=False)
c0 = res.waveguide_ports[0]
nv = int(np.asarray(c0.n_steps_recorded)); dt = float(c0.dt)
Vr = np.asarray(_rect_dft(c0.v_ref_t, freqs, dt, nv))      # complex (nf,) at ref plane
Vp = np.asarray(_rect_dft(c0.v_probe_t, freqs, dt, nv))    # at probe plane
d = abs(float(c0.probe_x_m) - float(c0.reference_x_m))     # plane separation (m)
beta = np.sqrt((2*np.pi*BAND/C0)**2 - (np.pi/W)**2)        # analytic TE10
e = np.exp(1j*beta*d); denom = (e - 1/e)
a = (Vr*e - Vp)/denom; b = (Vp - Vr/e)/denom
S11 = np.abs(b/a)
# MEEP smooth reference
z = np.load("scripts/diagnostics/_artifacts/meep_tjunction_farport_r500_drive0.npz")
M = np.interp(BAND, z["freqs_hz"], np.abs(z["col"])[0])
print(f"[TWO-PLANE dx={A.dx_mm}mm] plane sep d={d*1000:.1f}mm  βd(5/6/7GHz)={beta[0]*d:.2f}/{beta[5]*d:.2f}/{beta[10]*d:.2f} rad")
print("GHz   |S11|2plane   MEEP   |sinβd|(cond)")
for k in range(11):
    print(f"{BAND[k]/1e9:.1f}    {S11[k]:.3f}        {M[k]:.3f}   {abs(np.sin(beta[k]*d)):.3f}")
print(f"|S11| 2-plane: std={S11.std():.3f} mean={S11.mean():.3f}  (MEEP std={M.std():.3f} mean={M.mean():.3f})")
print(f"cross-FDTD |2plane - MEEP|: max={np.abs(S11-M).max():.3f} mean={np.abs(S11-M).mean():.3f}")
print(f"  [5.0-6.5GHz only] cross-FDTD max={np.abs(S11-M)[:8].max():.3f}  |S11| std={S11[:8].std():.3f}")