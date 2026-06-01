"""Generalization across GUIDE WIDTH: confirm the CPML far-end |Gamma| follows the
cutoff RATIO (f/fc), independent of absolute frequency. Parametrized W and band.
Compare the |Gamma|(f/fc) curve to the W=0.04 result. --w_mm --nc --fmin_ghz --fmax_ghz."""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (WaveguidePort, init_waveguide_port,
    _rect_dft, _compute_mode_impedance, _co_located_current_spectrum)
from rfx.simulation import run as run_sim

ap = argparse.ArgumentParser()
ap.add_argument("--w_mm", type=float, required=True); ap.add_argument("--nc", type=int, required=True)
ap.add_argument("--fmin_ghz", type=float, required=True); ap.add_argument("--fmax_ghz", type=float, required=True)
A = ap.parse_args()
dx = 0.002; nc = A.nc; W = A.w_mm*1e-3; C0 = 299792458.0; fc = C0/(2*W)
Y0 = 0.10; LX, LY, LZ = 0.30, Y0+W+0.10, 0.02
BAND = np.linspace(A.fmin_ghz*1e9, A.fmax_ghz*1e9, 31)
f0 = 0.5*(A.fmin_ghz+A.fmax_ghz)*1e9; bw = (A.fmax_ghz-A.fmin_ghz)/(A.fmin_ghz+A.fmax_ghz)*2*1.2
grid = Grid(freq_max=A.fmax_ghz*1e9+3e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
mat = pec([Box((0,0,0),(LX,Y0,LZ)), Box((0,Y0+W,0),(LX,LY,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=80)
ys = (yi(Y0), yi(Y0+W)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04),    y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(LX-0.04), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=f0, bandwidth=bw, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left,right)]
cfgs = [c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]
res = run_sim(grid, mat, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z", waveguide_ports=cfgs, return_state=False)
c = res.waveguide_ports[0]; nv = int(np.asarray(c.n_steps_recorded)); dt = float(c.dt)
Vr = np.asarray(_rect_dft(c.v_ref_t, freqs, dt, nv)); Ir = np.asarray(_co_located_current_spectrum(c, _rect_dft(c.i_ref_t, freqs, dt, nv)))
Z = np.asarray(_compute_mode_impedance(c.freqs, c.f_cutoff, c.mode_type, dt=c.dt, dx=c.dx))
G = np.abs((Vr - Z*Ir)/(Vr + Z*Ir))
print(f"[FREQSCAN2 W={A.w_mm}mm fc={fc/1e9:.2f}GHz nc={nc}] far-end |Gamma| vs cutoff ratio:", flush=True)
for k in range(0,31,4):
    print(f"   f={BAND[k]/1e9:.2f}GHz  x{BAND[k]/fc:.2f}cutoff  |G|={G[k]:.3f}", flush=True)