"""Frequency-generalization check of the CPML root cause. Map the matched-guide
far-end |Gamma| across a BROAD single-mode band (4.3-7.3 GHz; TE10 cutoff 3.75,
TE20 onset 7.5) at a given CPML count. Physical prediction: CPML absorbs worst near
the TE10 cutoff (low vg) -> |Gamma| should rise toward 4.3 GHz, and nc=24 should
reduce it across the whole band. --nc layers."""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (WaveguidePort, init_waveguide_port,
    _rect_dft, _compute_mode_impedance, _co_located_current_spectrum)
from rfx.simulation import run as run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--nc", type=int, required=True); A = ap.parse_args()
dx = 0.002; nc = A.nc; W = 0.04; LX, LY, LZ = 0.30, 0.24, 0.02
fc = 299792458.0/(2*W)
BAND = np.linspace(4.3e9, 7.3e9, 31)
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
mat = pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=80)
ys = (yi(0.10), yi(0.14)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
# wide source: f0=5.8GHz, bandwidth 0.62 -> fwidth~3.6 covers ~4.0-7.6
left  = WaveguidePort(x_index=xi(0.04),    y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(LX-0.04), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=5.8e9, bandwidth=0.62, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left,right)]
cfgs = [c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]
res = run_sim(grid, mat, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z", waveguide_ports=cfgs, return_state=False)
c = res.waveguide_ports[0]; nv = int(np.asarray(c.n_steps_recorded)); dt = float(c.dt)
Vr = np.asarray(_rect_dft(c.v_ref_t, freqs, dt, nv)); Ir = np.asarray(_co_located_current_spectrum(c, _rect_dft(c.i_ref_t, freqs, dt, nv)))
Z = np.asarray(_compute_mode_impedance(c.freqs, c.f_cutoff, c.mode_type, dt=c.dt, dx=c.dx))
G = np.abs((Vr - Z*Ir)/(Vr + Z*Ir))
np.savez(f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_cpmlfreq_nc{nc}.npz", G=G, band=BAND)
print(f"[FREQSCAN nc={nc}] matched-guide far-end |Gamma| vs f (cutoff ratio f/3.75):", flush=True)
for k in range(0,31,3):
    print(f"   {BAND[k]/1e9:.2f}GHz (x{BAND[k]/fc:.2f}): |G|={G[k]:.3f}", flush=True)
print(f"   band|G|: mean={G.mean():.3f} max={G.max():.3f} @ {BAND[G.argmax()]/1e9:.2f}GHz", flush=True)