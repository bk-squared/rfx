"""Pin the far-end reflection to CPML vs port-plane: sweep CPML layer count on a
matched guide and measure the far-end |Gamma| (apparent reflection). If |Gamma|
drops with more CPML -> it's CPML absorption of the TE10 mode (fix = more/better
CPML). If unchanged -> it's the port-plane structure. --nc layers."""
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
BAND = np.linspace(5.0e9, 7.0e9, 21)
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
mat = pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04),    y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(LX-0.04), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left,right)]
cfgs = [c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]
res = run_sim(grid, mat, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z", waveguide_ports=cfgs, return_state=False)
c = res.waveguide_ports[0]; nv = int(np.asarray(c.n_steps_recorded)); dt = float(c.dt)
Vr = np.asarray(_rect_dft(c.v_ref_t, freqs, dt, nv)); Vp = np.asarray(_rect_dft(c.v_probe_t, freqs, dt, nv))
Ir = np.asarray(_co_located_current_spectrum(c, _rect_dft(c.i_ref_t, freqs, dt, nv)))
Z = np.asarray(_compute_mode_impedance(c.freqs, c.f_cutoff, c.mode_type, dt=c.dt, dx=c.dx))
G = (Vr - Z*Ir)/(Vr + Z*Ir)
print(f"[CPML nc={nc} ({nc*dx*1000:.0f}mm)] far-end |Gamma|: band-mean={np.abs(G).mean():.3f} max={np.abs(G).max():.3f}  |Vp|/|Vr| std(SW depth)={np.abs(Vp/Vr).std():.3f}", flush=True)