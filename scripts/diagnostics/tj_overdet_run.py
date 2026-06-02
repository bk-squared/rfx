"""Over-determined SOL validation runner. Drives port 0, runs via run(), and saves
the COMPLEX modal V and I DFTs at BOTH recorded planes (ref, probe) for port 0,
plus geometry/mode metadata. The solver (tj_overdet_solve.py) then: extracts
NUMERICAL beta from the load two-plane phase, builds S11 per standard, does an
over-determined least-squares SOL solve with held-out validation, and applies it
to the device. --mode load | s<delta_mm> | device.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (WaveguidePort, init_waveguide_port,
    _rect_dft, _compute_mode_impedance, _co_located_current_spectrum)
from rfx.simulation import run as run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True)  # load | s0 s3 ... | device
ap.add_argument("--dx_mm", type=float, default=2.0); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc = 10; W = 0.04
BAND = np.linspace(5.0e9, 7.0e9, 11)
LX, LY, LZ = 0.30, 0.24, 0.02
X_CAL = 0.115
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
base_h = [Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))]
dev_blocks = [Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(0.13,LY,LZ)), Box((0.17,0.14,0),(LX,LY,LZ))]
wall = 3*dx
if A.mode == "load":     mat = pec(base_h); is_dev = False
elif A.mode == "device": mat = pec(dev_blocks); is_dev = True
elif A.mode.startswith("s"):
    delta = float(A.mode[1:]) * 1e-3
    mat = pec(base_h + [Box((X_CAL+delta,0.10,0),(X_CAL+delta+wall,0.14,LZ))]); is_dev = False
else: raise SystemExit("bad mode")
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); xs = (xi(0.13), xi(0.17)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top   = WaveguidePort(x_index=yi(0.21), y_slice=None, z_slice=None, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
plist = (left,right,top) if is_dev else (left,right)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in plist]
cfgs = [c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]
res = run_sim(grid, mat, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z", waveguide_ports=cfgs, return_state=False)
c = res.waveguide_ports[0]; nv = int(np.asarray(c.n_steps_recorded)); dt = float(c.dt)
Vr = np.asarray(_rect_dft(c.v_ref_t, freqs, dt, nv)); Vp = np.asarray(_rect_dft(c.v_probe_t, freqs, dt, nv))
Ir = np.asarray(_co_located_current_spectrum(c, _rect_dft(c.i_ref_t, freqs, dt, nv)))
Ip = np.asarray(_co_located_current_spectrum(c, _rect_dft(c.i_probe_t, freqs, dt, nv)))
Z = np.asarray(_compute_mode_impedance(c.freqs, c.f_cutoff, c.mode_type, dt=c.dt, dx=c.dx))
out = f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_od_{A.mode}_dx{A.dx_mm:.1f}.npz"
np.savez(out, Vr=Vr, Vp=Vp, Ir=Ir, Ip=Ip, Z=Z, band=BAND,
         ref_x_m=float(c.reference_x_m), probe_x_m=float(c.probe_x_m), f_cutoff=float(c.f_cutoff))
# quick S11 via V/I at ref plane (forward/backward), +x port -> S11 = backward/forward
fwd = 0.5*(Vr + Z*Ir); bwd = 0.5*(Vr - Z*Ir)
print(f"[OD {A.mode} dx={A.dx_mm}] |S11|=|bwd/fwd|: "+" ".join(f"{abs(bwd[k]/fwd[k]):.3f}" for k in range(11))+f" -> {out}", flush=True)