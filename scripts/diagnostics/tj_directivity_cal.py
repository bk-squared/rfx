"""Directivity error-correction for the far-port T-junction reflection.

Diagnosis (tj_straight_vi_check): a matched straight guide gives |S21|=1.000 but
|S11|~0.14 SMOOTH -> rfx's V/I reflection extraction carries a spurious directivity
floor e00 (forward->backward leak). With a real junction reflection present, the long
de-embedding arm rotates Gamma_junction's phase with frequency, and |Gamma + e00|
ripples. Standard VNA directivity correction removes it:

    Gamma_corrected(f) = S_dev[i,i](f) - e00_i(f)

where e00_i is the COMPLEX matched-guide reflection for port i (same plane). Uses the
matched-guide reference we already run -- no source change.

--mode device|horiz|vert  saves COMPLEX S for that run.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["device","horiz","vert"])
ap.add_argument("--dx_mm", type=float, default=2.0); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc = 10
BAND = np.linspace(5.0e9, 7.0e9, 11)
LX, LY, LZ = 0.30, 0.24, 0.02
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
mats = dict(
    device=pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(0.13,LY,LZ)), Box((0.17,0.14,0),(LX,LY,LZ))]),
    horiz =pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))]),                       # matched horiz guide
    vert  =pec([Box((0,0,0),(0.13,LY,LZ)), Box((0.17,0,0),(LX,LY,LZ))]))                       # matched vert guide
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); xs = (xi(0.13), xi(0.17)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top   = WaveguidePort(x_index=yi(0.21), y_slice=None, z_slice=None, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
ports = dict(device=(left,right,top), horiz=(left,right), vert=(top,))[A.mode]
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in ports]
S = np.asarray(extract_waveguide_s_matrix(grid, mats[A.mode], cfgs, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z"))  # COMPLEX
out = f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_cal_{A.mode}_dx{A.dx_mm:.1f}.npz"
np.savez(out, S=S, band=BAND)
print(f"[{A.mode} dx={A.dx_mm}] saved complex S {S.shape} -> {out}", flush=True)
print(f"  |S_ii|(f): "+" ".join(f"{abs(S[i,i,5]):.3f}" for i in range(S.shape[0]))+" (mid-band per diag)", flush=True)