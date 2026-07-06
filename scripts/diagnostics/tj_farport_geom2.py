"""Far-port H-plane T-junction — GEOMETRY 2 (W=0.036 m) for the broad claim.

Second, genuinely different junction geometry demanded by the single-geometry
honesty locks (tests/test_waveguide_tjunction_e4e5_gates.py): the numeric
breadth bars require >= 2 junction geometries before any "broad" junction
claim. Mirrors scripts/diagnostics/tj_farport_test.py (geometry 1, W=0.04)
with the same far-port discipline on the SAME domain/CPML:

  * guide width W = 0.036 m -> TE10 cutoff 4.163 GHz, TE20 cutoff 8.327 GHz;
    single-mode band used: 5.2-7.3 GHz (span 1.404 >= the 1.4 breadth floor;
    band max = 0.877 * fc2, same regime as geometry 1's 0.933).
  * far-port arms: probe-to-junction clearance left/right ~55 mm (~6.3 mid-band
    TE20 decay lengths L=8.7mm), top ~35 mm (~4.0 L) — all above geometry 1's
    validated floor (3.8 L).
  * CPML 48 mm = 0.50 * lambda_g at the 5.2 GHz band edge (geometry 1: 0.53).

Run:  python scripts/diagnostics/tj_farport_geom2.py --dx_mm 1.0 --nc 48
Output: scripts/diagnostics/_artifacts/tj_farport_geom2_dx{dx}_nc{nc}.npz
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import numpy as np
import jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux)

ap = argparse.ArgumentParser()
ap.add_argument("--dx_mm", type=float, required=True)
ap.add_argument("--nf", type=int, default=11)
ap.add_argument("--nc", type=int, default=48)
a = ap.parse_args()
dx = a.dx_mm * 1e-3
nc = a.nc

W = 0.036                      # guide width (geometry 2)
Y_LO, Y_HI = 0.102, 0.138      # horizontal channel (centre y=0.12, same as geom1)
X_LO, X_HI = 0.132, 0.168      # vertical stub (centre x=0.15, same as geom1)
BAND = np.linspace(5.2e9, 7.3e9, a.nf)
F0 = 6.25e9
LX, LY, LZ = 0.30, 0.24, 0.02

grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p / dx))
yi = lambda p: py + int(round(p / dx))


def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes:
        m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m


# H-plane tee: horizontal channel y in [Y_LO, Y_HI] all x; vertical stub
# x in [X_LO, X_HI], y in [Y_HI, LY] (open top). Junction at the overlap.
dev = pec([Box((0, 0, 0), (LX, Y_LO, LZ)),
           Box((0, Y_HI, 0), (X_LO, LY, LZ)),
           Box((X_HI, Y_HI, 0), (LX, LY, LZ))])
mat_h = pec([Box((0, 0, 0), (LX, Y_LO, LZ)), Box((0, Y_HI, 0), (LX, LY, LZ))])
mat_v = pec([Box((0, 0, 0), (X_LO, LY, LZ)), Box((X_HI, 0, 0), (LX, LY, LZ))])

freqs = jnp.asarray(BAND)
n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(Y_LO), yi(Y_HI) + 1)
xs = (xi(X_LO), xi(X_HI) + 1)
zs = (pz, grid.nz - pz)
ro = max(3, int(round(0.006 / dx)))
po = max(8, int(round(0.030 / dx)))
# ports: left x=0.04 (junction edge 0.132 -> 92mm), right x=0.26 (-> 92mm),
# top y=0.21 (junction edge 0.138 -> 72mm)
left = WaveguidePort(x_index=xi(0.04), y_slice=ys, z_slice=zs, a=W, b=LZ,
                     mode=(1, 0), mode_type="TE", direction="+x",
                     normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26), y_slice=ys, z_slice=zs, a=W, b=LZ,
                      mode=(1, 0), mode_type="TE", direction="-x",
                      normal_axis="x", u_slice=ys, v_slice=zs)
top = WaveguidePort(x_index=yi(0.21), y_slice=None, z_slice=None, a=W, b=LZ,
                    mode=(1, 0), mode_type="TE", direction="-y",
                    normal_axis="y", u_slice=xs, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=F0, ref_offset=ro, probe_offset=po,
                            dft_total_steps=n_steps) for p in (left, right, top)]
print(f"[geom2 dx={a.dx_mm}mm nc={nc}] W={W}, band {BAND[0]/1e9:.1f}-{BAND[-1]/1e9:.1f} GHz, "
      f"n_steps={n_steps}, clearance left~{(X_LO-(0.04+0.03))*1000:.0f}mm "
      f"top~{((0.21-0.03)-Y_HI)*1000:.0f}mm", flush=True)
S = np.abs(np.asarray(extract_waveguide_s_matrix_flux(
    grid, dev, init_materials(grid.shape), cfgs, n_steps,
    boundary="cpml", cpml_axes="xy", pec_axes="z",
    ref_materials_per_port=[mat_h, mat_h, mat_v])))
art = os.path.join(_HERE, "_artifacts")
os.makedirs(art, exist_ok=True)
out = os.path.join(art, f"tj_farport_geom2_dx{a.dx_mm:.1f}_nc{nc}.npz")
np.savez(out, S=S, band=BAND, dx_mm=a.dx_mm, W=W)
cps = np.sum(S ** 2, axis=0)
recip = max(np.mean(np.abs(S[i, j] - S[j, i])) for (i, j) in ((1, 0), (2, 0), (2, 1)))
print(f"[GEOM2 dx={a.dx_mm}mm] passivity_max={cps.max():.3f} "
      f"band-mean-colsum={np.mean(cps):.3f} recip={recip:.3f}", flush=True)
print(" per-freq passivity: " + " ".join(
    f"{BAND[k]/1e9:.1f}:{cps[:, k].max():.3f}" for k in range(len(BAND))), flush=True)
print(f" -> {out}", flush=True)
