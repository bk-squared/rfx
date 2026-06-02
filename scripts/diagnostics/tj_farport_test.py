"""DISCRIMINATING TEST: is the T-junction mesh-divergence a SOLVER corner error
(H1/H3) or just PORTS sitting in the junction's evanescent near-field (H2)?

Long-arm T-junction: ports placed >=5 evanescent decay-lengths from the junction
(old test had them at 1-3 lengths). If passivity now -> 1.0 AND stops worsening
with mesh -> H2 (port placement, a measurement artifact; SOLVER IS FINE). If it
still diverges with mesh -> H1/H3 (genuine solver/energy problem -> solver work
justified). Passivity (sum_i |S_ij|^2 <= 1 for a lossless PEC junction) is an
absolute physical bound, so this is decisive without needing MEEP as truth.

Geometry (W=0.04, single-mode 3.75-7.5 GHz): long arms so probe planes are
~50-60mm (>=4-5 decay lengths at mid-band) from the junction edges.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux)

ap = argparse.ArgumentParser(); ap.add_argument("--dx_mm", type=float, required=True)
ap.add_argument("--nf", type=int, default=11); ap.add_argument("--nc", type=int, default=10); a = ap.parse_args()
dx = a.dx_mm * 1e-3; nc = a.nc
BAND = np.linspace(5.0e9, 7.0e9, a.nf)   # full band incl. the old 7GHz spike zone
LX, LY, LZ = 0.30, 0.24, 0.02
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p / dx)); yi = lambda p: py + int(round(p / dx))

def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m

# Long-arm H-plane T: horizontal channel y in [0.10,0.14] all x; vertical stub
# x in [0.13,0.17], y in [0.14,0.24] (open top). Junction at x[0.13,0.17] y[0.10,0.14].
dev = pec([Box((0,0,0),(LX,0.10,LZ)),
           Box((0,0.14,0),(0.13,LY,LZ)),
           Box((0.17,0.14,0),(LX,LY,LZ))])
mat_h = pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))])       # straight horiz guide
mat_v = pec([Box((0,0,0),(0.13,LY,LZ)), Box((0.17,0,0),(LX,LY,LZ))])       # straight vert guide

freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); xs = (xi(0.13), xi(0.17)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
# ports: left x=0.04 (junction edge 0.13 -> 90mm), right x=0.26 (-> 90mm), top y=0.21 (-> 70mm)
left  = WaveguidePort(x_index=xi(0.04),     y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26),     y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top   = WaveguidePort(x_index=yi(0.21),     y_slice=None, z_slice=None, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left, right, top)]
print(f"[dx={a.dx_mm}mm] domain {LX}x{LY}, n_steps={n_steps}, probe ~{0.04+0.03:.2f}/{0.21-0.03:.2f}m from boundary; "
      f"junction-clearance left~{(0.13-(0.04+0.03))*1000:.0f}mm top~{((0.21-0.03)-0.14)*1000:.0f}mm", flush=True)
S = np.abs(np.asarray(extract_waveguide_s_matrix_flux(grid, dev, init_materials(grid.shape), cfgs, n_steps,
    boundary="cpml", cpml_axes="xy", pec_axes="z", ref_materials_per_port=[mat_h, mat_h, mat_v])))
out = f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_farport_dx{a.dx_mm:.1f}_nc{nc}.npz"
np.savez(out, S=S, band=BAND, dx_mm=a.dx_mm)
cps = np.sum(S**2, axis=0)
print(f"[FARPORT dx={a.dx_mm}mm] passivity_max={cps.max():.3f} band-mean-colsum={np.mean(cps):.3f} "
      f"recip={max(np.mean(np.abs(S[i,j]-S[j,i])) for (i,j) in ((1,0),(2,0),(2,1))):.3f}", flush=True)
print(" per-freq passivity: " + " ".join(f"{BAND[k]/1e9:.1f}:{cps[:,k].max():.3f}" for k in range(len(BAND))), flush=True)
print(f" -> {out}", flush=True)