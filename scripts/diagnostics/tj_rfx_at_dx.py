"""Run the flux T-junction (per-port matched-guide reference) at a given dx,
save |S| over the full 5-7 GHz band for the convergence study."""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux)

ap = argparse.ArgumentParser(); ap.add_argument("--dx_mm", type=float, required=True)
ap.add_argument("--nf", type=int, default=11); a = ap.parse_args()
dx = a.dx_mm * 1e-3; nc = 10
BAND = np.linspace(5.0e9, 7.0e9, a.nf)
grid = Grid(freq_max=10e9, domain=(0.12, 0.12, 0.02), dx=dx, cpml_layers=nc, cpml_axes="xy")

def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
dev = pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.04,0.12,0.02)), Box((0.08,0.08,0),(0.12,0.12,0.02))])
mat_h = pec([Box((0,0,0),(0.12,0.04,0.02)), Box((0,0.08,0),(0.12,0.12,0.02))])
mat_v = pec([Box((0,0,0),(0.04,0.12,0.02)), Box((0.08,0,0),(0.12,0.12,0.02))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=30)
px, py, pz = grid.axis_pads
xs = (px+int(round(0.04/dx)), px+int(round(0.08/dx))+1); ys = (py+int(round(0.04/dx)), py+int(round(0.08/dx))+1); zs = (pz, grid.nz-pz)
left = WaveguidePort(x_index=nc+5, y_slice=ys, z_slice=zs, a=0.04, b=0.02, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=grid.nx-nc-6, y_slice=ys, z_slice=zs, a=0.04, b=0.02, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top = WaveguidePort(x_index=grid.ny-nc-6, y_slice=None, z_slice=None, a=0.04, b=0.02, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=max(3,int(round(0.006/dx))), probe_offset=max(8,int(round(0.030/dx))), dft_total_steps=n_steps) for p in (left, right, top)]
S = np.abs(np.asarray(extract_waveguide_s_matrix_flux(grid, dev, init_materials(grid.shape), cfgs, n_steps,
    boundary="cpml", cpml_axes="xy", pec_axes="z", ref_materials_per_port=[mat_h, mat_h, mat_v])))
out = f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_rfx_dx{a.dx_mm:.1f}.npz"
np.savez(out, S=S, band=BAND, dx_mm=a.dx_mm)
cps = np.sum(S**2, axis=0)
print(f"dx={a.dx_mm}mm: passivity max={cps.max():.3f} band-mean-colsum={np.mean(cps):.3f} "
      f"recip={max(np.mean(np.abs(S[i,j]-S[j,i])) for (i,j) in ((1,0),(2,0),(2,1))):.3f} -> {out}")
