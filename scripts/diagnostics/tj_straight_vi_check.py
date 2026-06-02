"""Decisive control: STRAIGHT matched guide (no junction), long arm, V/I extractor.
A matched straight guide must give |S11|~0 smooth. If |S11| instead ripples,
the rfx port/extraction itself injects a reflection/standing-wave artifact over
the long arm (independent of any junction) -> that is the ripple source. If clean,
the T-junction ripple is junction-specific and the mechanism is elsewhere.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--dx_mm", type=float, default=2.0); A = ap.parse_args()
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
# straight horizontal matched guide y in [0.10,0.14], all x (NO junction)
straight = pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left, right)]
S = np.abs(np.asarray(extract_waveguide_s_matrix(grid, straight, cfgs, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z")))
print(f"[STRAIGHT-VI dx={A.dx_mm}mm] matched thru guide, long arm:")
print(f"  |S11|(f): "+" ".join(f"{S[0,0,k]:.3f}" for k in range(11))+f"  std={S[0,0].std():.3f} mean={S[0,0].mean():.3f}")
print(f"  |S21|(f): "+" ".join(f"{S[1,0,k]:.3f}" for k in range(11))+f"  mean={S[1,0].mean():.3f}")
print("  -> |S11|~0 smooth = port clean (junction-specific ripple); |S11| ripples = port/extraction artifact over long arm")