"""Short-Open-Load (here: Load + 2 offset-Shorts) 1-port error correction for the
rfx far-port waveguide reflection. 3-term model:  Gamma_m = e00 + T*Gamma_a/(1 - e11*Gamma_a).

Standards on the horizontal guide (port 0 drive), cal reference plane = short1 location:
  load   : matched thru guide                 -> Gamma_a = 0      => e00 = Gamma_m
  short1 : PEC wall across channel at x_cal    -> Gamma_a = -1
  short2 : PEC wall at x_cal + DELTA           -> Gamma_a = -exp(-2j*beta*DELTA)
Defining Gamma_a at the short1 plane lets the error terms ABSORB the (unknown) rfx
reference-plane phase; beta is only needed over the small known offset DELTA.
--std load|short1|short2  saves complex S[0,0] (drive port 0). DELTA=8mm.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--std", required=True, choices=["load","short1","short2"])
ap.add_argument("--dx_mm", type=float, default=2.0); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc = 10
BAND = np.linspace(5.0e9, 7.0e9, 11)
LX, LY, LZ = 0.30, 0.24, 0.02
X_CAL = 0.115; DELTA = 0.008          # short1 plane and offset to short2
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
base = [Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))]   # matched horizontal guide walls
wall = 3*dx
if A.std == "load":   mat = pec(base)
elif A.std == "short1": mat = pec(base + [Box((X_CAL,0.10,0),(X_CAL+wall,0.14,LZ))])
else:                   mat = pec(base + [Box((X_CAL+DELTA,0.10,0),(X_CAL+DELTA+wall,0.14,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(0.04), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(0.26), y_slice=ys, z_slice=zs, a=0.04, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left, right)]
S = np.asarray(extract_waveguide_s_matrix(grid, mat, cfgs, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z"))
out = f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_sol_{A.std}_dx{A.dx_mm:.1f}.npz"
np.savez(out, S11=S[0,0], band=BAND, x_cal=X_CAL, delta=DELTA)
print(f"[SOL {A.std} dx={A.dx_mm}] |S11|(f): "+" ".join(f"{abs(S[0,0,k]):.3f}" for k in range(11))+f"  -> {out}", flush=True)