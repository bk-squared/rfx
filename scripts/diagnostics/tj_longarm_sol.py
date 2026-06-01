"""Longer-arm probe of the upper-band (near-TE20-cutoff) residual. If pushing the
ports to ~140mm (~8 TE20 decay-lengths at 7GHz) from the junction cleans the
6.6-7.0GHz SOL-corrected |S11|, the residual is TE20 evanescent contamination of
rfx's TE10 modal projection. Same SOL standards (load/short1/short2) at the longer
arm. |S11| magnitude is reference-plane independent -> compare to the same matched
MEEP bare-junction reference. --mode device|load|short1|short2.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix

ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["device","load","short1","short2"])
ap.add_argument("--dx_mm", type=float, default=2.0); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc = 10; W = 0.04
BAND = np.linspace(5.0e9, 7.0e9, 11)
PB, AL = 0.04, 0.14                      # port buffer, arm length (junction edge -> port)
YC = 0.12; XL = PB+AL; XR = XL+W         # junction horiz channel center; junction x in [XL,XR]
LX = XR+AL+PB; LY = (YC+W/2)+AL+PB       # = 0.40 x 0.32
X_CAL = XL; DELTA = 0.008                # short1 at junction-left plane
grid = Grid(freq_max=10e9, domain=(LX, LY, 0.02), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
yb0, yb1 = YC-W/2, YC+W/2                 # channel y in [0.10,0.14]
base_h = [Box((0,0,0),(LX,yb0,0.02)), Box((0,yb1,0),(LX,LY,0.02))]      # matched horiz guide
dev_blocks = [Box((0,0,0),(LX,yb0,0.02)), Box((0,yb1,0),(XL,LY,0.02)), Box((XR,yb1,0),(LX,LY,0.02))]
wall = 3*dx
mats = dict(device=pec(dev_blocks), load=pec(base_h),
            short1=pec(base_h+[Box((X_CAL,yb0,0),(X_CAL+wall,yb1,0.02))]),
            short2=pec(base_h+[Box((X_CAL+DELTA,yb0,0),(X_CAL+DELTA+wall,yb1,0.02))]))
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(yb0), yi(yb1)+1); xs = (xi(XL), xi(XR)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(PB),   y_slice=ys, z_slice=zs, a=W, b=0.02, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(XR+AL),y_slice=ys, z_slice=zs, a=W, b=0.02, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top   = WaveguidePort(x_index=yi((YC+W/2)+AL), y_slice=None, z_slice=None, a=W, b=0.02, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
ports = (left,right,top) if A.mode=="device" else (left,right)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in ports]
S = np.asarray(extract_waveguide_s_matrix(grid, mats[A.mode], cfgs, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z"))
out = f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_la_{A.mode}_dx{A.dx_mm:.1f}.npz"
np.savez(out, S=S, S11=S[0,0], band=BAND, delta=DELTA)
print(f"[LA {A.mode} dx={A.dx_mm} arm={AL*1000:.0f}mm dom={LX:.2f}x{LY:.2f}] |S11|(f): "+" ".join(f"{abs(S[0,0,k]):.3f}" for k in range(11))+f" -> {out}", flush=True)