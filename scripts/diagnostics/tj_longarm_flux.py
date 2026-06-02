"""Clean TE20 test for the 7.0 GHz device residual: longer-arm (140mm) T-junction,
FLUX extractor, nc=24 (CPML no longer confounding). If the junction-excited TE20
(evanescent, 7.5 GHz cutoff) is the cause, the much longer arm (~6 TE20 decay-lengths
at 7.0 GHz vs ~3 at 90mm) should sharply reduce the 7.0 GHz cross-FDTD vs MEEP.
Compare to the 90mm nc=24 result (7.0 residual 0.25). --nc."""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix_flux)

ap = argparse.ArgumentParser(); ap.add_argument("--dx_mm", type=float, default=2.0); ap.add_argument("--nc", type=int, default=24)
ap.add_argument("--fmax_ghz", type=float, default=7.0); ap.add_argument("--nf", type=int, default=11); A = ap.parse_args()
dx = A.dx_mm*1e-3; nc = A.nc; W = 0.04
BAND = np.linspace(5.0e9, A.fmax_ghz*1e9, A.nf)
PB, AL = 0.04, 0.14; YC = 0.12; XL = PB+AL; XR = XL+W
LX = XR+AL+PB; LY = (YC+W/2)+AL+PB; LZ = 0.02          # 0.40 x 0.32 (140mm arms)
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
yb0, yb1 = YC-W/2, YC+W/2
dev   = pec([Box((0,0,0),(LX,yb0,LZ)), Box((0,yb1,0),(XL,LY,LZ)), Box((XR,yb1,0),(LX,LY,LZ))])
mat_h = pec([Box((0,0,0),(LX,yb0,LZ)), Box((0,yb1,0),(LX,LY,LZ))])
mat_v = pec([Box((0,0,0),(XL,LY,LZ)), Box((XR,0,0),(LX,LY,LZ))])
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(yb0), yi(yb1)+1); xs = (xi(XL), xi(XR)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
left  = WaveguidePort(x_index=xi(PB),    y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(XR+AL), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
top   = WaveguidePort(x_index=yi((YC+W/2)+AL), y_slice=None, z_slice=None, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-y", normal_axis="y", u_slice=xs, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left,right,top)]
S = np.abs(np.asarray(extract_waveguide_s_matrix_flux(grid, dev, init_materials(grid.shape), cfgs, n_steps,
    boundary="cpml", cpml_axes="xy", pec_axes="z", ref_materials_per_port=[mat_h, mat_h, mat_v])))
np.savez(f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_longarm_flux_nc{nc}_fmax{A.fmax_ghz:.1f}.npz", S=S, band=BAND)
k70 = int(np.argmin(np.abs(BAND - 7.0e9)))   # index of the 7.0 GHz point
M = np.zeros((3,3,len(BAND)))
for d in range(3):
    z=np.load(f"scripts/diagnostics/_artifacts/meep_tjunction_farport_r500_drive{d}.npz")
    for j in range(3): M[j,d]=np.interp(BAND,z["freqs_hz"],np.abs(z["col"])[j])
cps=np.sum(S**2,axis=0)
print(f"[LONGARM-FLUX nc={nc} arm=140mm] passivity_max={cps.max():.3f} recip={max(np.mean(np.abs(S[i,j]-S[j,i])) for (i,j) in ((1,0),(2,0),(2,1))):.3f}")
print(" per-freq cross-FDTD |S-MEEP| max: "+" ".join(f"{BAND[k]/1e9:.1f}:{np.abs(S[:,:,k]-M[:,:,k]).max():.3f}" for k in range(len(BAND))))
print(f" |S11|(f): "+" ".join(f"{S[0,0,k]:.3f}" for k in range(len(BAND))))
print(f" MEEP|S11|: "+" ".join(f"{M[0,0,k]:.3f}" for k in range(len(BAND))))
_pos = "EDGE" if k70 == len(BAND)-1 else "INTERIOR"
print(f" 7.0GHz [{_pos}]: |S11|={S[0,0,k70]:.3f} (MEEP {M[0,0,k70]:.3f}) cross-FDTD={float(np.abs(S[:,:,k70]-M[:,:,k70]).max()):.3f}; 7.0-as-edge was 0.25")