"""Localize the matched-guide standing wave. V/I are spatially anti-correlated
=> real backward wave (confirmed). Now: does the standing-wave period scale with
guide length? Vary the receiving-port distance (LX), keep source+measurement fixed
at the left, fine frequency grid. If the apparent-reflection phase slope
d(phase Gamma)/d(beta) = -2*L_refl tracks the measurement->receiving-port distance,
the reflection is at the receiving port (far end). --lx in meters.
"""
import sys, argparse; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np, jax.numpy as jnp
from rfx.grid import Grid
from rfx.materials import init_materials
from rfx.geometry.csg import Box
from rfx.sources.waveguide_port import (WaveguidePort, init_waveguide_port,
    _rect_dft, _compute_mode_impedance, _co_located_current_spectrum)
from rfx.simulation import run as run_sim

ap = argparse.ArgumentParser(); ap.add_argument("--lx", type=float, required=True); A = ap.parse_args()
dx = 0.002; nc = 10; W = 0.04; LX = A.lx; LY = 0.24; LZ = 0.02
BAND = np.linspace(5.0e9, 7.0e9, 41)          # FINE grid to resolve the period
grid = Grid(freq_max=10e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=nc, cpml_axes="xy")
px, py, pz = grid.axis_pads
xi = lambda p: px + int(round(p/dx)); yi = lambda p: py + int(round(p/dx))
def pec(boxes):
    m = init_materials(grid.shape)
    for b in boxes: m = m._replace(sigma=jnp.where(b.mask(grid), 1e10, m.sigma))
    return m
mat = pec([Box((0,0,0),(LX,0.10,LZ)), Box((0,0.14,0),(LX,LY,LZ))])   # matched horiz guide
freqs = jnp.asarray(BAND); n_steps = grid.num_timesteps(num_periods=60)
ys = (yi(0.10), yi(0.14)+1); zs = (pz, grid.nz-pz)
ro = max(3, int(round(0.006/dx))); po = max(8, int(round(0.030/dx)))
x_left = 0.04; x_right = LX - 0.04
left  = WaveguidePort(x_index=xi(x_left),  y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="+x", normal_axis="x", u_slice=ys, v_slice=zs)
right = WaveguidePort(x_index=xi(x_right), y_slice=ys, z_slice=zs, a=W, b=LZ, mode=(1,0), mode_type="TE", direction="-x", normal_axis="x", u_slice=ys, v_slice=zs)
cfgs = [init_waveguide_port(p, dx, freqs, f0=6e9, ref_offset=ro, probe_offset=po, dft_total_steps=n_steps) for p in (left,right)]
cfgs = [c._replace(src_amp=c.src_amp if i==0 else 0.0) for i,c in enumerate(cfgs)]
res = run_sim(grid, mat, n_steps, boundary="cpml", cpml_axes="xy", pec_axes="z", waveguide_ports=cfgs, return_state=False)
c = res.waveguide_ports[0]; nv = int(np.asarray(c.n_steps_recorded)); dt = float(c.dt)
Vr = np.asarray(_rect_dft(c.v_ref_t, freqs, dt, nv)); Vp = np.asarray(_rect_dft(c.v_probe_t, freqs, dt, nv))
Ir = np.asarray(_co_located_current_spectrum(c, _rect_dft(c.i_ref_t, freqs, dt, nv)))
Z = np.asarray(_compute_mode_impedance(c.freqs, c.f_cutoff, c.mode_type, dt=c.dt, dx=c.dx))
# apparent reflection at ref plane via V/I
G = (Vr - Z*Ir)/(Vr + Z*Ir)
# beta (numerical, analytic) for phase-slope localization
C0=299792458.0; beta=np.sqrt((2*np.pi*BAND/C0)**2-(np.pi/W)**2)
ph = np.unwrap(np.angle(G))
slope = np.polyfit(beta, ph, 1)[0]   # d(phase)/d(beta) = -2 L_refl
L_refl = -slope/2
x_ref_m = float(c.reference_x_m)
print(f"[SWSCALE LX={LX:.2f}] x_left={x_left} x_right={x_right:.2f} ref_plane={x_ref_m:.3f}")
print(f"  meas->right-port dist = {x_right-x_ref_m:.3f} m;  meas->left-bndry = {x_ref_m:.3f} m")
print(f"  |G| band-mean={np.abs(G).mean():.3f};  phase-slope L_refl = {L_refl:.3f} m")
print(f"  |Vp|/|Vr| std (standing-wave depth) = {np.abs(Vp/Vr).std():.3f}")
np.savez(f"/tmp/rfx-tj/scripts/diagnostics/_artifacts/tj_sw_lx{LX:.2f}.npz", G=G, band=BAND, x_right=x_right, x_ref=x_ref_m, L_refl=L_refl)