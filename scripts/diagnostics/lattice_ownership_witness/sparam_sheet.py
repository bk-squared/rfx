"""S-parameter entry point: is a declared conductor realized on the
waveguide S-matrix lane, and does a SHEET short read the same as a VOLUME?

Oracle: a full-cross-section PEC short in a WR-90 guide must give |S11|~1,
|S21|~0.  An empty guide gives |S11|~0, |S21|~1.  Nothing subtle: if the
declaration is dropped on this lane the numbers swap.
"""
import warnings, numpy as np
warnings.simplefilter("ignore")
from rfx import Simulation, Box
from rfx.boundaries.spec import Boundary, BoundarySpec

A, B = 22.86e-3, 10.16e-3
DX = 1.0e-3
LX = 60e-3

def guide(build):
    sim = Simulation(freq_max=13e9, domain=(LX, A, B), dx=DX,
                     boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                          y=Boundary(lo="pec", hi="pec"),
                                          z=Boundary(lo="pec", hi="pec")))
    build(sim)
    for x, d in ((10e-3, "+x"), (LX-10e-3, "-x")):
        sim.add_waveguide_port(x, direction=d, f0=10e9, bandwidth=0.3, n_freqs=9)
    return sim

# The grid realizes the guide by ceil, so draw the plug to the REALIZED walls
# (design note §6, the cv11 rule).  Read them off the node lines.
probe = guide(lambda s: None)
g = probe._build_grid()
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
co = coords_from_uniform_grid(g)
YW = float(np.asarray(co.y)[-1]); ZW = float(np.asarray(co.z)[-1])
print(f"realized guide walls: y_hi={YW*1e3:.3f} mm  z_hi={ZW*1e3:.3f} mm  (declared {A*1e3:.2f} x {B*1e3:.2f})")

cases = {
  "empty guide (control)": lambda s: None,
  "PEC VOLUME plug 2 cells": lambda s: s.add(
        Box((30e-3, 0, 0), (32e-3, YW, ZW)), material="pec"),
  "PEC SHEET short":        lambda s: s.add_thin_conductor(
        Box((30e-3, 0, 0), (30e-3, YW, ZW))),
}
for name, b in cases.items():
    sim = guide(b)
    r = sim.compute_waveguide_s_matrix(n_steps=2500)
    S = np.asarray(r.s_params)
    f = np.asarray(r.freqs)
    k = int(np.argmin(np.abs(f - 10e9)))
    if S.shape[0] == f.size:
        s11 = abs(S[k,0,0]); s21 = abs(S[k,1,0])
    else:
        s11 = abs(S[0,0,k]); s21 = abs(S[1,0,k])
    print('   S shape', S.shape, 'f size', f.size)
    print(f"  {name:26s} @ {f[k]/1e9:5.2f} GHz  |S11|={s11:.4f}  |S21|={s21:.4f}")
