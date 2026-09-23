"""Reviewer check of decision 6: magnetic wall by odd image of tangential H.

Cavity 24 x 20 x 4 mm, x faces PMC, y/z PEC, vacuum, run() on CPU.
Arm half : main (tangential H zeroed at Yee index 0 / N-2; E zeroed on x faces).
Arm image: monkeypatch -- x faces: no E zeroing; hi side ghost H_t[N-1] = -H_t[N-2]
           after the H step; lo side E_t[0] += cb * (+/-)H_t[0]/dx after the E step
           (the missing half of 2*H_t[0]/dx), cb = dt/(eps0 dx) (vacuum at the face).
Analytic: f(0,1) = 7.4948 GHz; f(1,1) = 9.7547 GHz (walls 24 mm apart),
9.9321 GHz (23 mm), 9.8420 GHz (23.5 mm).
"""
import sys
import numpy as np
import warnings
warnings.simplefilter("ignore")
import rfx.simulation as S
import rfx.boundaries.pmc as P
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.boundaries.pec import apply_pec as _orig_pec
from rfx.core.yee import EPS_0
from rfx.harminv import harminv

arm = sys.argv[1]
dx = float(sys.argv[2]) * 1e-3

def build():
    sim = Simulation(freq_max=12e9, domain=(0.024, 0.020, 0.004), dx=dx,
                     boundary=BoundarySpec(x=Boundary("pmc", "pmc"), y="pec", z="pec"))
    sim.add_source(position=(0.007, 0.006, 0.002), component="ez")
    sim.add_probe(position=(0.017, 0.013, 0.002), component="ez")
    return sim

sim = build()
dt = sim._build_grid().dt
cb = dt / (EPS_0 * dx)

if arm == "image":
    def pmc_image(st, faces):
        hy, hz = st.hy, st.hz
        # hi side: odd image into the stored ghost row
        hy = hy.at[-1, :, :].set(-hy[-2, :, :])
        hz = hz.at[-1, :, :].set(-hz[-2, :, :])
        return st._replace(hy=hy, hz=hz)
    P.apply_pmc_faces = pmc_image
    def pec_image(st, axes="xyz"):
        st = st._replace(
            ez=st.ez.at[0, :, :].add(cb * st.hy[0, :, :]),
            ey=st.ey.at[0, :, :].add(-cb * st.hz[0, :, :]))
        return _orig_pec(st, axes="".join(a for a in axes if a != "x"))
    S.apply_pec = pec_image

n_steps = int(6000 * (1e-3 / dx))
res = sim.run(n_steps=n_steps, skip_preflight=True)
ts = np.asarray(res.time_series)[:, 0]
modes = harminv(ts[n_steps // 8:], float(res.dt), 5e9, 12e9)
big = max(abs(m.amplitude) for m in modes)
fs = sorted(round(m.freq / 1e9, 4) for m in modes if abs(m.amplitude) > 1e-2 * big)
C0 = 299792458.0
f11 = [f for f in fs if f > 9.0]
if f11:
    k = 2 * f11[0] * 1e9 / C0
    a_der = 1.0 / np.sqrt(k * k - 1.0 / 0.02 ** 2)
    print(f"{arm} dx={dx*1e3} mm modes {fs} GHz; derived wall separation from f11: {a_der*1e3:.3f} mm")
else:
    print(f"{arm} dx={dx*1e3} mm modes {fs} GHz")
