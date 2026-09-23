"""Reviewer check: can CI reach the baked fast path on CPU, and what cavity does it solve?

Cavity 24 x 20 x 4 mm, x faces PMC, y/z PEC (the B0c geometry), dx = 1 mm.
z-invariant TM modes (Ez): f(0,1) = c/(2b) = 7.4948 GHz exists only with
magnetic x walls; f(1,1) = 9.7547 GHz for walls 24 mm apart, 9.9321 GHz for 23 mm.
Arm std : run() on CPU (standard path).
Arm fast: run() on CPU with rfx.simulation's backend query answering "gpu",
          which makes run() take update_he_fast with precompute_coeffs(pec_axes).
"""
import sys
import numpy as np
import warnings
warnings.simplefilter("ignore")
import jax
import rfx.simulation as S
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.harminv import harminv

fake = len(sys.argv) > 1 and sys.argv[1] == "fast"
calls = {"fast": 0}
if fake:
    real_backend = jax.default_backend
    S.jax.default_backend = lambda: "gpu"
    _orig_uhf = S.update_he_fast
    def _spy(st, c):
        calls["fast"] += 1
        return _orig_uhf(st, c)
    S.update_he_fast = _spy

sim = Simulation(freq_max=12e9, domain=(0.024, 0.020, 0.004), dx=1e-3,
                 boundary=BoundarySpec(x=Boundary("pmc", "pmc"), y="pec", z="pec"))
sim.add_source(position=(0.007, 0.006, 0.002), component="ez")
sim.add_probe(position=(0.017, 0.013, 0.002), component="ez")
res = sim.run(n_steps=6000, skip_preflight=True)
ts = np.asarray(res.time_series)[:, 0]
modes = harminv(ts[800:], float(res.dt), 5e9, 12e9)
big = max(abs(m.amplitude) for m in modes)
fs = sorted((round(m.freq / 1e9, 4), round(abs(m.amplitude) / big, 3)) for m in modes if abs(m.amplitude) > 1e-2 * big)
print(("FAST(update_he_fast traced %d)" % calls["fast"]) if fake else "STD", "modes GHz (rel amp):", fs)
