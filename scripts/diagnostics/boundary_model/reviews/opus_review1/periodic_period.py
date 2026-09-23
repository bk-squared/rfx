"""Reviewer check: what period does a declared periodic axis realize?

Parallel-plate TEM line between PEC plates at z = 0 and z = Lz, closed on
itself along x by a periodic boundary (a ring resonator of circumference P).
The TEM ring resonances are f_n = n c / P.  Declared P = 24 mm:
f_1 = 12.491 GHz; if the wrap spans the node count (25 nodes, 25 mm):
f_1 = 11.992 GHz.  y periodic (2 mm), z PEC (2 mm).  run() on CPU.
"""
import numpy as np
from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.harminv import harminv

C0 = 299_792_458.0
for dx in (1e-3, 0.5e-3):
    sim = Simulation(freq_max=30e9, domain=(0.024, 0.002, 0.002), dx=dx,
                     boundary=BoundarySpec(x="periodic", y="periodic", z="pec"),
                     cpml_layers=0)
    sim.add_source(position=(0.005, 0.001, 0.001), component="ez")
    sim.add_probe(position=(0.017, 0.001, 0.001), component="ez")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=6000, skip_preflight=True)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    shape = None
    try:
        shape = res.state.ex.shape
    except Exception:
        pass
    modes = harminv(ts[1000:], dt, 8e9, 16e9)
    fs = sorted(m.freq for m in modes if abs(m.amplitude) > 1e-2 * max(abs(mm.amplitude) for mm in modes))
    print(f"dx={dx*1e3} mm shape={shape} modes in 8-16 GHz: {[round(f/1e9, 4) for f in fs]}"
          f"  c/24mm={C0/0.024/1e9:.4f}  c/(24mm+dx)={C0/(0.024+dx)/1e9:.4f} GHz")
