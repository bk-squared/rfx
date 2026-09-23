"""Reviewer check: is 'E[0]; H[0]' on a PMC face physically different from 'H[0]'?

PMC x faces, PEC y/z, 24x20x16 mm, dx = 1 mm, interior Ez source and probes.
Arm E+H: main (apply_pec on xyz, PMC H zeroing at index 0 / N-2).
Arm H  : apply_pec restricted to y,z (no electric wall on the PMC x faces).
Also: probe on the x_lo face node plane (x = 0) in both arms.
"""
import numpy as np
import rfx.simulation as S
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.boundaries.pec import apply_pec as _orig

def build():
    sim = Simulation(freq_max=12e9, domain=(0.024, 0.020, 0.016), dx=1e-3,
                     boundary=BoundarySpec(x=Boundary("pmc", "pmc"), y="pec", z="pec"))
    sim.add_source(position=(0.007, 0.006, 0.008), component="ez")
    sim.add_probe(position=(0.017, 0.013, 0.008), component="ez")
    sim.add_probe(position=(0.0, 0.013, 0.008), component="ez")   # x_lo face node plane
    sim.add_probe(position=(0.024, 0.013, 0.008), component="ez") # x_hi face node plane
    return sim

def go(patch, entry):
    if patch:
        S.apply_pec = lambda st, axes="xyz": _orig(st, axes="".join(a for a in axes if a != "x"))
    try:
        sim = build()
        r = sim.forward(n_steps=600, skip_preflight=True) if entry == "forward" else sim.run(n_steps=600, skip_preflight=True)
        return np.asarray(r.time_series)
    finally:
        S.apply_pec = _orig

for entry in ("forward", "run"):
    a = go(False, entry); b = go(True, entry)
    print(entry, "interior probe bit-identical:", np.array_equal(a[:, 0], b[:, 0]),
          "max|interior|", float(np.abs(a[:, 0]).max()),
          "| x_lo plane max E+H / H:", float(np.abs(a[:, 1]).max()), float(np.abs(b[:, 1]).max()),
          "| x_hi plane max E+H / H:", float(np.abs(a[:, 2]).max()), float(np.abs(b[:, 2]).max()))
