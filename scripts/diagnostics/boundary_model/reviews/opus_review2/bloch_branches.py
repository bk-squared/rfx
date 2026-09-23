"""Reviewer check: is the Bloch (oblique-periodic) phase, which lives in the
shared difference operator today, seen by every E-update branch?

Oblique TFSF (angle 30 deg, the #404 complex Bloch path) over a slab that is
(a) plain dielectric -> update_e (sees bloch), (b) Debye -> update_e_debye
(own curl, no bloch argument). If (b) is accepted, the E update wraps the
periodic axis with a plain roll while the H update applies the phase.
Probe: record whether run() accepts it, and the field growth over 400 steps.
"""
import sys
import warnings
import numpy as np
from rfx import Simulation, Box
from rfx.materials.debye import DebyePole

kind = sys.argv[1]
warnings.simplefilter("ignore")
sim = Simulation(freq_max=10e9, domain=(0.030, 0.006, 0.002), dx=1e-3,
                 boundary="cpml", cpml_layers=8)
if kind == "debye":
    pass
if kind == "debye":
    sim.add_material("slab", eps_r=2.0, debye_poles=[DebyePole(2.0, 1e-11)])
else:
    sim.add_material("slab", eps_r=4.0)
sim.add(Box((0.018, 0.0, 0.0), (0.024, 0.006, 0.002)), material="slab")
sim.add_tfsf_source(f0=6e9, bandwidth=0.5, polarization="ez", angle_deg=30.0,
                    waveform="modulated_gaussian")
sim.add_probe((0.012, 0.003, 0.001), "ez")
try:
    res = sim.run(n_steps=400, skip_preflight=True, **({"subpixel_smoothing": True} if kind == "smooth" else {}))
    ts = np.abs(np.asarray(res.time_series)[:, 0])
    print(f"{kind}: ACCEPTED; max|Ez| first half {ts[:200].max():.3e}, last 50 steps {ts[-50:].max():.3e}; finite={np.all(np.isfinite(ts))}")
except Exception as e:
    print(f"{kind}: REFUSED {type(e).__name__}: {str(e)[:300]}")
