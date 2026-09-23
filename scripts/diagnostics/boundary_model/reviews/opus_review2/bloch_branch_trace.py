"""Which E-update branch does an oblique (Bloch) TFSF run take with
subpixel smoothing on, and does that branch receive the Bloch phase?"""
import warnings, numpy as np
import rfx.core.yee as Y
import rfx.simulation as S
from rfx import Simulation, Box
warnings.simplefilter("ignore")
calls = []
for name in ("update_e_aniso", "update_e_aniso_inv", "update_e"):
    orig = getattr(S, name, None) or getattr(Y, name)
    def wrap(*a, _o=orig, _n=name, **k):
        calls.append((_n, "bloch" in k and k["bloch"] is not None, str(getattr(a[0].ex, "dtype", "?"))))
        return _o(*a, **k)
    setattr(S, name, wrap)
sim = Simulation(freq_max=10e9, domain=(0.030, 0.006, 0.002), dx=1e-3, boundary="cpml", cpml_layers=8)
sim.add_material("slab", eps_r=4.0)
sim.add(Box((0.0185, 0.0, 0.0), (0.0243, 0.006, 0.002)), material="slab")
sim.add_tfsf_source(f0=6e9, bandwidth=0.5, polarization="ez", angle_deg=30.0, waveform="modulated_gaussian")
sim.add_probe((0.012, 0.003, 0.001), "ez")
res = sim.run(n_steps=5, skip_preflight=True, subpixel_smoothing=True)
print(sorted(set(calls)))
