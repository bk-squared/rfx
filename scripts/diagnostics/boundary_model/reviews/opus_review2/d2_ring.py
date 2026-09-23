"""D2 check: the R6 periodic ring with the periodic axis given L/dx nodes
(d2patch), and what the index map does with a coordinate on the wrap."""
import sys, warnings, time
import numpy as np
warnings.simplefilter("ignore")
if sys.argv[1] == "patched":
    import d2patch  # noqa: F401
from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.harminv import harminv
C0 = 299_792_458.0
dx = 1e-3
sim = Simulation(freq_max=30e9, domain=(0.024, 0.002, 0.002), dx=dx,
                 boundary=BoundarySpec(x="periodic", y="periodic", z="pec"), cpml_layers=0)
sim.add_source(position=(0.005, 0.001, 0.001), component="ez")
sim.add_probe(position=(0.017, 0.001, 0.001), component="ez")
t = time.time()
res = sim.run(n_steps=6000, skip_preflight=True)
ts = np.asarray(res.time_series)[:, 0]
modes = harminv(ts[1000:], float(res.dt), 8e9, 16e9)
big = max(abs(m.amplitude) for m in modes)
fs = sorted(round(m.freq / 1e9, 4) for m in modes if abs(m.amplitude) > 1e-2 * big)
print(f"{sys.argv[1]}: shape {res.state.ex.shape}; modes 8-16 GHz {fs}; c/24mm = {C0/0.024/1e9:.4f}; c/25mm = {C0/0.025/1e9:.4f} GHz ({time.time()-t:.0f} s)")
g = sim._build_grid()
for x in (0.0, 0.0235, 0.024):
    try:
        print(f"  position_to_index(x={x*1e3} mm) -> {g.position_to_index((x, 0.001, 0.001))}")
    except Exception as e:
        print(f"  position_to_index(x={x*1e3} mm) -> {type(e).__name__}: {str(e)[:120]}")
