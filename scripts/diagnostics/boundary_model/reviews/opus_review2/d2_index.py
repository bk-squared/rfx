import sys, warnings
warnings.simplefilter("ignore")
if sys.argv[1] == "patched":
    import d2patch  # noqa: F401
from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
sim = Simulation(freq_max=30e9, domain=(0.024, 0.002, 0.002), dx=1e-3,
                 boundary=BoundarySpec(x="periodic", y="periodic", z="pec"), cpml_layers=0)
g = sim._build_grid()
print(sys.argv[1], "shape", g.shape, "cells sum x =", g.cells("x").sum()*1e3, "mm; node_of(last) =", g.node_of("x", g.nx-1)*1e3, "mm")
for x in (0.0, 0.0235, 0.024):
    try:
        print(f"  position_to_index(x={x*1e3:.1f} mm) -> {g.position_to_index((x, 0.001, 0.001))}")
    except Exception as e:
        print(f"  position_to_index(x={x*1e3:.1f} mm) -> {type(e).__name__}")
try:
    sim.add_probe(position=(0.024, 0.001, 0.001), component="ez"); print("  add_probe at x = L accepted at add time")
except Exception as e:
    print("  add_probe at x = L:", type(e).__name__, str(e)[:100])
