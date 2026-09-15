"""Does the waveguide S-matrix lane realize BOTH faces of a 1-cell PEC slab?

Written for the cv18/cv19 migration under the #931 lattice-ownership contract.
Until stage C, ``compute_waveguide_s_matrix`` folded the PEC CELL mask into
``sigma = 1e10`` for the device run — a fourth realization of the geometry that
damped only the components indexed by the occupied cell, so a one-cell iris got
its lower face and never its far one. cv18 and cv19 both reach that lane, and
their whole migration arithmetic (drawn == realized) is only true if the lane
applies the realized edge set. This probe is the check, in two parts:

  * build only — the realized x wall planes of a 1-cell slab must be {k, k+1};
  * 200 steps through ``rfx.simulation.run`` with the same ``pec_edge_masks``
    the lane passes — E tangential must be exactly zero on BOTH planes and
    nonzero three planes upstream.

Run: ``python scripts/diagnostics/probe_wr90_lane_one_cell_slab_faces.py``
(~15 s CPU). Exits nonzero on failure.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.simulation import run as run_simulation, SourceSpec
import jax.numpy as jnp

A, B = 22.86e-3, 10.16e-3
cells = 30; DX = A / cells; glen_c = 40
sim = Simulation(freq_max=13.6e9, domain=(glen_c*DX, A, B), dx=DX,
                 boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                       y=Boundary(lo="pec", hi="pec"),
                                       z=Boundary(lo="pec", hi="pec")),
                 cpml_layers=8)
big = 1.0; i0 = 20
sim.add(Box((i0*DX, -big, -big), ((i0+1)*DX, 6*DX, big)), material="pec")
sim.add(Box((i0*DX, A-6*DX, -big), ((i0+1)*DX, big, big)), material="pec")
grid = sim._build_grid()
sheets, wires = [], []
asm = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
materials, pec_mask = asm[0], asm[3]
edges = realized_pec_edge_masks(pec_mask, sheets=tuple(sheets), wires=tuple(wires),
                                periodic=sim._periodic_flags())
xn = np.asarray(coords_from_uniform_grid(grid).x)
off = int(np.argmin(np.abs(xn - i0*DX)))
print("wall planes (fin region):",
      realized_wall_planes(edges, 0, region=(slice(None), slice(0, 6), slice(None))))

n = 200          # enough to fill the slab neighbourhood; no S-parameter is claimed
t = np.arange(n) * 1e-12
wf = jnp.asarray(np.sin(2*np.pi*10.3e9*np.arange(n)*0.5e-12), dtype=jnp.float32)
src = [SourceSpec(i=off-4, j=3, k=7, component="ey", waveform=wf)]
t0 = time.time()
res = run_simulation(grid, materials, n, boundary="cpml",
                     cpml_axes="x", pec_axes="yz",
                     sources=src, pec_edge_masks=edges)
print("200 steps in %.1fs" % (time.time()-t0))
ey = np.asarray(res.state.ey); ez = np.asarray(res.state.ez)
fin = slice(0, 6)
for p, label in ((off, "near face"), (off+1, "far face")):
    print(f"{label} x={p}: max|Ey|={np.abs(ey[p, fin, :]).max():.3e} "
          f"max|Ez|={np.abs(ez[p, fin, :]).max():.3e}")
free = off - 3
print(f"free plane x={free}: max|Ey|={np.abs(ey[free, fin, :]).max():.3e}")
assert np.abs(ey[off, fin, :]).max() == 0.0 and np.abs(ez[off, fin, :]).max() == 0.0
assert np.abs(ey[off+1, fin, :]).max() == 0.0 and np.abs(ez[off+1, fin, :]).max() == 0.0
assert np.abs(ey[free, fin, :]).max() > 0.0
print("FIELD WITNESS OK: both faces of the 1-cell slab are walls in this lane")
