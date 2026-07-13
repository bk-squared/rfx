"""Choosing boundaries — the four patterns you actually need.

Every simulation starts with one decision: what happens at the edge of the
domain? rfx expresses it with ``BoundarySpec`` (per-axis, per-face). This
tutorial builds the four patterns that cover almost all RF work and runs a
tiny simulation with each so you can see them execute.

THE RULE (worth memorising):
  * **CPML** for OPEN structures — antennas, scattering, anything that
    radiates. The absorber emulates infinite free space.
  * **PEC** for CLOSED structures — cavities, shielded fixtures. Energy is
    conserved by design; resonances ring forever (that is the physics, not
    a bug).
  * Do NOT mix the two roles: a cavity with one absorbing wall is neither a
    cavity nor an antenna, and its Q means nothing.
  * **PMC** is a symmetry tool: an E-field symmetric structure can be cut in
    half with a PMC wall on the symmetry plane — same physics, half the cells.
  * **periodic** for infinite arrays (metasurfaces / frequency-selective
    surfaces); the paired faces must be used together.

Run as::

    python examples/tutorials/boundary_spec_demo.py

Each pattern prints the resolved spec (``BoundarySpec.to_dict()``) and the
peak |Ez| after a short run, so you can confirm the configuration executed.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


DOMAIN = (0.01, 0.01, 0.005)   # 10 x 10 x 5 mm toy box
DX = 0.5e-3


def _run_and_report(label: str, spec: BoundarySpec) -> None:
    """Build a Simulation with ``spec``, run briefly, print what resolved."""
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary=spec)
    sim.add_source((0.005, 0.005, 0.0025), component="ez")
    sim.add_probe((0.006, 0.006, 0.0025), component="ez")
    result = sim.run(n_steps=120, compute_s_params=False)
    peak = float(np.max(np.abs(np.asarray(result.time_series))))
    print(f"[{label:>18}] spec = {spec.to_dict()}")
    print(f"[{label:>18}] peak |Ez| after 120 steps = {peak:.3e}")


if __name__ == "__main__":
    # 1) OPEN box — the default for anything that radiates.
    _run_and_report("open box", BoundarySpec.uniform("cpml"))

    # 2) Antenna over a ground plane — one PEC face (the ground), open
    #    everywhere else. Per-face control uses Boundary(lo=..., hi=...).
    #    (For a FINITE ground plane, prefer a PEC Box inside an all-CPML
    #    domain instead — an infinite boundary ground turns the antenna
    #    into a cavity and shifts its resonance.)
    _run_and_report("ground plane", BoundarySpec(
        x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")))

    # 3) CLOSED cavity — all PEC. Use with harminv for resonances; do not
    #    expect fields to decay (energy is conserved).
    _run_and_report("closed cavity", BoundarySpec.uniform("pec"))

    # 4) Periodic unit cell — infinite array in x/y, open in z.
    _run_and_report("periodic cell", BoundarySpec(
        x="periodic", y="periodic", z="cpml"))

    print("\nLegacy note: the old kwargs (boundary='cpml' + pec_faces={...} /")
    print("set_periodic_axes) still work but emit DeprecationWarning; new code")
    print("should construct a BoundarySpec as above.")
