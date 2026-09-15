"""Choosing boundaries — the four patterns that cover most RF work.

Every simulation starts with one decision: what happens at the edge of the
domain? rfx expresses it with ``BoundarySpec`` (per-axis, per-face). This
tutorial builds the four patterns and runs a tiny simulation with each.

THE RULE:
  * **CPML** for OPEN structures — antennas, scattering, anything that
    radiates. The absorber emulates infinite free space.
  * **PEC** for CLOSED structures — cavities, shielded fixtures. Energy is
    conserved by design, so resonances ring forever; that is the physics,
    not a defect.
  * Do NOT mix the two roles: a cavity with one absorbing wall is neither a
    cavity nor an antenna, and its Q means nothing.
  * **PMC** is a symmetry tool: an E-field symmetric structure can be cut in
    half with a PMC wall on the symmetry plane — same physics, half the cells.
  * **periodic** for infinite arrays (metasurfaces, frequency-selective
    surfaces); the paired faces must be used together.

A BOUNDARY IS NOT A BODY.  ``BoundarySpec`` PEC is a condition on the domain's
own faces: E tangential to the face is zeroed at the face plane, and the wall
is infinite in its own plane.  Metal drawn INSIDE the domain is a different
operator with its own rule — the lattice ownership contract (#931).  A
conductor there is exactly one of three things, and the declaration says which:

  * a **sheet** — ``sim.add_thin_conductor(shape)`` — a footprint on ONE node
    plane, zero thickness, owning no cell.  The two in-plane E components on
    that plane are zeroed and the E through the plane stays live.  This is the
    declaration for etched copper: a ground plane, a patch, a microstrip
    trace.  35 um of copper on a 1.5 mm board is a sheet, and the mesh does not
    have to resolve it.
  * a **volume** — ``sim.add(shape, material="pec")`` — the primal cells whose
    CENTRES lie inside the shape, with every E edge incident to one of them
    zeroed.  A drawn slab therefore realizes walls on BOTH of its faces with
    the interior shorted, and realized thickness = drawn thickness.  This is
    the declaration for a plate, an iris, a post, a machined cavity wall.
  * a **wire** — ``PolylineWire`` thinner than half a cell — the E edges of a
    lattice path.

The two worked contrasts, on the same 1 mm mesh:

  * a ground plane declared ``add_thin_conductor(Box((0,0,z0), (Lx,Ly,z0)))``
    realizes ONE wall plane at ``z0`` and adds no cell to the stack-up;
  * a 2 mm aluminium plate declared ``add(Box((0,0,z0), (Lx,Ly,z0+2mm)),
    material="pec")`` realizes walls at ``z0`` AND ``z0+2mm`` with the two
    cells between them shorted.

Declaring foil as a Box is the mistake this contract exists to stop: it puts
the metal's own cell into the board, and a one-cell Box is a filled slab, not a
film.  A PEC Box whose drawn extent is thinner than one local cell is refused
outright, with the physical thickness named in the message.

Run as::

    python examples/tutorials/boundary_spec_demo.py

Each pattern prints the resolved spec (``BoundarySpec.to_dict()``) and the peak
|Ez| after a short run, so the configuration is visibly executing.  All four
sources pass ``amplitude_kind="current"`` so the drive means the same thing
under every boundary; without it the source convention varies per boundary and
the four peaks are not comparable.
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
    sim.add_source((0.005, 0.005, 0.0025), component="ez",
                   amplitude_kind="current")
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
    #    For a FINITE ground plane declare a SHEET inside an all-CPML domain
    #    instead — sim.add_thin_conductor(Box(...)) with the two corners on the
    #    ground plane's own z, i.e. zero thickness (see the header).  A
    #    boundary ground is infinite, which turns the antenna into a cavity and
    #    shifts its resonance.  Do not reach for a PEC Box here: a Box is a
    #    VOLUME and a ground plane is foil.
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
