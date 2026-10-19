"""``pec_face_short_of_domain_wall``: a conductor stops one cell inside
the wall the mesh realizes (#931).

The nominal WR-90 plug remains 22.86 x 10.16 mm on a 1 mm mesh. The domain
lid is declared at 10.75 mm, inside its last solved cell, so both nearest
rounding and ceil give an 11 mm wall. The plug's 10.16 mm face rounds down
to 10 mm, leaving the one-cell vacuum slot this finding must identify.

Measured on the WR-90 waveguide-port case, 2026-09-07: that slot passed
|S21| 0.22-0.33 through what the fixture called a short,
and took the pec-short |S11| deficit from 0.0146 to 0.0560. Nothing said
so at build time; before #931 the node-half-open sampler happened to
include the top node and the question never arose.

The narrow (10.16 mm) axis is the one that fires here and the broad
(22.86 mm) axis is not: 22.86 rounds UP to the 23 mm wall (0.14 mm away,
under half a cell) while 10.16 rounds DOWN off the 11 mm wall (0.84 mm
away). The plug drawing and domain lid are separate declarations; the
finding reads their realized planes instead of comparing declared numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.preflight._common import PreflightWarning

MM = 1e-3
DX = 1 * MM
A_WG = 22.86 * MM          # WR-90 broad wall, declared
B_WG = 10.16 * MM          # WR-90 narrow wall, declared
B_DOMAIN = 10.75 * MM      # domain lid inside the 10..11 mm solved cell
A_REALIZED = float(np.ceil(A_WG / DX)) * DX     # 23 mm
B_REALIZED = float(np.ceil(B_WG / DX)) * DX     # 11 mm
CODE = "pec_face_short_of_domain_wall"


def _sim(guide_y: float, guide_z: float) -> Simulation:
    """A nominal WR-90 plug below a separately declared PEC domain lid."""
    sim = Simulation(
        freq_max=12.4e9, domain=(20 * MM, A_WG, B_DOMAIN), dx=DX,
        snap="declared",  # The drawn plug's open wall slot is intentional.
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=6)
    sim.add(Box((10 * MM, 0.0, 0.0), (12 * MM, guide_y, guide_z)),
            material="pec")
    return sim


def _preflight(sim):
    report = sim.preflight()
    warning, = report.by_code('wall_axis_off_node')
    assert warning.loc == 'z' and warning.severity == 'warning'
    return report


def test_plug_drawn_to_the_declared_cross_section_fires():
    """The cv11 drawing: the plug's z_hi face lands one node inside the wall."""
    rep = _preflight(_sim(A_WG, B_WG))
    hits = rep.by_code(CODE)
    assert len(hits) == 1, [str(h) for h in rep]
    text = str(hits[0])
    # exactly ONE face is short — the narrow axis; the broad axis rounded
    # up onto its wall and must not be reported.
    assert "1 PEC volume face(s)" in text, text
    assert "z_hi face" in text and "y_hi" not in text, text
    # both positions quoted in physical units, plus the gap
    assert "10mm" in text and "11mm" in text, text
    assert "ONE-CELL gap of 1mm" in text, text
    assert "REMEDY" in text and "REALIZED wall plane" in text, text
    assert hits[0].severity == "warning", hits[0].severity


def test_plug_drawn_to_the_realized_walls_does_not_fire():
    """The same plug drawn to ceil(declared/dx)*dx reaches both walls."""
    rep = _preflight(_sim(A_REALIZED, B_REALIZED))
    assert rep.by_code(CODE) == [], [str(h) for h in rep.by_code(CODE)]


def test_the_finding_reads_the_entry_s_own_realized_wall_planes():
    """The witness behind the message: the short drawing realizes its top
    at node 10 and the drawn-to-wall one at node 11, on the same mesh."""
    from rfx.boundaries.pec import realized_wall_planes

    tops = {}
    for label, (gy, gz) in (("declared", (A_WG, B_WG)),
                            ("realized", (A_REALIZED, B_REALIZED))):
        sim = _sim(gy, gz)
        with pytest.warns(PreflightWarning, match="wall axis 'z'") as caught:
            ctx = sim._campaign_ctx()
        assert [w.message.code for w in caught] == ['wall_axis_off_node']
        (entry,) = [e for e in ctx.entry_realizations() if e.kind == "volume"]
        edges = entry.edges(ctx.periodic, tuple(ctx.grid.shape))
        tops[label] = max(realized_wall_planes(edges, 2))
    wall = int(ctx.grid.interior[2].stop) - 1
    assert tops == {"declared": wall - 1, "realized": wall}, (tops, wall)


def test_ordinary_models_stay_silent():
    """The finding's false-positive boundary, measured.

    It is a WARNING on every model that has a wall to short to, so it has
    to be quiet on the models that do not: a body inside an absorbing box
    (no wall), and a body that neither reaches nor approaches a wall. A
    body deliberately parked one cell off a PEC wall DOES fire — that is a
    slot, and the message says so rather than guessing at intent.
    """
    # (a) patch antenna in a CPML box: every face absorbs, no wall exists
    cpml = Simulation(freq_max=10e9, domain=(30 * MM, 30 * MM, 12 * MM),
                      dx=0.5 * MM, boundary="cpml", cpml_layers=8)
    cpml.add_material("sub", eps_r=2.2)
    cpml.add(Box((0, 0, 0), (30 * MM, 30 * MM, 1.5 * MM)), material="sub")
    cpml.add(Box((10 * MM, 10 * MM, 1.5 * MM), (20 * MM, 20 * MM, 2 * MM)),
             material="pec")
    assert cpml.preflight().by_code(CODE) == []

    # (b) a post well inside a PEC cavity: nowhere near a wall
    post = Simulation(freq_max=10e9, domain=(20 * MM, 20 * MM, 20 * MM),
                      dx=1 * MM, boundary="pec")
    post.add(Box((8 * MM, 8 * MM, 5 * MM), (12 * MM, 12 * MM, 15 * MM)),
             material="pec")
    assert post.preflight().by_code(CODE) == []

    # (c) a slab drawn wall to wall: reaching IS the correct drawing
    walls = BoundarySpec(x=Boundary(lo="pec", hi="pec"),
                         y=Boundary(lo="pec", hi="pec"),
                         z=Boundary(lo="cpml", hi="cpml"))
    reaching = Simulation(freq_max=10e9, domain=(20 * MM, 20 * MM, 20 * MM),
                          dx=1 * MM, boundary=walls, cpml_layers=8)
    reaching.add(Box((0, 0, 8 * MM), (20 * MM, 20 * MM, 10 * MM)),
                 material="pec")
    assert reaching.preflight().by_code(CODE) == []

    # (d) the same slab parked one cell off both x walls: two faces short
    short = Simulation(freq_max=10e9, domain=(20 * MM, 20 * MM, 20 * MM),
                       dx=1 * MM, boundary=walls, cpml_layers=8)
    short.add(Box((1 * MM, 0, 8 * MM), (19 * MM, 20 * MM, 10 * MM)),
              material="pec")
    hits = short.preflight().by_code(CODE)
    assert len(hits) == 1, hits
    assert "2 PEC volume face(s)" in str(hits[0]), str(hits[0])
    assert "x_lo face" in str(hits[0]) and "x_hi face" in str(hits[0])
