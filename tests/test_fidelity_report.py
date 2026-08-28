"""Input-fidelity report (rfx.fidelity): declared-vs-realized, input units only.

Pins the three finding classes the tool exists for, on a fixture built to
contain each trap exactly once:

* a one-cell PEC ground with dielectric ABOVE it — the sheet's own cell is
  vacuum inside the cavity ("sheet-own-cell-live", the #693/#702 class);
* a patch with deliberately off-lattice x-faces ("off-lattice-face");
* a dielectric later overwritten by another entity
  ("materialization-overridden");
* an entity thinner than any cell ("absent" — the #369 silent-drop class).

Also locks the non-findings: exactly-on-lattice faces report 0.0 um
residuals (the report must not cry wolf), and coordinates are DOMAIN
coordinates (the CPML pad offset is compensated).
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box


def _fixture():
    sim = Simulation(freq_max=15e9, domain=(10e-3, 10e-3, 6e-3), dx=0.5e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("sub", eps_r=3.38, sigma=0.0)
    sim.add_material("override", eps_r=2.2, sigma=0.0)
    sim.add(Box((0, 0, 2.0e-3), (10e-3, 10e-3, 2.5e-3)), material="pec")     # [0] ground sheet
    sim.add(Box((0, 0, 2.5e-3), (10e-3, 10e-3, 4.0e-3)), material="sub")     # [1] substrate
    sim.add(Box((3.13e-3, 3.0e-3, 4.0e-3), (6.87e-3, 7.0e-3, 4.5e-3)),
            material="pec")                                                  # [2] patch, off-lattice x
    sim.add(Box((1.0e-3, 1.0e-3, 2.5e-3), (3.0e-3, 3.0e-3, 3.0e-3)),
            material="override")                                             # [3] overwrites part of [1]
    sim.add_material("thinfilm", eps_r=8.0, sigma=0.0)
    sim.add(Box((4.0e-3, 4.0e-3, 5.10e-3), (6.0e-3, 6.0e-3, 5.15e-3)),
            material="thinfilm")   # [4] 50 um dielectric film mid-cell -> absent
            # (a thin PEC box would be SNAPPED to a sheet by the thin-conductor
            #  realization — correct behaviour, so the absent class needs a
            #  dielectric, which has no snap path — and even that lands in
            #  its containing cell, realized 10x thicker: entity [5] below is
            #  the true absent case, declared outside the domain)
    sim.add(Box((4.0e-3, 4.0e-3, 9.0e-3), (6.0e-3, 6.0e-3, 9.2e-3)),
            material="thinfilm")   # [5] beyond even the CPML pad -> absent
    return sim



def _geo(rep, i):
    """The report leads with a domain pseudo-entity and may append an
    out-of-scope note, so tests address entities by name."""
    for item in rep:
        if item["entity"].startswith(f"geometry[{i}]"):
            return item
    raise AssertionError(f"geometry[{i}] missing from report: "
                         f"{[x['entity'] for x in rep]}")


def _tc(rep, i):
    for item in rep:
        if item["entity"].startswith(f"thin_conductor[{i}]"):
            return item
    raise AssertionError("thin_conductor missing")

def _kinds(item):
    return [f["kind"] for f in item["findings"]]


def test_finding_classes_are_detected():
    rep = _fixture().fidelity_report(print_report=False)

    ground = _geo(rep, 0)
    assert "one-plane sheet" in ground["realization"]
    assert "sheet-own-cell-live" in _kinds(ground)
    assert ground["own_cell_eps_r"][1] <= 1.0 + 1e-6, "ground own-cell must read vacuum"

    patch = _geo(rep, 2)
    assert "off-lattice-face" in _kinds(patch)
    ax_x = patch["axes"][0]
    assert max(ax_x["face_residual_um"]) > 100.0

    sub = _geo(rep, 1)
    assert "materialization-overridden" in _kinds(sub)
    assert sub["assembled_matches_declared_frac"] < 0.999
    assert min(sub["assembled_eps_r"]) < 3.0   # the 2.2 override is visible

    film = _geo(rep, 4)
    # Rasterization convention: a sub-cell box lands in its containing cell,
    # so a 50 um film is realized 500 um thick — caught as extent/face
    # distortion (the tool's job), not as absence.
    assert film["n_cells"] > 0
    assert film["axes"][2]["realized_extent_um"] > 5 * film["axes"][2]["declared_extent_um"]
    # A sub-cell film is reported against the CELL, not as a percentage of its
    # own thickness (the 2026-08-28 convention fix).
    assert "sub-cell-placement" in _kinds(film)

    clamped = _geo(rep, 5)
    # Measured rasterization behaviour: an out-of-range box CLAMPS to the
    # boundary cells rather than vanishing — the report makes that visible as
    # a face residual of millimetres (declared z 9.0 mm vs realized at the
    # array edge). The "absent" branch stays in the tool defensively for
    # shapes whose mask is genuinely empty.
    assert clamped["n_cells"] > 0
    # Classified as a domain clip (+ absorber overlap), NOT as a lattice
    # misalignment: comparing a deliberately oversized body against its
    # un-clipped declaration produced 99%-residual noise in the crossval
    # sweep, so residuals are now measured against the clipped declaration.
    assert "clipped-by-domain" in _kinds(clamped)
    assert "inside-absorber" in _kinds(clamped)


def test_exact_faces_report_zero_residual_in_domain_coords():
    rep = _fixture().fidelity_report(print_report=False)
    ground = _geo(rep, 0)
    for ax in ground["axes"]:
        assert max(ax["face_residual_um"]) < 1e-6, (
            f"exact face read nonzero residual on {ax['axis']}: {ax}")
    # domain coordinates, not padded-array coordinates:
    assert abs(ground["axes"][2]["realized_um"][0] - 2000.0) < 1e-6


def test_remedies_are_present_and_mechanical():
    rep = _fixture().fidelity_report(print_report=False)
    for item in rep:
        for f in item["findings"]:
            assert f.get("remedy"), f"finding without a remedy: {f}"
            assert "df/f" not in f["detail"], (
                "fidelity findings must never predict result-side impact")


# ---------------------------------------------------------------------------
# Adversarial traps (2026-08-27): models built to be WRONG in ways a user
# would not notice. Every one of these was a MISS on the first implementation
# — the tool reported "all clean" for models with invisible metal, silently
# dropped conductivity, and conductors inside the absorber.
# ---------------------------------------------------------------------------


def _plain(dx=1e-3):
    return Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=dx,
                      boundary="cpml", cpml_layers=4)


def test_thin_conductor_entities_are_audited():
    """add_thin_conductor is a second declaration surface; auditing only
    sim._geometry returned an EMPTY report — a false all-clear."""
    sim = _plain()
    sim.add_thin_conductor(Box((2e-3, 2e-3, 5e-3), (8e-3, 8e-3, 5.035e-3)),
                           sigma_bulk=5.8e7)
    rep = sim.fidelity_report(print_report=False)
    item = _tc(rep, 0)
    assert "sheet" in item["realization"]


def test_conductivity_drop_is_caught_when_eps_is_unchanged():
    """A later entity with the SAME eps but different sigma is invisible in
    eps alone — sigma is a declared input and gets its own check."""
    sim = _plain()
    sim.add_material("lossy", eps_r=4.0, sigma=5.0)
    sim.add_material("clean", eps_r=4.0, sigma=0.0)
    sim.add(Box((2e-3, 2e-3, 2e-3), (8e-3, 8e-3, 8e-3)), material="lossy")
    sim.add(Box((2e-3, 2e-3, 2e-3), (8e-3, 8e-3, 5e-3)), material="clean")
    item = _geo(sim.fidelity_report(print_report=False), 0)
    assert item["assembled_matches_declared_frac"] == 1.0, "eps is unchanged"
    assert "sigma-mismatch" in [f["kind"] for f in item["findings"]]
    assert item["assembled_sigma_matches_declared_frac"] < 0.6


def test_dielectric_claimed_by_a_later_conductor_is_caught():
    """PEC is a mask, not an eps change: the dielectric's cells still read the
    declared eps while the conductor owns them."""
    sim = _plain()
    sim.add_material("sub", eps_r=9.0, sigma=0.0)
    sim.add(Box((3e-3, 3e-3, 3e-3), (7e-3, 7e-3, 7e-3)), material="sub")
    sim.add(Box((3e-3, 3e-3, 3e-3), (7e-3, 7e-3, 7e-3)), material="pec")
    kinds = [f["kind"] for f in _geo(sim.fidelity_report(print_report=False), 0)["findings"]]
    assert "claimed-by-conductor" in kinds


def test_body_inside_the_absorber_is_caught():
    sim = _plain()
    sim.add(Box((-3e-3, 2e-3, 2e-3), (4e-3, 8e-3, 8e-3)), material="pec")
    kinds = [f["kind"] for f in _geo(sim.fidelity_report(print_report=False), 0)["findings"]]
    assert "inside-absorber" in kinds


def test_inert_two_plane_request_is_reported():
    sim = _plain()
    sim.add(Box((2e-3, 2e-3, 3e-3), (8e-3, 8e-3, 6e-3)), material="pec",
            two_plane=True)
    item = _geo(sim.fidelity_report(print_report=False), 0)
    assert "volumetric" in item["realization"]
    assert "two-plane-inert" in [f["kind"] for f in item["findings"]]


def test_multi_axis_thin_body_names_every_thin_axis():
    sim = _plain()
    sim.add(Box((2e-3, 5e-3, 5e-3), (8e-3, 5.5e-3, 5.5e-3)), material="pec")
    item = _geo(sim.fidelity_report(print_report=False), 0)
    assert "y+z" in item["realization"], item["realization"]


def test_dispersive_material_states_that_poles_are_not_verified():
    """Honesty over silence: the report audits instantaneous eps/sigma, so a
    dispersive material must say what it does NOT check."""
    sim = _plain()
    from rfx.api._spec import DebyePole
    sim.add_material("debye", eps_r=2.0, debye_poles=[DebyePole(delta_eps=3.0, tau=1e-11)])
    sim.add(Box((2e-3, 2e-3, 2e-3), (8e-3, 8e-3, 8e-3)), material="debye")
    kinds = [f["kind"] for f in _geo(sim.fidelity_report(print_report=False), 0)["findings"]]
    assert "dispersion-not-audited" in kinds


# ---------------------------------------------------------------------------
# Readout conventions (2026-08-28). Both of these were shipped wrong and were
# caught by an external review of the issues this tool's output produced: a
# sphere sitting dead centre of the grid it is solved on was reported as
# "offset half a cell", and a 17 um sheet landing in its cell was reported as
# "181% of the declared extent".
# ---------------------------------------------------------------------------


def test_curved_body_offset_reports_the_convention_free_midpoint():
    from rfx.geometry.csg import Sphere
    sim = Simulation(freq_max=10e9, domain=(20e-3, 20e-3, 20e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add(Sphere(center=(10.5e-3, 10.5e-3, 10.5e-3), radius=4e-3),
            material="pec")
    item = _geo(sim.fidelity_report(print_report=False), 0)
    offs = [f for f in item["findings"] if f["kind"] == "bbox-offset"]
    assert offs, "a half-cell-shifted sphere should still be reported"
    f = offs[0]
    assert "midpoint_shift_um" in f, (
        "a node-sampled mask compared against cell edges carries up to half a "
        "cell of pure convention; the midpoint shift must be reported so the "
        "reader can separate it from a real displacement")
    assert "readout convention" in f["detail"]


def test_sub_cell_body_is_measured_against_the_cell_not_its_own_thickness():
    sim = Simulation(freq_max=40e9, domain=(2e-3, 2e-3, 2e-3), dx=50e-6,
                     boundary="cpml", cpml_layers=4)
    sim.add(Box((0.5e-3, 0.5e-3, 0.98e-3), (1.5e-3, 1.5e-3, 0.997e-3)),
            material="pec")            # 17 um sheet in a 50 um cell
    item = _geo(sim.fidelity_report(print_report=False), 0)
    kinds = _kinds(item)
    assert "sub-cell-placement" in kinds
    f = [x for x in item["findings"] if x["kind"] == "sub-cell-placement"][0]
    assert "cell" in f["detail"] and "not a meaningful measure" in f["detail"]
    # and the z axis must NOT also produce a percentage-of-extent finding
    z_off = [x for x in item["findings"]
             if x["kind"] == "off-lattice-face" and x.get("axis") == "z"]
    assert not z_off, (
        "a sub-cell body must not be reported as a percentage of its own "
        f"thickness as well: {z_off}")


# ---------------------------------------------------------------------------
# #729 site 1: the DOMAIN row must count grid CELLS, not NODES.
#
# `_node_arrays` builds one array entry per NODE (grid.shape[a] entries for
# n cells, since an n-cell span needs n+1 nodes — rfx/grid.py's fence-post
# comment). Summing that whole array as if each entry were a cell inflated
# an exactly-commensurate domain's "realized" length by one dx on every
# axis and raised a false [domain-extent-quantized] finding, while the
# entity row for a Box filling the identical span (which reads `nodes`,
# not `sizes`) was correct. The domain row is the ONLY row for
# cavity/waveguide models where the box IS the geometry (cv09, cv14, cv21),
# so this defect landed exactly where the report is trusted most.
# ---------------------------------------------------------------------------


def _dom(rep):
    for item in rep:
        if item["entity"].startswith("domain"):
            return item
    raise AssertionError(f"domain pseudo-entity missing from report: "
                         f"{[x['entity'] for x in rep]}")


def test_commensurate_domain_reports_zero_domain_findings():
    """An exactly-commensurate domain must raise NO domain finding, and its
    realized extent must agree with the entity row for a Box filling it."""
    sim = Simulation(freq_max=10e9, domain=(20e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="pec")
    sim.add_material("m", eps_r=2.0)
    sim.add(Box((0, 0, 0), (20e-3, 10e-3, 10e-3)), material="m")
    rep = sim.fidelity_report(print_report=False)
    dom, geo = _dom(rep), _geo(rep, 0)
    assert [f["kind"] for f in dom["findings"]] == []
    for a, declared_um in enumerate((20000.0, 10000.0, 10000.0)):
        assert abs(dom["axes"][a]["realized_extent_um"] - declared_um) < 1e-3
        assert abs(dom["axes"][a]["realized_extent_um"]
                   - geo["axes"][a]["realized_extent_um"]) < 1e-3


def test_incommensurate_domain_reports_the_ceil_realized_length():
    """WR-90 at dx=1mm. The expected extents below are hand-computed from
    the grid's own construction rule (rfx/grid.py:151, "N cells need N+1
    nodes"): ceil(22.86/1)=23 cells -> 23.000 mm, ceil(10.16/1)=11 -> 11.000
    mm, 30/1=30 -> 30.000 mm exactly. Literals, not a re-run of the code
    under test: a node-count regression reports 24/12/31 and fails here."""
    sim = Simulation(freq_max=12e9, domain=(22.86e-3, 10.16e-3, 30e-3),
                     dx=1e-3, boundary="pec")
    dom = _dom(sim.fidelity_report(print_report=False))
    assert dom["axes"][0]["n_cells"] == 23
    assert abs(dom["axes"][0]["realized_extent_um"] - 23000.0) < 1e-3
    assert abs(dom["axes"][1]["realized_extent_um"] - 11000.0) < 1e-3
    assert abs(dom["axes"][2]["realized_extent_um"] - 30000.0) < 1e-3
    assert [(f["kind"], f["axis"]) for f in dom["findings"]] == [
        ("domain-extent-quantized", "x"), ("domain-extent-quantized", "y")]


def test_domain_row_is_cells_on_every_pad_mesh_and_uniformity():
    """Enumerate-and-classify: for every (boundary, pad symmetry, mesh
    resolution, uniformity, dimensionality) the MESH extent must be
    ceil(L/dx)*dx, never grid.shape[a]*dx (the node-count bug's
    signature). This checks ``mesh_extent_um`` (the node-to-node span),
    not ``realized_extent_um`` -- the "asymmetric pads" case below puts
    PMC on y_hi, and since the Q2 PMC-plane-convention change (#722 ninth
    surface) ``realized_extent_um`` on a PMC-faced axis is dx/2 SMALLER
    than the mesh span by design (rfx/fidelity.py; see
    test_pmc_face_reports_the_half_cell_realized_wall below for that
    convention's own test). Checking the mesh span here keeps this test's
    original purpose -- cell-vs-node counting, issue #729 -- decoupled
    from the PMC convention it would otherwise be conflated with."""
    from rfx.boundaries.spec import Boundary, BoundarySpec

    L = (20e-3, 10e-3, 10e-3)
    asym = BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                        y=Boundary(lo="cpml", hi="pmc"), z="pec")
    cases = []
    for dx in (1e-3, 0.5e-3, 0.25e-3):           # mesh 1x / 2x / 4x
        cases.append((f"pec dx={dx}", dict(boundary="pec", dx=dx)))
        cases.append((f"cpml4 dx={dx}",
                      dict(boundary="cpml", cpml_layers=4, dx=dx)))
    cases.append(("nonuniform dz", dict(boundary="pec", dx=1e-3,
                                        dz_profile=np.full(10, 1e-3))))
    cases.append(("asymmetric pads", dict(boundary=asym, cpml_layers=6,
                                          dx=1e-3)))
    cases.append(("2d_tmz", dict(boundary="pec", dx=1e-3, mode="2d_tmz")))

    bad = []
    for tag, kw in cases:
        dxv = kw["dx"]
        sim = Simulation(freq_max=10e9, domain=L, **kw)
        dom = _dom(sim.fidelity_report(print_report=False))
        for a in range(3):
            if tag == "2d_tmz" and a == 2:
                # z is not solved in 2D (grid.nz == 1): must not raise a
                # domain-extent-quantized finding on it.
                if any(f["axis"] == "z" for f in dom["findings"]):
                    bad.append((tag, "z", "unexpected finding on z"))
                continue
            want_cells = int(np.ceil(L[a] / dxv))
            got_cells = dom["axes"][a]["n_cells"]
            got_um = dom["axes"][a]["mesh_extent_um"]
            if (got_cells != want_cells
                    or abs(got_um - want_cells * dxv * 1e6) > 1e-2):
                bad.append((tag, "xyz"[a], got_cells, want_cells, got_um))
    assert not bad, f"domain row counted NODES as cells: {bad}"


def test_pmc_face_reports_the_half_cell_realized_wall():
    """Q2 (#722 ninth surface, decided 2026-08-28): apply_pmc_faces zeros
    H_tan a half-cell INSIDE the declared wall on every PMC face
    (rfx/boundaries/pmc.py, pinned by tests/test_boundary_pmc_hi_faces.py
    -- untouched here). The domain row must report that realized H_tan
    wall, not the raw mesh line, with a finding naming the convention so a
    PMC-mirror script cannot ship the offset silently."""
    from rfx.boundaries.spec import Boundary, BoundarySpec

    dx = 1e-3
    sim = Simulation(
        freq_max=10e9, domain=(20e-3, 10e-3, 10e-3), dx=dx,
        boundary=BoundarySpec(
            x=Boundary(lo="pec", hi="pmc"),
            y=Boundary(lo="pmc", hi="pec"),
            z="pec",
        ),
        cpml_layers=0,
    )
    dom = _dom(sim.fidelity_report(print_report=False))
    x, y, z = dom["axes"]

    # x_hi is pmc: the realized hi wall sits dx/2 (500 um) inside the
    # 20000.0 um mesh line; x_lo (pec) is untouched.
    assert x["realized_um"][0] == 0.0
    assert abs(x["realized_um"][1] - (20000.0 - 500.0)) < 1e-6
    assert abs(x["mesh_extent_um"] - 20000.0) < 1e-6
    assert abs(x["face_residual_um"][1] - 500.0) < 1e-6

    # y_lo is pmc: the realized lo wall sits dx/2 inside 0.0, i.e. at
    # +500 um; y_hi (pec) is untouched.
    assert abs(y["realized_um"][0] - 500.0) < 1e-6
    assert abs(y["realized_um"][1] - 10000.0) < 1e-6
    assert abs(y["face_residual_um"][0] - 500.0) < 1e-6

    # z has no pmc face: unaffected, no finding.
    assert abs(z["realized_um"][0] - 0.0) < 1e-6
    assert abs(z["realized_um"][1] - 10000.0) < 1e-6

    kinds_by_axis = {f["axis"]: f["kind"] for f in dom["findings"]}
    assert kinds_by_axis.get("x") == "pmc-wall-half-cell-inside"
    assert kinds_by_axis.get("y") == "pmc-wall-half-cell-inside"
    assert "z" not in kinds_by_axis
    for f in dom["findings"]:
        assert "CONVENTION" in f["detail"]


def test_pmc_face_finding_skips_the_axis_not_solved_2d_z():
    """The PMC convention finding must not fire on an axis the solve does
    not have (2D z) -- guarded the same way domain-extent-quantized is."""
    from rfx.boundaries.spec import Boundary, BoundarySpec

    sim = Simulation(
        freq_max=10e9, domain=(20e-3, 10e-3, 10e-3), dx=1e-3,
        boundary=BoundarySpec(x="pec", y="pec",
                              z=Boundary(lo="pmc", hi="pmc")),
        mode="2d_tmz", cpml_layers=0,
    )
    dom = _dom(sim.fidelity_report(print_report=False))
    assert not any(f["axis"] == "z" for f in dom["findings"])


def test_2d_not_solved_axis_note_reaches_the_printed_report(capsys):
    """A dict-only note is not a mitigation. Without it rendered, the 2D z
    row prints `extent 10000.0 -> 0.0 um` with no finding beside it — a
    silent all-clear over a 100% displayed gap. This fails if `_print`
    stops rendering `ax['note']`, or if the note stops naming the declared
    length it is declining to compare."""
    sim = Simulation(freq_max=10e9, domain=(20e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="pec", mode="2d_tmz")
    dom = _dom(sim.fidelity_report(print_report=True))
    printed = capsys.readouterr().out
    z = dom["axes"][2]
    assert "axis-not-solved" in z.get("note", ""), z
    assert "10000.0 um" in z["note"], (
        "the note must quote the declared Lz it is not comparing")
    assert z["note"] in printed, (
        "the note is in the returned dict but not in the printed report:\n"
        + printed)
    # the x/y rows ARE compared, so they must NOT carry a note
    assert "note" not in dom["axes"][0] and "note" not in dom["axes"][1]
