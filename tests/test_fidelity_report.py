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
    assert "off-lattice-face" in _kinds(film)

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
# Domain extent: N interior CELLS, not N+1 interior NODES (2026-08-28).
# The first implementation summed the node-count slice of the cell-size array,
# so every domain read one cell too long -- cv11's WR-90 guide reported
# 24000/12000 um where #722 measures 23000/11000, and
# differentiable_s11_design reported 26000/14000 um where #738 measures
# 24000/12000. The gate that pins these numbers is about fence-post errors,
# so its own fence post has to be right.
# ---------------------------------------------------------------------------

def _domain(sim):
    for item in sim.fidelity_report(print_report=False):
        if item["entity"].startswith("domain"):
            return item
    raise AssertionError("domain row missing from report")


def test_commensurate_domain_reports_exactly_its_declared_extent():
    """10 mm at dx = 1 mm is 10 cells bounded by 11 nodes -> 10.000 mm."""
    dom = _domain(_plain(dx=1e-3))  # domain=(10 mm)^3
    for ax in dom["axes"]:
        assert ax["n_cells"] == 10, ax
        assert abs(ax["realized_extent_um"] - 10000.0) < 1e-6, ax
        assert abs(ax["declared_extent_um"] - 10000.0) < 1e-6, ax
    assert [f for f in dom["findings"] if f["kind"] == "domain-extent-quantized"] == [], (
        "an exactly commensurate domain must raise no quantization finding")


def test_incommensurate_domain_still_reports_the_rounded_up_extent():
    """The finding must still fire when the cell size does NOT divide the
    declared length -- 10.5 mm at dx = 1 mm is 11 cells = 11.000 mm."""
    sim = Simulation(freq_max=10e9, domain=(10.5e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    dom = _domain(sim)
    ax = dom["axes"][0]
    assert ax["n_cells"] == 11, ax
    assert abs(ax["realized_extent_um"] - 11000.0) < 1e-6, ax
    kinds = [(f["kind"], f.get("axis")) for f in dom["findings"]]
    assert ("domain-extent-quantized", "x") in kinds, kinds


def test_two_dimensional_mode_reports_its_single_periodic_plane():
    """2-D mode has ONE Yee plane on z; the N-nodes-bound-N-1-cells rule
    does not apply to it, and reporting 0 cells / 0 um there would be a
    false 'this axis does not exist'."""
    sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 1e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4, mode="2d_tmz")
    ax_z = _domain(sim)["axes"][2]
    assert ax_z["n_cells"] == 1, ax_z
    assert abs(ax_z["realized_extent_um"] - 1000.0) < 1e-6, ax_z
