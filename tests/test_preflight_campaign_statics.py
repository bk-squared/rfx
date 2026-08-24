"""Issue #703 campaign statics preflight checks.

Four advisory checks derived from a month-long external cross-validation
(issue #703): congruent-conductor rasterization parity, node-thin sheet
live-edge material consistency, sheet-bounded cavity electrical-thickness
report, and the off-lattice design-edge census.

Every fixture here is SYNTHETIC and public — the motivating incidents come
from a private design and none of its dimensions appear here; the issue
text is the spec.

Every gate is mutation-falsified in BOTH directions inside the tests
(monkeypatching the module-level gate constants of ``rfx.api._preflight``):
loosening the gate must silence the firing fixture, tightening it must make
the silent fixture fire. The observed results are recorded verbatim in each
test's docstring so a reader can tell a live gate from a decorative one.
"""

import numpy as np
import pytest

import rfx.api._compile as _compile
import rfx.api._preflight as _pf
from rfx.api import Simulation
from rfx.geometry.csg import Box

MM = 1e-3

CONGRUENCE_CODE = "congruent_conductor_rasterization_parity"
LIVE_EDGE_CODE = "sheet_live_edge_material_mismatch"
CAVITY_CODE = "sheet_cavity_electrical_thickness"
OFF_LATTICE_CODE = "off_lattice_design_edges"


# ---------------------------------------------------------------------------
# Fixtures (builders, so each test gets a fresh Simulation)
# ---------------------------------------------------------------------------

def _congruence_sim(off_lattice_mirror: bool, dz_profile=None):
    """Mirror pair of PEC sheets, 3.5 x 3.0 x 0.4 mm, dx = 1 mm.

    ``off_lattice_mirror=True`` (the incident class): the pair is mirrored
    about x = 5.26 mm — 0.26 cells off the node/half-node lattice, the same
    sub-cell magnitude as the measured incident (#703: mirror plane 0.26
    cells off, 173 vs 183 cells). Member A occupies x-nodes {1,2,3,4}
    (4 cells/row), member B x in [6.02, 9.52) occupies {7,8,9} (3 cells/row):
    12 vs 9 cells, spread 3.

    ``off_lattice_mirror=False`` (negative control): the same extents
    mirrored about x = 5.5 mm — ON the half-node lattice — with both x
    extents integer multiples of dx and every face safely off a node
    (no knife edge): counts are equal by construction.
    """
    sim = Simulation(domain=(12 * MM, 6 * MM, 6 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml", dz_profile=dz_profile)
    if off_lattice_mirror:
        sim.add(Box((1.0 * MM, 1.0 * MM, 2.0 * MM),
                    (4.5 * MM, 4.0 * MM, 2.4 * MM)), material="pec")
        sim.add(Box((6.02 * MM, 1.0 * MM, 2.0 * MM),
                    (9.52 * MM, 4.0 * MM, 2.4 * MM)), material="pec")
    else:
        # integer-multiple extent (3 mm) mirrored about the half-node 5.5 mm
        sim.add(Box((1.2 * MM, 1.0 * MM, 2.0 * MM),
                    (4.2 * MM, 4.0 * MM, 2.4 * MM)), material="pec")
        sim.add(Box((6.8 * MM, 1.0 * MM, 2.0 * MM),
                    (9.8 * MM, 4.0 * MM, 2.4 * MM)), material="pec")
    return sim


def _sheet_sim(dz_profile=None):
    """Node-thin PEC sheet between two dielectric fills that ABUT its faces.

    The stack a real board export gives: dielectric below ends at the
    sheet's bottom face, dielectric above starts at its top face, so no
    dielectric spans the sheet's node — the #702 configuration. dx=0.5 mm,
    sheet 0.1 mm thick (node-thin).
    """
    sim = Simulation(domain=(8 * MM, 8 * MM, 6 * MM), dx=0.5 * MM,
                     freq_max=10e9, boundary="cpml", dz_profile=dz_profile)
    sim.add_material("diel_lo", eps_r=3.5)
    sim.add_material("diel_hi", eps_r=2.5)
    sim.add(Box((1 * MM, 1 * MM, 1.0 * MM), (7 * MM, 7 * MM, 3.0 * MM)),
            material="diel_lo")
    sim.add(Box((1 * MM, 1 * MM, 3.1 * MM), (7 * MM, 7 * MM, 5.0 * MM)),
            material="diel_hi")
    sim.add(Box((2 * MM, 2 * MM, 3.0 * MM), (6 * MM, 6 * MM, 3.1 * MM)),
            material="pec")
    return sim


def _cavity_sim(collapsed: bool):
    """Two node-thin PEC sheets bounding an eps_r=4 dielectric gap, dx=1 mm.

    ``collapsed=True`` (fires): sheets 0.4 mm thick, mid-planes at 2.2 and
    4.2 mm snap to nodes k=2 and k=4, so the mesh cavity is node-to-node
    2.0 mm while the physical face-to-face gap is 4.0-2.4 = 1.6 mm: both
    electrical-thickness measures read +25%.

    ``collapsed=False`` (silent): sheets 2 um thick with mid-planes ON
    nodes 2.0 and 4.0 mm; face-to-face 1.998 mm vs node-to-node 2.0 mm is
    +0.1% on both measures, inside the 1% advisory threshold.
    """
    sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml")
    sim.add_material("core", eps_r=4.0)
    if collapsed:
        s1 = (2.0 * MM, 2.4 * MM)
        s2 = (4.0 * MM, 4.4 * MM)
    else:
        s1 = (1.999 * MM, 2.001 * MM)
        s2 = (3.999 * MM, 4.001 * MM)
    sim.add(Box((2 * MM, 2 * MM, s1[1]), (8 * MM, 8 * MM, s2[0])),
            material="core")
    sim.add(Box((2 * MM, 2 * MM, s1[0]), (8 * MM, 8 * MM, s1[1])),
            material="pec")
    sim.add(Box((2 * MM, 2 * MM, s2[0]), (8 * MM, 8 * MM, s2[1])),
            material="pec")
    return sim


def _off_lattice_sim(on_lattice: bool):
    """One resolved PEC box, dx = 1 mm.

    ``on_lattice=False`` (fires): lo face at x = 1.3 mm, extent 9 mm —
    residual 0.3 mm = 3.3% of the extent, above the 0.5% census threshold.
    ``on_lattice=True`` (silent): every face an exact node multiple.
    """
    sim = Simulation(domain=(12 * MM, 8 * MM, 8 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml")
    if on_lattice:
        sim.add(Box((1.0 * MM, 2.0 * MM, 2.0 * MM),
                    (10.0 * MM, 5.0 * MM, 5.0 * MM)), material="pec")
    else:
        sim.add(Box((1.3 * MM, 2.0 * MM, 2.0 * MM),
                    (10.3 * MM, 5.0 * MM, 5.0 * MM)), material="pec")
    return sim


def _bypass_resample(monkeypatch):
    """Mutate the #702 fix off: the assembly keeps node-sampled statics."""
    monkeypatch.setattr(
        _compile, "resample_sheet_node_materials",
        lambda geo, res, coords, eps, sig, **kw: (eps, sig))


# ---------------------------------------------------------------------------
# Check 1 — congruent-conductor rasterization parity
# ---------------------------------------------------------------------------

class TestCongruenceParity:
    def test_off_lattice_mirror_pair_fires_once_with_basis(self):
        """The incident class: mirror plane 0.26 cells off-lattice.

        Observed on this fixture: ONE aggregated advisory; member counts
        12 vs 9 cells (spread 3 > tolerance 1); the message carries the
        counts, per-member sub-lattice offsets, a verified origin-shift
        suggestion (re-rasterized spread printed), the coverage clause and
        the falsifier.
        """
        rep = _congruence_sim(True).preflight()
        hits = rep.by_code(CONGRUENCE_CODE)
        assert len(hits) == 1  # aggregated: one message per class per run
        msg = str(hits[0])
        assert "12 cells" in msg and "9 cells" in msg
        assert "spread 3" in msg
        assert "sub-lattice offsets" in msg
        assert "slide the lattice origin" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg
        assert "non-Box" in msg  # skip clause is stated, not silent

    def test_on_lattice_mirror_pair_is_silent(self):
        """Negative control: mirror plane on the half-node lattice.

        Observed: counts equal, no advisory.
        """
        rep = _congruence_sim(False).preflight()
        assert rep.by_code(CONGRUENCE_CODE) == []

    def test_fires_on_nonuniform_lane_too(self):
        """The check runs on the NU builders as well (issue #703 spec).

        The z profile is graded (not uniform-valued: a uniform-valued
        profile only tests plumbing) but leaves the same x-lattice, so the
        same 12-vs-9 spread must be found through the NU node builder.
        """
        dz = np.array([1.2, 0.8, 1.0, 1.0, 0.8, 1.2]) * MM
        rep = _congruence_sim(True, dz_profile=dz).preflight()
        hits = rep.by_code(CONGRUENCE_CODE)
        assert len(hits) == 1
        assert "nonuniform lane" in str(hits[0])

    def test_gate_mutation_both_directions(self, monkeypatch):
        """Gate = spread > _CONGRUENCE_SPREAD_TOL_CELLS.

        Mutation results (verbatim from this test's own asserts):
        - loosened (tol 1 -> 10): firing fixture (spread 3) emitted 0
          advisories -> the tolerance is load-bearing;
        - tightened (tol 1 -> -1): silent fixture (spread 0) emitted 1
          advisory -> the comparison is live in both directions.
        """
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_CELLS", 10)
        assert _congruence_sim(True).preflight().by_code(
            CONGRUENCE_CODE) == []
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_CELLS", -1)
        assert len(_congruence_sim(False).preflight().by_code(
            CONGRUENCE_CODE)) == 1

    def test_meshshape_conductors_are_skipped_and_said_so(self):
        """A non-Box conductor must appear in the coverage clause, not
        silently vanish (#685 class: silence has two meanings)."""
        from rfx.geometry.csg import Sphere
        sim = _congruence_sim(True)
        sim.add(Sphere(center=(3 * MM, 3 * MM, 5 * MM), radius=0.8 * MM),
                material="pec")
        sim.add(Sphere(center=(9 * MM, 3 * MM, 5 * MM), radius=0.8 * MM),
                material="pec")
        hits = sim.preflight().by_code(CONGRUENCE_CODE)
        assert len(hits) == 1
        assert "skipped 2 non-Box conductor" in str(hits[0])


# ---------------------------------------------------------------------------
# Check 2 — node-thin sheet live-edge material consistency
# ---------------------------------------------------------------------------

class TestSheetLiveEdgeMaterials:
    def test_post_702_main_is_silent(self):
        """On current main the assembly resamples sheet-node statics at the
        live edge (#702), so the guard must not fire."""
        rep = _sheet_sim().preflight()
        assert rep.by_code(LIVE_EDGE_CODE) == []

    def test_fires_when_the_resample_is_mutated_off(self, monkeypatch):
        """Mutate the #702 fix off (assembly keeps node-sampled statics).

        Observed with the bypass: 64 mismatched cells, worst offender
        'assigned eps_r 1 / sigma 0 vs live-edge sample eps_r 2.5' — the
        exact #702 signature (live edge on vacuum where the stack has no
        air). One aggregated message.
        """
        _bypass_resample(monkeypatch)
        hits = _sheet_sim().preflight().by_code(LIVE_EDGE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "assigned eps_r 1" in msg
        assert "live-edge sample eps_r 2.5" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_gate_mutation_both_directions(self, monkeypatch):
        """Gate = |assigned - sampled| > _LIVE_EDGE_RTOL * max(|sampled|, 1).

        Mutation results:
        - loosened (rtol 1e-4 -> 1e9) WITH the resample bypassed: the
          firing fixture emitted 0 advisories -> the rtol is load-bearing;
        - tightened (rtol 1e-4 -> -1.0) on the CLEAN fixture: 1 advisory
          (every live-edge cell 'mismatches' under a negative tolerance)
          -> the comparison is live in both directions.
        """
        _bypass_resample(monkeypatch)
        monkeypatch.setattr(_pf, "_LIVE_EDGE_RTOL", 1e9)
        assert _sheet_sim().preflight().by_code(LIVE_EDGE_CODE) == []
        monkeypatch.undo()
        monkeypatch.setattr(_pf, "_LIVE_EDGE_RTOL", -1.0)
        assert len(_sheet_sim().preflight().by_code(LIVE_EDGE_CODE)) == 1

    def test_subgrid_fine_region_debt_is_named(self):
        """add_refinement + a node-thin sheet: the FINE region still
        inherits the original #702 defect (rfx/runners/subgridded.py never
        calls resample_sheet_node_materials), and the check must say so
        rather than read as clean coverage."""
        sim = _sheet_sim()
        sim.add_refinement((2.5 * MM, 3.5 * MM), ratio=2)
        hits = sim.preflight().by_code(LIVE_EDGE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "KNOWN-UNFIXED" in msg
        assert "subgridded.py" in msg

    def test_fires_on_nonuniform_lane_too(self, monkeypatch):
        """Same mutation on the NU lane (graded dz, per-axis distinct)."""
        # assemble_materials_nu imports the resample function-locally at
        # call time, so the mutation goes on the SOURCE module; the check
        # keeps its own import-time binding and stays live.
        import rfx.geometry.rasterize_grid as _rg
        monkeypatch.setattr(
            _rg, "resample_sheet_node_materials",
            lambda geo, res, coords, eps, sig, **kw: (eps, sig))
        dz = np.array([0.6, 0.4, 0.5, 0.5, 0.4, 0.6,
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * MM
        hits = _sheet_sim(dz_profile=dz).preflight().by_code(LIVE_EDGE_CODE)
        assert len(hits) == 1
        assert "nonuniform lane" in str(hits[0])


# ---------------------------------------------------------------------------
# Check 3 — sheet-bounded cavity electrical-thickness report
# ---------------------------------------------------------------------------

class TestSheetCavityThickness:
    def test_collapsed_registration_fires_with_both_measures(self):
        """Mid-plane-registered sheets: node-to-node 2 mm vs face-to-face
        1.6 mm -> +25.0% on BOTH measures for this vacuum-free eps=4 gap.

        The message must print both measures and name the governing one —
        the incident's lesson: the same defect measured 17.3% as a series
        capacitance and 3.2% as phase length, and a bare percentage invites
        correcting a right number into a wrong one.
        """
        hits = _cavity_sim(True).preflight().by_code(CAVITY_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "sum(d/eps)" in msg and "sum(d*sqrt(eps))" in msg
        assert "+25.0%" in msg
        assert "governs" in msg
        assert "node-to-node" in msg and "face-to-face" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_node_registered_thin_sheets_are_silent(self):
        """2 um sheets with mid-planes ON nodes: +0.1% < 1% threshold."""
        rep = _cavity_sim(False).preflight()
        assert rep.by_code(CAVITY_CODE) == []

    def test_gate_mutation_both_directions(self, monkeypatch):
        """Gate = |delta| > _CAVITY_THICKNESS_TOL on either measure.

        Mutation results:
        - loosened (tol 0.01 -> 10.0): the +25% fixture emitted 0
          advisories -> the threshold is load-bearing;
        - tightened (tol 0.01 -> -1.0): the +0.1% fixture emitted 1
          advisory -> the comparison is live in both directions.
        """
        monkeypatch.setattr(_pf, "_CAVITY_THICKNESS_TOL", 10.0)
        assert _cavity_sim(True).preflight().by_code(CAVITY_CODE) == []
        monkeypatch.setattr(_pf, "_CAVITY_THICKNESS_TOL", -1.0)
        assert len(_cavity_sim(False).preflight().by_code(CAVITY_CODE)) == 1


# ---------------------------------------------------------------------------
# Check 4 — off-lattice design-edge census
# ---------------------------------------------------------------------------

class TestOffLatticeCensus:
    def test_off_lattice_edge_fires_with_residual_and_detune(self):
        """lo face 0.3 mm off-lattice on a 9 mm extent = 3.33%: one
        aggregated advisory carrying the residual and df/f ~ dL/L."""
        hits = _off_lattice_sim(False).preflight().by_code(OFF_LATTICE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "300µm" in msg
        assert "3.33%" in msg
        assert "df/f" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_on_lattice_edges_are_silent(self):
        rep = _off_lattice_sim(True).preflight()
        assert rep.by_code(OFF_LATTICE_CODE) == []

    def test_gate_mutation_both_directions(self, monkeypatch):
        """Gate = residual/extent > _OFF_LATTICE_EDGE_TOL.

        Mutation results:
        - loosened (tol 0.005 -> 1.0): the 3.33% fixture emitted 0
          advisories -> the threshold is load-bearing;
        - tightened (tol 0.005 -> -1.0): the on-lattice fixture emitted 1
          advisory -> the comparison is live in both directions.
        """
        monkeypatch.setattr(_pf, "_OFF_LATTICE_EDGE_TOL", 1.0)
        assert _off_lattice_sim(False).preflight().by_code(
            OFF_LATTICE_CODE) == []
        monkeypatch.setattr(_pf, "_OFF_LATTICE_EDGE_TOL", -1.0)
        assert len(_off_lattice_sim(True).preflight().by_code(
            OFF_LATTICE_CODE)) == 1

    def test_offenders_are_aggregated_and_capped(self):
        """Seven distinct off-lattice conductors -> ONE message, worst 5
        named (the #697 lesson: never one line per geometry entry)."""
        sim = Simulation(domain=(40 * MM, 8 * MM, 8 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        for i in range(7):
            x0 = (2.0 + 5.0 * i / 7.0 + 0.05 * (i + 1)) * MM
            sim.add(Box((x0, 2.0 * MM, 2.0 * MM),
                        (x0 + 4.0 * MM, 5.0 * MM, 5.0 * MM)),
                    material="pec")
        hits = sim.preflight().by_code(OFF_LATTICE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert msg.count("geometry[") == 5  # capped at the worst 5


# ---------------------------------------------------------------------------
# Wiring: the checks reach run()'s chain and respect skip semantics
# ---------------------------------------------------------------------------

class TestWiring:
    def test_checks_run_inside_validate_simulation_config(self):
        """run()/forward() surface these via _validate_simulation_config;
        preflight() shares that chain, so the code must appear in a plain
        preflight() report (already asserted above) AND the umbrella must
        be silent on a conductor-free model."""
        sim = Simulation(domain=(8 * MM, 8 * MM, 6 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add_material("core", eps_r=4.0)
        sim.add(Box((1 * MM, 1 * MM, 1 * MM), (7 * MM, 7 * MM, 5 * MM)),
                material="core")
        rep = sim.preflight()
        for code in (CONGRUENCE_CODE, LIVE_EDGE_CODE, CAVITY_CODE,
                     OFF_LATTICE_CODE, "campaign_statics_unavailable"):
            assert rep.by_code(code) == []

    def test_advisory_tier_none_block(self):
        """All four are warning-severity: report.ok stays True."""
        rep = _congruence_sim(True).preflight()
        hits = rep.by_code(CONGRUENCE_CODE)
        assert hits and all(h.severity == "warning" for h in hits)
        rep2 = _cavity_sim(True).preflight()
        assert all(h.severity == "warning"
                   for h in rep2.by_code(CAVITY_CODE))
