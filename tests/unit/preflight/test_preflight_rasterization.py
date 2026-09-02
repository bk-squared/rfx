"""Preflight — the rasterization / statics stage: campaign statics, graded-mesh
fine-band displacement, thin metal on a non-uniform axis.

One file per preflight stage (tier 3b of the 2026-09 test-corpus
reorganisation, see ``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``).
Sections, each formerly its own file:

1. **Issue #703 campaign statics checks** — was
   ``test_preflight_campaign_statics.py``. Four advisory checks derived from
   a month-long external cross-validation: congruent-conductor rasterization
   parity, node-thin sheet live-edge material consistency, sheet-bounded
   cavity electrical-thickness report, and the off-lattice design-edge
   census. Every fixture is SYNTHETIC and public — the motivating incidents
   come from a private design and none of its dimensions appear here. Every
   gate is mutation-falsified in BOTH directions inside the tests
   (monkeypatching the module-level gate constants of ``rfx.api._preflight``):
   loosening the gate must silence the firing fixture, tightening it must
   make the silent fixture fire; the observed results are recorded verbatim
   in each test's docstring.
2. **Boxes displaced from a graded-mesh fine band** — was
   ``test_preflight_graded_rasterization.py``: the advisory fires with the
   ACTUAL and implied z-cell counts, is silent for a box pinned to the real
   fine band and on a uniform-dz simulation; and — because the validator
   MODELS the rasterizer — its predicted count must agree with the
   production rasterize path in both directions (#562 F2, #568 item 1).
3. **Issue #48 thin PEC on a NU axis** — was ``test_preflight_thin_metal_nu.py``:
   preflight must warn when a thin PEC sits on a non-uniform axis without
   symmetric neighbouring cells (Meep/OpenEMS convention), and stay silent
   on a uniform profile.

Every assertion, tolerance, fixture value and parametrisation of the
absorbed files is kept verbatim (the identical ``_has`` helper is defined
once).
"""

from __future__ import annotations

import math
import warnings as _w

import numpy as np
import pytest

import rfx.api._compile as _compile
import rfx.api._preflight as _pf
from rfx import Box, Simulation


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_rasterization.py
# ===========================================================================

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


class _PatternedSheet:
    """A patterned metal LAYER through the public ``Shape`` protocol.

    Why a test needs one instead of a ``Box``. A metal layer with
    clearance holes cannot BE a Box — a Box fills the holes with metal and
    shorts whatever the holes clear — so a CAD layer arrives as a
    user-defined ``Shape``: in-plane pattern from the design, thickness
    collapsed onto the one node nearest the sheet's mid-height (which is
    what rfx already does for a sub-cell Box). Implements exactly the
    ``rfx.geometry.csg.Shape`` members the congruence census uses:
    ``bounding_box`` and ``mask`` / ``mask_on_coords``.

    Footprint = the ``lo/hi`` rectangle minus the ``hole_lo/hole_hi`` one.
    """

    def __init__(self, lo, hi, hole_lo, hole_hi):
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        self._hlo = np.asarray(hole_lo, dtype=float)
        self._hhi = np.asarray(hole_hi, dtype=float)

    def bounding_box(self):
        return tuple(self._lo), tuple(self._hi)

    def mask(self, grid):
        from rfx.geometry.csg import _grid_coords
        return self.mask_on_coords(*_grid_coords(grid))

    def mask_on_coords(self, x, y, z):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        in_x = (x >= self._lo[0]) & (x < self._hi[0])
        in_y = (y >= self._lo[1]) & (y < self._hi[1])
        hole_x = (x >= self._hlo[0]) & (x < self._hhi[0])
        hole_y = (y >= self._hlo[1]) & (y < self._hhi[1])
        foot = ((in_x[:, None] & in_y[None, :])
                & ~(hole_x[:, None] & hole_y[None, :]))
        out = np.zeros((x.size, y.size, z.size), dtype=bool)
        k = int(np.argmin(np.abs(z - 0.5 * (self._lo[2] + self._hi[2]))))
        out[:, :, k] = foot
        return out


class _UnboundedSheet(_PatternedSheet):
    """Same layer, but declining to report bounds.

    ``rfx.geometry.csg.Shape.bounding_box`` raises by default, so a shape
    that never overrides it is a supported case — and one the congruence
    census cannot key. It must show up in the coverage clause, not vanish.
    """

    def bounding_box(self):
        raise NotImplementedError("this shape does not report bounds")


def _sheet_congruence_sim(off_lattice_mirror: bool, cls=_PatternedSheet):
    """Mirror pair of PATTERNED node-thin sheets, dx = 1 mm.

    The shape of the motivating incident: sub-cell-thick metal layers
    (0.4 mm on a 1 mm cell) that are NOT Boxes, mirrored about a plane
    that does or does not sit on the lattice.

    ``off_lattice_mirror=True`` (fires): mirror plane x = 5.26 mm, 0.26
    cells off the node lattice — the incident's sub-cell magnitude.
    Member A holds x-nodes {1,2,3,4}, member B x in [6.02, 9.52) holds
    {7,8,9}; both hold y-nodes {1,2,3} and lose one node to the mirrored
    clearance hole: 11 vs 8 cells, spread 3.

    ``off_lattice_mirror=False`` (negative control): the same pair
    mirrored about x = 5.0 mm — ON a node — with 3 mm x extents: 8 vs 8.
    """
    sim = Simulation(domain=(12 * MM, 6 * MM, 6 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml")
    z_lo, z_hi = 2.2 * MM, 2.6 * MM      # 0.4 mm: node-thin on a 1 mm cell

    def _add(x0, x1, hx0, hx1):
        sim.add(cls((x0 * MM, 1.0 * MM, z_lo), (x1 * MM, 4.0 * MM, z_hi),
                    (hx0 * MM, 2.0 * MM, z_lo), (hx1 * MM, 3.0 * MM, z_hi)),
                material="pec")

    if off_lattice_mirror:
        _add(1.0, 4.5, 2.0, 3.0)
        _add(6.02, 9.52, 7.52, 8.52)
    else:
        _add(1.5, 4.5, 2.5, 3.5)
        _add(5.5, 8.5, 6.5, 7.5)
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


def _face_registered_cavity_sim(upper: bool = False):
    """Two sheets that each FILL one sub-cell cell, dielectric on the faces.

    The mesh a face-registered export gives: both faces of every sheet are
    registered as nodes (1 um off the face, so the sheet's mid-height is
    unambiguously nearer one of them — an exactly-centred sheet is a coin
    flip between two equidistant nodes), and the dielectric between starts
    and ends on those same faces.

    ``upper=False``: nodes 1 um BELOW each face, so each sheet's PEC node
    is its LOWER face and the cell it fills is the cell above that node —
    inside the cavity. ``upper=True``: nodes 1 um ABOVE, so the PEC node
    is the UPPER face; the filled cell is now BELOW the node, the live
    edge is the cell above it, and for the lower sheet that cell is
    ordinary dielectric (the #702 resample takes its eps from the live
    edge) while for the upper sheet it is outside the cavity entirely.

    z cells (mm) 0.4 0.4 | 0.1 | 0.5 0.5 | 0.1 | 0.4 0.4, so the interior
    nodes sit at 0, 0.4, 0.8, 0.9, 1.4, 1.9, 2.0, 2.4, 2.8. Sheet A fills
    the cell [0.8, 0.9], sheet B fills [1.9, 2.0], and the eps_r=4 core
    runs face to face between them.

    The cavity here is 1.0 mm of solid eps_r=4: sum(d/eps) = 250 um,
    sum(d*sqrt(eps)) = 2 mm, and the mesh reproduces both exactly — the
    sheets' own cells are conductor and belong to neither side.
    """
    dz = np.array([0.4, 0.4, 0.1, 0.5, 0.5, 0.1, 0.4, 0.4]) * MM
    bias = 0.001 if upper else -0.001          # mm, node offset from a face
    sim = Simulation(domain=(10 * MM, 10 * MM, 2.8 * MM), dx=0.5 * MM,
                     freq_max=10e9, boundary="cpml", dz_profile=dz)
    sim.add_material("core", eps_r=4.0)
    lo_a, hi_a = (0.8 + bias) * MM, (0.9 + bias) * MM
    lo_b, hi_b = (1.9 + bias) * MM, (2.0 + bias) * MM
    sim.add(Box((2 * MM, 2 * MM, hi_a), (8 * MM, 8 * MM, lo_b)),
            material="core")
    sim.add(Box((2 * MM, 2 * MM, lo_a), (8 * MM, 8 * MM, hi_a)),
            material="pec")
    sim.add(Box((2 * MM, 2 * MM, lo_b), (8 * MM, 8 * MM, hi_b)),
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
        assert "no bounding box" in msg  # skip clause stated, not silent

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

    def test_conductors_without_bounds_are_skipped_and_said_so(self):
        """A conductor the census cannot key must appear in the coverage
        clause, not silently vanish (#685 class: silence has two
        meanings). Being a non-Box is NOT such a case any more — only
        declining to report bounds is."""
        sim = _congruence_sim(True)
        for x0 in (1.0, 6.02):
            sim.add(_UnboundedSheet(
                (x0 * MM, 1.0 * MM, 4.2 * MM),
                ((x0 + 3.5) * MM, 4.0 * MM, 4.6 * MM),
                ((x0 + 1.0) * MM, 2.0 * MM, 4.2 * MM),
                ((x0 + 2.0) * MM, 3.0 * MM, 4.6 * MM)), material="pec")
        hits = sim.preflight().by_code(CONGRUENCE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "skipped 2 conductor entr(y/ies) whose shape reports no "\
               "bounding box" in msg
        assert "examined 2 conductor entr(y/ies)" in msg

    def test_patterned_sheet_mirror_pair_fires(self):
        """DEFECT A regression. The incident class is a mirror pair of
        node-thin PATTERNED LAYERS, not Boxes, and a Box-only entry census
        put every member into 'skipped' — the check stayed silent on the
        board that motivated it (three mirror pairs, 173 vs 183 cells).

        Observed on this fixture: ONE aggregated advisory, counts 11 vs 8
        (spread 3 > tolerance 1), the members named, and the coverage
        clause admitting that a bounding box bounds congruence rather than
        proving it for shapes that are not Boxes.
        """
        hits = _sheet_congruence_sim(True).preflight().by_code(
            CONGRUENCE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "_PatternedSheet" in msg          # keyed by shape class
        assert "11 cells" in msg and "8 cells" in msg
        assert "spread 3" in msg
        assert "INFERRED: 2 member(s)" in msg    # honesty clause present
        assert "not analytic Boxes" in msg       # no origin-shift guess
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_patterned_sheet_on_lattice_mirror_pair_is_silent(self):
        """Negative control: the same pair mirrored about a NODE.

        Observed: both members 8 cells, no advisory — so the firing case
        above is the off-lattice mirror plane, not merely 'the check now
        looks at non-Box shapes'.
        """
        rep = _sheet_congruence_sim(False).preflight()
        assert rep.by_code(CONGRUENCE_CODE) == []

    def test_sheet_pair_gate_mutation_both_directions(self, monkeypatch):
        """Gate = spread > _CONGRUENCE_SPREAD_TOL_CELLS, on the sheet pair.

        Mutation results (verbatim from this test's own asserts):
        - loosened (tol 1 -> 10): the firing sheet pair (spread 3) emitted
          0 advisories -> the tolerance is load-bearing here too;
        - tightened (tol 1 -> -1): the on-node sheet pair (spread 0)
          emitted 1 advisory -> the pair IS being examined, so its silence
          above is an equal count and not a skipped entry.
        """
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_CELLS", 10)
        assert _sheet_congruence_sim(True).preflight().by_code(
            CONGRUENCE_CODE) == []
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_CELLS", -1)
        assert len(_sheet_congruence_sim(False).preflight().by_code(
            CONGRUENCE_CODE)) == 1


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

    def test_face_registered_sheet_cell_is_a_live_edge(self):
        """The fact the cavity number rests on, pinned at field level.

        On this stack each sheet fills ONE 100 um cell. ``apply_pec_mask``
        zeroes only TANGENTIAL E on a one-cell PEC sheet, so that cell's
        normal-E edge survives (it carries the sheet's surface charge) —
        and with the dielectric abutting the sheet's faces, the live edge
        sees vacuum.

        Observed: PEC at exactly 2 node layers; at each, eps_r 1.000,
        Ex and Ey masked, Ez NOT masked. A face-registered sheet
        therefore puts a live 100 um vacuum gap INSIDE the cavity, which
        is why the advisory below is a true reading and not a mis-pairing
        (it is also what rfx-known-issues records: face registration
        moves the PEC plane, it does not shorten the cavity).
        """
        from rfx.boundaries.pec import tangential_edge_masks

        sim = _face_registered_cavity_sim()
        ctx = _pf._CampaignStaticsContext(sim)
        mats, pec = ctx.assembled()
        pec = np.asarray(pec)
        eps = np.asarray(mats.eps_r)
        m_ex, m_ey, m_ez = (np.asarray(m) for m in
                            tangential_edge_masks(pec, (False, False, False)))
        i = j = pec.shape[0] // 2
        ks = np.flatnonzero(pec[i, j, :])
        assert ks.size == 2                       # one node layer per sheet
        for k in ks:
            assert eps[i, j, k] == pytest.approx(1.0)   # live edge on vacuum
            assert m_ex[i, j, k] and m_ey[i, j, k]      # tangential: zeroed
            assert not m_ez[i, j, k]                    # normal: LIVE

    def test_face_registered_stack_reports_the_live_vacuum_cell(self):
        """A face-registered stack is NOT electrically flush, and the
        message must name which of the two mechanisms it hit.

        Observed: 1 advisory, 'sum(d/eps) mesh 350um vs physical 250um
        (+40.0%)' — the 100 um excess is exactly the lower sheet's own
        cell at eps_r 1.000, and the message attributes it there rather
        than to the mid-plane collapse (a different mechanism with a
        different remedy: the collapse is a modelling trade, the vacuum
        cell is fixable by extending the abutting dielectric).
        """
        hits = _face_registered_cavity_sim().preflight().by_code(CAVITY_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "sum(d/eps) mesh 350µm vs physical 250µm (+40.0%)" in msg
        assert "is geometry[1]'s OWN cell (100µm at eps_r 1.000)" in msg
        assert "normal-E edge stays live" in msg
        assert "TWO MECHANISMS" in msg

    def test_midplane_stack_names_no_own_cell(self):
        """Negative control for the attribution: a mid-plane registered
        pair owns no cell, so its +25.0% carries NO own-cell clause and
        keeps the collapse story."""
        hits = _cavity_sim(True).preflight().by_code(CAVITY_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "+25.0%" in msg
        assert "'s OWN cell (" not in msg     # prose says "OWN cell" too

    def test_upper_face_registration_attributes_nothing(self):
        """Boundary of the attribution rule, in the other direction.

        With the nodes 1 um ABOVE each face, each sheet's PEC node is its
        UPPER face, so the cell it fills is below that node: the lower
        sheet's live edge is then an ordinary dielectric cell (the #702
        resample takes its eps from the live edge) and the upper sheet's
        live edge is outside the cavity. Nothing in the cavity is a
        conductor's own cell, so nothing may be attributed to one.

        Observed: 1 advisory, 'sum(d/eps) mesh 275um vs physical 250um
        (+10.0%)' — the collapse cost of moving both PEC planes up by a
        sheet's thickness — and NO own-cell clause.
        """
        hits = _face_registered_cavity_sim(upper=True).preflight().by_code(
            CAVITY_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "sum(d/eps) mesh 275µm vs physical 250µm (+10.0%)" in msg
        assert "'s OWN cell (" not in msg

    def test_own_cell_attribution_mutation_both_directions(self, monkeypatch):
        """Gate = _CAVITY_SHEET_CELL_FILL_FRAC decides whether a sheet
        FILLS a cell (attribute the excess to its live vacuum edge) or
        straddles two (attribute it to the mid-plane collapse).

        Mutation results (verbatim from this test's own asserts):
        - raised (0.9 -> 1.5, no sheet can fill a cell): the
          face-registered stack still reported the same
          'sum(d/eps) mesh 350um vs physical 250um (+40.0%)' — the NUMBER
          does not depend on this constant, which is the point — but the
          own-cell clause vanished, so the reader loses the mechanism;
        - lowered (0.9 -> 0.1, a sheet 'fills' any cell it touches): the
          mid-plane fixture's +25.0% gained an own-cell clause it has no
          right to, '250um is geometry[1]'s OWN cell (1mm at eps_r
          4.000)'. Live in both directions, and 0.9 is what separates
          'fills a cell' from 'straddles two'.
        """
        monkeypatch.setattr(_pf, "_CAVITY_SHEET_CELL_FILL_FRAC", 1.5)
        msg = str(_face_registered_cavity_sim().preflight().by_code(
            CAVITY_CODE)[0])
        assert "sum(d/eps) mesh 350µm vs physical 250µm (+40.0%)" in msg
        assert "'s OWN cell (" not in msg
        monkeypatch.setattr(_pf, "_CAVITY_SHEET_CELL_FILL_FRAC", 0.1)
        msg2 = str(_cavity_sim(True).preflight().by_code(CAVITY_CODE)[0])
        assert "+25.0%" in msg2
        assert "250µm is geometry[1]'s OWN cell (1mm at eps_r 4.000)" in msg2


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


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_rasterization.py
# ===========================================================================

def _issues(sim):
    return sim.preflight()


def _has(issues, substring):
    return any(substring in issue for issue in issues)


def _graded_sim(z_lo: float, z_hi: float) -> Simulation:
    dz = np.array([1.0e-3, 1.0e-3] + [0.25e-3] * 6 + [1.0e-3] * 2)
    sim = Simulation(
        freq_max=10e9,
        domain=(20e-3, 20e-3, float(np.sum(dz))),
        dx=1e-3,
        dz_profile=dz,
        cpml_layers=2,
    )
    sim.add_material("substrate", eps_r=3.5)
    sim.add(Box((5e-3, 5e-3, z_lo), (15e-3, 15e-3, z_hi)),
            material="substrate")
    sim.add_source((10e-3, 10e-3, 1e-3), "ez")
    return sim


def test_shifted_box_warns_with_actual_and_implied_counts():
    sim = _graded_sim(0.6e-3, 2.1e-3)

    issues = _issues(sim)

    # 2, not 1: the advisory reports what the RUN realizes, and the run puts
    # this box on the nodes at z = 1.0 and 2.0 mm (measured on the production
    # rasterize path). The former "1" came from the validator modelling the
    # rasterizer with cell centres — of which exactly one, z = 1.5 mm, fell in
    # the span. #562 made coordinates nodes and this validator now calls
    # Box.mask_on_coords instead of imitating it, so the number is the true
    # one. The ADVISORY still fires, which is what this test is about: 2 is
    # still below ceil(0.5 * 6.0) = 3.
    assert _has(issues, "rasterizes to 2 z cells (implied 6.0)"), issues
    assert _has(issues, "smooth_grading transition cells may have shifted"), issues


def test_box_pinned_to_actual_fine_band_is_silent():
    sim = _graded_sim(2.0e-3, 3.5e-3)

    assert not _has(_issues(sim), "smooth_grading transition cells may have shifted")


def test_uniform_dz_simulation_skips_check():
    sim = Simulation(
        freq_max=10e9,
        domain=(20e-3, 20e-3, 6e-3),
        dx=1e-3,
        cpml_layers=2,
    )
    sim.add_material("substrate", eps_r=3.5)
    sim.add(Box((5e-3, 5e-3, 0.5e-3), (15e-3, 15e-3, 2.0e-3)),
            material="substrate")
    sim.add_source((10e-3, 10e-3, 1e-3), "ez")

    assert not _has(_issues(sim), "smooth_grading transition cells may have shifted")


# --------------------------------------------------------------------------- #
# The validator MODELS the rasterizer, so it has to agree with it (#562 F2).
# --------------------------------------------------------------------------- #
_AGREEMENT_CASES = [
    # (z_lo_mm, z_hi_mm) — the 4.50-5.50 case is the reviewer's: it straddles a
    # grading transition, is classified THIN by Box.mask_on_coords (extent ==
    # one local cell) and so realizes ONE plane, and the hand-rolled
    # centre-model counted three and stayed silent where it should warn.
    (4.50, 5.50),
    (5.00, 6.00),
    (5.00, 7.00),
    (4.00, 5.00),
    (5.25, 6.75),
    (0.00, 5.00),
]


def _real_rasterized_z_count(sim) -> int:
    """The z-cell count the RUN actually produces, from the production path."""
    import numpy as _np
    from rfx.geometry.rasterize_grid import (rasterize_geometry,
                                             coords_from_nonuniform_grid)
    grid = sim._build_nonuniform_grid()
    coords = coords_from_nonuniform_grid(grid)
    out = rasterize_geometry(sim._geometry, sim._resolve_material, coords,
                             pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)
    eps = _np.asarray(getattr(out[0], "eps_r", out[0]))
    # the substrate is the only non-background material in these fixtures
    return int(_np.count_nonzero((eps > 1.0 + 1e-6).max(axis=(0, 1))))


def _validator_counts(sim):
    """(reported cell count, reported `implied`) or (None, None) when silent."""
    for issue in _issues(sim):
        if "rasterizes to" in issue:
            count = int(issue.split("rasterizes to")[1].split()[0])
            implied = float(issue.split("(implied")[1].split(")")[0])
            return count, implied
    return None, None


def _implied_cells(dz, z_lo, z_hi):
    """The validator's own `implied` figure: span thickness over the finest
    cell in a +-5-cell neighbourhood of the span (the #325 shifted-band
    recipe). Duplicated here on purpose so the SILENT direction can be
    justified rather than skipped; it is cross-checked against the
    validator's reported value on every case that fires.
    """
    edges = np.concatenate(([0.0], np.cumsum(dz)))
    local = (edges[:-1] < z_hi) & (edges[1:] > z_lo)
    idx = np.flatnonzero(local)
    lo_i = max(0, int(idx[0]) - 5)
    hi_i = min(dz.size, int(idx[-1]) + 6)
    return (z_hi - z_lo) / float(np.min(dz[lo_i:hi_i]))




@pytest.mark.parametrize("z_lo_mm,z_hi_mm", _AGREEMENT_CASES,
                         ids=[f"{a:.2f}-{b:.2f}mm" for a, b in _AGREEMENT_CASES])
def test_validator_count_matches_the_real_rasterizer(z_lo_mm, z_hi_mm):
    """This advisory predicts what the rasterizer will do, and a prediction
    that disagrees with the thing it predicts is worse than no advisory: it
    reads as a clean bill of health.

    Two separate ways the hand-rolled model was wrong before #562's review:
    it sampled cell CENTRES where the rasterizer samples E-NODES, and it knew
    nothing of the THIN-SHEET branch that snaps a box no thicker than its
    local cell onto a single nearest node. The validator now calls
    ``Box.mask_on_coords`` on node positions built by the same composition the
    grid builder uses, so agreement is by construction rather than by a copy
    that can drift — but only a test that runs both can say it stayed that way.
    """
    dz = np.array([1.0e-3] * 5 + [0.25e-3] * 8 + [1.0e-3] * 5)
    sim = Simulation(freq_max=30e9, domain=(4e-3, 4e-3, float(np.sum(dz))),
                     dx=1e-3, boundary="pec", dz_profile=dz)
    sim.add_material("substrate", eps_r=4.0)
    sim.add(Box((0.0, 0.0, z_lo_mm * 1e-3), (4e-3, 4e-3, z_hi_mm * 1e-3)),
            material="substrate")
    sim.add_source((2e-3, 2e-3, 1e-3), "ez")

    real = _real_rasterized_z_count(sim)
    predicted, implied = _validator_counts(sim)

    # Both directions, because the SILENT one is the F2 failure mode. The
    # first version of this test only asserted when the advisory fired, so
    # four of six cases skipped the assertion entirely — and "validator quiet
    # where it should warn" is exactly what F2 was (#568 item 1).
    implied_local = _implied_cells(dz, z_lo_mm * 1e-3, z_hi_mm * 1e-3)
    under_resolved = real < math.ceil(0.5 * implied_local)
    # The validator ALSO requires `actual <= 4`. That cutoff is its policy, not
    # this test's contract (#569 review, finding 4): hard-coding it here would
    # red this test the day the advisory is widened. So assert the two directions
    # the resolution condition settles, and stay silent about the band where the
    # cutoff alone decides.
    should_warn = under_resolved and real <= 4
    if should_warn:
        assert predicted is not None, (
            f"advisory SILENT for z-span [{z_lo_mm}, {z_hi_mm}) mm, but the "
            f"rasterizer realizes {real} cells against "
            f"{_implied_cells(dz, z_lo_mm * 1e-3, z_hi_mm * 1e-3):.1f} implied "
            f"— that silence reads as a clean bill of health")
        assert predicted == real, (
            f"validator predicts {predicted} z cells, rasterizer realizes "
            f"{real} for z-span [{z_lo_mm}, {z_hi_mm}) mm")
        # cross-check this test's own `implied` formula against the validator's
        assert implied == pytest.approx(implied_local, abs=0.05)
    elif not under_resolved:
        assert predicted is None, (
            f"advisory fired for z-span [{z_lo_mm}, {z_hi_mm}) mm where the "
            f"realized count {real} is not under-resolved against "
            f"{implied_local:.1f} implied")

    # and the reviewer's case must actually fire: 1 realized against 4 implied
    if (z_lo_mm, z_hi_mm) == (4.50, 5.50):
        assert real == 1, real
        assert predicted == 1, predicted


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_rasterization.py
# ===========================================================================

def _build(dz_profile):
    h_sub = 1.5e-3
    sim = Simulation(
        freq_max=4e9, domain=(0.08, 0.075, 0), dx=1e-3,
        dz_profile=dz_profile, cpml_layers=8,
    )
    sim.add_material("fr4", eps_r=4.3)
    z_gnd_lo = 12e-3 - 0.25e-3
    z_sub_lo = 12e-3
    z_sub_hi = 12e-3 + h_sub
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + 0.25e-3
    sim.add(Box((0.010, 0.010, z_gnd_lo), (0.070, 0.065, z_sub_lo)),
            material="pec")
    sim.add(Box((0.010, 0.010, z_sub_lo), (0.070, 0.065, z_sub_hi)),
            material="fr4")
    sim.add(Box((0.025, 0.018, z_patch_lo), (0.054, 0.057, z_patch_hi)),
            material="pec")
    return sim


def test_asymmetric_metal_on_nu_triggers_warning():
    # Raw profile with sharp 1mm → 0.25mm → 1mm transitions. Metal planes
    # sit in cells with 4x larger neighbours — should warn.
    dz = np.concatenate([np.full(12, 1e-3), np.full(6, 0.25e-3),
                         np.full(25, 1e-3)])
    sim = _build(dz)
    issues = sim.preflight()
    assert _has(issues, "issue #48"), (
        f"expected issue #48 warning, got: {issues!r}"
    )


def test_symmetric_metal_on_nu_is_silent():
    # All-uniform 0.25mm z profile. Metal cells have symmetric neighbours.
    dz = np.full(60, 0.25e-3)
    sim = _build(dz)
    issues = sim.preflight()
    assert not _has(issues, "issue #48"), (
        f"uniform-dz profile triggered the asymmetric-metal warning; "
        f"issues: {issues!r}"
    )
