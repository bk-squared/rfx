"""Preflight — the rasterization / statics stage: campaign statics, graded-mesh
fine-band displacement, thin metal on a non-uniform axis.

One file per preflight stage (tier 3b of the 2026-09 test-corpus
reorganisation, see ``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``).
Sections, each formerly its own file:

1. **Issue #703 campaign statics checks and the #931 realization findings**
   — was ``test_preflight_campaign_statics.py``. The four advisory checks
   derived from a month-long external cross-validation, re-pinned on the
   lattice ownership contract (issue #931,
   ``docs/design_notes/20260906_plan_realign_lattice_ownership.md``):
   congruent-conductor realization parity (realized PEC EDGE counts),
   the sheet-slot vacuum check (which replaces the #702 live-edge resample
   guard — a sheet owns no cell, so there is nothing to resample), the
   conductor-bounded cavity electrical-thickness report (adjacent REALIZED
   wall planes, sheet planes and volume faces alike), and the off-lattice
   design-edge census. Plus the design note's §3 findings:
   ``pec_box_subcell`` / ``pec_zero_cells`` / ``pec_realization_refused``
   (errors), ``pec_box_one_cell`` (warning), ``sheet_plane_realized``
   (notice; warning on a half-cell tie). Every fixture is SYNTHETIC and
   public — the motivating incidents come from a private design and none of
   its dimensions appear here. Every gate is mutation-falsified in BOTH
   directions inside the tests (monkeypatching the module-level gate
   constants of ``rfx.api._preflight``): loosening the gate must silence the
   firing fixture, tightening it must make the silent fixture fire; the
   observed results are recorded verbatim in each test's docstring.
2. **Boxes displaced from a graded-mesh fine band** — was
   ``test_preflight_graded_rasterization.py``: the advisory fires with the
   ACTUAL and implied z-cell counts, is silent for a box pinned to the real
   fine band and on a uniform-dz simulation; and — because the validator
   MODELS the rasterizer — its predicted count must agree with the
   production rasterize path in both directions (#562 F2, #568 item 1).
3. **Issue #48 thin PEC on a NU axis** — was ``test_preflight_thin_metal_nu.py``:
   preflight must warn when a realized metal PLANE sits on a non-uniform
   axis without symmetric neighbouring cells (Meep/OpenEMS convention), and
   stay silent on a uniform profile. The foils are sheet declarations —
   under #931 a 0.25 mm PEC Box on a 1 mm cell is refused, not snapped.

Section 2 is kept verbatim from the absorbed file (the identical ``_has``
helper is defined once).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import rfx.api._preflight as _pf
from rfx import Box, Simulation
from rfx.geometry.csg import Cylinder


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_campaign_statics.py
# ===========================================================================

MM = 1e-3

CONGRUENCE_CODE = "congruent_conductor_rasterization_parity"
SLOT_CODE = "sheet_slot_vacuum"
CAVITY_CODE = "sheet_cavity_electrical_thickness"
OFF_LATTICE_CODE = "off_lattice_design_edges"
ONE_CELL_CODE = "pec_box_one_cell"
PLANE_CODE = "sheet_plane_realized"
SUBCELL_CODE = "pec_box_subcell"
ZERO_CELLS_CODE = "pec_zero_cells"
REFUSED_CODE = "pec_realization_refused"
UNAVAILABLE_CODE = "campaign_statics_unavailable"


# ---------------------------------------------------------------------------
# Fixtures (builders, so each test gets a fresh Simulation)
# ---------------------------------------------------------------------------

def _congruence_sim(off_lattice_mirror: bool, dz_profile=None):
    """Mirror pair of PEC VOLUMES, 3.5 x 3.0 x 1.0 mm, dx = 1 mm.

    ``off_lattice_mirror=True`` (the incident class): the pair is mirrored
    about x = 5.26 mm — 0.26 cells off the node/half-node lattice, the same
    sub-cell magnitude as the measured incident (#703: mirror plane 0.26
    cells off, 173 vs 183 cells). Under the centre-sampled volume rule
    (#931 §1.1) member A ``x in [1.0, 4.5)`` holds the cell centres 1.5,
    2.5, 3.5 (3 columns); member B ``x in [6.02, 9.52)`` holds 6.5, 7.5,
    8.5 AND 9.5 (4 columns). Realized PEC edges of a 3x3x1 vs a 4x3x1
    block: 64 vs 82, spread 18.

    ``off_lattice_mirror=False`` (negative control): the same extents
    mirrored about x = 5.5 mm — ON the half-node lattice — with both x
    extents integer multiples of dx and every face safely off a centre:
    3 columns each, counts equal by construction.

    The z extent is a full cell on node planes (2 -> 3 mm): under #931 a
    sub-cell PEC Box is refused at add()/assembly, so the former 0.4 mm
    "sheet" version of this fixture cannot be built as a volume any more.
    """
    sim = Simulation(domain=(12 * MM, 6 * MM, 6 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml", dz_profile=dz_profile)
    if off_lattice_mirror:
        sim.add(Box((1.0 * MM, 1.0 * MM, 2.0 * MM),
                    (4.5 * MM, 4.0 * MM, 3.0 * MM)), material="pec")
        sim.add(Box((6.02 * MM, 1.0 * MM, 2.0 * MM),
                    (9.52 * MM, 4.0 * MM, 3.0 * MM)), material="pec")
    else:
        # integer-multiple extent (3 mm) mirrored about the half-node 5.5 mm
        sim.add(Box((1.2 * MM, 1.0 * MM, 2.0 * MM),
                    (4.2 * MM, 4.0 * MM, 3.0 * MM)), material="pec")
        sim.add(Box((6.8 * MM, 1.0 * MM, 2.0 * MM),
                    (9.8 * MM, 4.0 * MM, 3.0 * MM)), material="pec")
    return sim


class _PatternedSheet:
    """A patterned metal LAYER through the public ``Shape`` protocol.

    Why a test needs one instead of a ``Box``. A metal layer with
    clearance holes cannot BE a Box — a Box fills the holes with metal and
    shorts whatever the holes clear — so a CAD layer arrives as a
    user-defined ``Shape``: in-plane pattern from the design. Declared as
    a SHEET through ``add_thin_conductor`` (#931 §1.3: a non-Box sheet's
    footprint is its cross-section at its own mid-plane, placed on the
    nearest node plane). Implements exactly the ``rfx.geometry.csg.Shape``
    members the sheet builder and the congruence census use:
    ``bounding_box`` and ``mask`` / ``mask_on_coords``.

    Footprint = the ``lo/hi`` rectangle minus the ``hole_lo/hole_hi`` one,
    half-open on x and y as the design's own raster is.
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
    Under #931 such a shape can only be a VOLUME via ``add()`` (a sheet
    needs its bounds to find its plane), so it is added that way.
    """

    def bounding_box(self):
        raise NotImplementedError("this shape does not report bounds")


def _sheet_congruence_sim(off_lattice_mirror: bool):
    """Mirror pair of PATTERNED sheets declared through add_thin_conductor,
    dx = 1 mm — the shape of the motivating incident: patterned metal
    layers that are NOT Boxes, mirrored about a plane that does or does
    not sit on the lattice.

    ``off_lattice_mirror=True`` (fires): mirror plane x = 5.26 mm, 0.26
    cells off the node lattice — the incident's sub-cell magnitude.
    Member A (x in [1.0, 4.5)) holds x-nodes {1,2,3,4}, member B
    (x in [6.02, 9.52)) holds {7,8,9}; both hold y-nodes {1,2,3} and lose
    the node (2,2) / (8,2) to the mirrored clearance hole. Realized
    in-plane PEC edges (§1.3, both end nodes in the footprint): A 13
    (Mx 7 + My 6), B 8 (Mx 4 + My 4), spread 5.

    ``off_lattice_mirror=False`` (negative control): the same pair
    mirrored about x = 5.0 mm — ON a node — with 3 mm x extents: equal.
    """
    sim = Simulation(domain=(12 * MM, 6 * MM, 6 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml")
    z_lo, z_hi = 2.2 * MM, 2.6 * MM      # 0.4 mm foil: mid-plane 2.4 -> node 2

    def _add(x0, x1, hx0, hx1):
        sim.add_thin_conductor(
            _PatternedSheet((x0 * MM, 1.0 * MM, z_lo), (x1 * MM, 4.0 * MM, z_hi),
                            (hx0 * MM, 2.0 * MM, z_lo), (hx1 * MM, 3.0 * MM, z_hi)),
            sigma_bulk=5.8e7, thickness=0.4 * MM)

    if off_lattice_mirror:
        _add(1.0, 4.5, 2.0, 3.0)
        _add(6.02, 9.52, 7.52, 8.52)
    else:
        _add(1.5, 4.5, 2.5, 3.5)
        _add(5.5, 8.5, 6.5, 7.5)
    return sim


def _slot_sim(slot: bool, dz_profile=None):
    """A PEC sheet on z = 3.0 mm between two dielectrics, dx = 0.5 mm.

    ``slot=True`` (the #702 configuration, now reported instead of
    re-sampled): the dielectric below ends at 3.0 mm, the one above starts
    at 3.1 mm — the stack a board export gives when it leaves a slot for
    the foil. The node sampler is half-open, so the node at 3.0 mm gets
    neither dielectric; the sheet owns no cell and writes nothing; the
    normal E edge from the sheet plane into the cell above runs on vacuum.

    ``slot=False`` (silent): the upper dielectric is drawn from the sheet
    plane itself, so the plane's node carries eps_r 2.5.
    """
    sim = Simulation(domain=(8 * MM, 8 * MM, 6 * MM), dx=0.5 * MM,
                     freq_max=10e9, boundary="cpml", dz_profile=dz_profile)
    sim.add_material("diel_lo", eps_r=3.5)
    sim.add_material("diel_hi", eps_r=2.5)
    sim.add(Box((1 * MM, 1 * MM, 1.0 * MM), (7 * MM, 7 * MM, 3.0 * MM)),
            material="diel_lo")
    z_hi_lo = 3.1 * MM if slot else 3.0 * MM
    sim.add(Box((1 * MM, 1 * MM, z_hi_lo), (7 * MM, 7 * MM, 5.0 * MM)),
            material="diel_hi")
    sim.add(Box((2 * MM, 2 * MM, 3.0 * MM), (6 * MM, 6 * MM, 3.0 * MM)),
            material="pec")                      # zero thickness = a sheet
    return sim


def _cavity_sim(faces: bool):
    """Two foils declared with FACES (add_thin_conductor) bounding an
    eps_r=4 core, dx = 1 mm.

    ``faces=True`` (fires): foils 0.4 mm thick, faces at 2.0/2.4 and
    4.0/4.4 mm, mid-planes 2.2 / 4.2 -> realized on node planes 2 and 4
    (#931 §1.3, nearest node). The core is drawn from plane to plane
    (2.0 -> 4.0), so the mesh cavity is 2 cells of eps_r 4 = 2.0 mm while
    the physical face-to-face stack is 4.0 - 2.4 = 1.6 mm: both
    electrical-thickness measures read +25.0%. That is the sheet model's
    honest cost — a foil has no thickness on this lattice — and the
    message prints the lower plane's snap (-400 um: realized 2.0 mm,
    declared upper face 2.4 mm).

    ``faces=False`` (silent): 2 um foils with mid-planes ON nodes 2.0 and
    4.0 mm; face-to-face 1.998 mm vs plane-to-plane 2.0 mm is +0.1% on
    both measures, inside the 1% advisory threshold.
    """
    sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml")
    sim.add_material("core", eps_r=4.0)
    if faces:
        s1 = (2.0 * MM, 2.4 * MM)
        s2 = (4.0 * MM, 4.4 * MM)
    else:
        s1 = (1.999 * MM, 2.001 * MM)
        s2 = (3.999 * MM, 4.001 * MM)
    sim.add(Box((2 * MM, 2 * MM, 2.0 * MM), (8 * MM, 8 * MM, 4.0 * MM)),
            material="core")
    for lo, hi in (s1, s2):
        sim.add_thin_conductor(Box((2 * MM, 2 * MM, lo), (8 * MM, 8 * MM, hi)),
                               sigma_bulk=5.8e7, thickness=hi - lo)
    return sim


def _stack_sim(patch_z_mm: float, ground_cells: int = 1):
    """A VOLUME ground under an eps_r=4 core under a SHEET patch, dx = 1 mm.

    Ground: Box z in [2, 2 + ground_cells] mm (a slab with walls on both
    faces, #931 §1.2). Core: [3, patch_z). Patch: zero-thickness Box at
    ``patch_z_mm`` — the sheet declaration.

    ``patch_z_mm=5.0``: every face on a node, the cavity reads FLUSH (the
    #767 closure: the ground's far face is a realized wall the check can
    see). ``patch_z_mm=5.3``: the patch snaps to node 5, so the mesh
    cavity is 2 mm against a declared 2.3 mm: -13.0% on both measures.
    """
    sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                     freq_max=10e9, boundary="cpml")
    sim.add_material("core", eps_r=4.0)
    z_top = (2 + ground_cells) * MM
    sim.add(Box((2 * MM, 2 * MM, 2 * MM), (8 * MM, 8 * MM, z_top)),
            material="pec")
    sim.add(Box((2 * MM, 2 * MM, z_top), (8 * MM, 8 * MM, patch_z_mm * MM)),
            material="core")
    sim.add(Box((2 * MM, 2 * MM, patch_z_mm * MM),
                (8 * MM, 8 * MM, patch_z_mm * MM)), material="pec")
    return sim


def _tie_stack_sim():
    """Two face-registered foils on a graded z mesh — the half-cell TIE.

    z cells (mm) 0.4 0.4 | 0.1 | 0.5 0.5 | 0.1 | 0.4 0.4, so the interior
    nodes sit at 0, 0.4, 0.8, 0.9, 1.4, 1.9, 2.0, 2.4, 2.8. Foil A is
    declared with faces at 0.8/0.9 mm (both on nodes): its mid-plane
    0.85 mm is EXACTLY equidistant from the two, and the contract resolves
    the tie to the LOWER plane, 0.8 mm — with a WARNING naming both
    candidates. Foil B at 1.9/2.0 mm likewise lands on 1.9 mm. The eps_r=4
    core runs face to face, 0.9 -> 1.9 mm.

    Cavity: planes 0.8 and 1.9 mm bracket the 0.1 mm cell [0.8, 0.9]
    (vacuum — the core starts at 0.9) plus two 0.5 mm cells of eps_r 4:
    sum(d/eps) = 100 + 125 + 125 = 350 um against the physical 1.0 mm / 4
    = 250 um, +40.0%; the message prints foil A's snap (-100 um).
    """
    dz = np.array([0.4, 0.4, 0.1, 0.5, 0.5, 0.1, 0.4, 0.4]) * MM
    sim = Simulation(domain=(10 * MM, 10 * MM, 2.8 * MM), dx=0.5 * MM,
                     freq_max=10e9, boundary="cpml", dz_profile=dz)
    sim.add_material("core", eps_r=4.0)
    sim.add(Box((2 * MM, 2 * MM, 0.9 * MM), (8 * MM, 8 * MM, 1.9 * MM)),
            material="core")
    for lo, hi in ((0.8 * MM, 0.9 * MM), (1.9 * MM, 2.0 * MM)):
        sim.add_thin_conductor(Box((2 * MM, 2 * MM, lo), (8 * MM, 8 * MM, hi)),
                               sigma_bulk=5.8e7, thickness=hi - lo)
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


def _by_code(rep, code):
    return rep.by_code(code)


# ---------------------------------------------------------------------------
# Check 1 — congruent-conductor realization parity
# ---------------------------------------------------------------------------

class TestCongruenceParity:
    def test_off_lattice_mirror_pair_fires_once_with_basis(self):
        """The incident class: mirror plane 0.26 cells off-lattice.

        Observed on this fixture: ONE aggregated advisory; member counts
        64 vs 82 realized PEC edges (spread 18 > tolerance 1); the message
        carries the counts, per-member sub-lattice offsets, a verified
        origin-shift suggestion (re-realized spread printed), the
        coverage clause and the falsifier.
        """
        rep = _congruence_sim(True).preflight()
        hits = rep.by_code(CONGRUENCE_CODE)
        assert len(hits) == 1  # aggregated: one message per class per run
        msg = str(hits[0])
        assert "64 PEC edges" in msg and "82 PEC edges" in msg
        assert "spread 18" in msg
        assert "sub-lattice offsets" in msg
        assert "slide the lattice origin" in msg
        assert "spread drops to 0 edge(s)" in msg
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
        profile only tests plumbing) but keeps the members' z cell
        [2, 3] mm as one cell and leaves the same x-lattice, so the same
        64-vs-82 spread must be found through the NU node builder.
        """
        dz = np.array([1.2, 0.8, 1.0, 1.0, 0.8, 1.2]) * MM
        rep = _congruence_sim(True, dz_profile=dz).preflight()
        hits = rep.by_code(CONGRUENCE_CODE)
        assert len(hits) == 1
        assert "nonuniform lane" in str(hits[0])
        assert "64 PEC edges" in str(hits[0]) and "82 PEC edges" in str(hits[0])

    def test_gate_mutation_both_directions(self, monkeypatch):
        """Gate = spread > _CONGRUENCE_SPREAD_TOL_EDGES.

        Mutation results (verbatim from this test's own asserts):
        - loosened (tol 1 -> 100): firing fixture (spread 18) emitted 0
          advisories -> the tolerance is load-bearing;
        - tightened (tol 1 -> -1): silent fixture (spread 0) emitted 1
          advisory -> the comparison is live in both directions.
        """
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_EDGES", 100)
        assert _congruence_sim(True).preflight().by_code(
            CONGRUENCE_CODE) == []
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_EDGES", -1)
        assert len(_congruence_sim(False).preflight().by_code(
            CONGRUENCE_CODE)) == 1

    def test_conductors_without_bounds_are_skipped_and_said_so(self):
        """A conductor the census cannot key must appear in the coverage
        clause, not silently vanish (#685 class: silence has two
        meanings). Being a non-Box is NOT such a case — only declining to
        report bounds is. A bound-less shape is a VOLUME via add()."""
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
        assert "examined 2 conductor declaration(s)" in msg

    def test_patterned_sheet_mirror_pair_fires(self):
        """DEFECT A regression. The incident class is a mirror pair of
        PATTERNED LAYERS, not Boxes, and a Box-only entry census put every
        member into 'skipped' — the check stayed silent on the board that
        motivated it (three mirror pairs, 173 vs 183 cells).

        Under #931 the layers are SHEETS (add_thin_conductor) and the
        quantity compared is the realized in-plane PEC edge count.
        Observed on this fixture: ONE aggregated advisory, counts 13 vs 8
        edges (spread 5 > tolerance 1), the members named with their
        realization kind, and the coverage clause admitting that a
        bounding box bounds congruence rather than proving it for shapes
        that are not Boxes.
        """
        hits = _sheet_congruence_sim(True).preflight().by_code(
            CONGRUENCE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "_PatternedSheet" in msg          # keyed by shape class
        assert "(sheet)" in msg                   # ... and realization kind
        assert "13 PEC edges" in msg and "8 PEC edges" in msg
        assert "spread 5" in msg
        assert "INFERRED: 2 member(s)" in msg    # honesty clause present
        assert "not analytic Boxes" in msg       # no origin-shift guess
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_patterned_sheet_on_lattice_mirror_pair_is_silent(self):
        """Negative control: the same pair mirrored about a NODE.

        Observed: equal edge counts, no advisory — so the firing case
        above is the off-lattice mirror plane, not merely 'the check now
        looks at non-Box shapes'.
        """
        rep = _sheet_congruence_sim(False).preflight()
        assert rep.by_code(CONGRUENCE_CODE) == []

    def test_sheet_pair_gate_mutation_both_directions(self, monkeypatch):
        """Gate = spread > _CONGRUENCE_SPREAD_TOL_EDGES, on the sheet pair.

        Mutation results (verbatim from this test's own asserts):
        - loosened (tol 1 -> 100): the firing sheet pair (spread 5)
          emitted 0 advisories -> the tolerance is load-bearing here too;
        - tightened (tol 1 -> -1): the on-node sheet pair (spread 0)
          emitted 1 advisory -> the pair IS being examined, so its silence
          above is an equal count and not a skipped entry.
        """
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_EDGES", 100)
        assert _sheet_congruence_sim(True).preflight().by_code(
            CONGRUENCE_CODE) == []
        monkeypatch.setattr(_pf, "_CONGRUENCE_SPREAD_TOL_EDGES", -1)
        assert len(_sheet_congruence_sim(False).preflight().by_code(
            CONGRUENCE_CODE)) == 1


# ---------------------------------------------------------------------------
# Check 2 — sheet-slot vacuum (replaces the #702 live-edge resample guard)
# ---------------------------------------------------------------------------

class TestSheetSlotVacuum:
    def test_slotted_stack_fires_and_names_the_remedy(self):
        """The #702 configuration under the contract: nothing is re-sampled,
        the slot is reported. Observed: ONE aggregated ERROR-grade warning
        naming the plane (z = 3 mm), the vacuum on it, the two dielectrics
        (3.5 below, 2.5 above) and the remedy (extend the dielectric to the
        sheet plane)."""
        rep = _slot_sim(True).preflight()
        hits = rep.by_code(SLOT_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert msg.startswith("ERROR-GRADE")
        assert hits[0].severity == "warning"     # error-grade, not a refusal
        assert "z = 3mm" in msg
        assert "eps_r 3.5" in msg and "eps_r 2.5" in msg
        assert "extend the dielectric boxes to the sheet plane" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_dielectric_drawn_to_the_plane_is_silent(self):
        """The remedy applied: the upper dielectric starts ON the sheet
        plane, the plane's node carries eps_r 2.5, no slot."""
        rep = _slot_sim(False).preflight()
        assert rep.by_code(SLOT_CODE) == []

    def test_fires_on_nonuniform_lane_too(self):
        """Same slot on a graded z mesh with a node at 3.0 mm. The cell
        sizes are dyadic (0.75 / 0.25 / 0.5 mm) so every node is exact in
        float32 too: the NU lane samples dielectrics on float32 node
        coordinates, and a node at 2.9999998 mm would fall INSIDE the
        lower dielectric's half-open span — a different (buried-sheet)
        geometry, reported by the assembly's own warning, not a slot."""
        dz = np.array([0.75, 0.25, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * MM
        hits = _slot_sim(True, dz_profile=dz).preflight().by_code(SLOT_CODE)
        assert len(hits) == 1
        assert "nonuniform lane" in str(hits[0])

    def test_a_sheet_in_air_is_not_a_slot(self):
        """Vacuum on the plane with vacuum on at least one side is a sheet
        in air, not a slot: the check needs a dielectric on BOTH sides."""
        sim = Simulation(domain=(8 * MM, 8 * MM, 6 * MM), dx=0.5 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add_material("diel_hi", eps_r=2.5)
        sim.add(Box((1 * MM, 1 * MM, 3.1 * MM), (7 * MM, 7 * MM, 5.0 * MM)),
                material="diel_hi")
        sim.add(Box((2 * MM, 2 * MM, 3.0 * MM), (6 * MM, 6 * MM, 3.0 * MM)),
                material="pec")
        assert sim.preflight().by_code(SLOT_CODE) == []


# ---------------------------------------------------------------------------
# Check 3 — conductor-bounded cavity electrical-thickness report
# ---------------------------------------------------------------------------

class TestSheetCavityThickness:
    def test_foils_declared_with_faces_fire_with_both_measures(self):
        """Foils with faces realize on one plane each: plane-to-plane 2 mm
        vs face-to-face 1.6 mm -> +25.0% on BOTH measures for this eps=4
        gap, and the lower plane's snap (-400 um) is printed.

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
        assert "plane-to-plane" in msg and "face-to-face" in msg
        assert "snap -400µm" in msg
        assert "(sheet, plane k=" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    def test_node_registered_thin_foils_are_silent(self):
        """2 um foils with mid-planes ON nodes: +0.1% < 1% threshold."""
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

    def test_volume_far_face_and_sheet_plane_read_flush(self):
        """The #767 closure. A one-cell VOLUME ground has a wall on its
        FAR face (#931 §1.2) and the check reads it: the cavity between
        that face (3 mm) and the patch sheet (5 mm) equals the declared
        core, so nothing fires. The old check read the cell mask, which
        never contains a volume's far face, and mis-paired the cavity."""
        rep = _stack_sim(5.0).preflight()
        assert rep.by_code(CAVITY_CODE) == []
        # ... and the ground IS a one-cell slab, said so by its own finding
        assert len(rep.by_code(ONE_CELL_CODE)) == 1

    def test_snapped_sheet_plane_is_the_whole_difference(self):
        """Patch declared at 5.3 mm snaps to node 5: the mesh cavity is
        2 mm against a declared 2.3 mm, -13.0% on both measures, and the
        message attributes it to the printed snap (-300 um) — the
        difference IS the snap, nothing else."""
        hits = _stack_sim(5.3).preflight().by_code(CAVITY_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "-13.0%" in msg
        assert "snap -300µm" in msg
        assert "(volume, plane k=" in msg and "(sheet, k=" in msg

    def test_face_registered_foil_on_a_tie_reads_its_thickness_as_cavity(self):
        """The half-cell tie. Foil A (faces 0.8/0.9 mm) realizes on the
        LOWER plane 0.8 mm; the 0.1 mm cell above it is vacuum (the core
        starts at 0.9), so the cavity reads 350 um vs 250 um (+40.0%) and
        the snap printed for foil A is -100 um. No 'own cell' story: a
        sheet owns no cell, the number is the plane snap and the drawing."""
        hits = _tie_stack_sim().preflight().by_code(CAVITY_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "sum(d/eps) mesh 350µm vs physical 250µm (+40.0%)" in msg
        assert "snap -100µm" in msg
        assert "OWN cell" not in msg


# ---------------------------------------------------------------------------
# Check 4 — off-lattice design-edge census
# ---------------------------------------------------------------------------

class TestOffLatticeCensus:
    def test_off_lattice_edge_fires_with_alignment_residual(self):
        """lo face 0.3 mm off-lattice on a 9 mm extent = 3.33%: one
        aggregated advisory carrying the measured alignment residual."""
        hits = _off_lattice_sim(False).preflight().by_code(OFF_LATTICE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "300µm" in msg
        assert "3.33%" in msg
        assert "df/f" not in msg
        assert "frequency sensitivity depends on the mode" in msg
        assert "COVERAGE:" in msg and "STALE IF:" in msg

    @pytest.mark.parametrize("kind,lo,hi,z_hi,realized_mm,residual_mm", [
        ("sheet", 1.2, 6.2, 2.0, 4.0, 0.2),
        ("volume", 1.3, 6.7, 4.0, 6.0, 0.3),
    ])
    def test_nearest_node_residual_is_not_an_extent_error_bound(
        self, kind, lo, hi, z_hi, realized_mm, residual_mm,
    ):
        from rfx.boundaries.pec import realized_pec_edge_masks
        from rfx.geometry.rasterize_grid import coords_from_uniform_grid

        sim = Simulation(domain=(12 * MM, 8 * MM, 8 * MM), dx=MM,
                         freq_max=10e9, boundary="cpml")
        sim.add(Box((lo * MM, 2 * MM, 2 * MM),
                    (hi * MM, 5 * MM, z_hi * MM)), material="pec")
        hits = sim.preflight().by_code(OFF_LATTICE_CODE)
        assert len(hits) == 1 and hits[0].severity == "warning"

        grid = sim._build_grid()
        sheets, wires = [], []
        assembled = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
        edges = realized_pec_edge_masks(assembled[3], sheets=tuple(sheets),
                                       wires=tuple(wires), periodic=sim._periodic_flags())
        nodes = np.asarray(coords_from_uniform_grid(grid).x)
        # Ey lies on x nodes, so its occupied columns bound the physical span.
        columns = np.flatnonzero(np.asarray(edges[1]).any(axis=(1, 2)))
        actual = float(nodes[columns[-1]] - nodes[columns[0]])
        residual = max(float(np.min(abs(nodes - face * MM))) for face in (lo, hi))
        assert actual == pytest.approx(realized_mm * MM, rel=0, abs=1e-15)
        assert residual == pytest.approx(residual_mm * MM, rel=0, abs=1e-15)
        assert abs(actual - (hi - lo) * MM) > residual
        message = str(hits[0])
        assert f"({kind}) x:" in message
        assert "declared-face alignment only" in message
        assert "up to the printed residual" not in message
        assert "df/f" not in message

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

    def test_sheet_in_plane_edges_are_examined_and_its_normal_is_not(self):
        """A sheet's in-plane rim is a real design edge (its footprint is
        sampled closed on the nodes it covers); its NORMAL axis has no
        extent and is reported by sheet_plane_realized instead — the
        message says so."""
        sim = Simulation(domain=(12 * MM, 8 * MM, 8 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add(Box((1.3 * MM, 2.0 * MM, 2.0 * MM),
                    (10.3 * MM, 5.0 * MM, 2.0 * MM)), material="pec")
        hits = sim.preflight().by_code(OFF_LATTICE_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "(sheet) x:" in msg and "300µm" in msg
        assert "1 sheet normal axis/axes reported by sheet_plane_realized" in msg


# ---------------------------------------------------------------------------
# #931 §3 — per-declaration realization findings
# ---------------------------------------------------------------------------

class TestRealizationFindings:
    def test_one_cell_volume_warns_with_both_walls(self):
        """A PEC Box exactly one cell thick is a filled slab with walls on
        both faces; the warning prints both planes and names the sheet
        declaration as the foil remedy."""
        hits = _stack_sim(5.0).preflight().by_code(ONE_CELL_CODE)
        assert len(hits) == 1
        msg = str(hits[0])
        assert "walls at 2mm and 3mm" in msg
        assert "add_thin_conductor" in msg
        assert hits[0].severity == "warning"

    def test_two_cell_volume_is_silent(self):
        assert _stack_sim(6.0, ground_cells=2).preflight().by_code(
            ONE_CELL_CODE) == []

    def test_sheet_plane_notice_prints_declared_realized_and_offset(self):
        """Per sheet: declared mid-plane, realized node plane, offset; a
        NOTICE (info severity) so it never blocks and never counts as a
        warning. A sheet declared at 6.3 mm realizes on 6 mm: -0.300 cell."""
        sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add(Box((2 * MM, 2 * MM, 5.0 * MM), (8 * MM, 8 * MM, 5.0 * MM)),
                material="pec")
        sim.add(Box((2 * MM, 2 * MM, 6.3 * MM), (8 * MM, 8 * MM, 6.3 * MM)),
                material="pec")
        rep = sim.preflight()
        hits = rep.by_code(PLANE_CODE)
        assert len(hits) == 1
        assert hits[0].severity == "info"
        msg = str(hits[0])
        assert "2 PEC sheet(s) realized" in msg and "1 of them off" in msg
        assert "declared mid-plane 6.3mm, realized node plane" in msg
        assert "(offset -0.300 cell = 300µm)" in msg
        assert "declared mid-plane 5mm" in msg and "(offset +0.000 cell" in msg
        assert rep.ok

    def test_half_cell_tie_is_a_warning_naming_both_planes(self):
        """A face-registered foil (both faces on nodes) has its mid-plane
        on an exact tie; the contract resolves it LOWER and says so."""
        rep = _tie_stack_sim().preflight()
        hits = [h for h in rep.by_code(PLANE_CODE) if h.severity == "warning"]
        assert len(hits) == 1
        msg = str(hits[0])
        assert "2 PEC sheet(s) declared with a mid-plane on an exact half-cell TIE" in msg
        assert "equidistant from node planes" in msg
        assert "realized on the LOWER one" in msg
        assert "declared mid-plane 850µm" in msg

    def test_sub_cell_box_is_an_error_and_run_raises(self):
        """§1.5: a 0.4 mm PEC Box on a 1 mm cell is refused. Preflight
        reports it as pec_box_subcell (error severity, report.ok False)
        with the physical thickness and the local cell, and run() raises
        the same refusal."""
        sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add(Box((2 * MM, 2 * MM, 2.0 * MM), (8 * MM, 8 * MM, 2.4 * MM)),
                material="pec")
        sim.add_source((5 * MM, 5 * MM, 5 * MM), "ez")
        rep = sim.preflight()
        hits = rep.by_code(SUBCELL_CODE)
        assert len(hits) == 1
        assert hits[0].severity == "error"
        assert not rep.ok
        msg = str(hits[0])
        assert "0.0004 m against a local cell of 0.001 m" in msg
        assert "declare a sheet" in msg
        # the checks that need the assembly stay silent rather than
        # announcing 'unavailable' — the refusal IS the reason
        assert rep.by_code(UNAVAILABLE_CODE) == []
        with pytest.raises(ValueError, match="thinner than one cell"):
            sim.run(n_steps=2)

    def test_zero_cell_volume_is_an_error(self):
        """A post between cell centres (radius 0.55 mm centred on a node,
        dx = 1 mm: the nearest centres are 0.707 mm away) rasterizes to
        no cell — the #369 vaporized-metal class, now pec_zero_cells."""
        sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add(Cylinder(center=(5 * MM, 5 * MM, 4 * MM), radius=0.55 * MM,
                         height=2 * MM), material="pec")
        rep = sim.preflight()
        hits = rep.by_code(ZERO_CELLS_CODE)
        assert len(hits) == 1
        assert hits[0].severity == "error"
        assert "PolylineWire" in str(hits[0])

    def test_line_box_is_a_refusal(self):
        """Two zero-extent axes: a line is not a conductor (§1.5)."""
        sim = Simulation(domain=(10 * MM, 10 * MM, 8 * MM), dx=1 * MM,
                         freq_max=10e9, boundary="cpml")
        sim.add(Box((2 * MM, 5 * MM, 4 * MM), (8 * MM, 5 * MM, 4 * MM)),
                material="pec")
        hits = sim.preflight().by_code(REFUSED_CODE)
        assert len(hits) == 1 and hits[0].severity == "error"
        assert "a line or a point is not a conductor" in str(hits[0])


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
        for code in (CONGRUENCE_CODE, SLOT_CODE, CAVITY_CODE,
                     OFF_LATTICE_CODE, ONE_CELL_CODE, PLANE_CODE,
                     SUBCELL_CODE, ZERO_CELLS_CODE, REFUSED_CODE,
                     UNAVAILABLE_CODE):
            assert rep.by_code(code) == []

    def test_advisory_tier_none_block(self):
        """The campaign checks and the one-cell/slot findings are
        warning-severity, the plane notice is info: report.ok stays True."""
        rep = _congruence_sim(True).preflight()
        hits = rep.by_code(CONGRUENCE_CODE)
        assert hits and all(h.severity == "warning" for h in hits)
        assert rep.ok
        rep2 = _cavity_sim(True).preflight()
        assert all(h.severity == "warning"
                   for h in rep2.by_code(CAVITY_CODE))
        assert all(h.severity == "info" for h in rep2.by_code(PLANE_CODE))
        assert rep2.ok
        rep3 = _slot_sim(True).preflight()
        assert all(h.severity == "warning" for h in rep3.by_code(SLOT_CODE))
        assert rep3.ok

    def test_context_is_built_once_per_configuration(self):
        """The shared context is reused across the checks of one preflight
        and rebuilt after an add(): a cache keyed on the entries, not a
        stale instance attribute (the add_box-after-preflight lesson)."""
        sim = _stack_sim(5.0)
        c1 = sim._campaign_ctx()
        assert sim._campaign_ctx() is c1
        sim.add(Box((2 * MM, 2 * MM, 6 * MM), (8 * MM, 8 * MM, 6 * MM)),
                material="pec")
        c2 = sim._campaign_ctx()
        assert c2 is not c1
        assert sum(1 for e in c2.entry_realizations() if e.kind == "sheet") == 2


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
# formerly tests/unit/preflight/test_preflight_thin_metal_nu.py
# ===========================================================================

def _build(dz_profile):
    """FR4 patch on a graded z mesh, foils declared as SHEETS (#931).

    The ground sits on the substrate floor (z = 12 mm) and the patch on its
    top (z = 13.5 mm): zero-thickness Boxes via add(), i.e. sheet
    declarations realized on those node planes. (Before #931 both were
    0.25 mm PEC Boxes on a 1 mm cell — a sub-cell volume the contract
    refuses rather than snaps, so the fixture states its intent instead.)
    """
    h_sub = 1.5e-3
    sim = Simulation(
        freq_max=4e9, domain=(0.08, 0.075, 0), dx=1e-3,
        dz_profile=dz_profile, cpml_layers=8,
    )
    sim.add_material("fr4", eps_r=4.3)
    z_sub_lo = 12e-3
    z_sub_hi = 12e-3 + h_sub
    sim.add(Box((0.010, 0.010, z_sub_lo), (0.070, 0.065, z_sub_lo)),
            material="pec")
    sim.add(Box((0.010, 0.010, z_sub_lo), (0.070, 0.065, z_sub_hi)),
            material="fr4")
    sim.add(Box((0.025, 0.018, z_sub_hi), (0.054, 0.057, z_sub_hi)),
            material="pec")
    return sim


def test_asymmetric_metal_on_nu_triggers_warning():
    # Raw profile with sharp 1mm → 0.25mm → 1mm transitions. The ground
    # plane (z = 12 mm) has a 1 mm cell below and a 0.25 mm cell above,
    # the patch plane (13.5 mm) the reverse — both 4x asymmetric.
    dz = np.concatenate([np.full(12, 1e-3), np.full(6, 0.25e-3),
                         np.full(25, 1e-3)])
    sim = _build(dz)
    issues = sim.preflight()
    assert _has(issues, "issue #48"), (
        f"expected issue #48 warning, got: {issues!r}"
    )
    hits = [i for i in issues if "issue #48" in i]
    assert any("realized as a sheet" in i and "node" in i for i in hits), hits


def test_symmetric_metal_on_nu_is_silent():
    # All-uniform 0.25mm z profile. Metal planes have symmetric neighbours.
    dz = np.full(60, 0.25e-3)
    sim = _build(dz)
    issues = sim.preflight()
    assert not _has(issues, "issue #48"), (
        f"uniform-dz profile triggered the asymmetric-metal warning; "
        f"issues: {issues!r}"
    )
