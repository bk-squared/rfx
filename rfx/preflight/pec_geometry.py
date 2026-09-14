"""PEC-geometry preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 2. The conductor-realization family: the issue-#703
campaign-statics checks, the #931 per-declaration realization findings, the
#669 Leontovich surface-impedance advisories and the conformal-fine-dx guard.
Everything here was relocated byte for byte out of ``rfx/api/_preflight.py``
-- same text, same order, same indentation, same docstrings, nothing renamed,
reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first so all ten check bodies are witnessed rather than merely counted.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

The five gate constants below are the reason this module has to exist as a
patch target rather than as a name the facade re-exports.
``tests/unit/preflight/test_preflight_rasterization.py`` proves each gate is
load-bearing by ``monkeypatch.setattr``-ing it in BOTH directions and
watching a firing fixture go silent and a silent one fire. Their readers are
the check bodies, which resolve them in THIS module's globals, so a patch
aimed at the ``rfx.api._preflight`` re-export would rebind a name nobody
consults and both arms of every such test would quietly pass on an
unmutated gate. The facade re-exports them anyway: 6 names move out and 6
come back, so the module namespace
``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
exactly as wide after this leg as before it -- still 55.
"""


# --------------------------------------------------------------------------
# Issue #703: campaign statics checks — tunables + shared lazy context.
#
# Four failure classes a month-long external cross-validation hit, all
# statically detectable before the first time step (issue #703; message
# design per docs/design_notes/preflight_lessons_from_a_long_crossval.md:
# every finding carries OBSERVED / WHY / COST / REMEDY / STALE-IF plus a
# COVERAGE clause, and each check aggregates into ONE message per run —
# the #697 failure mode was 84 advisories with 93% duplication).
#
# The gate values are module-level on purpose: the falsification tests
# monkeypatch them in BOTH directions (loosen -> firing fixture goes
# silent; tighten -> silent fixture fires) to prove each gate is
# load-bearing (tests/unit/preflight/test_preflight_rasterization*.py).
# --------------------------------------------------------------------------

# Check 1 — congruence key quantum (extents equal within 1e-9 m) and the
# tolerated realized-EDGE-count spread inside one congruence group. Under
# the lattice ownership contract (#931) every conductor is realized as a
# set of PEC E edges by one function (rfx.boundaries.pec
# .realized_pec_edge_masks); two congruent members that land at different
# sub-cell offsets realize different edge sets, and the edge count is the
# quantity that decides whether the lattice kept the design's symmetry.
# The tolerance is one edge: a symmetric pair realizes IDENTICAL counts,
# so any spread at all is an asymmetry, and one edge is the smallest
# spread a rounding tie on a single face can produce.
_CONGRUENCE_EXTENT_QUANTUM_M = 1e-9
_CONGRUENCE_SPREAD_TOL_EDGES = 1
# Check 3 — advisory threshold on either electrical-thickness measure of a
# cavity between two adjacent realized wall planes (sheet planes and the
# faces of volumes alike). Mesh sums run over the cells strictly between
# the two planes; the physical stack runs between the DECLARED faces, so
# the difference is exactly the plane snap (declared face vs realized
# node plane) plus any vacuum the declaration left at a sheet plane
# (``sheet_slot_vacuum`` names that separately).
_CAVITY_THICKNESS_TOL = 0.01
# Check 4 — off-lattice face residual as a fraction of the axis extent.
_OFF_LATTICE_EDGE_TOL = 5e-3
# Shared cap on named offenders per aggregated message.
_CAMPAIGN_MAX_OFFENDERS = 5
