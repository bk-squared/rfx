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

The ten check bodies are MODULE-LEVEL functions whose first parameter is still
named ``self`` and is still the :class:`~rfx.api.Simulation` instance.
``rfx/api/_preflight.py`` binds them back into the ``_PreflightMixin`` class
body AT THEIR ORIGINAL POSITIONS, so ``sim._validate_cfg_pec_realization(...)``
keeps its name, signature, ``__doc__`` and bound-method behaviour, and
``_validate_simulation_config``'s ordered call sequence -- the observable the
snapshot lock renders -- is unchanged.

``_congruence_origin_shift`` is the one that could not move as written. It was
a ``@staticmethod``, and a module-level ``@staticmethod`` produces a
staticmethod OBJECT rather than a callable, so the decorator stayed behind:
the facade re-applies it with ``_congruence_origin_shift =
staticmethod(_congruence_origin_shift)``. Its single caller,
``_validate_cfg_congruent_rasterization_parity``, invokes it as
``self._congruence_origin_shift(ctx, members, counts)``, which the wrapper
keeps at three arguments.

Two cross-family edges leave with the family and both are ``self.`` calls, so
both resolve through the composed ``Simulation`` MRO and survive untouched:
``_validate_cfg_campaign_statics`` -> ``self._campaign_ctx()`` (realization,
still in the facade) and ``_validate_cfg_pec_face_short_of_domain_wall`` ->
``self._preflight_face_layers()`` (absorber, which leg 4 moved on to
``rfx/preflight/absorber.py`` -- a ``self.`` edge does not care which module
the other end lives in, which is the property that let both legs land without
either package module importing the other).
``__qualname__`` is restored at the foot of this module.

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


from __future__ import annotations

import math

import numpy as np

from rfx.grid import C0
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box

from rfx.preflight._common import (
    _fmt_len,
    _fmt_signed,
    _sorted_box_corners,
    PreflightWarning,
)

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


# ---------------------------------------------------------------------------
# ``_validate_cfg_conformal_fine_dx`` (the ``conformal_nan`` advisory) was
# DELETED 2026-09-15 by #1043 / PR #1047. It was a self-detecting stale check
# by its own design: its comment named
# ``tests/unit/geometry/test_subpixel_pec.py::test_mesh_convergence_s21_with_conformal_pec``
# (xfail strict=True) as the tripwire that would XPASS when the NaN was fixed
# and said that XPASS is the signal to delete the guard. It XPASSed.
#
# The conformal-PEC fine-dx NaN was the #1043 CPML coefficient defect:
# ``conformal_eps_correction`` sets ``aniso_eps`` to ``eps_eff = eps/w`` at
# wall cells (``rfx/runners/uniform.py:279-303``), so the Yee half saw a
# HIGHER permittivity than ``apply_cpml_e``'s psi coefficient, which read the
# staircase ``materials.eps_r`` -- the amplifying direction. The wall cells run
# through the x CPML pads, which span every y and z.
#
# Measured on the tripwire's own three rungs, |S21| at dx 3 / 2 / 1.5 mm:
#   origin/main                          0.7564 / 0.6974 / nan   (xfail)
#   this parameter present but unthreaded 0.7564 / 0.6974 / nan   (xfail)
#   psi coefficient threaded             0.7796 / 0.6974 / 0.7274 (passes)
#
# The 2026-06 diagnosis -- "the E-update-only eps_eff makes the operator
# non-SPSD (discrete-adjointness break), dt cannot cure it" -- was a
# misdiagnosis of this symptom. Whether the conformal METHOD is ACCURATE is a
# separate, still-open question: the 2026-06-08 accuracy verdicts on the four
# conformal methods were all taken with this defect present.
# ---------------------------------------------------------------------------


def _validate_cfg_thin_conductor_surface_impedance(self, _w) -> None:
    """Advisories for Leontovich (surface_impedance_f0) sheets (#669).

    Two advisory-tier checks per f0-mode thin conductor, both on
    CONCRETE values only (traced f0/sigma_bulk/thickness skip them):

    (a) ``thickness < 3 * delta(f0)`` with skin depth
        ``delta = sqrt(2/(2*pi*f0*mu0*sigma_bulk))`` — the
        thick-conductor (Leontovich) model is invalid for thin films;
        the DC sheet path (omit ``surface_impedance_f0``) is the
        correct model there.
    (b) ``|f0 - source centre| / source centre > 0.20`` — Rs is frozen
        at ``f0`` with relative band error ``|sqrt(f/f0)-1|``; a source
        band centred far from f0 makes that error claims-relevant.

    Thresholds 3x and 20% are fixed by the issue #669 implementation
    contract.
    """
    from rfx.core.yee import MU_0 as _MU0

    f0_sheets = [tc for tc in getattr(self, "_thin_conductors", ())
                 if getattr(tc, "surface_impedance_f0", None) is not None]
    if not f0_sheets:
        return

    src_f0s: list[float] = []
    for family in ("_ports", "_msl_ports", "_waveguide_ports",
                   "_coaxial_ports", "_floquet_ports"):
        for entry in getattr(self, family, ()) or ():
            wf = getattr(entry, "waveform", None)
            wf0 = getattr(wf, "f0", None)
            if wf0 is not None and not is_tracer(wf0):
                try:
                    src_f0s.append(float(wf0))
                except (TypeError, ValueError):
                    pass

    for i, tc in enumerate(f0_sheets):
        f0 = tc.surface_impedance_f0
        sb = tc.sigma_bulk
        t = tc.thickness
        if is_tracer(f0) or is_tracer(sb):
            continue
        f0 = float(f0)
        sb = float(sb)
        delta = math.sqrt(2.0 / (2.0 * math.pi * f0 * _MU0 * sb))
        if not is_tracer(t) and 0.0 < float(t) < 3.0 * delta:
            _w.warn(
                PreflightWarning(
                    f"surface_impedance_f0 thin conductor #{i}: "
                    f"thickness {_fmt_len(float(t))} is below 3 skin "
                    f"depths ({_fmt_len(3.0 * delta)} at f0 = "
                    f"{f0:.4g} Hz, delta = {_fmt_len(delta)}). The "
                    f"thick-conductor (Leontovich) surface-resistance "
                    f"model is invalid for thin films — omit "
                    f"surface_impedance_f0 and use the DC sheet path "
                    f"(sigma_bulk*t/d), which is the correct model "
                    f"there.",
                    code="thin_conductor_leontovich_thin_film",
                    source="_validate_cfg_thin_conductor_surface_impedance",
                ),
                stacklevel=3,
            )
        for sf in src_f0s:
            if sf > 0.0 and abs(f0 - sf) / sf > 0.20:
                _w.warn(
                    PreflightWarning(
                        f"surface_impedance_f0 thin conductor #{i}: "
                        f"f0 = {f0:.4g} Hz is more than 20% away from "
                        f"the source centre frequency {sf:.4g} Hz. Rs "
                        f"is frozen at f0 with relative band error "
                        f"|sqrt(f/f0)-1| — at the source centre that "
                        f"is {abs(math.sqrt(sf / f0) - 1.0):.1%}. Set "
                        f"surface_impedance_f0 to the band centre you "
                        f"actually analyse.",
                        code="thin_conductor_leontovich_band_offset",
                        source=(
                            "_validate_cfg_thin_conductor_"
                            "surface_impedance"),
                    ),
                    stacklevel=3,
                )
                break

def _validate_cfg_campaign_statics(self, _w) -> None:
    """Umbrella for the conductor-realization checks; builds the shared context.

    Two families share the context. The #931 realization findings
    (design note §3: ``pec_box_subcell`` / ``pec_zero_cells`` /
    ``pec_realization_refused`` errors, ``pec_box_one_cell`` warning,
    ``sheet_plane_realized`` notice, ``sheet_slot_vacuum`` and
    ``pec_face_short_of_domain_wall`` warnings) say per declaration
    what the lattice realized; the issue-#703 campaign checks
    (congruent-conductor realization parity, sheet-cavity electrical
    thickness, off-lattice design-edge census) say what that
    realization does to the design's symmetries and stack-ups.

    Skips silently when the model has no conductor at all, and on a
    traced mesh (no concrete node positions — the
    ``_validate_cfg_graded_box_rasterization`` precedent). On a
    context-build failure it says so instead of reading as clean: a
    guard that cannot evaluate the model must not be indistinguishable
    from a guard that found nothing (#685 class).
    """
    try:
        has_conductor = any(
            self._resolve_material(e.material_name).sigma
            >= self._PEC_SIGMA_THRESHOLD
            for e in self._geometry
        ) or bool(self._thin_conductors)
    except KeyError:
        return  # unresolved material name; add_box/run raise elsewhere
    if not has_conductor:
        return
    ctx = self._campaign_ctx()
    if ctx.error == "traced-mesh":
        return
    if ctx.error is not None:
        _w.warn(PreflightWarning(
            "the conductor-realization checks (#931 realization "
            "findings; issue-#703 congruent-conductor parity, "
            "sheet-cavity electrical thickness, off-lattice design-edge "
            f"census) could NOT run: {ctx.error}. Their silence on this "
            "run means 'not evaluated', not 'clean' (#685 class: a guard "
            "that cannot see the model must say so).",
            code="campaign_statics_unavailable",
            source="_validate_cfg_campaign_statics",
        ))
        return
    self._validate_cfg_pec_realization(_w, ctx)
    self._validate_cfg_congruent_rasterization_parity(_w, ctx)
    self._validate_cfg_off_lattice_design_edges(_w, ctx)
    self._validate_cfg_pec_face_short_of_domain_wall(_w, ctx)
    realized = ctx.realized()
    if realized is None:
        if any(e.kind == "refused" for e in ctx.entry_realizations()):
            return  # the refusal above IS the reason; run() raises it
        _w.warn(PreflightWarning(
            "the sheet-slot and sheet-cavity checks could NOT run (the "
            f"production assembly failed: {ctx.assembly_error}); their "
            "silence means 'not evaluated', not 'clean' (#685 class).",
            code="campaign_statics_unavailable",
            source="_validate_cfg_campaign_statics",
        ))
        return
    self._validate_cfg_sheet_slot_vacuum(_w, ctx)
    self._validate_cfg_sheet_cavity_thickness(_w, ctx)

# ------------------------------------------------------------------
# #931 §3 — what the lattice realized, per PEC declaration
# ------------------------------------------------------------------

def _validate_cfg_pec_realization(self, _w, ctx) -> None:
    """Design note §3: the per-declaration realization findings.

    Everything here is read from the shared classifier
    (:meth:`_CampaignStaticsContext.entry_realizations`) and the
    realized edge set; nothing is inferred from an extent. Input
    fidelity only: each message says what was declared and what the
    lattice realized, in physical units, and never predicts a
    result-side number (``feedback_preflight_input_fidelity_only``).

    * ``pec_box_subcell`` (ERROR) — the §1.5 refusal: a PEC shape
      with ``0 < extent < one local cell`` on some axis. A Box passed
      to ``add()`` is a volume; declare a sheet or resolve the
      thickness. The classifier's own message carries the physical
      thickness and the local cell.
    * ``pec_zero_cells`` (ERROR) — a PEC volume no cell centre falls
      inside (the #369 vaporized-metal class), or a sheet whose
      footprint reaches no node.
    * ``pec_realization_refused`` (ERROR) — every other refusal the
      classifier raises (a line/point Box, a sheet declared further
      than half a cell from the node line, an ``add_thin_conductor``
      thicker than one local cell).
    * ``pec_box_one_cell`` (WARNING) — a PEC volume exactly one cell
      thick along some axis: realized as a filled slab with walls on
      BOTH faces (§1.2). Every crossval foil used to be drawn this
      way, so this fires on unmigrated scripts by design; the remedy
      names the sheet declaration.
    * ``sheet_plane_realized`` (NOTICE) — per sheet, the declared
      mid-plane, the realized node plane and the offset between them;
      a WARNING instead when the mid-plane sits on an exact half-cell
      tie, naming both candidate planes and the one chosen (lower).

    Aggregated per code (the #697 lesson: never one line per geometry
    entry), worst offenders first, capped at
    ``_CAMPAIGN_MAX_OFFENDERS``; the refusals are per entry because
    each is a hard stop with its own remedy.
    """
    entries = ctx.entry_realizations()
    shape = tuple(ctx.grid.shape)

    # --- refusals: ERROR, one per entry -----------------------------
    for e in entries:
        if e.kind != "refused":
            continue
        msg = e.error or "refused"
        text = (
            f"{e.label} '{e.name}' cannot be realized on this lattice "
            f"and run()/forward() will raise: {msg} (lattice ownership "
            "contract #931 §1.5; preflight reports it here so the whole "
            "configuration is audited before the raise).")
        # Three literal constructions on purpose: the emission-site
        # freeze (test_preflight_advisory_emission_contract) reads
        # ``code=`` literals off the AST, and a computed slug would
        # register as an uncoded-at-source site.
        if "thinner than one cell" in msg:
            _w.warn(PreflightWarning(
                text, code="pec_box_subcell", severity="error",
                source="_validate_cfg_pec_realization"))
        elif "ZERO cells" in msg or "ZERO nodes" in msg:
            _w.warn(PreflightWarning(
                text, code="pec_zero_cells", severity="error",
                source="_validate_cfg_pec_realization"))
        else:
            _w.warn(PreflightWarning(
                text, code="pec_realization_refused", severity="error",
                source="_validate_cfg_pec_realization"))

    # --- pec_box_one_cell: WARNING, aggregated ------------------------
    one_cell = []
    for e in entries:
        if e.kind != "volume":
            continue
        thin_axes = []
        for a in range(3):
            if shape[a] == 1:
                continue        # the 2-D lane's flat axis is not a slab
            planes = e.wall_planes(a, ctx.periodic, shape)
            if len(planes) >= 2 and max(planes) - min(planes) == 1:
                k_lo, k_hi = min(planes), max(planes)
                thin_axes.append((a, k_lo, k_hi))
        if thin_axes:
            one_cell.append((e, thin_axes))
    if one_cell:
        desc = []
        for e, thin_axes in one_cell[:_CAMPAIGN_MAX_OFFENDERS]:
            per_axis = "; ".join(
                f"{'xyz'[a]}: walls at {_fmt_len(float(ctx.nodes[a][k_lo]))} "
                f"and {_fmt_len(float(ctx.nodes[a][k_hi]))}"
                + (f" (drawn {_fmt_len(float(e.lo[a]))} -> "
                   f"{_fmt_len(float(e.hi[a]))})"
                   if e.lo is not None else " (shape reports no bounds)")
                for a, k_lo, k_hi in thin_axes)
            desc.append(f"{e.label} '{e.name}' ({type(e.shape).__name__}) "
                        f"is one cell thick along "
                        f"{'/'.join('xyz'[a] for a, _k1, _k2 in thin_axes)} "
                        f"[{per_axis}]")
        _w.warn(PreflightWarning(
            f"{len(one_cell)} PEC volume(s) are exactly ONE cell thick "
            f"along some axis and are realized as a filled slab with "
            f"walls on BOTH faces (every E edge between the two faces "
            f"is shorted; lattice ownership contract #931 §1.2): "
            + "; ".join(desc)
            + ". If this is foil (a ground plane, patch or trace), "
            "declare it as a SHEET — add_thin_conductor(shape), or a "
            "zero-thickness Box via add() — which realizes on ONE node "
            "plane with the normal E edge through it live; a slab and a "
            "sheet are different conductors on this lattice. If it is a "
            "plate, an iris or a wall drawn one cell thick on purpose, "
            "no action is needed (report-only). COVERAGE: examined "
            f"{sum(1 for e in entries if e.kind == 'volume')} PEC "
            f"volume(s) on the {ctx.lane} lane through "
            "rfx.boundaries.pec.realized_wall_planes. STALE IF: "
            "realized_wall_planes on the named entry's own edge masks "
            "returns planes more than one index apart.",
            code="pec_box_one_cell",
            source="_validate_cfg_pec_realization",
        ))

    # --- sheet_plane_realized: NOTICE per sheet (aggregated), WARNING on a tie
    sheets = [e for e in entries if e.kind == "sheet"]
    if not sheets:
        return
    rows = []
    ties = []
    for e in sheets:
        a = int(e.sheet.normal_axis)
        k = int(e.sheet.plane)
        realized_z = float(ctx.nodes[a][k])
        declared = e.declared_mid
        if declared is None:
            rows.append((0.0, e, a, k, realized_z, None, 0.0))
            continue
        off = realized_z - float(declared)
        d_loc = ctx.local_spacing(a, float(declared))
        rows.append((abs(off), e, a, k, realized_z, float(declared),
                     off / d_loc if d_loc > 0 else 0.0))
        if e.tie_planes is not None:
            ties.append((e, a, k, float(declared)))
    rows.sort(key=lambda t: -t[0])
    desc = "; ".join(
        f"{e.label} '{e.name}' normal {'xyz'[a]}: declared mid-plane "
        + ("n/a" if declared is None else _fmt_len(declared))
        + f", realized node plane {k} at {_fmt_len(realized_z)}"
        + ("" if declared is None else
           f" (offset {off_cells:+.3f} cell = "
           f"{_fmt_len(abs(realized_z - declared))})")
        for _o, e, a, k, realized_z, declared, off_cells
        in rows[:_CAMPAIGN_MAX_OFFENDERS])
    n_off = sum(1 for r in rows if r[0] > 1e-12)
    # Say something only when there IS something to say. A model whose
    # sheets all landed on the plane they declared is the expected case,
    # and preflight(strict=True) escalates EVERY finding including
    # info-severity (the historical contract at :2999), so an
    # unconditional "N sheets realized, 0 of them off" line failed every
    # strict run on a correct board — measured on
    # docs/public/guide/first-patch.mdx. The realized planes stay
    # available from fidelity_report(); this finding reports a CONDITION.
    if n_off == 0 and not ties:
        rows = []
    if rows:
        _w.warn(PreflightWarning(
          f"{len(sheets)} PEC sheet(s) realized (lattice ownership "
          f"contract #931 §1.3: one node plane each, closed footprint, "
          f"normal E through the plane live), {n_off} of them off their "
          f"declared mid-plane; worst first: {desc}. A sheet snaps to the "
          "node plane nearest its declared mid-plane (an exact half-cell "
          "tie resolves LOWER); an offset means the declared plane — a "
          "laminate face, a ground level — is not on this mesh's node "
          "line, and the conductor sits that far from where it was "
          "drawn. REMEDY when the offset matters: put a mesh node on the "
          "declared plane (dx = h/N for an interface at height h, or a "
          "preserved region on the non-uniform lane). COVERAGE: every "
          f"sheet declaration on the {ctx.lane} lane (zero-thickness PEC "
          "Boxes via add() and PEC add_thin_conductor entries). STALE "
          "IF: the named sheet's SheetSpec.plane is not the printed node.",
          code="sheet_plane_realized", severity="info",
          source="_validate_cfg_pec_realization",
      ))
    if ties:
        desc_t = "; ".join(
            f"{e.label} '{e.name}' normal {'xyz'[a]}: declared mid-plane "
            f"{_fmt_len(declared)} is equidistant from node planes {k} "
            f"({_fmt_len(float(ctx.nodes[a][k]))}) and {k + 1} "
            f"({_fmt_len(float(ctx.nodes[a][k + 1]))}); realized on the "
            f"LOWER one, plane {k}"
            for e, a, k, declared in ties[:_CAMPAIGN_MAX_OFFENDERS])
        _w.warn(PreflightWarning(
            f"{len(ties)} PEC sheet(s) declared with a mid-plane on an "
            f"exact half-cell TIE between two node planes: {desc_t}. A "
            "face-registered one-cell foil (both faces on nodes, "
            "mid-plane at the cell centre) is this case; the contract "
            "resolves the tie to the lower plane so existing "
            "declarations land where they always did, but the choice "
            "is a rule, not the drawing. REMEDY: declare the sheet ON "
            "the plane you mean (a zero-thickness Box at that "
            "coordinate) so no tie is resolved for you.",
            code="sheet_plane_realized", severity="warning",
            source="_validate_cfg_pec_realization",
        ))

def _validate_cfg_sheet_slot_vacuum(self, _w, ctx) -> None:
    """Design note §3 ``sheet_slot_vacuum``: a sheet plane left in a slot.

    A stack-up drawn with a SLOT for the foil — the dielectric below
    ends at the foil's bottom face, the dielectric above starts at its
    top face — leaves the node plane the sheet realizes on with NO
    dielectric: the node sampler is half-open ``[lo, hi)`` and neither
    box contains the plane. The one E edge that reads that node's
    material is the normal edge from the sheet plane into the upper
    cell, half a cell inside a dielectric the drawing says is there,
    so the cavity above the sheet gains a vacuum cell in series (the
    #702 measurement: 17 % on a 127 um stack). Nothing is re-sampled
    silently any more (§2 deleted the #702 resample); this check names
    the slot and the remedy — extend the dielectric boxes to the sheet
    plane. ERROR-grade: the realized stack is not the drawn stack.

    Read from the run's ASSEMBLED ``eps_r`` (the production arrays,
    with the sheet/wire collectors) at the sheet's own plane: fires
    for the footprint nodes where ``eps_r`` on the plane is vacuum
    while the cells on BOTH sides carry a dielectric.
    """
    realized = ctx.realized()
    if realized is None:
        return
    sheets = [e for e in ctx.entry_realizations() if e.kind == "sheet"]
    if not sheets:
        return
    eps = np.asarray(realized.materials.eps_r, dtype=np.float64)
    if eps.ndim != 3:
        return
    rows = []
    for e in sheets:
        sp = e.sheet
        a = int(sp.normal_axis)
        k = int(sp.plane)
        if k - 1 < 0 or k + 1 >= eps.shape[a]:
            continue
        foot = np.any(np.asarray(sp.footprint, dtype=bool), axis=a)
        on = np.take(eps, k, axis=a)
        below = np.take(eps, k - 1, axis=a)
        above = np.take(eps, k + 1, axis=a)
        slot = (foot & (on <= 1.0 + 1e-9)
                & (below > 1.0 + 1e-9) & (above > 1.0 + 1e-9))
        n = int(slot.sum())
        if n == 0:
            continue
        d_k = float(ctx.spacings[a][k])
        rows.append((n, e, a, k, float(below[slot][0]),
                     float(above[slot][0]), d_k))
    if not rows:
        return
    rows.sort(key=lambda t: -t[0])
    desc = "; ".join(
        f"{e.label} '{e.name}' at node plane {k} "
        f"({'xyz'[a]} = {_fmt_len(float(ctx.nodes[a][k]))}): eps_r on "
        f"the plane 1.0 over {n} footprint node(s) while the cell below "
        f"carries eps_r {eb:.4g} and the cell above eps_r {ea:.4g}; the "
        f"normal E edge from the plane into the upper cell "
        f"({_fmt_len(d_k)} long) runs on vacuum"
        for n, e, a, k, eb, ea, d_k in rows[:_CAMPAIGN_MAX_OFFENDERS])
    _w.warn(PreflightWarning(
        f"ERROR-GRADE: {len(rows)} PEC sheet(s) sit in a SLOT of the "
        f"dielectric stack: {desc}. WHY: the stack was drawn with a gap "
        "for the foil (dielectric below ends at its bottom face, "
        "dielectric above starts at its top face), the node sampler "
        "is half-open [lo, hi) so neither box reaches the sheet's node "
        "plane, and a sheet owns no cell and writes no material "
        "(lattice ownership contract #931 §1.3) — so the one cell "
        "whose normal edge reads that node is vacuum in series with "
        "the cavity (the #702 measurement: 17% on a 127um stack). "
        "REMEDY: extend the dielectric boxes to the sheet plane (draw "
        "the laminates edge-to-edge at the foil's plane; a foil has no "
        "thickness on this lattice). Nothing is re-sampled for you. "
        f"COVERAGE: examined {len(sheets)} sheet(s) on the {ctx.lane} "
        "lane against the run's own assembled eps_r. STALE IF: "
        "eps_r at the named plane is not 1.0 on the assembled arrays.",
        code="sheet_slot_vacuum", severity="warning",
        source="_validate_cfg_sheet_slot_vacuum",
    ))

def _validate_cfg_pec_face_short_of_domain_wall(self, _w, ctx) -> None:
    """Design note §6 (cv11): a PEC face one cell shy of a domain wall.

    ``Grid`` realizes a declared extent by ``ceil(extent / dx)`` cells,
    so a 22.86 x 10.16 mm WR-90 declared on a 1 mm mesh is a
    23 x 11 mm guide and the domain-face PEC (§1.8, BC-owned) stands
    on the last INTERIOR node. A conductor meant to reach that wall —
    a shorting plug, a full-width iris — must be drawn to THAT plane,
    because a volume's face rounds to the NEAREST node (§1.1): drawn
    to the declared 10.16 mm the plug's top realizes at 10.000 mm
    under a wall at 11.000 mm, and the one-cell gap between them is a
    parallel-plate line along the broad wall, open at both ends.

    Measured on cv11 2026-09-07 (``scripts/diagnostics/
    pec_short_lane_ab.py``): that slot passed |S21| 0.22-0.33 through
    a "short" and took the pec-short |S11| deficit from 0.0146 to
    0.0560. Pre-#931 the node-half-open sampler included the top node
    by accident, so nothing in the repo had ever had to say this.

    Fires per volume entry, per axis, per side: the entry's own
    realized wall plane (``realized_wall_planes`` on its own edges —
    not a bounding box, not a cell mask) sits exactly one node inside
    a NON-absorbing domain face. Absorbing faces are excluded: there
    is no wall there to short to, and a body grazing an absorber is
    ``geometry_in_absorber``'s finding, not this one.
    """
    interior = getattr(ctx.grid, "interior", None)
    if interior is None:
        return          # non-uniform lane: no interior slices
    from rfx.boundaries.pec import realized_wall_planes

    face_layers = self._preflight_face_layers()
    shape = tuple(ctx.grid.shape)
    rows = []
    for e in ctx.pec_entries():
        if e.kind != "volume":
            continue    # a sheet has no face to draw to a wall
        edges = e.edges(ctx.periodic, shape)
        for a in range(3):
            if ctx.periodic[a] or shape[a] < 3:
                continue
            planes = realized_wall_planes(edges, a)
            if not planes:
                continue
            sl = interior[a]
            for side, face, wall in (
                    ("lo", min(planes), int(sl.start)),
                    ("hi", max(planes), int(sl.stop) - 1)):
                step = 1 if side == "lo" else -1
                if face - wall != step:
                    continue
                if face_layers.get(f"{'xyz'[a]}_{side}", 0):
                    continue    # absorbing face: no wall
                rows.append((e, a, side, face, wall))
    if not rows:
        return
    desc = "; ".join(
        f"{e.label} '{e.name}' {'xyz'[a]}_{side} face realized at node "
        f"{face} ({'xyz'[a]} = {_fmt_len(float(ctx.nodes[a][face]))}) "
        f"against the domain wall at node {wall} "
        f"({_fmt_len(float(ctx.nodes[a][wall]))}) — a ONE-CELL gap of "
        f"{_fmt_len(float(ctx.spacings[a][min(face, wall)]))}"
        for e, a, side, face, wall in rows[:_CAMPAIGN_MAX_OFFENDERS])
    _w.warn(PreflightWarning(
        f"{len(rows)} PEC volume face(s) stop ONE cell short of a "
        f"domain wall instead of reaching it: {desc}. WHY: Grid "
        "realizes a declared domain by ceil(extent/dx) cells, so the "
        "wall stands where the mesh puts it, not at the declared "
        "number; a volume's face rounds to the NEAREST node (lattice "
        "ownership contract #931 §1.1), so a body drawn to the "
        "DECLARED cross-section realizes short of the realized wall. "
        "The vacuum cell left between them is a parallel-plate line "
        "along that wall, open at both ends (measured on cv11 "
        "2026-09-07: |S21| 0.22-0.33 past a PEC short, and the "
        "pec-short |S11| deficit 0.0146 -> 0.0560). REMEDY: draw the "
        "face to the REALIZED wall plane quoted above, not to the "
        "declared dimension (tests/_realized_geometry."
        "domain_wall_positions reads it off the grid the run builds). "
        "If the gap is intended, it is a slot and this says where it "
        f"is. COVERAGE: examined the realized wall planes of "
        f"{len([e for e in ctx.pec_entries() if e.kind == 'volume'])} "
        f"PEC volume(s) on the {ctx.lane} lane, on non-periodic axes "
        "with a non-absorbing face; sheets and wires are not examined "
        "(they have no face to draw to a wall). STALE IF: "
        "realized_wall_planes on the named entity does not reproduce "
        "the printed node.",
        code="pec_face_short_of_domain_wall", severity="warning",
        source="_validate_cfg_pec_face_short_of_domain_wall",
    ))

# ------------------------------------------------------------------
# issue #703 campaign checks, on the realized edge set
# ------------------------------------------------------------------

def _congruence_origin_shift(ctx, members, counts):
    """Predict the best lattice-origin slide for a flagged group.

    Uniform lane only (a per-axis "origin shift" is well-defined only
    when the spacing is one number). Candidates are the shifts that
    snap a member's lo face onto a node, plus the shifts that snap
    each pairwise symmetry plane onto a node or half-node (a mirror
    pair realizes symmetrically when its mirror plane sits on a node
    or half-node). Each candidate is scored by RE-REALIZING the
    members through the production rule on shifted node coordinates
    — the §1.1 centre-sampled volume rule or the §1.3 closed sheet
    footprint, then ``realized_pec_edge_masks`` — no second copy of
    the occupancy rule.

    Restricted to groups whose members are all analytic Boxes: the
    scoring re-realizes every member once per candidate, and for a
    shape whose occupancy is a point-in-mesh query that is a preflight
    that runs for minutes. A group with an inexact member gets the
    geometry-move remedy instead, and the message says which.

    Returns ``(axis_index, shift_m, predicted_spread)`` or ``None``.
    """
    if ctx.lane != "uniform":
        return None
    if not all(exact for (_e, _lo, _hi, exact) in members):
        return None
    d = float(ctx.grid.dx)
    fracs = np.array([ctx.sub_lattice_offsets(lo)
                      for (_e, lo, _hi, _x) in members])

    def _wrap_spread(col):
        s = np.sort(np.asarray(col))
        gaps = np.diff(np.concatenate([s, s[:1] + 1.0]))
        return 1.0 - float(np.max(gaps))

    ax = int(np.argmax([_wrap_spread(fracs[:, a]) for a in range(3)]))
    cands: set[float] = set()
    for f in fracs[:, ax]:
        cands.add(round(-float(f) * d, 15))
        cands.add(round((1.0 - float(f)) * d, 15))
    centers = [0.5 * float(lo[ax] + hi[ax])
               for (_e, lo, hi, _x) in members]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            plane = 0.5 * (centers[i] + centers[j])
            r = plane % (d / 2.0)
            cands.add(round(-r, 15))
            cands.add(round(d / 2.0 - r, 15))
    cands.discard(0.0)
    cand_list = sorted(c for c in cands if abs(c) <= d)[:16]

    best = None
    for s in cand_list:
        cnts = [e.edge_count_shifted(ctx, ax, s)
                for (e, _lo, _hi, _x) in members]
        if any(c is None for c in cnts):
            continue
        spread = max(cnts) - min(cnts)
        if best is None or (spread, abs(s)) < (best[2], abs(best[1])):
            best = (ax, s, spread)
    if best is None or best[2] >= (max(counts) - min(counts)):
        return None
    return best

def _validate_cfg_congruent_rasterization_parity(self, _w, ctx) -> None:
    """#703 check 1: congruent conductors must realize congruently.

    Groups conductor entries by congruence (same shape class and
    realization kind, sorted bounding-box extents equal within
    ``_CONGRUENCE_EXTENT_QUANTUM_M`` — mirror images and right-angle
    rotations share the key by construction) and compares each
    member's REALIZED PEC EDGE COUNT from the shared realization
    (``rfx.boundaries.pec.realized_pec_edge_masks`` on that member's
    own cells / sheet, #931 §1.7). A spread beyond
    ``_CONGRUENCE_SPREAD_TOL_EDGES`` means the lattice broke a symmetry
    the design has. Runs on BOTH the uniform and the non-uniform lane
    (the counts and offsets come from that lane's own builders); the
    origin-shift suggestion is uniform-lane only.

    The census is :meth:`_CampaignStaticsContext.congruence_entries`,
    NOT a Box-only one: a patterned metal LAYER — the incident class
    — cannot be a ``Box`` (a Box fills its clearance holes), so it
    arrives as a user-defined ``Shape`` and a Box-only census skipped
    every member of every mirror pair. For a member whose bounding box
    is not the shape, equal bounds do not PROVE congruence, so the
    message says how many members were keyed that way.
    """
    keyed, unkeyed = ctx.congruence_entries()
    if len(keyed) < 2:
        return
    shape = tuple(ctx.grid.shape)
    groups: dict[tuple, list] = {}
    for e, lo, hi, exact in keyed:
        ext = np.sort(hi - lo)
        key = (
            type(e.shape).__name__, e.kind,
            tuple(int(round(float(x) / _CONGRUENCE_EXTENT_QUANTUM_M))
                  for x in ext),
        )
        groups.setdefault(key, []).append((e, lo, hi, exact))

    flagged = []
    n_groups = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        n_groups += 1
        counts = [e.edge_count(ctx.periodic, shape)
                  for (e, _lo, _hi, _x) in members]
        spread = max(counts) - min(counts)
        if spread > _CONGRUENCE_SPREAD_TOL_EDGES:
            flagged.append((key, members, counts, spread))
    if not flagged:
        return

    flagged.sort(key=lambda t: -t[3])
    key, members, counts, spread = flagged[0]
    ext_key = key[2]
    n_inexact = sum(1 for (_e, _lo, _hi, x) in members if not x)
    member_desc = "; ".join(
        f"{e.label} '{e.name}' ({e.kind}) {c} PEC edges, "
        "lo-corner sub-lattice offsets (x,y,z)=("
        + ", ".join(f"{o:.3f}" for o in ctx.sub_lattice_offsets(lo))
        + ") cells"
        for (e, lo, _hi, _x), c in zip(members, counts)
    )
    inferred = (
        f"INFERRED: {n_inexact} member(s) of the worst group are keyed "
        "by a bounding box that is not the shape itself, so equal "
        "bounds bound congruence rather than proving it — read the "
        "named members before acting. " if n_inexact else "")
    shift = self._congruence_origin_shift(ctx, members, counts)
    if shift is not None:
        ax_name = "xyz"[shift[0]]
        remedy = (
            f"REMEDY: slide the lattice origin by {_fmt_len(shift[1])} "
            f"along {ax_name} (re-realized with that slide, the "
            f"group's spread drops to {shift[2]} edge(s)), or move the "
            "members' shared symmetry plane onto a node or half-node."
        )
    elif ctx.lane == "nonuniform":
        remedy = (
            "REMEDY: place the members at positions congruent modulo "
            "the local cell size (no origin-shift prediction on the "
            "non-uniform lane — a single per-axis slide is not "
            "well-defined when the spacing varies)."
        )
    elif n_inexact:
        remedy = (
            "REMEDY: place the members at positions congruent modulo "
            "the cell size (no origin-shift prediction for this group "
            f"— {n_inexact} member(s) are not analytic Boxes, and "
            "scoring candidate slides would re-realize each of them "
            "16 times inside preflight)."
        )
    else:
        remedy = (
            "REMEDY: place the members at positions congruent modulo "
            "the cell size (no candidate origin slide improved the "
            "spread — the members' offsets differ on more than one "
            "axis, so equalizing them needs a geometry move)."
        )
    _w.warn(PreflightWarning(
        f"{len(flagged)} congruent-conductor group(s) realize to "
        "UNEQUAL PEC edge sets on this lattice (design-identical "
        f"conductors, different meshes). Worst group ({key[0]}, "
        f"{key[1]}, sorted extents "
        + " x ".join(_fmt_len(k * _CONGRUENCE_EXTENT_QUANTUM_M)
                     for k in ext_key)
        + f"): {member_desc}. OBSERVED: edge-count spread {spread} > "
        f"tolerance {_CONGRUENCE_SPREAD_TOL_EDGES} edge (a symmetric "
        "pair realizes identical counts; one edge is the smallest "
        "spread a rounding tie on one face can produce). WHY: "
        "congruent conductors whose faces sit at different sub-cell "
        "offsets are realized from different cell centres (volume) or "
        "node sets (sheet footprint), so the mesh invents an asymmetry "
        "the design does not have — a mirror pair whose mirror plane "
        "is off-lattice realizes asymmetrically with the same sign in "
        "every pair. COST (measured, issue #703): mirror pairs 173 vs "
        "183 cells (5.6%) from a mirror plane 0.26 cells off the "
        "lattice; an A/B run pair differing ONLY by a 13um "
        "lattice-origin slide moved |S11| up to 3.5 dB per bin and "
        f"improved every aggregate agreement metric. {remedy} "
        f"COVERAGE: examined {len(keyed)} conductor declaration(s) "
        f"that report a bounding box, in {n_groups} congruence group(s) "
        f"of >=2 members on the {ctx.lane} lane; skipped "
        f"{len(unkeyed)} conductor entr(y/ies) whose shape reports no "
        f"bounding box (no congruence key). {inferred}STALE IF: "
        "re-realizing the named members through "
        "rfx.boundaries.pec.realized_pec_edge_masks gives equal edge "
        "counts (spread <= tolerance).",
        code="congruent_conductor_rasterization_parity",
        source="_validate_cfg_congruent_rasterization_parity",
    ))

def _validate_cfg_sheet_cavity_thickness(self, _w, ctx) -> None:
    """#703 check 3: each conductor-bounded cavity's electrical thickness.

    Census = every realized wall plane the contract puts on the
    lattice: a sheet's node plane along its normal, and a volume's
    two faces along every axis (``realized_wall_planes`` on the
    entry's own edge masks, #931 §1.7 — the check that used to read
    ``pec_mask`` could not see a volume's far face, issue #767). A
    cavity is two ADJACENT planes of two different conductors over a
    shared footprint with no conductor cell between them. For each,
    compare the MESH electrical thickness — the cells strictly between
    the two planes, on the run's own spacings and assembled ``eps_r``
    at the pair's shared column — against the PHYSICAL stack between
    the DECLARED faces (the lower conductor's upper face to the upper
    conductor's lower face, through the dielectric Box spans), in
    BOTH measures: ``sum(d/eps)`` (series capacitance) and
    ``sum(d*sqrt(eps))`` (phase length). Advisory above
    ``_CAVITY_THICKNESS_TOL`` on either.

    What the difference IS under the contract: each plane's snap
    (declared face vs realized node plane, printed per plane) plus
    any vacuum the drawing left at a sheet plane (named separately by
    ``sheet_slot_vacuum``). A volume's faces are realized where drawn
    when they lie on nodes, so a node-aligned stack reads flush; a
    sheet has no thickness, so a foil declared with faces reads its
    thickness as cavity — the sheet model's honest, quantified cost.
    """
    entries = [e for e in ctx.pec_entries()
               if e.kind in ("volume", "sheet") and e.lo is not None]
    if len(entries) < 2:
        return
    realized = ctx.realized()
    if realized is None:
        return
    shape = tuple(ctx.grid.shape)
    eps_arr = np.asarray(realized.materials.eps_r, dtype=np.float64)
    pec_np = (realized.pec_mask if realized.pec_mask is not None
              else np.zeros(shape, dtype=bool))

    # plane census: (entry, axis, k, footprint2d, declared face coord)
    planes_by_axis: dict[int, list] = {0: [], 1: [], 2: []}
    for e in entries:
        if e.kind == "sheet":
            a = int(e.sheet.normal_axis)
            k = int(e.sheet.plane)
            foot = e.footprint_on_plane(a, k, ctx.periodic, shape)
            # a sheet's plane is both its lower and its upper face
            planes_by_axis[a].append(
                (e, k, foot, float(e.lo[a]), float(e.hi[a])))
            continue
        for a in range(3):
            if shape[a] == 1:
                continue
            pl = e.wall_planes(a, ctx.periodic, shape)
            if not pl:
                continue
            k_lo, k_hi = min(pl), max(pl)
            planes_by_axis[a].append(
                (e, k_lo, e.footprint_on_plane(a, k_lo, ctx.periodic, shape),
                 float(e.lo[a]), float(e.lo[a])))
            if k_hi != k_lo:
                planes_by_axis[a].append(
                    (e, k_hi,
                     e.footprint_on_plane(a, k_hi, ctx.periodic, shape),
                     float(e.hi[a]), float(e.hi[a])))

    nonbox_diel = sum(
        1 for g in self._geometry
        if not isinstance(g.shape, Box)
        and self._resolve_material(g.material_name).sigma
        < self._PEC_SIGMA_THRESHOLD)

    results = []
    n_pairs = 0
    n_skipped_pec_between = 0
    lam0 = C0 / float(self._freq_max)
    for a in range(3):
        axis_planes = sorted(planes_by_axis[a], key=lambda t: t[1])
        inplane = [x for x in range(3) if x != a]
        for si in range(len(axis_planes)):
            e1, k1, foot1, _f1_lo, f1_hi = axis_planes[si]
            covered = np.zeros(foot1.shape, dtype=bool)
            for sj in range(si + 1, len(axis_planes)):
                e2, k2, foot2, f2_lo, _f2_hi = axis_planes[sj]
                if k2 <= k1:
                    continue
                overlap = foot1 & foot2 & ~covered
                if not overlap.any():
                    continue
                covered |= (foot1 & foot2)
                if e2 is e1:
                    continue          # a volume's own two faces
                n_pairs += 1
                idxs = np.argwhere(overlap)
                cen = idxs.mean(axis=0)
                rep = idxs[int(np.argmin(
                    ((idxs - cen) ** 2).sum(axis=1)))]
                col = [0, 0, 0]
                col[inplane[0]] = int(rep[0])
                col[inplane[1]] = int(rep[1])
                # Mesh sums over the cells strictly between the two
                # realized planes: the normal E edges E_a[k1 .. k2-1],
                # each on its own cell's eps_r (#931 §1.1: node k is
                # the lower corner of cell k).
                mesh_cap = mesh_phase = 0.0
                pec_between = False
                for kk in range(k1, k2):
                    col[a] = kk
                    t = tuple(col)
                    if pec_np[t]:
                        pec_between = True
                        break
                    d_loc = float(ctx.spacings[a][kk])
                    ee = float(eps_arr[t])
                    mesh_cap += d_loc / ee
                    mesh_phase += d_loc * math.sqrt(ee)
                if pec_between:
                    n_skipped_pec_between += 1
                    continue
                # A slot at the lower plane (sheet_slot_vacuum's own
                # test on this column): the cell above the plane is
                # vacuum while the cells on both sides carry dielectric.
                slot = False
                if e1.kind == "sheet" and k2 - k1 > 1 and k1 - 1 >= 0:
                    col[a] = k1
                    e_on = float(eps_arr[tuple(col)])
                    col[a] = k1 - 1
                    e_below = float(eps_arr[tuple(col)])
                    col[a] = k1 + 1
                    e_above = float(eps_arr[tuple(col)])
                    slot = (e_on <= 1.0 + 1e-9 and e_below > 1.0 + 1e-9
                            and e_above > 1.0 + 1e-9)
                g_lo, g_hi = f1_hi, f2_lo
                if g_hi <= g_lo:
                    continue
                # in-plane physical point: centre of the two entries'
                # in-plane bounding-box intersection
                p = [0.0, 0.0, 0.0]
                for ia in inplane:
                    lo_ov = max(float(e1.lo[ia]), float(e2.lo[ia]))
                    hi_ov = min(float(e1.hi[ia]), float(e2.hi[ia]))
                    p[ia] = 0.5 * (lo_ov + hi_ov)
                cuts = {g_lo, g_hi}
                diel_boxes = []
                for g in self._geometry:
                    mat = self._resolve_material(g.material_name)
                    if mat.sigma >= self._PEC_SIGMA_THRESHOLD:
                        continue
                    blo, bhi = _sorted_box_corners(g.shape)
                    if blo is None:
                        continue  # non-Box: counted in coverage
                    if not all(blo[ia] <= p[ia] < bhi[ia]
                               for ia in inplane):
                        continue
                    diel_boxes.append((blo, bhi, float(mat.eps_r)))
                    for c in (float(blo[a]), float(bhi[a])):
                        if g_lo < c < g_hi:
                            cuts.add(c)
                edges = sorted(cuts)
                phys_cap = phys_phase = 0.0
                for e_lo, e_hi in zip(edges[:-1], edges[1:]):
                    zm = 0.5 * (e_lo + e_hi)
                    ee = 1.0
                    for blo, bhi, be in diel_boxes:
                        if blo[a] <= zm < bhi[a]:
                            ee = be  # entry order: later wins
                    phys_cap += (e_hi - e_lo) / ee
                    phys_phase += (e_hi - e_lo) * math.sqrt(ee)
                if phys_cap <= 0.0 or phys_phase <= 0.0:
                    continue
                d_cap = mesh_cap / phys_cap - 1.0
                d_phase = mesh_phase / phys_phase - 1.0
                if (abs(d_cap) > _CAVITY_THICKNESS_TOL
                        or abs(d_phase) > _CAVITY_THICKNESS_TOL):
                    gap = g_hi - g_lo
                    z1 = float(ctx.nodes[a][k1])
                    z2 = float(ctx.nodes[a][k2])
                    governs = ("the capacitance measure (sum d/eps) "
                               "governs (gap << lambda at freq_max)"
                               if gap < 0.1 * lam0 else
                               "the phase measure (sum d*sqrt(eps)) "
                               "governs (gap not << lambda)")
                    vac_clause = (
                        f"; the cell above {e1.label}'s plane is "
                        "vacuum between two dielectrics (a slot in "
                        "the drawn stack — see sheet_slot_vacuum)"
                        if slot else "")
                    results.append((max(abs(d_cap), abs(d_phase)), (
                        f"[{'xyz'[a]}] {e1.label} ({e1.kind}, plane "
                        f"k={k1} at {_fmt_len(z1)}, declared face "
                        f"{_fmt_len(g_lo)}, snap {_fmt_signed(z1 - g_lo)})/"
                        f"{e2.label} ({e2.kind}, k={k2} at "
                        f"{_fmt_len(z2)}, declared face "
                        f"{_fmt_len(g_hi)}, snap {_fmt_signed(z2 - g_hi)}) at "
                        f"in-plane column ({int(rep[0])},{int(rep[1])}): "
                        f"sum(d/eps) mesh {_fmt_len(mesh_cap)} vs "
                        f"physical {_fmt_len(phys_cap)} ({d_cap:+.1%}); "
                        f"sum(d*sqrt(eps)) mesh {_fmt_len(mesh_phase)} "
                        f"vs physical {_fmt_len(phys_phase)} "
                        f"({d_phase:+.1%}); plane-to-plane "
                        f"{_fmt_len(z2 - z1)} vs face-to-face "
                        f"{_fmt_len(gap)}; {governs}{vac_clause}")))
    if not results:
        return
    results.sort(key=lambda t: -t[0])
    lines = " | ".join(r[1] for r in results[:_CAMPAIGN_MAX_OFFENDERS])
    _w.warn(PreflightWarning(
        f"{len(results)} conductor-bounded cavit(y/ies) differ from the "
        "physical stack by more than "
        f"{_CAVITY_THICKNESS_TOL:.0%} in electrical thickness: {lines}. "
        "OBSERVED: mesh sums run over the cells strictly between two "
        "adjacent REALIZED wall planes (rfx.boundaries.pec"
        ".realized_wall_planes on each conductor's own edges) on the "
        "run's own spacings and assembled eps_r; physical sums run "
        "between the DECLARED faces through the geometry Box spans at "
        "the pair's shared column. The difference is exactly the "
        "printed per-plane snap (a declared face that is not on a "
        "node realizes on the nearest node plane; a sheet realizes on "
        "one plane, so a foil declared with two faces reads its "
        "thickness as cavity — the sheet model's honest cost) plus any "
        "vacuum cell the drawing left at a sheet plane. WHY BOTH "
        "MEASURES: the same defect class measured 17.3% as a series "
        "capacitance and 3.2% as phase length (#703) — a bare "
        "percentage invites 'correcting' a right number into a wrong "
        "one, so both are printed and the governing one is named. "
        "REMEDY: put the declared faces on mesh nodes (dx = h/N, or "
        "preserved regions on the non-uniform lane) and, for a foil, "
        "declare the sheet ON the interface it bounds so no snap is "
        "resolved for you; a vacuum cell is fixed by extending the "
        "dielectric to the sheet plane. Nothing else is a defect: "
        "a plane-to-plane cavity that equals the declared stack is "
        "what the contract promises. COVERAGE: examined "
        f"{n_pairs} adjacent plane pair(s) from {len(entries)} "
        f"conductor(s) on the {ctx.lane} lane (a sheet's normal plane, "
        "a volume's two faces per axis; coplanar sheet rims are not "
        "paired); physical stack computed from Box entries only — "
        f"{nonbox_diel} non-Box dielectric entr(y/ies) ignored (said "
        f"so, per #703); {n_skipped_pec_between} pair(s) skipped "
        "(conductor cell between). STALE IF: re-summing the printed "
        "column disagrees with these numbers, or realized_wall_planes "
        "on the named entries does not return the printed planes.",
        code="sheet_cavity_electrical_thickness",
        source="_validate_cfg_sheet_cavity_thickness",
    ))

#: A sheet's electrical size differing from its drawing by more than this is
#: reported. 1 % is the frequency tolerance the project judges results by; a
#: resonant dimension moves the resonance by about its own relative error.
_SHEET_EFFECTIVE_SIZE_TOL = 1e-2


def _warn_sheet_effective_size(_w, ctx, boxes) -> None:
    """A PEC sheet is solved about ``EDGE_OFFSET`` of a cell LONGER at each
    in-plane end than the nodes it covers (measured:
    ``scripts/diagnostics/pec_sheet_edge_offset.py``), so the size the solve
    sees is ``covered node span + EDGE_OFFSET * (cell beyond each end)``.
    Reported in input units against the drawn size; an end that lies on the
    domain wall is a wall, not an edge, and adds nothing. This fires for a
    sheet drawn exactly ON the lattice too -- that sheet is 0.6 cell long."""
    from rfx.mesh_edges import EDGE_OFFSET
    domain = tuple(float(v) for v in getattr(ctx.sim, "_domain", (0.0,) * 3))
    rows = []
    sheets = [e for e in boxes if e.kind == "sheet"]
    union = None
    for e in sheets:
        fp = np.asarray(e.sheet.footprint, dtype=bool)
        union = fp.copy() if union is None else (union | fp)

    def _continues(fp, a, i_end, i_next):
        """The metal goes on past this end: every footprint node of the end
        row has sheet metal (another sheet's) on the next node. A seam
        between two abutting sheets is interior metal, not a free edge."""
        end = np.take(fp, i_end, axis=a)
        return bool(end.any()) and bool(np.take(union, i_next, axis=a)[end].all())

    for e in sheets:
        fp = np.asarray(e.sheet.footprint, dtype=bool)
        for a in range(3):
            if a == int(e.sheet.normal_axis):
                continue
            ext = float(e.hi[a] - e.lo[a])
            if ext <= 0.0:
                continue
            other = tuple(i for i in range(3) if i != a)
            idx = np.flatnonzero(fp.any(axis=other))
            if idx.size == 0:
                continue
            nodes = np.asarray(ctx.nodes[a], dtype=float)
            i0, i1 = int(idx[0]), int(idx[-1])
            span = float(nodes[i1] - nodes[i0])
            # An end drawn at (or past) the declared domain boundary is not
            # a free edge: it meets a wall, or it continues into the
            # absorber pad. Only ends strictly inside the domain count.
            dom_hi = float(domain[a])
            tol = 1e-9 * max(dom_hi, 1e-12)
            add = 0.0
            if (i0 > 0 and float(e.lo[a]) > tol
                    and not _continues(fp, a, i0, i0 - 1)):
                add += EDGE_OFFSET * float(nodes[i0] - nodes[i0 - 1])
            if (i1 + 1 < nodes.size and float(e.hi[a]) < dom_hi - tol
                    and not _continues(fp, a, i1, i1 + 1)):
                add += EDGE_OFFSET * float(nodes[i1 + 1] - nodes[i1])
            if add == 0.0:
                continue
            eff = span + add
            rel = (eff - ext) / ext
            if abs(rel) > _SHEET_EFFECTIVE_SIZE_TOL:
                rows.append((abs(rel), rel, e, a, ext, span, eff))
    if not rows:
        return
    rows.sort(key=lambda t: -t[0])
    lines = "; ".join(
        f"{e.label} '{e.name}' {'xyz'[a]}: drawn {_fmt_len(ext)}, nodes "
        f"cover {_fmt_len(span)}, solved as {_fmt_len(eff)} ({rel:+.2%})"
        for _, rel, e, a, ext, span, eff in rows[:_CAMPAIGN_MAX_OFFENDERS])
    _w.warn(PreflightWarning(
        f"{len(rows)} conductor sheet dimension(s) are solved more than "
        f"{_SHEET_EFFECTIVE_SIZE_TOL:.0%} off their drawn size (worst "
        f"{min(len(rows), _CAMPAIGN_MAX_OFFENDERS)} listed): {lines}. "
        "OBSERVED: the node span the sheet's realized footprint covers on "
        f"this run's grid, plus {EDGE_OFFSET:.2f} of the adjacent cell at "
        "each free in-plane edge -- a PEC sheet's edge is solved that far "
        "beyond its last node (measured on this solver, both "
        "polarizations; a flat wall has no such offset). REMEDY: build the "
        "in-plane profiles with rfx.mesh_edges.edge_aware_profiles(domain, "
        "dx, sheets=[...], solids=[...]) and pass them as "
        "Simulation(dx_profile=..., dy_profile=...); a node ON the edge is "
        "not the fix. STALE IF: the footprint's node span on the run's node "
        "coordinates does not reproduce the printed numbers.",
        code="sheet_effective_size",
        source="_validate_cfg_off_lattice_design_edges",
    ))


def _validate_cfg_off_lattice_design_edges(self, _w, ctx) -> None:
    """#703 check 4: census of conductor design edges landing off-lattice.

    Per conductor Box and axis, the largest distance from a declared
    face to its nearest E-node, relative to the axis extent. Under
    the lattice ownership contract a PEC volume's faces realize on
    the cell-centre rule (#931 §1.1: a face off a node plane rounds to
    the nearest plane) and a sheet's footprint is sampled closed on
    the nodes, so every off-node face is a real displacement of the
    realized conductor — there is no sub-cell exclusion (the former
    "node-thin snap" is gone with the rule that produced it). A
    nearest-node residual measures alignment; the realized extent
    depends on both faces and on the conductor's sampling rule.
    Frequency sensitivity requires a model of the affected dimension
    and mode. A sheet's NORMAL axis has no extent to compare
    against and is reported by ``sheet_plane_realized`` instead. One
    aggregated advisory above ``_OFF_LATTICE_EDGE_TOL``, worst
    offenders first.
    """
    boxes = [e for e in ctx.pec_entries()
             if e.kind in ("volume", "sheet") and isinstance(e.shape, Box)]
    others = [e for e in ctx.pec_entries()
              if not (e.kind in ("volume", "sheet")
                      and isinstance(e.shape, Box))]
    if not boxes:
        return
    offenders = []
    n_axes = 0
    n_normal_axes = 0
    for e in boxes:
        lo, hi = e.lo, e.hi
        for a in range(3):
            ext = float(hi[a] - lo[a])
            if e.kind == "sheet" and a == int(e.sheet.normal_axis):
                n_normal_axes += 1
                continue
            if ext <= 0.0:
                continue
            n_axes += 1
            nodes = ctx.nodes[a]
            res = max(
                float(np.min(np.abs(nodes - float(lo[a])))),
                float(np.min(np.abs(nodes - float(hi[a])))),
            )
            rel = res / ext
            if rel > _OFF_LATTICE_EDGE_TOL:
                offenders.append((rel, e, a, ext, res))
    _warn_sheet_effective_size(_w, ctx, boxes)
    if not offenders:
        return
    offenders.sort(key=lambda t: -t[0])
    lines = "; ".join(
        f"{e.label} '{e.name}' ({e.kind}) {'xyz'[a]}: extent "
        f"{_fmt_len(ext)}, worst face residual {_fmt_len(res)} "
        f"({rel:.2%} of the extent)"
        for rel, e, a, ext, res
        in offenders[:_CAMPAIGN_MAX_OFFENDERS])
    _w.warn(PreflightWarning(
        f"{len(offenders)} conductor-Box design edge(s) sit off-lattice "
        f"by more than {_OFF_LATTICE_EDGE_TOL:.1%} of their extent "
        f"(worst {min(len(offenders), _CAMPAIGN_MAX_OFFENDERS)} "
        f"listed): {lines}. OBSERVED: distance from each declared face "
        "to its nearest E-node on this run's own node coordinates; a "
        "PEC volume's face realizes on the nearest node plane and a "
        "sheet footprint on the nodes it covers (lattice ownership "
        "contract #931 §1.1/§1.3). Reported residuals describe "
        "declared-face alignment only. A sheet's extent can change "
        "by more than this nearest-node residual, and an extent "
        "change involves both faces. Read the realized bounds from "
        "fidelity_report(); frequency sensitivity depends on the "
        "mode and the affected dimension. "
        "COST (measured, #703): a uniform-mesh sweep rounded ONE "
        "substrate thickness by 8-10% across three 'convergence' "
        "points — three different boards solved under one name; the "
        "same campaign's board survived at dx=50um only because every "
        "patterned dimension happened to be an exact multiple of 50um. "
        "REMEDY: for a PEC VOLUME choose dx commensurate with its "
        "dimensions, slide the lattice origin onto the worst face, or "
        "(non-uniform lane) place mesh nodes on its faces; for a SHEET see "
        "sheet_effective_size -- a node on a sheet's edge is not the fix. "
        "COVERAGE: "
        f"examined {n_axes} axis extent(s) on {len(boxes)} conductor "
        f"Box declaration(s) on the {ctx.lane} lane; {n_normal_axes} "
        "sheet normal axis/axes reported by sheet_plane_realized "
        f"instead; {len(others)} non-Box conductor entr(y/ies) skipped "
        "(no analytic face coordinates). STALE IF: |face - nearest "
        "node| on the run's node coordinates does not reproduce the "
        "printed residuals.",
        code="off_lattice_design_edges",
        source="_validate_cfg_off_lattice_design_edges",
    ))


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the ten functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py`` states the rule but only
# walks PUBLIC members, and all ten names here are private, so
# ``tests/locks/test_preflight_split_snapshot.py`` pins them directly.
#
# ``_congruence_origin_shift`` is the exception, and it is restored to the
# SAME value for the opposite reason. It was a ``@staticmethod``, so
# ``vars(_PreflightMixin)`` held a ``staticmethod`` OBJECT, and that rewrite
# loop tests ``inspect.isfunction`` -- it has always skipped this one, and
# ``Simulation._congruence_origin_shift.__qualname__`` has always read
# ``_PreflightMixin._congruence_origin_shift``. The facade re-wraps it in
# ``staticmethod(...)``, so the loop keeps skipping it and the pre-move string
# survives. Measured both ways before and after the move.
# ---------------------------------------------------------------------------
_validate_cfg_thin_conductor_surface_impedance.__qualname__ = (
    "_PreflightMixin._validate_cfg_thin_conductor_surface_impedance"
)
_validate_cfg_campaign_statics.__qualname__ = (
    "_PreflightMixin._validate_cfg_campaign_statics"
)
_validate_cfg_pec_realization.__qualname__ = (
    "_PreflightMixin._validate_cfg_pec_realization"
)
_validate_cfg_sheet_slot_vacuum.__qualname__ = (
    "_PreflightMixin._validate_cfg_sheet_slot_vacuum"
)
_validate_cfg_pec_face_short_of_domain_wall.__qualname__ = (
    "_PreflightMixin._validate_cfg_pec_face_short_of_domain_wall"
)
_congruence_origin_shift.__qualname__ = (
    "_PreflightMixin._congruence_origin_shift"
)
_validate_cfg_congruent_rasterization_parity.__qualname__ = (
    "_PreflightMixin._validate_cfg_congruent_rasterization_parity"
)
_validate_cfg_sheet_cavity_thickness.__qualname__ = (
    "_PreflightMixin._validate_cfg_sheet_cavity_thickness"
)
_validate_cfg_off_lattice_design_edges.__qualname__ = (
    "_PreflightMixin._validate_cfg_off_lattice_design_edges"
)
