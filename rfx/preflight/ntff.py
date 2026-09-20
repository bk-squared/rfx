"""Near-to-far-field preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 6. The NTFF family asks whether the far-field
transform can mean anything on the configuration as drawn: whether the
requested NTFF box overlaps declared PEC (an ERROR, the #334 inverse-design
structural gate) or sits inside the radiator's near field at lambda/4,
whether a finite ground plane under the radiator is electrically small enough
to shape the pattern by edge diffraction rather than by the antenna, and
whether an NTFF corner has crossed out of the domain into the absorber
(#500). Everything here was relocated byte for byte out of
``rfx/api/_preflight.py`` -- same text, same order, same indentation, same
docstrings, nothing renamed, reordered, tidied or rewritten. Issue #1030
later DELETED a fourth body from this module; see below.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first, and the CALL CENSUS that chose the extension found the family split
cleanly in two. ``_validate_ntff_inverse_design`` (60 of 65 fixtures) and
``_validate_ntff_small_ground_plane`` (4, because the umbrella only calls it
once the PEC-overlap and near-field arms are past) were already witnessed
three codes over. ``_validate_cfg_ntff_absorber_overlap`` ran on 63 and spoke
on none -- every NTFF box in the corpus was interior -- which the
``ntff_absorber_overlap`` fixture closes.

DELETED, issue #1030: ``_validate_cfg_ntff_min_steps``. Leg 6 moved it here
verbatim and recorded that it was the one rebound name in the whole split
with NO emission site -- it only computed a CFL-based step estimate and wrote
``self._ntff_min_steps_hint``, an instance attribute. The follow-up census
(#1030) found that attribute had NO consumer anywhere in ``rfx/``, ``tests/``,
``validation/``, ``scripts/`` or ``examples/``: its only readers were
``rfx/interop/_design.py``'s ``EXCLUDED_SIMULATION_ATTRS``, which named it to
keep it OUT of the design document, and the test asserting that exclusion --
readers that exist only because the attribute did.

Deleted rather than given a consumer, on three measured grounds. (1) The only
place a consumer could live is ``run()`` AFTER ``_dispatch_plan`` resolves
``n_steps`` (``rfx/api/_execute.py``: ``_auto_preflight`` is called first,
which is why the producer's own comment read "Can't check n_steps here"), and
at that point ``grid.dt`` is concrete -- so a consumer would recompute the
quantity exactly instead of reading a stored estimate. (2) The estimate was
``dx / (C0 * 1.732) * 0.99``, a CUBIC-cell CFL guess, which this project's
engineering rule 2 forbids: on a rectangular or non-uniform mesh it is not the
grid's ``dt``. (3) "10 periods of the lowest NTFF frequency" is an unsourced
heuristic, while the project's evidence-based answer to "was the run long
enough" is the #885 energy ring-down settling witness (``Result.settling_db``
against the -40 dB bar), measured on the run that happened rather than
guessed from ``dx`` before it. Deleting the producer moved NO frozen emission
number: measured 113 sites / 74 literal codes / {preflight: 2,
preflight_sparameters: 2, finding: 1} before and after, because the body
constructed no issue class at all.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic. This module moves no module-level name.

Two edges leave this module and both are attribute lookups on the composed
``Simulation``, so neither costs an import: ``_validate_ntff_inverse_design``
calls ``self._campaign_ctx`` (realization, still in the facade) and
``self._validate_ntff_small_ground_plane`` (intra-module). Its one
cross-family module-global read, ``_absorber_boundary_for_axis``, is a
``_common`` leaf that leg 0 already moved -- which is why
``rfx/preflight/absorber.py``'s docstring used to say this check "stays in
the facade" and, as of this leg, no longer does.
"""

from __future__ import annotations

import jax.numpy as jnp

from rfx.grid import C0

from rfx.preflight._common import (
    _absorber_boundary_for_axis,
    PreflightConfigError,
    PreflightWarning,
)


def _validate_ntff_inverse_design(
    self, *, include_pec_overlap_error: bool = True,
) -> None:
    """NTFF checks: PEC overlap (error) and λ/4 gap (warn).

    CHECK 2: NTFF face plane strictly intersecting a PEC bbox
    (hard error; skipped when ``include_pec_overlap_error=False`` —
    the ``run()`` advisory tier, issue #303).
    CHECK 3: NTFF face closer than λ/4 to any geometry or port/source.
    Passive DFT probes are NOT counted: they read field state without
    perturbing it, so a probe on a box face is a measurement choice,
    not a radiating/scattering culprit (issue #303).
    CHECK 4: source backed by a PEC sheet under ~1λ across
    (warning-severity advisory; both tiers, issue #334) — the far-field
    pattern will be shaped by ground-plane edge diffraction. Expected
    physics, not a solver defect; the advisory exists so a resonance
    fixture is not mistaken for a pattern fixture.
    """
    import warnings as _w

    if self._ntff is None:
        return

    corner_lo, corner_hi, freqs = self._ntff
    # face = (axis, sign, coord, tangential bbox: [(lo_a, hi_a), (lo_b, hi_b)])
    faces = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        tang = ((corner_lo[other[0]], corner_hi[other[0]]),
                (corner_lo[other[1]], corner_hi[other[1]]))
        faces.append(("lo", axis, corner_lo[axis], tang))
        faces.append(("hi", axis, corner_hi[axis], tang))

    # CHECK 2: PEC intersection, CLOSED on the REALIZED wall planes
    # (#931 §1.7). A PEC volume drawn z_a -> z_b realizes walls at
    # BOTH planes, so an NTFF face lying exactly ON a wall plane sits
    # on a conductor surface (tangential E zeroed there) and is an
    # overlap; a sheet (thin conductor or zero-thickness Box) is one
    # plane and is in the census too. Bounds along the normal axis are
    # the entry's own realized wall planes in physical units, read
    # through the shared context; the tangential overlap uses the
    # declared bounds. A traced mesh has no concrete planes, so that
    # lane keeps the declared-bounds test (closed).
    try:
        has_pec_geom = any(
            self._resolve_material(e.material_name).sigma
            >= self._PEC_SIGMA_THRESHOLD
            for e in self._geometry)
    except KeyError:
        has_pec_geom = False   # unresolved material: add()/run() raise
    realized_entries = []
    if include_pec_overlap_error and (
            has_pec_geom or getattr(self, "_thin_conductors", None)):
        ctx = self._campaign_ctx()
        if ctx.error is None:
            shape = tuple(ctx.grid.shape)
            for e in ctx.pec_entries():
                if e.kind == "wire" or e.lo is None:
                    continue
                walls = []
                for a in range(3):
                    planes = e.wall_planes(a, ctx.periodic, shape)
                    if not planes:
                        walls = None
                        break
                    nodes = ctx.nodes[a]
                    walls.append((float(nodes[min(planes)]),
                                  float(nodes[max(planes)]),
                                  ctx.local_spacing(a, float(nodes[min(planes)]))))
                if walls is not None:
                    realized_entries.append((e, walls))
        else:
            for e in ctx.entry_realizations():
                if e.kind in ("wire", "lossy") or e.lo is None:
                    continue
                realized_entries.append((e, [
                    (float(e.lo[a]), float(e.hi[a]), 0.0)
                    for a in range(3)]))
    for side, axis, coord, tang in faces:
        for e, walls in realized_entries:
            w_lo, w_hi, d_loc = walls[axis]
            tol = 1e-9 * d_loc
            if not (w_lo - tol <= coord <= w_hi + tol):
                continue
            # Tangential overlap along the other two axes (declared)
            other = [a for a in range(3) if a != axis]
            overlap = True
            for idx, (tlo, thi) in zip(other, tang):
                if e.hi[idx] <= tlo or e.lo[idx] >= thi:
                    overlap = False
                    break
            if overlap:
                raise PreflightConfigError(
                    f"NTFF face {'xyz'[axis]}_{side} at {coord*1e3:.2f}mm "
                    f"lies on or inside PEC {e.label} '{e.name}' "
                    f"(realized as a {e.kind}: {'xyz'[axis]} walls at "
                    f"[{w_lo*1e3:.3f}, {w_hi*1e3:.3f}] mm; declared "
                    f"bbox {tuple(e.lo)}–{tuple(e.hi)}). NTFF box must "
                    f"enclose all radiators with no conductor surface "
                    f"on or crossing any face. Shrink or move the NTFF "
                    f"box.",
                    code="ntff_pec_overlap",
                    source="_validate_ntff_inverse_design",
                )

    # CHECK 3: λ/2 (Huygens) and λ/4 (reactive-near-field) gaps to any
    # geometry/source (probes excluded, issue #303). Issue #77: the λ/2 Huygens-equivalence rule
    # was documented but only the λ/4 strong
    # tier was enforced; a face at λ/30 above a ground-plane PEC silently
    # ran and produced corrupted directivity. The two-tier check below
    # warns mildly in [λ/4, λ/2) (results may degrade) and strongly in
    # < λ/4 (directivity / pattern likely corrupted).
    if freqs is None:
        return
    try:
        f_max = float(jnp.max(jnp.asarray(freqs)))
    except Exception:
        f_max = float(self._freq_max)
    lam_min = C0 / max(f_max, 1.0)
    gap_thresh = lam_min / 4.0
    huygens_thresh = lam_min / 2.0

    # Collect candidate bboxes and point positions
    bboxes: list[tuple[str, tuple, tuple]] = []
    for entry in self._geometry:
        try:
            c1, c2 = entry.shape.bounding_box()
            bboxes.append((entry.material_name, c1, c2))
        except (NotImplementedError, TypeError, AttributeError):
            continue
    # #931: PEC thin conductors are sheets — conductors in the census.
    from rfx.materials.thin_conductor import sheet_bounds as _sheet_bounds
    for _ti, tc in enumerate(getattr(self, "_thin_conductors", ())):
        if not getattr(tc, "is_pec", False):
            continue
        try:
            c1, c2 = _sheet_bounds(tc.shape)
        except (NotImplementedError, TypeError, AttributeError):
            continue
        if c1 is None or c2 is None:
            continue
        bboxes.append((f"thin_conductor[{_ti}]", tuple(c1), tuple(c2)))
    points: list[tuple[str, tuple]] = []
    for pe in self._ports:
        points.append(("port/source", tuple(pe.position)))
    # Probes intentionally excluded (issue #303): a DFT probe is a
    # passive observer and does not radiate or scatter.

    for side, axis, coord, tang in faces:
        other = [a for a in range(3) if a != axis]
        min_gap = float("inf")
        culprit = None
        # bbox distances
        for name, c1, c2 in bboxes:
            # tangential overlap check — only meaningful gap if the face
            # is "above" the feature in the normal direction
            overlap = True
            for idx, (tlo, thi) in zip(other, tang):
                if c2[idx] <= tlo or c1[idx] >= thi:
                    overlap = False
                    break
            if not overlap:
                continue
            if coord <= c1[axis]:
                d = c1[axis] - coord
            elif coord >= c2[axis]:
                d = coord - c2[axis]
            else:
                d = 0.0  # already handled by CHECK 2 for PEC; skip
                continue
            if d < min_gap:
                min_gap, culprit = d, f"geometry '{name}'"
        # points
        for name, pos in points:
            # require tangential in-box for relevance
            in_tang = all(
                tang[i][0] <= pos[other[i]] <= tang[i][1] for i in range(2)
            )
            if not in_tang:
                continue
            d = abs(coord - pos[axis])
            if d < min_gap:
                min_gap, culprit = d, f"{name} at {pos}"

        if culprit is not None and min_gap < gap_thresh:
            _w.warn(
                PreflightWarning(
                    f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                    f"from {culprit} — below λ/4 = {gap_thresh*1e3:.2f}mm at "
                    f"f_max={f_max/1e9:.2f}GHz. NTFF will integrate reactive "
                    f"near-field; directivity / pattern likely corrupted. "
                    f"Move NTFF box ≥ λ/2 from any radiating/scattering "
                    f"structure (Huygens-equivalence rule).",
                    code="ntff_near_field",
                    source="_validate_ntff_inverse_design",
                ),
                stacklevel=3,
            )
        elif culprit is not None and min_gap < huygens_thresh:
            _w.warn(
                PreflightWarning(
                    f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                    f"from {culprit} — below λ/2 = {huygens_thresh*1e3:.2f}mm "
                    f"at f_max={f_max/1e9:.2f}GHz. Close to reactive near-"
                    f"field; far-field pattern accuracy may degrade. Move "
                    f"NTFF box ≥ λ/2 from radiating/scattering structures.",
                    code="ntff_near_field",
                    source="_validate_ntff_inverse_design",
                ),
                stacklevel=3,
            )

    # CHECK 4 (issue #334): electrically small ground plane under a
    # radiator. Advisory in BOTH tiers — it is pattern physics, not an
    # inverse-design structural gate.
    self._validate_ntff_small_ground_plane(f_max, lam_min)

def _validate_ntff_small_ground_plane(
    self, f_max: float, lam: float,
) -> None:
    """CHECK 4 (issue #334): finite PEC sheet backing a radiator that is
    under ~1λ across → edge-diffraction-shaped far-field pattern.

    Background: a 0.48λ × 0.44λ ground plane produces a pattern dominated
    by ground-plane edge diffraction (broadside dip, off-axis side peaks)
    — correct physics for that geometry, but a trap when the fixture was
    built for resonance/impedance work and its pattern is then read as a
    solver defect. The advisory names the mechanism up front.

    Predicate (warning-severity, fires at most ONCE per preflight):
    - a PEC geometry entry is sheet-like: thin-axis extent
      ``t <= max(λ/20, L_small/10)`` with both lateral extents >= λ/8;
    - a radiator backs it: an ``add_source()`` / lumped-wire
      ``add_port()`` entry sits laterally inside the sheet footprint and
      within half a wavelength of the sheet along the thin axis
      (image-theory coupling zone);
    - among qualifying sheets only the LARGEST footprint is judged — in a
      patch stack that is the ground plane, never the (intentionally
      sub-wavelength) resonant patch element itself;
    - fire iff that sheet's smaller lateral extent < 1λ at the highest
      requested NTFF frequency (sub-wavelength across the whole
      requested pattern band — the conservative direction).

    λ is evaluated at ``f_max`` of the NTFF frequencies: if the sheet is
    sub-wavelength even at the shortest requested wavelength, every bin
    of the requested pattern carries the edge-diffraction shaping.

    TFSF and MSL/waveguide/coax excitations are not counted as radiators
    here (same scope as the CHECK 3 point list): a sub-wavelength PEC
    plate as a scattering target is a legitimate RCS fixture, not a
    ground-plane misuse.
    """
    import warnings as _w

    ports = [tuple(pe.position) for pe in self._ports]
    if not ports:
        return

    best = None  # (lateral_area, L_small, L_big, c1, c2)
    # #931: a ground plane is canonically a SHEET declaration
    # (add_thin_conductor / zero-thickness Box), so PEC thin
    # conductors are in the census beside PEC geometry entries.
    from rfx.materials.thin_conductor import sheet_bounds as _sheet_bounds
    candidates = []
    for entry in self._geometry:
        if entry.material_name != "pec":
            continue
        try:
            c1, c2 = entry.shape.bounding_box()
        except (NotImplementedError, TypeError, AttributeError):
            continue
        candidates.append((c1, c2))
    for tc in getattr(self, "_thin_conductors", ()):
        if not getattr(tc, "is_pec", False):
            continue
        try:
            c1, c2 = _sheet_bounds(tc.shape)
        except (NotImplementedError, TypeError, AttributeError):
            continue
        if c1 is None or c2 is None:
            continue
        candidates.append((tuple(c1), tuple(c2)))
    for c1, c2 in candidates:
        ext = [c2[a] - c1[a] for a in range(3)]
        thin = min(range(3), key=lambda a: ext[a])
        lat = [a for a in range(3) if a != thin]
        l_small = min(ext[lat[0]], ext[lat[1]])
        l_big = max(ext[lat[0]], ext[lat[1]])
        # sheet-like: electrically thin, or thin relative to its own
        # footprint (covers coarse-meshed few-cell-thick ground planes)
        if ext[thin] > max(lam / 20.0, l_small / 10.0):
            continue
        # electrically non-negligible in BOTH lateral dims — wires,
        # narrow straps and tiny pads are not ground planes
        if l_small < lam / 8.0:
            continue
        backed = False
        for pos in ports:
            if not all(c1[a] <= pos[a] <= c2[a] for a in lat):
                continue
            d = max(c1[thin] - pos[thin], pos[thin] - c2[thin], 0.0)
            if d <= lam / 2.0:
                backed = True
                break
        if not backed:
            continue
        area = ext[lat[0]] * ext[lat[1]]
        if best is None or area > best[0]:
            best = (area, l_small, l_big, c1, c2)

    if best is None:
        return
    _, l_small, l_big, c1, c2 = best
    if l_small >= lam:
        return  # ground plane >= ~1λ both ways: clean-pattern regime

    _w.warn(
        PreflightWarning(
            f"Far-field pattern advisory: the PEC sheet backing a source "
            f"(bbox ({c1[0]*1e3:.1f}, {c1[1]*1e3:.1f}, {c1[2]*1e3:.1f})"
            f"–({c2[0]*1e3:.1f}, {c2[1]*1e3:.1f}, {c2[2]*1e3:.1f}) mm) "
            f"spans {l_big*1e3:.1f}mm × {l_small*1e3:.1f}mm = "
            f"{l_big/lam:.2f}λ × {l_small/lam:.2f}λ at "
            f"f_max={f_max/1e9:.2f}GHz — a ground plane under ~1λ "
            f"across. Expect the radiation pattern to be shaped by "
            f"ground-plane edge diffraction (broadside dip, off-axis "
            f"side peaks). This is expected physics, not a solver "
            f"defect, and the fixture stays fine for resonance / "
            f"impedance work. For a clean broadside pattern enlarge the "
            f"ground plane to at least ~1.4λ; if the small ground plane "
            f"is intentional, interpret the pattern accordingly.",
            code="ntff_small_ground_plane",
            source="_validate_ntff_small_ground_plane",
        ),
        stacklevel=4,
    )


def _validate_cfg_ntff_absorber_overlap(
    self,
    _w,
    cpml_thickness: float,
    cpml_thick_lo: list[float],
    cpml_thick_hi: list[float],
    absorber_label: str,
) -> None:
    """P1.4: NTFF box overlap with absorber.

    Issue #500: uses :func:`_absorber_boundary_for_axis` — the CPML
    pad is EXTERIOR to ``[0, domain_extent]`` (see that helper), so an
    NTFF corner is only in the absorber when it is genuinely outside
    the requested domain, not merely within ``ct_{lo,hi}`` of an edge.
    """
    if self._ntff is not None and cpml_thickness > 0:
        corner_lo, corner_hi, _ = self._ntff
        for ax in range(3):
            domain_ext = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
            ax_i = min(ax, 2)
            ct_lo = cpml_thick_lo[ax_i]
            ct_hi = cpml_thick_hi[ax_i]
            lo_b, hi_b = _absorber_boundary_for_axis(domain_ext, ct_lo, ct_hi)
            if (lo_b is not None and corner_lo[ax] < lo_b) or (
                hi_b is not None and corner_hi[ax] > hi_b
            ):
                _w.warn(
                    PreflightWarning(
                        f"NTFF box extends into {absorber_label} region along "
                        f"{'xyz'[ax]}-axis. Far-field results will be "
                        f"corrupted. Shrink NTFF box to interior.",
                        code="absorber_overlap",
                        source="_validate_cfg_ntff_absorber_overlap",
                    ),
                    stacklevel=3,
                )
                break

    # P1.5: non-uniform + NTFF is SUPPORTED (stale "unsupported" note removed
    # 2026-07-02). The NU runner accumulates the NTFF box and
    # compute_far_field handles graded-z per-cell dS + z-edges; a graded-z
    # dipole directivity benchmarks within ~0.05 dB of theory
    # (tests/unit/farfield/test_farfield_nonuniform.py). No guard needed.


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the three functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all three names here are private, so
# ``tests/locks/test_preflight_split_snapshot.py`` pins these three directly.
#
# None of the three was a ``@staticmethod`` -- the one leg 6 moved is in
# ``rfx/preflight/sources.py`` -- so this module has no decorator the facade
# has to re-apply and every restored qualname below becomes
# ``Simulation.<name>`` after composition.
# ---------------------------------------------------------------------------
_validate_ntff_inverse_design.__qualname__ = (
    "_PreflightMixin._validate_ntff_inverse_design"
)
_validate_ntff_small_ground_plane.__qualname__ = (
    "_PreflightMixin._validate_ntff_small_ground_plane"
)
_validate_cfg_ntff_absorber_overlap.__qualname__ = (
    "_PreflightMixin._validate_cfg_ntff_absorber_overlap"
)
