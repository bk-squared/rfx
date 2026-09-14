"""Mesh-quality and non-uniform-lane preflight, moved verbatim out of
``rfx.api._preflight``.

Issue #980 Phase 3, leg 5. The mesh family: whether the grid resolves what is
drawn on it (cells per wavelength in every material, imported-CAD thin
features, the #743 per-body coarsest-cell rule and the Taflove Ch.4
phase-velocity estimate), whether a GRADED mesh is inside the validated
SPEC-01 WP6 envelope and what that envelope excludes, what lands ON a grading
transition (the #669 lossy sheet, the #672 current source, the #688 lumped /
wire port -- each reported on its Ampere-loop axes only, through the shared
``_graded_node_report``), how a Box rasterizes across a graded z column, and
which features the non-uniform and SBP-SAT subgridded lanes do not implement.
Everything here was relocated byte for byte out of ``rfx/api/_preflight.py``
-- same text, same order, same indentation, same docstrings, nothing renamed,
reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first. The extension was chosen by a CALL CENSUS, as legs 3 and 4 were, and it
came back leg 4's answer six times over: all twelve bodies below were entered
-- the family sits on ``_validate_simulation_config``'s unconditional spine,
so 46 to 50 of the 50 fixtures called each -- while SIX of them emitted
nothing. The three graded-node advisories each need a specific object standing
on a grading step; ``_validate_cfg_nonuniform_limitations`` needed two
fixtures because its TFSF finding raises and aborts the body before its
CPML-thickness finding is reached; ``_validate_cfg_subgrid_limitations``
needed a refinement region carrying a refused feature, a pair no test in the
repo had built; and ``_validate_cfg_floquet_nonuniform`` is reachable only
through AUTO-MESH, because ``add_floquet_port`` refuses a DECLARED
``dz_profile`` at registration. Seven fixtures now close all six. The lock
module's own docstring carries the measurement.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

THE THREE CLASS-BODY CONSTANTS DO NOT MOVE. ``_MULTIBAND_RATIO_CAP``,
``_INPLANE_RATIO_CAP`` and ``_AXIS_OF_COMPONENT`` are assignments in the
``_PreflightMixin`` class body, read as ``self.<NAME>`` by four of the bodies
below and -- for the first two -- read off the CLASS by
``tests/unit/nonuniform/test_multiband_nu_envelope.py:329``, which asserts the
two caps are DIFFERENT numbers so that a single shared constant cannot
silently re-couple the in-plane lock to the z one. Moving them here would keep
every ``self.`` read working (the method travels with the constant) while
breaking that test, and no module-level namespace lock would see it happen,
because they are not module-level names. They stay in the facade, the moved
bodies keep reaching them through the composed ``Simulation`` MRO, and
``tests/locks/test_preflight_split_snapshot.py``'s
``test_preflight_mixin_keeps_its_class_body_constants`` pins exactly those
three. The long SPEC-01 WP6 provenance comment stays with the two caps it
documents, for the same reason.

ONE module-level name moves with this leg: ``_local_cell``, re-exported by
``rfx/api/_preflight.py`` so its module namespace stays exactly as wide. The
split inventory filed it as SHARED between ``_validate_mesh_quality`` and
``_CampaignStaticsContext``, i.e. as a leaf for ``_common``. Measured, it is
not shared -- it is not even the same function. ``_CampaignStaticsContext.
entry_realizations`` imports a DIFFERENT ``_local_cell`` from
``rfx.geometry.rasterize_grid`` function-locally, with a different signature
(``(nodes, d, pos)`` against this one's ``(profile, lo, hi, fallback)``), and
that local binding shadows the module global for every read inside it. An AST
scope walk over the whole facade finds four loads of the name: one inside
``entry_realizations``'s nested ``_tie``, resolving to its local import, and
three inside ``_validate_mesh_quality``, resolving to the module global. So
the #743 helper is mesh-family-local and comes here rather than to
``_common``. Nothing outside the facade imports it from there.

That shadowing is also the ONE edit this leg made outside a moved block, and
it is worth naming rather than burying. Once ``_validate_mesh_quality`` left,
the re-exported ``_local_cell`` had no reader inside ``rfx/api/_preflight.py``
at all, and ruff's F811 then read the function-local import as redefining an
unused one -- a green CI lint turning red on a pure code motion. The fix is
the alias ``_local_cell as _rasterize_local_cell`` at that import and at its
single call site: it binds the same object, called with the same arguments,
so no output changes, and it makes the file say what the scope walk found
instead of leaving two different functions sharing one name.
"""

from __future__ import annotations

import math

import numpy as np

from rfx.grid import C0
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box

from rfx.preflight._common import (
    _fmt_freq,
    _fmt_len,
    PreflightConfigError,
    PreflightWarning,
)


def _local_cell(profile, lo, hi, fallback):
    """Coarsest cell a body spans on one axis (#743).

    ``profile`` is a per-cell size array whose cumulative sum gives node
    positions from the padded array's origin; ``lo``/``hi`` are the body's
    physical bounds. Returns ``fallback`` when there is no profile or the
    span selects no cell, so callers keep their previous behaviour on a
    uniform axis.
    """
    if profile is None:
        return fallback
    import numpy as _np
    d = _np.asarray(profile, dtype=float)
    edges = _np.concatenate([[0.0], _np.cumsum(d)])
    inside = (edges[1:] > min(lo, hi)) & (edges[:-1] < max(lo, hi))
    if not inside.any():
        return fallback
    return float(d[inside].max())


def _validate_mesh_quality(self) -> None:
    """Pre-simulation mesh quality check (P0).

    Scans all geometry elements against the grid cell size and warns
    about under-resolved features. Prevents silent garbage results
    from mesh-related setup errors.
    """
    import warnings as _w

    # Tracer-valued profiles (mesh-as-design-variable gradient) cannot
    # participate in host-side min/len/indexing. Advisory warnings
    # are skipped in that case — correctness is preserved downstream.
    if any(
        p is not None and is_tracer(p)
        for p in (self._dx_profile, self._dy_profile, self._dz_profile)
    ):
        return

    dx = self._dx
    if dx is None:
        dx = C0 / self._freq_max / 20.0

    # Determine minimum cell size per axis — use profile min when
    # non-uniform xy is active, so we don't flag features that are
    # actually well-resolved in their local fine-mesh region.
    min_dx = float(min(self._dx_profile)) if self._dx_profile is not None else dx
    min_dy = float(min(self._dy_profile)) if self._dy_profile is not None else dx
    if self._dz_profile is not None:
        min_dz = min(self._dz_profile)
    else:
        min_dz = dx

    for entry in self._geometry:
        shape = entry.shape
        mat_name = entry.material_name

        # Imported CAD mesh (#358): warn when the thinnest dimension (bbox extent — a
        # tessellation-independent proxy, e.g. a plate/wall thickness) falls below ~2 cells;
        # rasterisation is a cell-centre staircase, so a sub-2-cell dimension is lost or
        # misregistered (the #330 thin-conductor class).
        if hasattr(shape, "min_feature_size"):
            try:
                feat = float(shape.min_feature_size())
                cell = float(min(min_dx, min_dy, min_dz))
                if 0.0 < feat < 2.0 * cell:
                    _w.warn(PreflightWarning(
                        f"Imported mesh (material '{mat_name}') thinnest dimension "
                        f"~{feat * 1e3:.3f} mm is below 2 cells "
                        f"(~{2 * cell * 1e3:.3f} mm at dx={cell * 1e3:.3f} mm); it will "
                        f"be lost or staircased by rasterisation. Refine the mesh region "
                        f"(finer dx / subpixel smoothing) or thicken the feature.",
                        code="mesh_import_underresolved", source="_validate_mesh_quality"))
            except (ValueError, AttributeError, TypeError):
                pass

        # Get bounding box dimensions
        if hasattr(shape, "bounding_box"):
            try:
                c1, c2 = shape.bounding_box()
                dims = [abs(c2[i] - c1[i]) for i in range(3)]
            except (NotImplementedError, TypeError):
                continue
        else:
            continue

        # Score against the LOCAL cell where this body actually sits,
        # not the global minimum. Using the finest cell anywhere made
        # the check vacuously green exactly where grading hurts — a
        # body in a coarse region judged by a fine cell it never sees
        # (#743). Falls back to the global minimum when a profile has
        # no usable extent for this body.
        cell_sizes = [
            _local_cell(self._dx_profile, c1[0], c2[0], min_dx),
            _local_cell(self._dy_profile, c1[1], c2[1], min_dy),
            _local_cell(self._dz_profile, c1[2], c2[2], min_dz),
        ]

        # FP1 refinement (2026-05-06): the partial-volume warning
        # at 3-5 cells along one axis is meaningful only for actual
        # *volumes* (≥3 cells in every axis).  A thin strip
        # (e.g. an MSL trace at LX × W_trace × dx → many × 4.7 × 1
        # cells) is a sheet, not a volume, and the per-axis 4.7
        # signal must not fire.  Compute cells on every axis up
        # front and gate the volume branch on the minimum.
        cells_per_axis = [
            (dim / cell) if cell > 0 else float("inf")
            for dim, cell in zip(dims, cell_sizes)
        ]
        is_thin_along_some_axis = min(cells_per_axis) < 3.0

        for axis, (dim, cell) in enumerate(zip(dims, cell_sizes)):
            mat = self._resolve_material(mat_name)
            is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
            axis_name = "xyz"[axis]
            if dim <= 0:
                # Zero-thickness geometry. For a PEC Box that IS the
                # sheet declaration (lattice ownership contract #931
                # §1.5: "zero thickness is a statement of intent") —
                # it is realized on one node plane and reported by
                # ``sheet_plane_realized``, so nothing to say here.
                # For a dielectric a zero-extent Box samples no node
                # and rasterizes to nothing.
                if is_pec:
                    continue
                _w.warn(
                    PreflightWarning(
                        f"Zero-thickness dielectric '{mat_name}' along "
                        f"{axis_name}-axis rasterizes to nothing (the "
                        f"node sampler is half-open [lo, hi), so a "
                        f"zero-extent span contains no node). Give it "
                        f"at least one cell of thickness "
                        f"({_fmt_len(cell)}). (A zero-thickness PEC Box "
                        f"is the sheet declaration and is fine.)",
                        code="mesh_resolution",
                        source="_validate_mesh_quality",
                    ),
                    stacklevel=3,
                )
            elif dim < cell:
                cells_count = dim / cell
                # #931 §1.5: a PEC Box (or any PEC shape with a
                # bounding box) thinner than one local cell is REFUSED
                # at assembly — nothing is inferred from raster
                # thickness. This advisory only documents the rule;
                # the ``pec_box_subcell`` ERROR carries the refusal.
                hint = (
                    " A PEC shape passed to sim.add() is a VOLUME "
                    "(realized with both faces and a shorted interior, "
                    "lattice ownership contract #931 §1.2), and a "
                    "volume thinner than one local cell is refused at "
                    "assembly (§1.5) rather than snapped to a plane. "
                    "If this is foil (a ground plane, patch, trace), "
                    "declare a SHEET: a zero-thickness Box via add(), "
                    "or add_thin_conductor(shape) — realized on the "
                    "node plane nearest its mid-plane, with the normal "
                    "E edge through it live. Otherwise resolve the "
                    "thickness with a finer local cell."
                    if is_pec else
                    " Use non-uniform mesh or reduce dx."
                )
                _w.warn(
                    PreflightWarning(
                        f"'{mat_name}' {axis_name}-extent {_fmt_len(dim)} = "
                        f"{cells_count:.1f} cells — below 1 cell resolution."
                        + hint,
                        code="mesh_resolution",
                        source="_validate_mesh_quality",
                    ),
                    stacklevel=3,
                )
            else:
                # Physics-based resolution thresholds (issue #37).
                # A PEC volume 1-2 cells thick is a FILLED SLAB with
                # walls on both faces (#931 §1.2 — ``pec_box_one_cell``
                # says so for the 1-cell case); its thickness is
                # realized as drawn, so there is no partial-volume
                # question. Only warn on partial volume: 3-5 cells
                # thick PEC bodies that are volumes on every axis.
                # Dielectric: cells per local λ_eff, not cells per
                # geometry extent.
                cells = dim / cell
                if is_pec:
                    if (3.0 <= cells < 5.0
                            and not is_thin_along_some_axis):
                        _w.warn(
                            PreflightWarning(
                                f"PEC '{mat_name}' {axis_name}-extent "
                                f"{_fmt_len(dim)} = {cells:.1f} cells — "
                                "volume under-resolved (a PEC volume's "
                                "curved/edge features need ≥5 cells; a "
                                "1-2 cell slab is realized as drawn, "
                                "with walls on both faces).",
                                code="mesh_resolution",
                                source="_validate_mesh_quality",
                            ),
                            stacklevel=3,
                        )
                else:
                    eps_r = float(mat.eps_r) if mat.eps_r else 1.0
                    lam_eff = (
                        C0 / self._freq_max / math.sqrt(max(eps_r, 1.0))
                    )
                    cells_per_lam = lam_eff / cell
                    # rfx's Yee update is 2nd-order in bulk but
                    # degrades to 1st-order at ε-discontinuities
                    # because subpixel smoothing is default OFF
                    # (Meep ships it ON and stays 2nd-order). For
                    # phase-accurate propagation we need ≥15 cells
                    # per λ_eff — the traditional λ/10 rule applies
                    # to subpixel-smoothed codes. S-parameter
                    # extraction with a port or flux monitor
                    # amplifies dielectric-interface phase error
                    # into |S| magnitude error (see
                    # validation/crossval/11 rfx-vs-analytic audit,
                    # 2026-04-24): at 17.7 cells/λ_eff we measure
                    # ~5% |S21| deficit at Fabry-Perot peaks; at
                    # 35 cells/λ_eff (dx halved) it halves to ~2%.
                    # Require 20 cells/λ_eff when S-param
                    # extraction is active.
                    sparam_active = bool(
                        self._waveguide_ports
                        or self._flux_monitors
                    )
                    threshold = 20.0 if sparam_active else 15.0
                    if cells_per_lam < threshold:
                        suffix = (
                            " S-parameter extraction amplifies "
                            "ε-interface phase error into |S| "
                            "magnitude error; ~5% |S21| deficit "
                            "expected at 17 cells/λ_eff."
                            if sparam_active else
                            " Yee without subpixel smoothing has "
                            "1st-order convergence at ε interfaces."
                        )
                        _w.warn(
                            PreflightWarning(
                                f"dielectric '{mat_name}' on {axis_name}: "
                                f"{cells_per_lam:.1f} cells per λ_eff "
                                f"(eps_r={eps_r:.2f}, freq_max="
                                f"{_fmt_freq(self._freq_max)}, "
                                f"dx={_fmt_len(cell)}). Need ≥"
                                f"{threshold:.0f} cells/λ_eff for "
                                f"phase-accurate propagation."
                                f"{suffix}",
                                code="mesh_resolution",
                                source="_validate_mesh_quality",
                            ),
                            stacklevel=3,
                        )

    # Check gaps between PEC structures
    pec_entries = [e for e in self._geometry if e.material_name == "pec"]
    if len(pec_entries) >= 2:
        for i in range(len(pec_entries)):
            for j in range(i + 1, min(i + 5, len(pec_entries))):
                try:
                    c1a, c2a = pec_entries[i].shape.bounding_box()
                    c1b, c2b = pec_entries[j].shape.bounding_box()
                    # Min gap along each axis
                    for ax in range(3):
                        gap = max(0, max(c1b[ax] - c2a[ax], c1a[ax] - c2b[ax]))
                        cell = [dx, dx, min_dz][ax]
                        if 0 < gap < 3 * cell:
                            _w.warn(
                                PreflightWarning(
                                    f"Gap between PEC structures: "
                                    f"{_fmt_len(gap)} = {gap/cell:.1f} cells "
                                    f"along {'xyz'[ax]} — coupling may be "
                                    f"under-resolved.",
                                    code="mesh_resolution",
                                    source="_validate_mesh_quality",
                                ),
                                stacklevel=3,
                            )
                except (NotImplementedError, TypeError, AttributeError):
                    continue

    # Physics-based numerical dispersion check (Taflove Ch. 4).
    # Instead of a fixed aspect-ratio heuristic, compute the actual
    # per-axis phase velocity error at freq_max from the FDTD
    # dispersion relation. This is application-independent.
    self._check_numerical_dispersion()

    # Thin-metal-on-NU-mesh symmetry (Meep/OpenEMS convention — issue #48).
    self._validate_thin_metal_on_nu_mesh()

def _check_numerical_dispersion(self) -> None:
    """Warn when per-axis FDTD phase velocity error at freq_max
    exceeds a threshold (Taflove Ch. 4 dispersion relation).

    For each axis the worst-case phase velocity is:
        v_ph = (omega·dt) / (2·arcsin(nu_i · sin(k·d_i/2)))
    where nu_i = c·dt/d_i, k = 2π/λ, d_i = cell size along axis i.

    Reports the per-axis error so the user sees which axis is under-
    resolved or has Courant mismatch — no arbitrary ratio threshold.
    """
    import warnings as _w

    # Skip host-side min when any profile is a tracer. The dispersion
    # warning is advisory only; mesh-as-design-variable optimisation
    # runs under tracing and the warning cannot fire correctly there.
    if any(
        p is not None and is_tracer(p)
        for p in (self._dx_profile, self._dy_profile, self._dz_profile)
    ):
        return

    dx_nom = self._dx or (C0 / self._freq_max / 20.0)
    d = [dx_nom, dx_nom, dx_nom]
    if self._dx_profile is not None:
        d[0] = float(np.min(self._dx_profile))
    if self._dy_profile is not None:
        d[1] = float(np.min(self._dy_profile))
    if self._dz_profile is not None:
        d[2] = float(np.min(self._dz_profile))

    inv_sq = sum(1.0 / di ** 2 for di in d)
    dt_cfl = 0.99 / (C0 * math.sqrt(inv_sq))
    omega = 2.0 * math.pi * self._freq_max

    errors = {}
    sin_wdt2 = math.sin(omega * dt_cfl / 2.0)
    for ax, (name, di) in enumerate(zip("xyz", d)):
        # Taflove Eq. 4.44: v_ph along axis i
        # = omega * d_i / (2 * arcsin(d_i * sin(omega*dt/2) / (c*dt)))
        arg = di * sin_wdt2 / (C0 * dt_cfl)
        if abs(arg) >= 1.0:
            errors[name] = float("inf")
            continue
        v_ph = omega * di / (2.0 * math.asin(arg))
        errors[name] = abs(v_ph - C0) / C0

    max_err = max(errors.values())
    if max_err > 0.02:
        parts = ", ".join(
            f"{name}={err*100:.1f}%" for name, err in errors.items()
        )
        worst = max(errors, key=errors.get)
        _w.warn(
            PreflightWarning(
                f"FDTD numerical dispersion at freq_max="
                f"{self._freq_max/1e9:.2f}GHz exceeds 2%: {parts}. "
                f"Worst axis: {worst} (cell {d['xyz'.index(worst)]*1e3:.3f}mm). "
                f"Phase velocity error causes resonance frequency bias. "
                f"Refine the coarse axis or co-refine all axes together "
                f"(Taflove Ch. 4).",
                code="numerical_dispersion",
                source="_check_numerical_dispersion",
            ),
            stacklevel=4,
        )

def _validate_thin_metal_on_nu_mesh(self) -> None:
    """Warn when a realized metal PLANE sits on a NU axis without
    symmetric neighbouring cells (Meep/OpenEMS require equal dz on
    both sides of a metal plane, else surface currents pick up O(1)
    error and the far-field pattern is corrupted — issue #48).

    Lattice ownership contract (#931): the plane examined is the one
    the run REALIZES — a sheet's node plane, or each wall plane of a
    thin volume (a slab at most two cells thick along the NU axis,
    whose two faces are both surface-current planes) — read from the
    shared realization, never from a bounding-box midpoint (which
    could name the wrong cell). Thick volumes are not "thin metal"
    and are skipped.
    """
    import warnings as _w
    profiles = (self._dx_profile, self._dy_profile, self._dz_profile)
    if all(p is None for p in profiles):
        return
    if any(p is not None and is_tracer(p) for p in profiles):
        # Tracer profiles can't be host-scanned for edge / ratio
        # checks. The warning is advisory only; correctness is
        # preserved downstream.
        return
    ctx = self._campaign_ctx()
    if ctx.error is not None:
        return
    shape = tuple(ctx.grid.shape)
    for axis_idx, prof in enumerate(profiles):
        if prof is None:
            continue
        axis_name = "xyz"[axis_idx]
        d = ctx.spacings[axis_idx]
        if d.size < 3:
            continue
        for e in ctx.pec_entries():
            if e.kind == "wire":
                continue
            planes = e.wall_planes(axis_idx, ctx.periodic, shape)
            if not planes:
                continue
            if e.kind == "volume":
                # thin slab: at most two cell layers between its faces
                if max(planes) - min(planes) > 2:
                    continue
            for k in planes:
                if k - 1 < 0 or k >= d.size:
                    continue
                d_below = float(d[k - 1])
                d_above = float(d[k])
                ratio = max(d_below, d_above) / min(d_below, d_above)
                if ratio <= 1.5:
                    continue
                _w.warn(
                    PreflightWarning(
                        f"Thin PEC {e.label} ('{e.name}', realized as a "
                        f"{e.kind}) has a metal plane at {axis_name} = "
                        f"{_fmt_len(float(ctx.nodes[axis_idx][k]))} "
                        f"(node {k}) with asymmetric neighbouring cells "
                        f"({_fmt_len(d_below)} below, {_fmt_len(d_above)} "
                        f"above, ratio {ratio:.2f}). Meep/OpenEMS require "
                        f"equal cell sizes across a metal plane; "
                        f"radiation pattern may be corrupted (issue #48). "
                        f"Put the metal plane on a preserved-region "
                        f"boundary or refine the neighbouring cell.",
                        code="thin_metal_nu_mesh",
                        source="_validate_thin_metal_on_nu_mesh",
                    ),
                    stacklevel=4,
                )


def _validate_cfg_graded_box_rasterization(self, _w) -> None:
    """Warn when a Box misses the fine cells implied by its z-span."""
    if self._dz_profile is None or is_tracer(self._dz_profile):
        return

    dz = np.asarray(self._dz_profile, dtype=np.float64)
    if dz.size == 0:
        return
    edges = np.concatenate(([0.0], np.cumsum(dz)))
    # Sample positions must be the ones the RUN uses, and the occupancy
    # rule must be the run's rule. This validator previously modelled both
    # by hand — cell centres plus a bare half-open span — and was wrong on
    # each count: dielectric coordinates are E-NODES (cell edges) since
    # #562 (the node sampler's thin-sheet branch snaps a sub-cell
    # dielectric Box onto its nearest node, #48/#75/#371). With the hand
    # model this check diverged from the rasterizer on boxes straddling a
    # grading transition — the #325 signature it exists to catch — and
    # went SILENT on one of them (#562 review, F2). So call the production
    # path instead of imitating it: node positions through the same two
    # steps the grid builder composes, and the shape's own mask for a
    # dielectric; for a PEC Box the production rule is the #931 §1.1
    # CENTRE-sampled volume rule (half-open on cell centres), spelled by
    # ``rasterize_grid._box_axis_volume`` and called here. x/y are passed
    # as single-sample arrays because only the z profile is needed here;
    # the mask admits the box's own midpoint, so the combined mask's z
    # profile is the z-axis mask.
    from rfx.nonuniform import node_positions_from_profile
    z_nodes = np.asarray(node_positions_from_profile(dz), dtype=np.float64)

    z_centres = z_nodes[:-1] + 0.5 * np.diff(z_nodes)
    for entry in self._geometry:
        if not isinstance(entry.shape, Box):
            continue
        c1, c2 = entry.shape.bounding_box()
        z_lo, z_hi = sorted((float(c1[2]), float(c2[2])))
        thickness = z_hi - z_lo
        if thickness <= 0.0:
            continue

        x_mid = np.array([0.5 * (float(c1[0]) + float(c2[0]))])
        y_mid = np.array([0.5 * (float(c1[1]) + float(c2[1]))])
        try:
            is_pec = (self._resolve_material(entry.material_name).sigma
                      >= self._PEC_SIGMA_THRESHOLD)
        except KeyError:
            is_pec = False
        if is_pec:
            # #931 §1.1: a PEC VOLUME is centre-sampled, half-open on
            # the cell centres — the production rule
            # (rasterize_grid._box_axis_volume), not the node sampler.
            from rfx.geometry.rasterize_grid import _box_axis_volume
            actual = int(np.count_nonzero(
                np.asarray(_box_axis_volume(z_centres, z_lo, z_hi))))
        else:
            mask = np.asarray(entry.shape.mask_on_coords(x_mid, y_mid, z_nodes))
            actual = int(np.count_nonzero(mask))
        local = (edges[:-1] < z_hi) & (edges[1:] > z_lo)
        if not np.any(local):
            continue
        # The #325 signature is a fine band SHIFTED OUT of the Box span by
        # smooth_grading's transition insertion — the intended fine cells
        # then sit ADJACENT to the span, so measure min dz over a padded
        # neighborhood (±5 cells), not the span alone (span-only misses
        # the shifted-substrate case entirely).
        idx = np.flatnonzero(local)
        lo_i = max(0, int(idx[0]) - 5)
        hi_i = min(dz.size, int(idx[-1]) + 6)
        implied = thickness / float(np.min(dz[lo_i:hi_i]))

        if actual < math.ceil(0.5 * implied) and actual <= 4:
            _w.warn(
                PreflightWarning(
                    f"Box material '{entry.material_name}' rasterizes to "
                    f"{actual} z cells (implied {implied:.1f}) over z-span "
                    f"[{_fmt_len(z_lo)}, {_fmt_len(z_hi)}). "
                    "smooth_grading transition cells may have shifted the "
                    "fine band — derive z coordinates from the actual "
                    "fine-band edges and assert the rasterized cell count "
                    "(issue #325)",
                    code="graded_box_rasterization",
                    loc=f"z=[{z_lo}, {z_hi})",
                    source="_validate_cfg_graded_box_rasterization",
                ),
                stacklevel=3,
            )


def _validate_cfg_floquet_nonuniform(self) -> None:
    """P1.1: Floquet + non-uniform mesh — no silent fallback allowed."""
    if self._floquet_ports and self._dz_profile is not None:
        raise PreflightConfigError(
            "Floquet ports do not support non-uniform z mesh (dz_profile). "
            "Use the uniform reference lane and set dx explicitly.",
            code="floquet_nonuniform",
            source="_validate_cfg_floquet_nonuniform",
        )


def _validate_cfg_multiband_grading(self, _w) -> None:
    """P2: multi-band graded-mesh envelope advisories (SPEC-01 WP6).

    Two advisory-tier checks on every explicit per-axis profile. A
    profile whose adjacent ratios are all <= 1.4 draws NEITHER — that
    is the "allow" half of WP6: a small-large-small-large profile is
    now a documented, witnessed configuration and must construct and
    preflight clean.

    1. ``nu_grading_ratio_beyond_validated_cap`` — some adjacent-cell
       ratio exceeds the 1.4 cap. Quotes the measured accuracy class
       on both sides of the cap so the reader can price the choice.
    2. ``nu_grading_reaches_absorber`` — an axis has an active
       absorber face and its adjacent interior runway is not uniform.

    Why check 2 exists, and where its DEPTH comes from. The absorber
    pad replicates the outermost interior cell
    (``rfx/nonuniform._pad_profile``), so the absorber itself is
    always uniformly meshed; what a boundary-adjacent transition does
    is change the discrete medium in the boundary-NORMAL direction
    right where the absorber begins. Normal-direction inhomogeneity
    is the documented breakdown class for PML generally (Meep's PML
    documentation: PML tolerates media varying only in the
    boundary-PARALLEL directions), and it leaves the transition's own
    reflection no interior runway in which to separate from the
    absorber's. The depth used is the FACE'S OWN allocated layer
    count — the only length the absorber itself defines, and the span
    over which its conductivity ramp acts. No measurement sets this
    depth and none could: no multi-band witness measures an accuracy
    observable with an absorber present (every accuracy witness is
    PEC-closed at cpml_layers = 0; the one absorber-bearing witness,
    F-S5, checks the AD path and scores no accuracy quantity), which
    is precisely why the combination is flagged rather than scored. Tolerance 1e-6 on the adjacent ratio, i.e. a runway is
    "uniform" up to 1 ppm of cell size — far below any grading a user
    can mean, and above float round-tripping in a computed profile.

    Inherited caveat: ``_preflight_face_layers`` over-reports the
    absorber on the non-port axes of a waveguide-port simulation (the
    grid drops those axes from ``cpml_axes``, the helper does not).
    This check inherits that, as every other consumer of the helper
    does; the effect is a possible advisory on an axis whose absorber
    the grid will not actually allocate. Advisory tier, so it costs a
    line of noise, never a run.
    """
    face_layers = None
    for ax_name, profile in (("x", self._dx_profile),
                             ("y", self._dy_profile),
                             ("z", self._dz_profile)):
        if profile is None or is_tracer(profile) or len(profile) < 2:
            continue
        p = np.asarray(profile, dtype=float)
        ratios = p[1:] / p[:-1]
        max_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
        cap = (self._MULTIBAND_RATIO_CAP if ax_name == "z"
               else self._INPLANE_RATIO_CAP)
        scope = ("validated multi-band grading cap"
                 if ax_name == "z" else
                 "in-plane grading threshold (the validated 1.4 "
                 "multi-band cap is a z-axis envelope: no witness in "
                 "it grades an in-plane axis)")
        if max_ratio > cap + 1e-6:
            _w.warn(
                PreflightWarning(
                    f"d{ax_name}_profile max adjacent cell ratio "
                    f"{max_ratio:.3f} exceeds the {scope} "
                    f"{cap:g} "
                    f"(docs/guides/support_matrix.md, 'Multi-band graded "
                    f"mesh'). WHY: stability is unaffected (dt is the "
                    f"global min-cell CFL, sufficient on any tensor "
                    f"grid), but per-transition accuracy is witnessed "
                    f"only along z and only up to ratio 1.4 — the -54 dB "
                    f"reflection class at 30 cells/wavelength (it scales "
                    f"as (dz/lambda)^2), against -44 dB measured at "
                    f"ratio 2.0. COST: an unquantified reflection and "
                    f"grading-dispersion error at that transition. "
                    f"REMEDY: split it into ratio<={cap:g} steps (e.g. "
                    f"rfx.smooth_grading) to stay inside the "
                    f"{'validated envelope' if ax_name == 'z' else 'pre-existing in-plane threshold (no in-plane envelope is validated)'}, "
                    f"or accept it and say so in the report. STALE-IF: "
                    f"the support-matrix row raises the cap.",
                    code="nu_grading_ratio_beyond_validated_cap",
                    source="_validate_cfg_multiband_grading",
                ),
                stacklevel=3,
            )
        if face_layers is None:
            face_layers = self._preflight_face_layers()
        for side in ("lo", "hi"):
            layers = face_layers.get(f"{ax_name}_{side}", 0)
            if layers < 2 or len(p) < layers:
                # layers < 2: "one uniform cell" is vacuous (no adjacent
                # ratio exists inside a single-cell runway).
                continue
            # The REMEDY asks for `layers` uniform interior cells
            # against the face, i.e. `layers - 1` adjacent ratios among
            # p[0..layers-1]; taking layers+1 cells here demanded one
            # uniform cell MORE than the remedy and fired on an exactly
            # compliant profile (review 2026-08-29).
            runway = p[:layers] if side == "lo" else p[-layers:]
            r_run = runway[1:] / runway[:-1]
            dev = float(np.max(np.abs(r_run - 1.0)))
            if dev > 1e-6:
                _w.warn(
                    PreflightWarning(
                        f"d{ax_name}_profile is not uniform within the "
                        f"{layers} interior cells adjacent to the "
                        f"{ax_name}_{side} absorber face (max adjacent "
                        f"cell-ratio deviation {dev:.3g}). WHY: the "
                        f"absorber pad replicates the outermost interior "
                        f"cell, so a transition in that runway changes "
                        f"the discrete medium in the boundary-NORMAL "
                        f"direction exactly where the absorber starts — "
                        f"the documented PML breakdown class — and gives "
                        f"the transition's own reflection no runway to "
                        f"separate from the absorber's. COST: the "
                        f"validated multi-band envelope does not cover "
                        f"it at all; no witness in it MEASURES an "
                        f"accuracy observable with an absorber present "
                        f"(every accuracy witness is PEC-closed at "
                        f"cpml_layers = 0; the one absorber-bearing "
                        f"witness checks the AD path only) "
                        f"(docs/guides/support_matrix.md, 'Multi-band "
                        f"graded mesh'). REMEDY: keep at least "
                        f"{layers} uniform interior cells against that "
                        f"face, or close it with PEC/PMC. STALE-IF: a "
                        f"witness that SCORES an accuracy observable "
                        f"with an absorber present is added to that "
                        f"row.",
                        code="nu_grading_reaches_absorber",
                        source="_validate_cfg_multiband_grading",
                    ),
                    stacklevel=3,
                )

def _validate_cfg_thin_conductor_graded_node(self, _w) -> None:
    """Advisory: a LOSSY thin conductor landing on a grading transition.

    A lossy sheet folds into ``materials.sigma`` at ONE E node along its
    normal, and the length that node's sigma acts over is the DUAL spacing
    ``(d[k-1]+d[k])/2`` — not either adjacent cell. Where the two adjacent
    cells are equal the distinction is invisible, which is precisely how
    the pre-#669-review fold (which divided by the primal cell ``d[k]``)
    stayed silent: every uniform mesh and every NU node away from a step
    agrees. This check surfaces the case that does not agree.

    Advisory tier, concrete profiles only, LOSSY sheets only — a PEC thin
    sheet is a :class:`~rfx.boundaries.pec.SheetSpec` (#931 §1.3) and
    folds no sigma, so it has no sheet resistance to normalize.
    Threshold: adjacent cells differing by more than 10%.

    The node is located through the lossy fold's OWN production path
    (node positions + the shape's node mask): the sigma-fill sheet is
    fenced out of the #931 contract (design note §1.8 — a lossy volume
    model, unchanged), so mirroring its mask here IS reading what the
    run realizes, not a re-derivation.
    """
    from rfx.nonuniform import node_positions_from_profile
    from rfx.materials.thin_conductor import sheet_bounds

    tcs = [tc for tc in getattr(self, "_thin_conductors", ())
           if not getattr(tc, "is_pec", False)]
    if not tcs:
        return
    profiles = (self._dx_profile, self._dy_profile, self._dz_profile)
    if all(p is None or is_tracer(p) for p in profiles):
        return

    for i, tc in enumerate(tcs):
        # Bounds source (issue #674): a surface-impedance sheet may be any
        # ``mask_on_coords`` shape, so read its bounding box; the legacy DC
        # fold is still Box-only on this lane (it warn-and-skips a non-Box
        # sheet), and advising about a sheet that is not folded would be
        # worse than silence.
        if getattr(tc, "surface_impedance_f0", None) is not None:
            lo, hi = sheet_bounds(tc.shape)
        else:
            lo = getattr(tc.shape, "corner_lo", None)
            hi = getattr(tc.shape, "corner_hi", None)
        if lo is None or hi is None:
            continue
        extents = [float(hi[a]) - float(lo[a]) for a in range(3)]
        n_axis = min(range(3), key=lambda a: extents[a])
        prof = profiles[n_axis]
        if prof is None or is_tracer(prof):
            continue
        d = np.asarray(prof, dtype=np.float64)
        if d.size < 2:
            continue

        # Locate the node the RUN will realize the sheet on, through the
        # production path (node positions + the shape's own mask), not a
        # hand-rolled nearest-node rule (#562 review F2, #568).
        nodes = np.asarray(node_positions_from_profile(d),
                           dtype=np.float64)
        _other = tuple(a for a in range(3) if a != n_axis)

        def _occupied_layers(fracs, _tc=tc, _lo=lo, _hi=hi,
                             _n_axis=n_axis, _nodes=nodes,
                             _other=_other):
            args = []
            for a in range(3):
                if a == _n_axis:
                    args.append(_nodes)
                    continue
                a_lo, a_hi = float(_lo[a]), float(_hi[a])
                args.append(np.array(
                    [a_lo + f * (a_hi - a_lo) for f in fracs],
                    dtype=np.float64))
            m = np.asarray(_tc.shape.mask_on_coords(*args))
            return np.flatnonzero(m.any(axis=_other))

        # Bounding-box centre first — for a Box that is the whole story,
        # and it is bit-identically the probe this check has always run.
        hit = _occupied_layers((0.5,))
        if hit.size == 0:
            # #674: a PATTERNED sheet can have its bbox centre inside a
            # clearance hole, which would read as "no sheet here" and
            # silently drop the advisory. Fan out before giving up.
            hit = _occupied_layers((0.1, 0.3, 0.5, 0.7, 0.9))
        if hit.size == 0:
            continue
        k = int(hit[0])

        # cells adjacent to node k: d[k-1] below, d[k] above. The lo face
        # (k == 0) and the last node (k == d.size, backed by the bounding
        # -node duplicate d[k] == d[k-1]) are matched by construction.
        if k == 0 or k >= d.size:
            continue
        d_below, d_above = float(d[k - 1]), float(d[k])
        small, large = sorted((d_below, d_above))
        if small <= 0.0 or (large / small) - 1.0 <= 0.10:
            continue
        axis_name = "xyz"[n_axis]
        dual = 0.5 * (d_below + d_above)
        _w.warn(
            PreflightWarning(
                f"lossy thin conductor #{i} sits at "
                f"{axis_name} = {_fmt_len(float(nodes[k]))}, an E node "
                f"whose adjacent cells differ by "
                f"{(large / small - 1.0):.0%} ({_fmt_len(d_below)} below, "
                f"{_fmt_len(d_above)} above). Its sheet fold is "
                f"normalized by the E-node DUAL spacing "
                f"{_fmt_len(dual)} — the length that node's sigma acts "
                f"over — which is neither adjacent cell. That IS the "
                f"correct normalization, but a sheet on a grading step is "
                f"where the realized sheet resistance is most "
                f"mesh-sensitive: move the sheet onto a locally uniform "
                f"node (or flatten the grading there) if its loss is "
                f"claims-bearing.",
                code="thin_conductor_graded_node",
                source="_validate_cfg_thin_conductor_graded_node",
            ),
            stacklevel=3,
        )


def _graded_node_report(self, axis: int, coord: float):
    """``(node_pos, d_below, d_above, dual, ratio)`` when the E node
    nearest ``coord`` on ``axis`` sits on a >10% grading step, else None.

    Concrete profiles only (a tracer profile is skipped, matching the
    #671 check). The node is located with ``node_positions_from_profile``
    — the production node convention — not a hand-rolled rule (#562
    review F2, #568). ``k == 0`` and ``k >= d.size`` are matched by
    construction and never report.
    """
    from rfx.nonuniform import node_positions_from_profile

    prof = (self._dx_profile, self._dy_profile, self._dz_profile)[axis]
    if prof is None or is_tracer(prof):
        return None
    d = np.asarray(prof, dtype=np.float64)
    if d.size < 2:
        return None
    nodes = np.asarray(node_positions_from_profile(d), dtype=np.float64)
    k = int(np.argmin(np.abs(nodes - float(coord))))
    if k == 0 or k >= d.size:
        return None
    d_below, d_above = float(d[k - 1]), float(d[k])
    small, large = sorted((d_below, d_above))
    if small <= 0.0 or (large / small) - 1.0 <= 0.10:
        return None
    return (float(nodes[k]), d_below, d_above,
            0.5 * (d_below + d_above), large / small)

def _validate_cfg_source_on_graded_node(self, _w) -> None:
    """Advisory: a current source sitting on a grading transition (#672).

    ``make_current_source`` divides the waveform by the E node's control
    volume, which is the PRIMAL per-cell width on the component's own
    axis and the DUAL spacing on the two TRANSVERSE axes. The parallel
    axis is exact by construction, so only the transverse axes are
    checked. Advisory tier: the normalization IS correct now — this
    flags where the realized current moment is most mesh-sensitive.

    The counterfactual ratio in the message is ``d[k] / dual``, where
    ``d[k]`` is the cell ABOVE the node (``d_above``) — that is the width
    the pre-#672 code actually used on every axis, so it is the one that
    makes the number and the sentence describe the same thing. Using
    ``max(d_below, d_above)`` instead, as this message did until the #673
    split review, silently reported the wrong cell on every DOWN-step
    node (where ``d_above < d_below``).
    """
    entries = [pe for pe in getattr(self, "_ports", ())
               if float(getattr(pe, "impedance", 0.0)) == 0.0]
    if not entries:
        return
    for pe in entries:
        axis = self._AXIS_OF_COMPONENT.get(pe.component)
        if axis is None:
            continue
        for a in (ax for ax in range(3) if ax != axis):
            rep = self._graded_node_report(a, pe.position[a])
            if rep is None:
                continue
            node_pos, d_below, d_above, dual, ratio = rep
            _w.warn(
                PreflightWarning(
                    f"current source at {pe.position} "
                    f"(component {pe.component}) sits at "
                    f"{'xyz'[a]} = {_fmt_len(node_pos)}, an E node whose "
                    f"adjacent cells differ by {(ratio - 1.0):.0%} "
                    f"({_fmt_len(d_below)} below, {_fmt_len(d_above)} "
                    f"above). {'xyz'[a]} is one of this component's two "
                    f"TRANSVERSE axes, so its control volume takes the "
                    f"DUAL spacing {_fmt_len(dual)} there — not the "
                    f"primal cell {_fmt_len(d_above)} — and the realized "
                    f"current moment would be off by "
                    f"{d_above / dual:.3f}x "
                    f"on this axis alone if the primal cell were used "
                    f"(issue #672). That IS handled, but a source on a "
                    f"grading step is where the injected amplitude is "
                    f"most mesh-sensitive: move it onto a locally "
                    f"uniform node if its amplitude is claims-bearing.",
                    code="source_on_graded_node",
                    source="_validate_cfg_source_on_graded_node",
                ),
                stacklevel=3,
            )

def _validate_cfg_wire_port_on_graded_node(self, _w) -> None:
    """Advisory: an impedance port whose control volume straddles a step.

    The port current is the discrete Ampere loop on the DUAL face
    pierced by the port's E edge, each leg weighted by the dual spacing
    along that H component's own axis — the two axes TRANSVERSE to the
    port component. ``V`` uses the primal edge length and is exact
    either way, so only the transverse axes are checked. An EXCITED port
    also drives through ``make_current_source``, so the source-side
    normalization above applies to it as well.

    Since #688 the same two dual spacings also size the TERMINATION
    conductance (``sigma = n_live * d_par / (Z0 * dual_b * dual_c)``),
    which is why this now covers single-cell lumped ports too: they
    carry no ``extent``, so the old filter skipped them entirely while
    they sat on the identical metric (measured, a lumped ez port on a
    2:1 step on both transverse axes printed ``[PREFLIGHT] All checks
    passed.`` while carrying the same 1.7778x conductance error).
    """
    entries = [pe for pe in getattr(self, "_ports", ())
               if getattr(pe, "extent", None) is not None
               or float(getattr(pe, "impedance", 0.0)) > 0.0]
    if not entries:
        return
    for pe in entries:
        axis = self._AXIS_OF_COMPONENT.get(pe.component)
        if axis is None:
            continue
        for a in (ax for ax in range(3) if ax != axis):
            rep = self._graded_node_report(a, pe.position[a])
            if rep is None:
                continue
            node_pos, d_below, d_above, dual, ratio = rep
            kind = ("wire port" if getattr(pe, "extent", None) is not None
                    else "lumped port")
            _w.warn(
                PreflightWarning(
                    f"{kind} at {pe.position} "
                    f"(component {pe.component}) sits at "
                    f"{'xyz'[a]} = {_fmt_len(node_pos)}, an E node whose "
                    f"adjacent cells differ by {(ratio - 1.0):.0%} "
                    f"({_fmt_len(d_below)} below, {_fmt_len(d_above)} "
                    f"above). {'xyz'[a]} is one of the port's two "
                    f"Ampere-loop axes, so the DUAL spacing "
                    f"{_fmt_len(dual)} weights BOTH that leg of the loop "
                    f"that measures I (issue #672) and the termination "
                    f"conductance that realizes Z0 (issue #688). The "
                    f"extracted Z_in = -V/I, and every S-parameter built "
                    f"on it, is most mesh-sensitive here: move the port "
                    f"onto a locally uniform node (or flatten the "
                    f"grading there) if its S-parameters are "
                    f"claims-bearing.",
                    code="wire_port_on_graded_node",
                    source="_validate_cfg_wire_port_on_graded_node",
                ),
                stacklevel=3,
            )


def _validate_cfg_nonuniform_limitations(
    self, _w, cpml_thickness: float
) -> None:
    """P2: Non-uniform mesh shadow-lane limitations."""
    if self._dz_profile is not None:
        # P2.3: TFSF on nonuniform mesh — narrowed scope.
        # Axis-aligned ±x incidence with angle_deg=0 runs the 1D
        # auxiliary along the uniform x axis and is supported. The
        # z-directed and oblique cases would need a z-nonuniform 1D
        # aux (resp. nonuniform 2D aux) and are deferred.
        if self._tfsf is not None:
            if self._tfsf.direction in ("+z", "-z"):
                raise PreflightConfigError(
                    "TFSF z-directed incidence is not yet supported on "
                    "nonuniform z mesh. Axis-aligned incidence along x "
                    "(direction='+x' or '-x') is supported.",
                    code="nonuniform_tfsf",
                    source="_validate_cfg_nonuniform_limitations",
                )
            if abs(self._tfsf.angle_deg) > 0.01:
                raise PreflightConfigError(
                    "TFSF oblique incidence is not yet supported on "
                    "nonuniform z mesh. Use angle_deg=0.",
                    code="nonuniform_tfsf",
                    source="_validate_cfg_nonuniform_limitations",
                )

        # P2.6: CPML z-thickness on non-uniform mesh.
        # Skip on tracer profiles — advisory warning only.
        # Issue #647: the cell count is the z faces' OWN allocation, not
        # the global budget. Keyed off `_boundary_spec` via
        # `_preflight_face_layers`, so a per-face spec whose z faces are
        # PEC/PMC (allocation 0) no longer reports a thin absorber that
        # does not exist, and a per-face `hi_thickness` is measured at the
        # thickness it actually allocates.
        _z_layers = max(self._preflight_face_layers()["z_lo"],
                        self._preflight_face_layers()["z_hi"])
        if (self._boundary == "cpml"
                and _z_layers > 0
                and not is_tracer(self._dz_profile)):
            cpml_z_thick = sum(float(d) for d in self._dz_profile[:_z_layers])
            if cpml_z_thick < cpml_thickness * 0.3:
                _w.warn(
                    PreflightWarning(
                        f"CPML z-thickness is {cpml_z_thick*1e3:.1f}mm "
                        f"({_z_layers} cells), much thinner than "
                        f"xy-thickness {cpml_thickness*1e3:.1f}mm. "
                        f"Absorbing performance may be asymmetric. "
                        f"Consider more z cells or fewer CPML layers.",
                        code="nonuniform_cpml_thin",
                        source="_validate_cfg_nonuniform_limitations",
                    ),
                    stacklevel=3,
                )

def _validate_cfg_subgrid_limitations(self, _w) -> None:
    """P4: Subgridded path limitations.

    P3 (Distributed path): distributed warnings are emitted at
    run() dispatch time in distributed_v2.py — no preflight check
    here.
    """
    if self._refinement is not None:
        if self._dft_planes:
            _w.warn(
                PreflightWarning(
                    "DFT plane probes are not supported with SBP-SAT "
                    "subgridding.",
                    code="subgrid_unsupported_feature",
                    source="_validate_cfg_subgrid_limitations",
                ),
                stacklevel=3,
            )
        if self._waveguide_ports:
            _w.warn(
                PreflightWarning(
                    "Waveguide ports are not supported with SBP-SAT "
                    "subgridding.",
                    code="subgrid_unsupported_feature",
                    source="_validate_cfg_subgrid_limitations",
                ),
                stacklevel=3,
            )
        if self._floquet_ports:
            _w.warn(
                PreflightWarning(
                    "Floquet ports are not supported with SBP-SAT subgridding.",
                    code="subgrid_unsupported_feature",
                    source="_validate_cfg_subgrid_limitations",
                ),
                stacklevel=3,
            )
        if self._tfsf is not None:
            _w.warn(
                PreflightWarning(
                    "TFSF source is not supported with SBP-SAT subgridding.",
                    code="subgrid_unsupported_feature",
                    source="_validate_cfg_subgrid_limitations",
                ),
                stacklevel=3,
            )
        if self._lumped_rlc:
            _w.warn(
                PreflightWarning(
                    "Lumped RLC elements are not supported with SBP-SAT "
                    "subgridding.",
                    code="subgrid_unsupported_feature",
                    source="_validate_cfg_subgrid_limitations",
                ),
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the twelve functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all twelve names here are private,
# so ``tests/locks/test_preflight_split_snapshot.py`` pins these twelve
# directly.
#
# None of the twelve was a ``@staticmethod``, so like legs 3 and 4 this module
# has no decorator the facade has to re-apply and every restored qualname
# below becomes ``Simulation.<name>`` after composition.
# ---------------------------------------------------------------------------
_validate_mesh_quality.__qualname__ = "_PreflightMixin._validate_mesh_quality"
_check_numerical_dispersion.__qualname__ = (
    "_PreflightMixin._check_numerical_dispersion"
)
_validate_thin_metal_on_nu_mesh.__qualname__ = (
    "_PreflightMixin._validate_thin_metal_on_nu_mesh"
)
_validate_cfg_graded_box_rasterization.__qualname__ = (
    "_PreflightMixin._validate_cfg_graded_box_rasterization"
)
_validate_cfg_floquet_nonuniform.__qualname__ = (
    "_PreflightMixin._validate_cfg_floquet_nonuniform"
)
_validate_cfg_multiband_grading.__qualname__ = (
    "_PreflightMixin._validate_cfg_multiband_grading"
)
_validate_cfg_thin_conductor_graded_node.__qualname__ = (
    "_PreflightMixin._validate_cfg_thin_conductor_graded_node"
)
_graded_node_report.__qualname__ = "_PreflightMixin._graded_node_report"
_validate_cfg_source_on_graded_node.__qualname__ = (
    "_PreflightMixin._validate_cfg_source_on_graded_node"
)
_validate_cfg_wire_port_on_graded_node.__qualname__ = (
    "_PreflightMixin._validate_cfg_wire_port_on_graded_node"
)
_validate_cfg_nonuniform_limitations.__qualname__ = (
    "_PreflightMixin._validate_cfg_nonuniform_limitations"
)
_validate_cfg_subgrid_limitations.__qualname__ = (
    "_PreflightMixin._validate_cfg_subgrid_limitations"
)
