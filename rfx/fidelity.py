"""Input-fidelity report: what you declared vs what will actually be solved.

Design rule (PI, 2026-08-27): a simulation exists because the RESULT is
unknown, so diagnostics must never rank findings by predicted impact on
observables. The one legitimate question is INPUT fidelity — does the
rasterized/materialized model faithfully realize the declared geometry,
materials and realization classes — reported per entity, in input units
(um, %, material values). Relevance is the user's judgment; this module
only puts clean numbers in front of it.

Everything here reuses the SAME grid build and material assembly the solve
uses, without time stepping.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fidelity_report"]


def _axis_names():
    return ("x", "y", "z")


def _node_arrays(sim, grid, nonuniform):
    if nonuniform:
        dx = np.asarray(grid.dx_arr, dtype=float)
        dy = np.asarray(grid.dy_arr, dtype=float)
        dz = np.asarray(grid.dz, dtype=float)
    else:
        n = grid.shape
        dx = np.full(n[0], float(grid.dx))
        dy = np.full(n[1], float(grid.dx))
        dz = np.full(n[2], float(grid.dx))
    nodes = tuple(np.concatenate([[0.0], np.cumsum(d)]) for d in (dx, dy, dz))
    return (dx, dy, dz), nodes


def _entity_mask(entry, sim, grid, nonuniform):
    if nonuniform:
        from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
        c = coords_from_nonuniform_grid(grid)
        return np.asarray(entry.shape.mask_on_coords(c.x, c.y, c.z), dtype=bool)
    return np.asarray(entry.shape.mask(grid), dtype=bool)


def _declared_material(sim, name):
    if name == "pec":
        return dict(kind="pec")
    spec = sim._materials.get(name)
    if spec is None:
        return dict(kind="unknown", name=name)
    return dict(kind="dielectric", name=name,
                eps_r=float(getattr(spec, "eps_r", 1.0)),
                sigma=float(getattr(spec, "sigma", 0.0)),
                n_debye=len(getattr(spec, "debye_poles", None) or ()),
                n_lorentz=len(getattr(spec, "lorentz_poles", None) or ()))


def _assembled_as_pec(sim, entry):
    """Does ``_assemble_materials`` route this geometry entry to the PEC
    branch? Keyed the way the assembly keys it (rfx/api/_compile.py: the
    RESOLVED material's ``sigma >= _PEC_SIGMA_THRESHOLD``), not on the
    literal name "pec" — a named copper with sigma 5.8e7 is PEC there too.
    """
    try:
        mat = sim._resolve_material(entry.material_name)
    except Exception:
        return entry.material_name == "pec"
    thr = float(getattr(sim, "_PEC_SIGMA_THRESHOLD", 1e6))
    return float(getattr(mat, "sigma", 0.0) or 0.0) >= thr


def _max_run_length(mask, axis):
    """Longest contiguous run of True along `axis` over occupied lines.

    Sheet detection must use the MAXIMUM, not the minimum: a solid sphere has
    single-cell runs at its rim, so the minimum is 1 for any curved body and
    the first implementation called a 1045-cell PEC sphere a "one-plane
    sheet" and claimed vacuum was substituted for its metal (found by the
    crossval sweep, rcs_scattering.py, 2026-08-27).
    """
    m = np.moveaxis(mask, axis, -1)
    flat = m.reshape(-1, m.shape[-1])
    best = 0
    for line in flat[flat.any(axis=1)]:
        padded = np.concatenate([[0], line.view(np.int8), [0]])
        edges = np.flatnonzero(np.diff(padded))
        runs = edges[1::2] - edges[0::2]
        best = max(best, int(runs.max()))
    return best


def fidelity_report(sim, print_report: bool = True):
    """Per-entity declared-vs-realized audit; returns a list of dicts.

    Findings never predict physics. Each entry reports, in input units:

    * rasterization — occupied cell count (0 = the entity is silently
      ABSENT from the solve), realized bounds from the run's own node
      coordinates, per-axis face residuals and extent deltas;
    * realization class for conductors — volumetric / one-plane sheet
      (electric wall on the LOWER node plane only) / two-plane sheet —
      with the wall coordinates, and the sheet's OWN-cell permittivity
      (a one-plane sheet's cell volume stays live inside adjacent
      cavities: eps_r 1.0 there means declared metal is realized as a
      vacuum layer plus one wall);
    * materialization for dielectrics — the assembled eps_r/sigma inside
      the entity's cells vs the declared values (later entities may have
      overwritten earlier ones);
    * a mechanical remedy per finding class (no impact estimates).

    The list LEADS with a ``"domain (the solved box)"`` pseudo-entity —
    the only row a cavity/waveguide model has, since there the box IS the
    geometry. Its per-axis ``n_cells`` / ``realized_extent_um`` are the
    wall-to-wall CELL count and length, i.e. ``ceil(L/dx)`` cells, not the
    ``ceil(L/dx) + 1`` node count the grid allocates. An axis the solve
    does not have — ``z`` under ``mode="2d_tmz"``, where ``grid.nz == 1``
    and the declared ``Lz`` is ignored — is not compared: that axis dict
    carries a ``note`` key (rendered by the printed report) instead of a
    finding.
    """
    nonuniform = any(getattr(sim, a, None) is not None
                     for a in ("_dx_profile", "_dy_profile", "_dz_profile"))
    if nonuniform:
        grid = sim._build_nonuniform_grid()
        from rfx.runners.nonuniform import assemble_materials_nu
        out = assemble_materials_nu(sim, grid)
    else:
        grid = sim._build_grid()
        out = sim._assemble_materials(grid)
    mats, pec_mask = out[0], out[3]
    eps = np.asarray(mats.eps_r, dtype=float)
    sigma_arr = np.asarray(mats.sigma, dtype=float)
    pec_mask = (np.asarray(pec_mask, dtype=bool)
                if pec_mask is not None else np.zeros(eps.shape, bool))
    sizes, nodes = _node_arrays(sim, grid, nonuniform)
    domain = tuple(float(v) for v in getattr(sim, "_domain", (0.0, 0.0, 0.0)))
    # Mask/material arrays live on the PADDED grid (CPML pad cells on every
    # side); declared coordinates are DOMAIN coordinates. Shift node arrays so
    # index pad_lo sits at domain 0.
    pads = (int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo))
    nodes = tuple(n - n[p] for n, p in zip(nodes, pads))

    # Declared conductors arrive through TWO surfaces: sim.add(...,
    # material="pec"/dielectric) and sim.add_thin_conductor(...). Auditing only
    # the first produced an EMPTY report — a false all-clear — for models built
    # the second way (found by the adversarial suite, 2026-08-27).
    entries = [("geometry", i, e) for i, e in enumerate(sim._geometry)]
    for j, tc in enumerate(getattr(sim, "_thin_conductors", ()) or ()):
        entries.append(("thin_conductor", j, tc))

    report = []

    # The domain box IS the geometry for cavity / waveguide models (no
    # entities at all), and the first implementation printed "0 entities, 0
    # findings" for them — a clean bill of health for an unaudited structure
    # (found by the crossval sweep: cv09/cv14/cv21, 2026-08-27).
    # The domain row must read `nodes`, the SAME fence-post-correct array
    # the entity rows read below (nodes[a][i0], nodes[a][i1 + 1]) — not
    # `sizes`, which has one entry per NODE. Summing `sizes` counted a span
    # of n nodes as n cells and inflated an exactly-commensurate domain's
    # "realized" length by one cell on every axis (issue #729 site 1,
    # 2026-08-27: a 21-node/20-cell span read back as 21000 um instead of
    # 20000 um, with a false [domain-extent-quantized] finding attached).
    # NonUniformGrid has no is_2d attribute, so `grid.is_2d` would raise on
    # the non-uniform lane; getattr keeps that lane running. KNOWN GAP, not
    # fixed here: Simulation(mode="2d_tmz", dz_profile=...) builds a
    # NonUniformGrid that carries no 2D marker at all, so the not-solved
    # guard below never fires there and that lane still compares a z it does
    # not solve. Pre-existing (the non-uniform builder ignores `mode`); the
    # fix belongs with the builder, not with this reporter.
    is_2d = getattr(grid, "is_2d", False)
    # PMC-plane convention (#722 ninth surface, decided 2026-08-28):
    # apply_pmc_faces zeros H_tan a HALF-CELL INSIDE the declared wall on
    # every PMC face (rfx/boundaries/pmc.py: index 0 on a `_lo` face, index
    # -2 on a `_hi` face -- both 0.5*dx inside the declared mesh line, pinned
    # by tests/test_boundary_pmc_hi_faces.py -- that placement is solver
    # physics, measured, and is NOT touched here). A PMC-mirrored model's
    # H_tan wall therefore sits half a cell inside the mesh line this report
    # would otherwise quote as "realized", so the domain row must read the
    # face list off the SAME BoundarySpec the solve compiled, not re-derive
    # it (sim._boundary_spec is set unconditionally in Simulation.__init__).
    _bspec = getattr(sim, "_boundary_spec", None)
    pmc_faces = _bspec.pmc_faces() if _bspec is not None else set()
    dom_item = dict(entity="domain (the solved box)", findings=[], axes=[])
    for a in range(3):
        axis_name = _axis_names()[a]
        p_lo = int(getattr(grid, f"pad_{axis_name}_lo"))
        p_hi = int(getattr(grid, f"pad_{axis_name}_hi"))
        i_lo = p_lo
        # DEFENSIVE ONLY — unreachable on both grid classes as they are
        # built today, and NOT evidence that "the pads consumed the axis"
        # is a state this code has ever seen. Grid: shape[a] = ceil(L/dx)
        # + 1 + p_lo + p_hi, so i_hi - i_lo = ceil(L/dx) >= 0
        # (rfx/grid.py:151). NonUniformGrid: shape[a] = p_lo + len(profile)
        # + p_hi + 1 (the trailing bounding node, rfx/nonuniform.py
        # _append_bounding_node), so i_hi - i_lo = len(profile) >= 1. The
        # clamp exists so a future grid layout cannot turn a negative index
        # into a silent wrap; it has no test because it has no reachable
        # input.
        i_hi = max(len(sizes[a]) - p_hi - 1, i_lo)
        n_int = max(i_hi - i_lo, 0)
        # mesh_extent is the NODE-to-NODE span (unchanged from before this
        # change) -- the commensurability finding below stays keyed on it,
        # since "does the cell size divide the declared length" is a mesh
        # question, independent of any PMC face on this axis.
        mesh_extent = float(nodes[a][i_hi] - nodes[a][i_lo])
        declared = float(domain[a]) if a < len(domain) else 0.0
        # z is not solved in 2D (grid.nz == 1, Lz is ignored — rfx/grid.py);
        # comparing it against the declared z would replace one meaningless
        # number (the old node/cell miscount) with another (a 0.0 um
        # "realized" length reported as a domain-extent-quantized finding).
        axis_not_solved = is_2d and _axis_names()[a] == "z"
        lo_is_pmc = f"{axis_name}_lo" in pmc_faces
        hi_is_pmc = f"{axis_name}_hi" in pmc_faces
        half_lo = (float(sizes[a][i_lo]) / 2.0
                   if (not axis_not_solved and n_int > 0 and lo_is_pmc)
                   else 0.0)
        half_hi = (float(sizes[a][max(i_hi - 1, i_lo)]) / 2.0
                   if (not axis_not_solved and n_int > 0 and hi_is_pmc)
                   else 0.0)
        eff_lo = half_lo
        eff_hi = mesh_extent - half_hi
        realized = eff_hi - eff_lo
        ax = dict(axis=axis_name, n_cells=int(n_int),
                  declared_um=(0.0, declared * 1e6),
                  mesh_extent_um=mesh_extent * 1e6,
                  realized_um=(eff_lo * 1e6, eff_hi * 1e6),
                  face_residual_um=(eff_lo * 1e6, abs(eff_hi - declared) * 1e6),
                  declared_extent_um=declared * 1e6,
                  realized_extent_um=realized * 1e6)
        if axis_not_solved:
            ax["note"] = (
                "axis-not-solved — 2D mode (grid.nz == 1): the declared "
                f"Lz = {declared * 1e6:.1f} um is IGNORED by the solve and "
                "the realized 0.0 um above is not a discrepancy to fix. "
                "Read this row as 'not compared', not as a pass")
        dom_item["axes"].append(ax)
        if (not axis_not_solved and declared > 0
                and abs(mesh_extent - declared) > 0.005 * declared):
            dom_item["findings"].append(dict(
                kind="domain-extent-quantized", axis=ax["axis"],
                detail=f"declared {declared * 1e6:.1f} um realized "
                       f"{mesh_extent * 1e6:.1f} um over {n_int} cells "
                       f"({(mesh_extent - declared) * 1e6:+.1f} um, "
                       f"{100 * (mesh_extent - declared) / declared:+.2f}%) — "
                       "the cell size does not divide the declared length",
                remedy="choose a cell size that divides this length, or "
                       "accept and quote the REALIZED length (for a "
                       "PEC/PMC-walled model the realized length is the "
                       "cavity/guide dimension the solve actually has)"))
        if not axis_not_solved and (lo_is_pmc or hi_is_pmc) and n_int > 0:
            pmc_face_labels = [f for f in (f"{axis_name}_lo", f"{axis_name}_hi")
                               if f in pmc_faces]
            dom_item["findings"].append(dict(
                kind="pmc-wall-half-cell-inside", axis=axis_name,
                detail=(
                    f"{'/'.join(pmc_face_labels)} zero H_tan a half-cell "
                    "INSIDE the declared mesh line (rfx/boundaries/pmc.py, "
                    "pinned by tests/test_boundary_pmc_hi_faces.py): the "
                    f"realized H_tan wall on this axis is "
                    f"[{eff_lo * 1e6:.1f}, {eff_hi * 1e6:.1f}] um against "
                    f"the declared mesh line at [0.0, {declared * 1e6:.1f}] "
                    f"um (mesh extent {mesh_extent * 1e6:.1f} um) — a "
                    "residual equal to HALF the boundary cell here is the "
                    "PMC-plane CONVENTION (realize-declared: declare a "
                    "mirror plane at plane + dx/2 so the H_tan zero lands "
                    "ON the intended plane, issue #722 ninth surface), not "
                    "a discrepancy to fix (#303 class: a displayed gap "
                    "needs its explanation attached, not silence)"),
                remedy="if this axis mirrors a symmetry plane, declare the "
                       "half-domain extent at (plane + dx/2) so the "
                       "realized H_tan wall above lands on the intended "
                       "plane; otherwise no action is needed"))
    report.append(dom_item)

    # Issue #589: ``_assemble_materials`` is PEC-OR-only (``pec_mask =
    # pec_mask | mask``; rfx/api/_compile.py) and there is no CSG
    # subtraction shape, so a dielectric declared AFTER a conductor cannot
    # carve it — its overlapping cells are a silent no-op. The committed
    # coax-MSL junction fixture declared its clearance hole exactly that way
    # (full-plane PEC ground, then a PTFE Cylinder) and the settled run
    # measured a short (S00 = -0.9928 at 6 GHz). This is the OR of every
    # PEC-assembled geometry entry seen so far, in declaration order; the
    # ordered finding below reads it. Geometry entries only: thin
    # conductors are OR-ed in after ALL geometry (a different ordering
    # question, not audited here).
    pec_before = np.zeros(eps.shape, dtype=bool)
    # Every PEC-assembled geometry entity's realized cells, rasterized ONCE
    # and kept as flat indices (memory scales with occupied cells, not with
    # the grid). The ordered finding below reads this instead of
    # re-rasterizing every earlier conductor for every overlapping
    # dielectric, which was O(n^2) rasterizations on a 20-box ground sheet.
    pec_cells_by_entity: dict = {}
    # Conductors whose mask could not be rasterized. They are NOT in
    # pec_before, so the ordered check cannot see them; the failure is
    # reported on the conductor's own row (entity name + exception class)
    # and named again on every later dielectric row, so a skipped check
    # never reads as a clean one (the #303 class: "All checks passed" with
    # a silently skipped family).
    pec_unrasterized: list = []

    for kind_src, i, entry in entries:
        if kind_src == "thin_conductor":
            sig = float(getattr(entry, "sigma_bulk", 0.0))
            mat_name = "pec" if sig >= 1e6 else "lossy-sheet"
            name = (f"thin_conductor[{i}] sigma_bulk={sig:.3g} S/m"
                    f" -> {mat_name}")
        else:
            name = f"geometry[{i}] '{entry.material_name}'"
            mat_name = entry.material_name
        try:
            lo, hi = entry.shape.bounding_box()
        except Exception:
            nb_item = dict(entity=name, findings=[dict(
                kind="no-analytic-bounds",
                detail="shape exposes no bounding_box(); declared-vs-realized "
                       "bounds cannot be audited",
                remedy="give the shape an axis-aligned bounding box")])
            report.append(nb_item)
            if kind_src == "geometry" and _assembled_as_pec(sim, entry):
                try:
                    nb_mask = _entity_mask(entry, sim, grid, nonuniform)
                except Exception as exc:
                    pec_unrasterized.append((i, name, type(exc).__name__))
                    nb_item["findings"].append(dict(
                        kind="rasterization-failed",
                        exception=type(exc).__name__,
                        detail=(f"mask() raised {type(exc).__name__}: {exc} — "
                                "this conductor's realized cells are unknown, "
                                "so it is NOT in the ordered PEC accumulator "
                                "and dielectric-after-conductor-no-op cannot "
                                "see it (every later dielectric row carries a "
                                "dielectric-after-conductor-unaudited finding "
                                "naming it)"),
                        remedy="fix the shape so it rasterizes on this grid; "
                               "until then the ordered-overlap audit is "
                               "incomplete for this conductor"))
                else:
                    pec_before |= nb_mask
                    pec_cells_by_entity[i] = np.flatnonzero(nb_mask)
            continue
        boxlike = type(entry.shape).__name__ == "Box"
        mask = _entity_mask(entry, sim, grid, nonuniform)
        pec_assembled = kind_src == "geometry" and _assembled_as_pec(sim, entry)
        item = dict(entity=name, material=_declared_material(sim, mat_name),
                    declared_lo=tuple(float(v) for v in lo),
                    declared_hi=tuple(float(v) for v in hi),
                    n_cells=int(mask.sum()), findings=[])
        if item["n_cells"] == 0:
            item["findings"].append(dict(
                kind="absent",
                detail="rasterizes to ZERO cells — this entity does not exist "
                       "in the solved model",
                remedy="thinner than a cell? place a mesh node inside its span "
                       "or refine the local cell size"))
            report.append(item)
            continue

        # A PEC mask does not change eps, so a dielectric whose cells were
        # later claimed by a conductor looks untouched in eps alone (trap T5).
        if mat_name not in ("pec", "lossy-sheet"):
            pec_frac = float(np.mean(pec_mask[mask]))
            _declared_sigma = float(item["material"].get("sigma", 0.0) or 0.0)
            # A metal-like declaration realized as PEC is its own finding
            # below; here we mean a DIELECTRIC whose cells metal took over.
            if pec_frac > 1e-9 and _declared_sigma < 1e6:
                item["findings"].append(dict(
                    kind="claimed-by-conductor",
                    detail=f"{100 * pec_frac:.1f}% of this dielectric's cells "
                           "are also PEC — the conductor wins, so that "
                           "fraction of the declared material is not solved",
                    remedy="check the overlap against the intended stack; "
                           "shrink whichever body is over-declared"))
        # Ordered case of the overlap above (issue #589): a non-PEC entity
        # sharing cells with a PEC entity declared EARLIER. claimed-by-
        # conductor stays byte-identical (it fires for either order); this
        # adds the statement that the assembly order makes these cells a
        # no-op — a hole "carved" by a later dielectric does not exist in
        # the solve. Report-only: a slab deliberately drawn through a
        # ground sheet is a legitimate pattern and lands here too.
        if kind_src == "geometry" and not pec_assembled:
            n_ov = int(np.count_nonzero(mask & pec_before))
            if n_ov > 0:
                flat = mask.ravel()
                contributors = []
                for m, cells_m in pec_cells_by_entity.items():
                    if m >= i:      # declaration order; defensive
                        continue
                    n_m = int(np.count_nonzero(flat[cells_m]))
                    if n_m > 0:
                        contributors.append((m, n_m))
                who = ", ".join(
                    f"geometry[{m}] '{sim._geometry[m].material_name}' "
                    f"({n_m} cells)" for m, n_m in contributors)
                item["findings"].append(dict(
                    kind="dielectric-after-conductor-no-op",
                    overlap_cells=n_ov,
                    conductor_entities=[m for m, _ in contributors],
                    detail=(f"{n_ov} of this entity's {item['n_cells']} cells "
                            f"({100.0 * n_ov / item['n_cells']:.1f}%) are "
                            f"already PEC from an entity declared EARLIER: "
                            f"{who}. _assemble_materials is PEC-OR-only, so a "
                            "dielectric declared after a conductor cannot "
                            f"carve it — these {n_ov} cells are a no-op (the "
                            "eps_r/sigma written there is never solved); if a "
                            "clearance/hole was intended, build the conductor "
                            "with the hole"),
                    remedy="if the overlap is intended (e.g. a slab drawn "
                           "through a ground sheet) no action is needed; if a "
                           "hole/clearance was intended, build the conductor "
                           "WITH the hole (split it into boxes around the "
                           "aperture) — no later entity can remove PEC"))
            missing = [(m, nm, ex) for m, nm, ex in pec_unrasterized if m < i]
            if missing:
                item["findings"].append(dict(
                    kind="dielectric-after-conductor-unaudited",
                    conductor_entities=[m for m, _, _ in missing],
                    detail=("the ordered-overlap audit above is INCOMPLETE "
                            "for this entity: " + ", ".join(
                                f"{nm} (mask() raised {ex})"
                                for _, nm, ex in missing)
                            + " declared earlier could not be rasterized, so "
                            "any cells shared with it are not counted — a "
                            "silent dielectric-after-conductor no-op against "
                            "that conductor is still possible"),
                    remedy="fix that conductor's shape so it rasterizes, "
                           "then re-run fidelity_report"))
        if pec_assembled:
            pec_before |= mask
            pec_cells_by_entity[i] = np.flatnonzero(mask)
        # Absorber overlap: cells outside [0, domain) live in the CPML pad.
        pad_hit = []
        for a in range(3):
            idx = np.where(mask.any(axis=tuple(x for x in range(3) if x != a)))[0]
            if len(idx) and (nodes[a][idx.min()] < -1e-12
                             or nodes[a][idx.max() + 1] > domain[a] + 1e-12):
                pad_hit.append(_axis_names()[a])
        if pad_hit:
            item["findings"].append(dict(
                kind="inside-absorber", axis=",".join(pad_hit),
                detail="the realized body extends beyond the declared domain "
                       f"on {','.join(pad_hit)} — those cells sit in the CPML "
                       "pad, where the update equations are the absorber's",
                remedy="keep the body inside the domain, or enlarge the "
                       "domain so the absorber stays empty"))

        occ = np.where(mask)
        axes = []
        clipped_axes = []
        for a in range(3):
            i0, i1 = int(occ[a].min()), int(occ[a].max())
            r_lo, r_hi = float(nodes[a][i0]), float(nodes[a][i1 + 1])
            d_lo, d_hi = float(lo[a]), float(hi[a])
            # A body drawn past the domain is CLIPPED by construction (a
            # common deliberate idiom: draw big, let the rasterizer cut).
            # Comparing the realized body against the un-clipped declaration
            # produced 99%-residual nonsense that drowned the real findings
            # (crossval sweep: cv18 4/6, cv19 20/30 findings, 2026-08-27).
            dom_hi = float(domain[a]) if a < len(domain) else 0.0
            if dom_hi > 0 and (d_lo < -1e-12 or d_hi > dom_hi + 1e-12):
                clipped_axes.append(_axis_names()[a])
                d_lo, d_hi = max(d_lo, 0.0), min(d_hi, dom_hi)
            ax = dict(axis=_axis_names()[a],
                      declared_um=(d_lo * 1e6, d_hi * 1e6),
                      realized_um=(r_lo * 1e6, r_hi * 1e6),
                      face_residual_um=(abs(r_lo - d_lo) * 1e6,
                                        abs(r_hi - d_hi) * 1e6),
                      declared_extent_um=(d_hi - d_lo) * 1e6,
                      realized_extent_um=(r_hi - r_lo) * 1e6)
            ext = ax["declared_extent_um"]
            # Local cell size on this axis over the body's span — the scale a
            # placement can actually be resolved to.
            cell_um = float(np.mean(sizes[a][i0:i1 + 1])) * 1e6
            ax["cell_um"] = cell_um
            ax["sub_cell"] = bool(ext < cell_um)
            if ext > 0:
                worst = max(ax["face_residual_um"])
                # A body thinner than a cell cannot be placed more precisely
                # than the lattice allows, so quoting its offset as a fraction
                # of its own sub-cell thickness produces numbers like "181% of
                # 17 um" that read as catastrophic and mean only "it landed in
                # a cell". For those, report against the CELL (found by the
                # crossval sweep and by an external review of issue #725).
                if ax["sub_cell"]:
                    if worst > 0.25 * cell_um:
                        item["findings"].append(dict(
                            kind="sub-cell-placement", axis=ax["axis"],
                            detail=f"declared extent {ext:.1f} um is smaller "
                                   f"than the local cell {cell_um:.1f} um; the "
                                   f"body lands {worst:.1f} um "
                                   f"({worst / cell_um:.2f} cell) from its "
                                   "declared face. Placement quantizes to the "
                                   "lattice — a percentage of the declared "
                                   "extent is not a meaningful measure here",
                            remedy="place a mesh node on the declared face, or "
                                   "resolve the body with >= 1 cell, if this "
                                   "placement is load-bearing"))
                    axes.append(ax)      # the axis record is still reported
                    continue
                if worst > 0.005 * ext:
                    d_ext = ax["realized_extent_um"] - ext
                    size_txt = (f"SIZE {ax['realized_extent_um']:.1f} um "
                                f"({d_ext:+.1f} um, "
                                f"{100 * d_ext / ext:+.2f}%)")
                    if abs(d_ext) <= 1e-9:
                        size_txt = "SIZE preserved"
                    if boxlike:
                        item["findings"].append(dict(
                            kind="off-lattice-face", axis=ax["axis"],
                            detail=f"PLACEMENT off by {worst:.1f} um "
                                   f"({100 * worst / ext:.2f}% of the declared "
                                   f"{ext:.1f} um extent); {size_txt}",
                            remedy="place a mesh node on this face "
                                   "(non-uniform profile) or choose the cell "
                                   "size / origin commensurate with it"))
                    else:
                        # A curved/implicit shape has no face to put a node on;
                        # its bounding box is sampled, so the honest statement
                        # is where the realized body sits and how big it is.
                        d_mid = (ax["declared_um"][0] + ax["declared_um"][1]) / 2
                        r_mid = (ax["realized_um"][0] + ax["realized_um"][1]) / 2
                        item["findings"].append(dict(
                            kind="bbox-offset", axis=ax["axis"],
                            midpoint_shift_um=r_mid - d_mid,
                            detail=f"PLACEMENT off by {worst:.1f} um "
                                   f"({100 * worst / ext:.2f}% of the declared "
                                   f"{ext:.1f} um extent); {size_txt}. Curved "
                                   f"boundary sampled cell-wise; midpoint "
                                   f"shift {r_mid - d_mid:+.1f} um "
                                   f"({(r_mid - d_mid) / cell_um:+.2f} cell). "
                                   "A non-Box mask samples NODE coordinates "
                                   "while these bounds are cell EDGES, so up "
                                   "to half a cell of the offset is a readout "
                                   "convention, not a displacement — the "
                                   "midpoint shift is the convention-free "
                                   "number",
                            remedy="refine the local cell size if either the "
                                   "placement or the realized size is "
                                   "load-bearing for the comparison"))
            axes.append(ax)
        item["axes"] = axes
        if clipped_axes:
            item["findings"].append(dict(
                kind="clipped-by-domain", axis=",".join(clipped_axes),
                detail="declared past the domain on "
                       f"{','.join(clipped_axes)}; the rasterizer clips it, so "
                       "the residuals above are measured against the CLIPPED "
                       "declaration (this is a common deliberate idiom, not "
                       "necessarily a defect)",
                remedy="none if the overhang is intentional; otherwise draw "
                       "the body to its physical bounds"))

        # Realization class follows the ASSEMBLY, not the material name: a
        # model may declare its metal as a named material with a finite
        # sigma (e.g. copper 5.8e7) and the assembly may still realize it as
        # a PEC mask. Keying on the literal name "pec" made every sheet
        # finding invisible on exactly such a model (the CST board,
        # 2026-08-27) — the tool's whole purpose, silently skipped.
        pec_frac_self = float(np.mean(pec_mask[mask]))
        realized_conductor = pec_frac_self > 0.5
        declared_sigma = float(item["material"].get("sigma", 0.0) or 0.0)
        if realized_conductor and mat_name != "pec" and declared_sigma >= 1e6:
            item["findings"].append(dict(
                kind="declared-lossy-realized-pec",
                detail=f"declared as material '{mat_name}' with sigma "
                       f"{item['material'].get('sigma', 0.0):.4g} S/m, but "
                       f"{100 * pec_frac_self:.0f}% of its cells are realized "
                       "as PEC — the solve has a LOSSLESS perfect conductor, "
                       "not the declared finite conductivity",
                remedy="if conductor loss matters, use the surface-impedance "
                       "path (add_thin_conductor(..., surface_impedance_f0=) "
                       "or the sheet operator); if not, declare 'pec' so the "
                       "model states what it solves"))
        if realized_conductor:
            runs = [_max_run_length(mask, a) for a in range(3)]
            thin_axes = [a for a, r in enumerate(runs) if r == 1]
            if not thin_axes:
                item["realization"] = "volumetric PEC (>= 2 cells on every axis)"
                if getattr(entry, "two_plane", False):
                    item["findings"].append(dict(
                        kind="two-plane-inert",
                        detail="two_plane=True was declared, but this body is "
                               ">= 2 cells thick on every axis, so the rule "
                               "adds nothing — the request has no effect",
                        remedy="drop the flag here, or check whether the body "
                               "was meant to be a one-cell sheet"))
            else:
                aname = "+".join(_axis_names()[a] for a in thin_axes)
                if getattr(entry, "two_plane", False):
                    item["realization"] = (
                        f"two-plane sheet (normal {aname}): electric walls on "
                        "BOTH bounding node planes; interior enclosed")
                else:
                    item["realization"] = (
                        f"one-plane sheet (normal {aname}): electric wall on "
                        "the LOWER node plane ONLY")
                    own_eps = eps[mask]
                    e_lo, e_hi = float(own_eps.min()), float(own_eps.max())
                    item["own_cell_eps_r"] = (e_lo, e_hi)
                    item["findings"].append(dict(
                        kind="sheet-own-cell-live", axis=aname,
                        detail=(f"the declared metal's cell volume stays live "
                                f"with eps_r {e_lo:.2f}"
                                + (f"..{e_hi:.2f}" if e_hi - e_lo > 1e-6 else "")
                                + " inside adjacent cavities"
                                + (" — VACUUM substituted for declared metal"
                                   if e_hi < 1.0 + 1e-6 else "")),
                        remedy="two_plane=True on this entry (walls on both "
                               "faces), or extend the abutting dielectric "
                               "across this cell, or resolve the thickness "
                               "with >= 2 cells"))
        else:
            m = item["material"]
            if m["kind"] == "dielectric":
                inside = eps[mask]
                frac = float(np.mean(np.isclose(inside, m["eps_r"], rtol=1e-6)))
                item["assembled_eps_r"] = (float(inside.min()),
                                           float(inside.max()))
                item["assembled_matches_declared_frac"] = frac
                # sigma is a declared input too: a dropped/zeroed conductivity
                # is invisible in eps alone (adversarial trap T1).
                s_in = sigma_arr[mask]
                item["assembled_sigma"] = (float(s_in.min()), float(s_in.max()))
                s_frac = float(np.mean(np.isclose(s_in, m["sigma"], rtol=1e-6,
                                                  atol=1e-12)))
                item["assembled_sigma_matches_declared_frac"] = s_frac
                if s_frac < 0.999:
                    item["findings"].append(dict(
                        kind="sigma-mismatch",
                        detail=f"only {100 * s_frac:.1f}% of the cells carry "
                               f"the declared sigma {m['sigma']:.4g} S/m "
                               f"(assembled {item['assembled_sigma'][0]:.4g}.."
                               f"{item['assembled_sigma'][1]:.4g} S/m)",
                        remedy="check for a later entity overwriting these "
                               "cells, or a material path that drops sigma"))
                if m["n_debye"] or m["n_lorentz"]:
                    item["findings"].append(dict(
                        kind="dispersion-not-audited",
                        detail=f"material declares {m['n_debye']} Debye and "
                               f"{m['n_lorentz']} Lorentz pole(s); this report "
                               "audits the instantaneous eps/sigma only, so "
                               "pole realization is NOT verified here",
                        remedy="verify the dispersive response separately "
                               "(a two-run R/T measurement or the material's "
                               "own oracle test)"))
                if frac < 0.999:
                    item["findings"].append(dict(
                        kind="materialization-overridden",
                        detail=f"only {100 * frac:.1f}% of this entity's cells "
                               f"carry the declared eps_r {m['eps_r']:.3f} "
                               f"(assembled range {item['assembled_eps_r'][0]:.3f}"
                               f"..{item['assembled_eps_r'][1]:.3f})",
                        remedy="a later entity overwrites these cells "
                               "(assembly is declaration-ordered) — check the "
                               "overlap, or reorder if the overlap is "
                               "unintended"))
        report.append(item)

    # Declared inputs this report does NOT audit — stated rather than omitted,
    # because silence reads as coverage (crossval sweep: cv21 and the ports
    # tutorial were 100% port-materialized geometry and reported "0 findings").
    unaudited = []
    for attr, label in (("_ports", "lumped/wire port"),
                        ("_msl_ports", "MSL port"),
                        ("_waveguide_ports", "waveguide port"),
                        ("_coaxial_ports", "coaxial port (pin/shield/caps)"),
                        ("_floquet_ports", "Floquet port"),
                        ("_sources", "source"),
                        ("_lumped_rlc", "lumped RLC")):
        n = len(getattr(sim, attr, ()) or ())
        if n:
            unaudited.append(f"{n} {label}(s)")
    if unaudited:
        report.append(dict(
            entity="NOT AUDITED by this report",
            findings=[dict(
                kind="out-of-scope",
                detail="; ".join(unaudited) + " — these declare geometry "
                       "and/or materials that this report does not walk "
                       "(port pins, shields, end caps, source cells)",
                remedy="audit those with their own port preflight checks; "
                       "do not read this report as covering them")]))

    if print_report:
        _print(report)
    return report


def _print(report):
    print("=== INPUT-FIDELITY REPORT (declared vs solved; input units only) ===")
    n_findings = sum(len(it.get("findings", [])) for it in report)
    print(f"  {len(report)} entities, {n_findings} findings\n")
    for it in report:
        head = it["entity"]
        if "realization" in it:
            head += f" — {it['realization']}"
        print(f"  {head}")
        if "n_cells" in it:
            print(f"    cells: {it['n_cells']}")
        for ax in it.get("axes", []):
            print(f"    {ax['axis']}: declared "
                  f"[{ax['declared_um'][0]:.1f}, {ax['declared_um'][1]:.1f}] um"
                  f" -> realized [{ax['realized_um'][0]:.1f}, "
                  f"{ax['realized_um'][1]:.1f}] um"
                  f" | face residuals ({ax['face_residual_um'][0]:.1f}, "
                  f"{ax['face_residual_um'][1]:.1f}) um"
                  f" | extent {ax['declared_extent_um']:.1f} -> "
                  f"{ax['realized_extent_um']:.1f} um")
            # An axis carrying a note is one the report deliberately does
            # NOT compare. Printing the row without the note leaves a
            # displayed declared-vs-realized gap with no finding and no
            # explanation next to it — a silent all-clear (#303 class).
            if ax.get("note"):
                print(f"       note: {ax['note']}")
        for f in it.get("findings", []):
            ax = f" {f['axis']}:" if f.get("axis") else ""
            print(f"    ! [{f['kind']}]{ax} {f['detail']}")
            print(f"      remedy: {f['remedy']}")
        print()
