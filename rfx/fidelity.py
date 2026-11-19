"""Input-fidelity report: what you declared vs what will actually be solved.

Design rule: a simulation exists because the RESULT is
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

from rfx.model.materials import assembled_as_pec
from rfx.model.thin_conductors import conductivity_envelope

import jax.numpy as jnp
import numpy as np

__all__ = ["fidelity_report"]

_SUB_CELL_TIE_REL = 1e-9


def _zero_tie(residual_m, cell_m):
    """Return a face residual with pure float64 round-off snapped to 0.0.

    A declared face that sits ON a node still leaves ``k*dz - face`` at the
    1e-18 m level (a cumsum / product ulp), and that last digit differs
    between hosts (measured: 3.5e-12 um on CI vs 0.0 here for the
    same variant), so a snapshot that records it is platform-dependent. The
    margin is the same 1e-9 of the LOCAL cell used for ``sub_cell``: far above
    round-off, far below any placement a user could intend.
    """
    if abs(float(residual_m)) < _SUB_CELL_TIE_REL * abs(float(cell_m)):
        return 0.0
    return float(residual_m)


def _axis_names():
    return ("x", "y", "z")


from rfx.realized_geometry import _node_arrays


def _entity_mask(entry, sim, grid, nonuniform, *, pec_volume: bool = False, record=None):
    """Read an entity mask from the public diagnostic record."""
    collection = sim._thin_conductors if hasattr(entry, "sigma_bulk") else sim._geometry
    prefix = "thin_conductor" if hasattr(entry, "sigma_bulk") else "geometry"
    index = next(i for i, candidate in enumerate(collection) if candidate is entry)
    if record is None:
        record = sim.realized_geometry()
    row = next(row for row in record.entities
               if row.label == f"{prefix}[{index}]")
    if row.mask_error is not None:
        raise ValueError(row.mask_error)
    return row.mask


def _declared_material(sim, name):
    if name == "pec":
        return dict(kind="pec")
    try:
        spec = sim._resolve_material(name)
    except KeyError:
        return dict(kind="unknown", name=name)
    return dict(kind="dielectric", name=name,
                eps_r=float(getattr(spec, "eps_r", 1.0)),
                sigma=float(getattr(spec, "sigma", 0.0)),
                n_debye=len(getattr(spec, "debye_poles", None) or ()),
                n_lorentz=len(getattr(spec, "lorentz_poles", None) or ()))


def _contract_coords(sim, grid, nonuniform):
    """(node coords, per-cell sizes) on the grid the report audits."""
    from rfx.geometry.rasterize_grid import (
        GridCoords, cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
        coords_from_nonuniform_grid)
    if nonuniform:
        return coords_from_nonuniform_grid(grid), cell_sizes_from_nonuniform_grid(grid)
    from rfx.geometry.csg import _grid_coords
    x, y, z = _grid_coords(grid)
    return (GridCoords(x=x, y=y, z=z, shape=tuple(grid.shape)),
            cell_sizes_from_uniform_grid(grid))


def _contract_refusals(sim, grid, nonuniform):
    """PEC geometry entries the classifier refuses (sub-cell Box, a
    line/point Box, a zero-cell volume): ``{index: message}``.  The report
    is input-fidelity-only and must not crash on the very input it exists
    to explain, so these entries are excluded from the audited assembly
    and carry a finding instead."""
    from rfx.geometry.rasterize_grid import cell_centres_from_nodes, classify_pec_entry
    out = {}
    coords, sizes = _contract_coords(sim, grid, nonuniform)
    centres = cell_centres_from_nodes(coords, sizes)
    for i, entry in enumerate(sim._geometry):
        if not assembled_as_pec(sim, entry):
            continue
        try:
            classify_pec_entry(entry.shape, coords, centres, sizes,
                               name=entry.material_name, grid=grid)
        except ValueError as exc:
            out[i] = str(exc)
    return out


def _max_run_length(mask, axis):
    """Longest contiguous run of True along `axis` over occupied lines.

    Sheet detection must use the MAXIMUM, not the minimum: a solid sphere has
    single-cell runs at its rim, so the minimum is 1 for any curved body and
    the first implementation called a 1045-cell PEC sphere a "one-plane
    sheet" and claimed vacuum was substituted for its metal (found by the
    crossval sweep, rcs_scattering.py).
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
    * realization class for conductors
      — volumetric PEC / one-cell PEC volume (walls on BOTH bounding node
      planes, interior shorted); a zero-thickness Box or a PEC thin
      conductor is a sheet and owns no cell;
    * materialization for dielectrics — the assembled eps_r/sigma inside
      the entity's cells vs the declared values (later entities may have
      overwritten earlier ones);
    * a mechanical remedy per finding class (no impact estimates).

    The list LEADS with a ``"domain (the solved box)"`` pseudo-entity —
    the only row a cavity/waveguide model has, since there the box IS the
    geometry. Its per-axis ``n_cells`` / ``realized_extent_um`` are the
    wall-to-wall CELL count and length, i.e. ``cells_spanning(L, dx)`` cells,
    not the ``+ 1`` node count the grid allocates. ``cells_spanning`` snaps ratios within a few ULP of an integer to it,
    avoiding a cell no declared Box reaches. An axis the solve
    does not have — ``z`` under ``mode="2d_tmz"``, where ``grid.nz == 1``
    and the declared ``Lz`` is ignored — is not compared: that axis dict
    carries a ``note`` key (rendered by the printed report) instead of a
    finding.

    A PMC domain face, realized by the odd tangential-H image, is a magnetic
    wall on the E-node mesh plane, not on the half-cell H samples.
    """
    record = sim.realized_geometry()
    from rfx.preflight.realization import context_from_conductors
    grid = getattr(record, "_report_grid", None)
    if grid is None:
        ctx = (context_from_conductors(sim, record.conductors) if record.conductors is not None
               else sim._campaign_ctx())
        grid = ctx.grid
        nonuniform = ctx.lane == "nonuniform"
    else:
        nonuniform = hasattr(grid, "dx_arr")
    refused = dict(record.refused)
    refused_tc = dict(record.refused_thin_conductors)
    assembled_sheets = record.sheets
    assembled_wires = record.wires
    pad_fill_findings = [dict(row) for row in record.pad_fill_findings]
    rows = {row.label: row for row in record.entities}
    table_rows = {row.label: row for row in record.realization_table.rows}
    mats = record.materials
    eps = np.asarray(mats.eps_r, dtype=float)
    sigma_arr = np.asarray(conductivity_envelope(mats), dtype=float)
    pec_mask = (record.pec_mask if record.pec_mask is not None else np.zeros(eps.shape, bool))
    nodes = record.nodes
    domain = tuple(d.declared_length_m for d in table_rows["domain"].domain_geometry)
    pads = (int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo))
    nodes = tuple(n - n[p] for n, p in zip(nodes, pads))

    entries = [("geometry", i, e) for i, e in enumerate(sim._geometry)]
    for j, tc in enumerate(getattr(sim, "_thin_conductors", ()) or ()):
        entries.append(("thin_conductor", j, tc))

    report = []

    is_2d = getattr(grid, "is_2d", False)
    dom_item = dict(entity="domain (the solved box)", findings=[], axes=[])
    dom_item["realization_table"] = record.realization_table
    if nonuniform and getattr(sim, "_interface_eps", "sampled") == "dual_average":
        dom_item["interface_eps_rule"] = sim._interface_eps
    for a in range(3):
        axis_name = _axis_names()[a]
        n_int = table_rows["domain"].domain_geometry[a].n_cells
        mesh_extent = table_rows["domain"].axes[a].realized_span_m
        declared = float(domain[a]) if a < len(domain) else 0.0
        axis_not_solved = is_2d and _axis_names()[a] == "z"
        eff_lo = 0.0
        eff_hi = mesh_extent
        realized = eff_hi - eff_lo
        cell_lo, cell_hi = table_rows["domain"].domain_face_cells_m[a]
        ax = dict(axis=axis_name, n_cells=int(n_int),
                  declared_um=(0.0, declared * 1e6),
                  mesh_extent_um=mesh_extent * 1e6,
                  realized_um=(eff_lo * 1e6, eff_hi * 1e6),
                  face_residual_um=(_zero_tie(eff_lo, cell_lo) * 1e6,
                                    _zero_tie(abs(eff_hi - declared), cell_hi) * 1e6),
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
    report.append(dom_item)

    pec_before = np.zeros(eps.shape, dtype=bool)
    pec_cells_by_entity: dict = {}
    pec_unrasterized: list = []
    pec_faces_by_entity: dict = {}

    for kind_src, i, entry in entries:
        label = f"{kind_src}[{i}]"
        if table_rows[label].status.startswith("not represented:"):
            report.append(dict(entity=label, findings=[dict(
                kind="not-represented", detail=table_rows[label].status,
                remedy="use a solver lane that carries these material declarations")]))
            continue
        geometry = rows[label]
        if kind_src == "thin_conductor":
            sig = float(getattr(entry, "sigma_bulk", 0.0))
            mat_name = "pec" if sig >= 1e6 else "lossy-sheet"
            name = (f"thin_conductor[{i}] sigma_bulk={sig:.3g} S/m"
                    f" -> {mat_name}")
        else:
            name = f"geometry[{i}] '{entry.material_name}'"
            mat_name = entry.material_name
        try:
            lo, hi = table_rows[geometry.label].declared_bounds_m
        except Exception:
            nb_item = dict(entity=name, findings=[dict(
                kind="no-analytic-bounds",
                detail="shape exposes no bounding_box(); declared-vs-realized "
                       "bounds cannot be audited",
                remedy="give the shape an axis-aligned bounding box")])
            report.append(nb_item)
            if kind_src == "geometry" and assembled_as_pec(sim, entry):
                try:
                    if geometry.mask_error is not None:
                        import builtins
                        cls, message = geometry.mask_error.split(": ", 1)
                        raise getattr(builtins, cls, ValueError)(message)
                    nb_mask = _entity_mask(entry, sim, grid, nonuniform, record=record)
                except (TimeoutError,): raise
                except Exception as exc:  # noqa: BLE001
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
        pec_assembled = kind_src == "geometry" and assembled_as_pec(sim, entry)
        sheet_spec = geometry.sheet
        sheet_fp = (np.asarray(sheet_spec.footprint, dtype=bool)
                    if sheet_spec is not None else None)
        mask = _entity_mask(entry, sim, grid, nonuniform, record=record)
        mask = mask if mask is not None else np.zeros(eps.shape, bool)
        item = dict(entity=name, material=_declared_material(sim, mat_name),
                    declared_lo=tuple(float(v) for v in lo),
                    declared_hi=tuple(float(v) for v in hi),
                    n_cells=table_rows[geometry.label].occupancy_count, findings=[])
        if (pec_assembled and i not in refused) or kind_src == "thin_conductor":
            item["continued_faces"] = list(table_rows[geometry.label].continued_faces)
        if sheet_fp is not None:
            item["n_cells"] = 0
            item["n_sheet_nodes"] = table_rows[geometry.label].occupancy_count
        if item["n_cells"] == 0 and sheet_fp is None:
            item["findings"].append(dict(
                kind="absent",
                detail="rasterizes to ZERO cells — this entity does not exist "
                       "in the solved model",
                remedy="thinner than a cell? place a mesh node inside its span "
                       "or refine the local cell size"))
            report.append(item)
            continue

        for _shortfall in pad_fill_findings:
            if _shortfall.get("shape") is not entry.shape:
                continue
            item["findings"].append(dict(
                kind="declared-span-short-of-padded-face",
                axis=_shortfall["axis"],
                detail=(
                    f"declared out to the {_shortfall['face']} face "
                    f"({_shortfall['declared_extent_m'] * 1e6:.1f} um = "
                    f"{_shortfall['declared_cells']!r} cells) but the "
                    f"realized mask stops "
                    f"{_shortfall['empty_interior_nodes']} interior nodes "
                    f"short of it, and that face carries a "
                    f"{_shortfall['pad_cells']}-cell absorber filled by "
                    f"replicating the interior edge — so the pad is vacuum "
                    f"and the structure ends in a facet inside its own "
                    f"absorber. One node short is the "
                    f"documented half-open [lo, hi) shortfall and is "
                    f"repaired; "
                    f"{_shortfall['empty_interior_nodes']} is not"),
                remedy=("choose a domain length commensurate with dx, or "
                        "check what put an interior node beyond the "
                        "declared span — rfx.grid.cells_spanning absorbs "
                        "float dust in domain/dx, so a shortfall with that "
                        "in place has another cause")))
        if mat_name not in ("pec", "lossy-sheet"):
            pec_frac = float(np.mean(pec_mask[mask]))
            _declared_sigma = float(item["material"].get("sigma", 0.0) or 0.0)
            if pec_frac > 1e-9 and _declared_sigma < 1e6:
                item["findings"].append(dict(
                    kind="claimed-by-conductor",
                    detail=f"{100 * pec_frac:.1f}% of this dielectric's cells "
                           "are also PEC — the conductor wins, so that "
                           "fraction of the declared material is not solved",
                    remedy="check the overlap against the intended stack; "
                           "shrink whichever body is over-declared"))
        if kind_src == "geometry" and not pec_assembled:
            n_cells = int(np.count_nonzero(mask & pec_before))
            flat = mask.ravel()
            contributors = []
            for m, cells_m in pec_cells_by_entity.items():
                if m < i:
                    n_m = int(np.count_nonzero(flat[cells_m]))
                    if n_m:
                        contributors.append((m, n_m))
            faces_by_normal = {}
            for m, (normal, faces) in pec_faces_by_entity.items():
                if m >= i:
                    continue
                coords = np.unravel_index(faces, mask.shape)
                valid = coords[normal] > 0
                faces = faces[valid]
                stride = int(np.prod(mask.shape[normal + 1:]))
                occupied = faces[flat[faces] & flat[faces - stride]]
                if occupied.size:
                    contributors.append((m, int(occupied.size)))
                    faces_by_normal.setdefault(normal, []).append(occupied)
            n_faces = sum(np.unique(np.concatenate(faces)).size
                          for faces in faces_by_normal.values())
            n_ov = n_cells + int(n_faces)
            if n_ov > 0:
                contributors.sort()
                who = ", ".join(
                    f"geometry[{m}] '{sim._geometry[m].material_name}' "
                    f"({n_m} "
                    f"{'faces' if m in pec_faces_by_entity else 'cells'})"
                    for m, n_m in contributors)
                if n_faces:
                    detail = (
                        f"{n_faces} metal faces have this entity's declared "
                        "dielectric in the cells on BOTH sides"
                        + (f"; {n_cells} cells are already PEC" if n_cells else "")
                        + f" from a conductor declared EARLIER: {who}. "
                        "_assemble_materials is PEC-OR-only and there is no "
                        "CSG subtraction, so a dielectric declared after a "
                        "conductor cannot carve it. A sheet contributor owns "
                        "no cell, so adjacent cells keep the eps_r/sigma "
                        "declared here — what does not exist is the "
                        "CLEARANCE: the sheet's metal stays on its node "
                        "plane and shorts the tangential E there. If a "
                        "clearance/hole was intended, build the conductor "
                        "with the hole")
                else:
                    detail = (
                        f"{n_ov} of this entity's {item['n_cells']} cells "
                        f"({100.0 * n_ov / item['n_cells']:.1f}%) are "
                        f"already PEC from an entity declared EARLIER: "
                        f"{who}. _assemble_materials is PEC-OR-only, so a "
                        "dielectric declared after a conductor cannot "
                        f"carve it — these {n_ov} cells are a no-op (the "
                        "eps_r/sigma written there is never solved); if a "
                        "clearance/hole was intended, build the conductor "
                        "with the hole")
                item["findings"].append(dict(
                    kind="dielectric-after-conductor-no-op",
                    overlap_cells=n_cells,
                    overlap_faces=int(n_faces),
                    conductor_entities=[m for m, _ in contributors],
                    detail=detail,
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
            if sheet_fp is None:
                pec_before |= mask
                pec_cells_by_entity[i] = np.flatnonzero(mask)
            else:
                normal = int(table_rows[geometry.label].plane[0])
                pec_faces_by_entity[i] = (
                    normal, np.flatnonzero(_sheet_contact_faces(sheet_fp, normal)))
        pad_hit = []
        for a in range(3):
            idx = np.where(mask.any(axis=tuple(x for x in range(3) if x != a)))[0]
            drawn_in_pad = (
                (float(lo[a]) < -1e-12 and getattr(grid, f"pad_{'xyz'[a]}_lo") > 0)
                or (float(hi[a]) > domain[a] + 1e-12
                    and getattr(grid, f"pad_{'xyz'[a]}_hi") > 0))
            if len(idx) and (drawn_in_pad
                             or nodes[a][idx.min()] < -1e-12
                             or nodes[a][idx.max() + (table_rows[geometry.label].record_axes[a].cell_range is not None)]
                             > domain[a] + 1e-12):
                pad_hit.append(_axis_names()[a])
        if pad_hit:
            item["findings"].append(dict(
                kind="inside-absorber", axis=",".join(pad_hit),
                detail="the realized body extends beyond the declared domain "
                       f"on {','.join(pad_hit)} — those cells sit in the CPML "
                       "pad, where the update equations are the absorber's",
                remedy="keep the body inside the domain, or enlarge the "
                       "domain so the absorber stays empty"))

        axes = []
        clipped_axes = []
        for a, comparison in enumerate(table_rows[geometry.label].record_axes):
            r_lo, r_hi = comparison.bounds_m
            d_lo, d_hi = float(lo[a]), float(hi[a])
            dom_hi = float(domain[a]) if a < len(domain) else 0.0
            if dom_hi > 0 and (d_lo < -1e-12 or d_hi > dom_hi + 1e-12):
                clipped_axes.append(_axis_names()[a])
                d_lo, d_hi = max(d_lo, 0.0), min(d_hi, dom_hi)
            cell_m = comparison.cell_size_m
            cell_um = cell_m * 1e6
            residual = comparison.face_residual_m
            if (d_lo, d_hi) != comparison.declared_bounds_m:
                residual = (r_lo - d_lo, r_hi - d_hi)
            ax = dict(axis=_axis_names()[a],
                      declared_um=(d_lo * 1e6, d_hi * 1e6),
                      realized_um=(r_lo * 1e6, r_hi * 1e6),
                      face_residual_um=(_zero_tie(abs(residual[0]), cell_m) * 1e6,
                                        _zero_tie(abs(residual[1]), cell_m) * 1e6),
                      declared_extent_um=(d_hi - d_lo) * 1e6,
                      realized_extent_um=comparison.extent_m * 1e6)
            ext = ax["declared_extent_um"]
            ax["cell_um"] = cell_um
            ax["sub_cell"] = bool(ext < cell_um * (1.0 - _SUB_CELL_TIE_REL))
            if ext > 0:
                worst = max(ax["face_residual_um"])
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
                    axes.append(ax)
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
                            remedy=(
                                "a sheet's in-plane edge is solved about 0.35 "
                                "cell beyond its last node, so a node ON the "
                                "edge is not the fix: put a node 0.35 cell "
                                "INSIDE the metal with "
                                "rfx.mesh_edges.edge_aware_profile "
                                "(non-uniform profile for this axis)"
                                if sheet_fp is not None else
                                "place a mesh node on this face "
                                "(non-uniform profile) or choose the cell "
                                "size / origin commensurate with it")))
                    else:
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

        if kind_src == "geometry" and i in refused:
            item["realization"] = "REFUSED by the lattice ownership contract"
            item["findings"].append(dict(
                kind="refused-by-contract",
                detail=refused[i],
                remedy="declare a sheet (a zero-thickness Box, or "
                       "add_thin_conductor), a radius=0 PolylineWire for a "
                       "legacy filament, or resolve the thickness and any "
                       "positive wire radius with the mesh (a >= 0.5*d for wires)"))
        if kind_src == "thin_conductor" and i in refused_tc:
            item["findings"].append(dict(
                kind="refused-by-run",
                detail="run() would refuse this model: " + refused_tc[i],
                remedy="draw it as a Box, or run on a uniform mesh"))
        if sheet_spec is not None:
            a, k = sheet_spec.normal_axis, sheet_spec.plane
            z_k = float(nodes[a][k])
            n_nodes = int(np.asarray(sheet_spec.footprint).sum())
            mid = 0.5 * (item["declared_lo"][a] + item["declared_hi"][a])
            from rfx._internal._periodic import plane_coordinate
            matched_mid = plane_coordinate(grid, a, mid)
            item["realization"] = (
                f"PEC sheet on node plane {_axis_names()[a]} = "
                f"{z_k * 1e6:.2f} um ({n_nodes} nodes, closed footprint, "
                "zero thickness, no cell; in-plane E zeroed on that plane, "
                "normal E live)")
            item["realized_plane"] = dict(axis=_axis_names()[a], index=int(k),
                                          coordinate=z_k,
                                          declared_midplane=float(mid),
                                          offset_um=(z_k - matched_mid) * 1e6)
            report.append(item)
            continue
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
            item["realized_wall_planes"] = {
                _axis_names()[a]: dict(planes=list(planes),
                                      planes_um=[v * 1e6 for v in table_rows[geometry.label].wall_positions_m[a]])
                for a, planes in enumerate(table_rows[geometry.label].wall_planes)}
            runs = [_max_run_length(mask, a) for a in range(3)]
            thin_axes = [a for a, r in enumerate(runs) if r == 1]
            if not thin_axes:
                item["realization"] = "volumetric PEC (>= 2 cells on every axis)"
            else:
                aname = "+".join(_axis_names()[a] for a in thin_axes)
                item["realization"] = (
                    f"one-cell PEC volume (thin along {aname}): electric "
                    "walls on BOTH bounding node planes, interior shorted")
        else:
            m = item["material"]
            if m["kind"] == "dielectric":
                inside = eps[mask]
                frac = float(np.mean(np.isclose(inside, m["eps_r"], rtol=1e-6)))
                item["assembled_eps_r"] = (float(inside.min()),
                                           float(inside.max()))
                item["assembled_matches_declared_frac"] = frac
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

    for j, ps in enumerate(getattr(sim, "_pinned_sheets", ()) or ()):
        geometry = rows[f"pinned_sheet[{j}]"]
        a, k, plane_m = table_rows[geometry.label].plane
        info = dict(normal_axis=a, plane_index=k, plane_m=plane_m,
                    n_nodes=table_rows[geometry.label].occupancy_count)
        for axis in table_rows[geometry.label].record_axes:
            info[f"{axis.axis}_node_indices"] = axis.node_range
            info[f"{axis.axis}_edges_m"] = axis.bounds_m
            info[f"{axis.axis}_span_m"] = axis.extent_m
        a = int(info["normal_axis"])
        k = int(info["plane_index"])
        others = tuple(b for b in range(3) if b != a)
        name = f"pinned_sheet[{j}]" + (f" '{ps.name}'" if ps.name else "")
        item = dict(entity=name,
                    material=dict(name="pec", eps_r=1.0, sigma=float("inf")),
                    declared_node_plane=int(ps.plane_index),
                    declared_node_ranges={
                        _axis_names()[others[0]]: tuple(ps.i_range),
                        _axis_names()[others[1]]: tuple(ps.j_range)},
                    n_cells=0, findings=[])
        item["realization"] = (
            f"PEC sheet on node plane {_axis_names()[a]} = "
            f"{info['plane_m'] * 1e6:.2f} um ({info['n_nodes']} nodes, "
            "declared by node index, zero thickness, no cell; in-plane E "
            "zeroed on that plane, normal E live)")
        item["realized_plane"] = dict(axis=_axis_names()[a], index=k,
                                      coordinate=info["plane_m"])
        item["axes"] = [
            dict(axis=_axis_names()[t],
                 node_indices=info[f"{_axis_names()[t]}_node_indices"],
                 realized_lo_um=info[f"{_axis_names()[t]}_edges_m"][0] * 1e6,
                 realized_hi_um=info[f"{_axis_names()[t]}_edges_m"][1] * 1e6,
                 realized_extent_um=info[f"{_axis_names()[t]}_span_m"] * 1e6)
            for t in others]
        report.append(item)

    reported_planes = sorted(
        (it["realized_plane"]["axis"], int(it["realized_plane"]["index"]))
        for it in report if "realized_plane" in it)
    assembled_planes = sorted(
        (_axis_names()[int(sp.normal_axis)], int(sp.plane))
        for sp in assembled_sheets)
    if reported_planes != assembled_planes:
        dom_item["findings"].append(dict(
            kind="sheet-report-assembly-drift",
            detail=f"this report names sheet planes {reported_planes} but the "
                   f"assembly the solve uses classified {assembled_planes} — "
                   "the audit and the run disagree about which conductors "
                   "exist and where",
            remedy="treat the assembly's list as authoritative and re-derive "
                   "the report's sheet resolution from "
                   "rfx.geometry.rasterize_grid.sheet_spec_from_shape; do not "
                   "trust either realization until they agree"))
    if assembled_wires:
        dom_item["findings"].append(dict(
            kind="wire-not-audited",
            detail=f"{len(assembled_wires)} PEC filament(s) (sub-cell "
                   "PolylineWire) are realized as lattice edges and own no "
                   "cell, so this report's per-entity cell audit says nothing "
                   "about them",
            remedy="check a filament with "
                   "rfx.boundaries.pec.realized_pec_edge_masks / edge_is_pec "
                   "on its declared path"))

    unaudited = []
    unaudited_entities = []
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
            unaudited_entities.append({"count": n, "class": label})
    if unaudited:
        report.append(dict(
            entity="NOT AUDITED by this report",
            findings=[dict(
                kind="out-of-scope",
                entities=unaudited_entities,
                detail="; ".join(unaudited) + " — these declare geometry "
                       "and/or materials that this report does not walk "
                       "(port pins, shields, end caps, source cells)",
                remedy="audit those with their own port preflight checks; "
                       "do not read this report as covering them")]))

    if print_report:
        if nonuniform:
            print(f"interface eps rule (NU lane): {getattr(sim, '_interface_eps', 'sampled')}")
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
        if it.get("continued_faces"):
            print("    continues through absorber faces: "
                  + ", ".join(it["continued_faces"]))
        if "n_sheet_nodes" in it:
            print(f"    sheet nodes: {it['n_sheet_nodes']} (owns no cell)")
        elif "n_cells" in it:
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
            if ax.get("note"):
                print(f"       note: {ax['note']}")
            walls = it.get("realized_wall_planes", {}).get(ax["axis"])
            if walls and walls["planes"]:
                vals = walls["planes_um"]
                um = (", ".join(f"{v:.1f}" for v in vals) if len(vals) <= 4
                      else f"{vals[0]:.1f} .. {vals[-1]:.1f}"
                           f" ({len(vals)} planes)")
                print(f"       realized {ax['axis']} walls at [{um}] um")
        for f in it.get("findings", []):
            ax = f" {f['axis']}:" if f.get("axis") else ""
            print(f"    ! [{f['kind']}]{ax} {f['detail']}")
            print(f"      remedy: {f['remedy']}")
        print()


def _sheet_contact_faces(footprint, normal):
    """Metal faces indexed at their lower in-plane corner on the node plane.

    All four in-plane corners must be metal; open edges never wrap.
    """
    metal = np.asarray(footprint, bool).copy()
    for a in range(3):
        if a == normal:
            continue
        shifted = np.zeros_like(metal)
        low, high = [slice(None)]*3, [slice(None)]*3
        low[a], high[a] = slice(None, -1), slice(1, None)
        shifted[tuple(low)] = metal[tuple(high)]
        metal &= shifted
    return metal
