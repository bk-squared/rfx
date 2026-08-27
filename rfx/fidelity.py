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
                sigma=float(getattr(spec, "sigma", 0.0)))


def _min_run_length(mask, axis):
    """Shortest contiguous run of True along `axis` over occupied lines."""
    m = np.moveaxis(mask, axis, -1)
    flat = m.reshape(-1, m.shape[-1])
    best = None
    for line in flat[flat.any(axis=1)]:
        padded = np.concatenate([[0], line.view(np.int8), [0]])
        edges = np.flatnonzero(np.diff(padded))
        runs = edges[1::2] - edges[0::2]
        r = int(runs.min())
        best = r if best is None else min(best, r)
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
    pec_mask = (np.asarray(pec_mask, dtype=bool)
                if pec_mask is not None else np.zeros(eps.shape, bool))
    sizes, nodes = _node_arrays(sim, grid, nonuniform)
    # Mask/material arrays live on the PADDED grid (CPML pad cells on every
    # side); declared coordinates are DOMAIN coordinates. Shift node arrays so
    # index pad_lo sits at domain 0.
    pads = (int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo))
    nodes = tuple(n - n[p] for n, p in zip(nodes, pads))

    report = []
    for i, entry in enumerate(sim._geometry):
        name = f"geometry[{i}] '{entry.material_name}'"
        try:
            lo, hi = entry.shape.bounding_box()
        except Exception:
            report.append(dict(entity=name, findings=[dict(
                kind="no-analytic-bounds",
                detail="shape exposes no bounding_box(); declared-vs-realized "
                       "bounds cannot be audited",
                remedy="give the shape an axis-aligned bounding box")]))
            continue
        mask = _entity_mask(entry, sim, grid, nonuniform)
        item = dict(entity=name, material=_declared_material(sim, entry.material_name),
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

        occ = np.where(mask)
        axes = []
        for a in range(3):
            i0, i1 = int(occ[a].min()), int(occ[a].max())
            r_lo, r_hi = float(nodes[a][i0]), float(nodes[a][i1 + 1])
            d_lo, d_hi = float(lo[a]), float(hi[a])
            ax = dict(axis=_axis_names()[a],
                      declared_um=(d_lo * 1e6, d_hi * 1e6),
                      realized_um=(r_lo * 1e6, r_hi * 1e6),
                      face_residual_um=(abs(r_lo - d_lo) * 1e6,
                                        abs(r_hi - d_hi) * 1e6),
                      declared_extent_um=(d_hi - d_lo) * 1e6,
                      realized_extent_um=(r_hi - r_lo) * 1e6)
            ext = ax["declared_extent_um"]
            if ext > 0:
                worst = max(ax["face_residual_um"])
                if worst > 0.005 * ext:
                    item["findings"].append(dict(
                        kind="off-lattice-face", axis=ax["axis"],
                        detail=f"face residual {worst:.1f} um = "
                               f"{100 * worst / ext:.2f}% of the declared "
                               f"{ext:.1f} um extent",
                        remedy="place a mesh node on this face (non-uniform "
                               "profile) or choose the cell size / origin "
                               "commensurate with it"))
            axes.append(ax)
        item["axes"] = axes

        if entry.material_name == "pec":
            runs = [_min_run_length(mask, a) for a in range(3)]
            thin_axes = [a for a, r in enumerate(runs) if r == 1]
            if not thin_axes:
                item["realization"] = "volumetric PEC (>= 2 cells on every axis)"
            else:
                aname = _axis_names()[thin_axes[0]]
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
        for f in it.get("findings", []):
            print(f"    ! [{f['kind']}] {f['detail']}")
            print(f"      remedy: {f['remedy']}")
        print()
