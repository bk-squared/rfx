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
    dom_item = dict(entity="domain (the solved box)", findings=[], axes=[])
    for a in range(3):
        n_int = len(sizes[a]) - int(getattr(grid, f"pad_{_axis_names()[a]}_lo"))            - int(getattr(grid, f"pad_{_axis_names()[a]}_hi"))
        realized = float(np.sum(sizes[a][
            int(getattr(grid, f"pad_{_axis_names()[a]}_lo")):
            len(sizes[a]) - int(getattr(grid, f"pad_{_axis_names()[a]}_hi"))]))
        declared = float(domain[a]) if a < len(domain) else 0.0
        ax = dict(axis=_axis_names()[a], n_cells=int(n_int),
                  declared_um=(0.0, declared * 1e6),
                  realized_um=(0.0, realized * 1e6),
                  face_residual_um=(0.0, abs(realized - declared) * 1e6),
                  declared_extent_um=declared * 1e6,
                  realized_extent_um=realized * 1e6)
        dom_item["axes"].append(ax)
        if declared > 0 and abs(realized - declared) > 0.005 * declared:
            dom_item["findings"].append(dict(
                kind="domain-extent-quantized", axis=ax["axis"],
                detail=f"declared {declared * 1e6:.1f} um realized "
                       f"{realized * 1e6:.1f} um over {n_int} cells "
                       f"({(realized - declared) * 1e6:+.1f} um, "
                       f"{100 * (realized - declared) / declared:+.2f}%) — "
                       "the cell size does not divide the declared length",
                remedy="choose a cell size that divides this length, or "
                       "accept and quote the REALIZED length (for a "
                       "PEC/PMC-walled model the realized length is the "
                       "cavity/guide dimension the solve actually has)"))
    report.append(dom_item)

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
            report.append(dict(entity=name, findings=[dict(
                kind="no-analytic-bounds",
                detail="shape exposes no bounding_box(); declared-vs-realized "
                       "bounds cannot be audited",
                remedy="give the shape an axis-aligned bounding box")]))
            continue
        boxlike = type(entry.shape).__name__ == "Box"
        mask = _entity_mask(entry, sim, grid, nonuniform)
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
            if 1e-9 < pec_frac <= 0.5:
                item["findings"].append(dict(
                    kind="claimed-by-conductor",
                    detail=f"{100 * pec_frac:.1f}% of this dielectric's cells "
                           "are also PEC — the conductor wins, so that "
                           "fraction of the declared material is not solved",
                    remedy="check the overlap against the intended stack; "
                           "shrink whichever body is over-declared"))
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
            if ext > 0:
                worst = max(ax["face_residual_um"])
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
                        item["findings"].append(dict(
                            kind="bbox-offset", axis=ax["axis"],
                            detail=f"PLACEMENT off by {worst:.1f} um "
                                   f"({100 * worst / ext:.2f}% of the declared "
                                   f"{ext:.1f} um extent); {size_txt}. Curved "
                                   "boundary sampled cell-wise: placement can "
                                   "shift by up to a cell, and the size "
                                   "changes with the staircase — both are "
                                   "reported above, neither is assumed",
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
        if realized_conductor and mat_name != "pec":
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
        for f in it.get("findings", []):
            ax = f" {f['axis']}:" if f.get("axis") else ""
            print(f"    ! [{f['kind']}]{ax} {f['detail']}")
            print(f"      remedy: {f['remedy']}")
        print()
