"""Realized placement of non-port sources and probes, and the mesh scales.

Every index here comes from the function the runners call to place the same
declaration: ``field_index`` for soft point sources and point probes (all
five paths), the lane's own ``position_to_index`` for DFT and flux planes,
``resolve_flux_region`` for finite flux windows, ``make_ntff_box`` for NTFF
on both meshes and ``tfsf_from_declaration`` for the TFSF
planes. Index boxes are inclusive ``(lo, hi)`` per axis on the padded grid;
``None`` on an axis means the region is not bounded there.
"""
from __future__ import annotations

import numpy as np

from rfx._grid_metric import field_index

C0 = 299_792_458.0


def _box(grid, fixed):
    """Full inclusive extent on every axis, overridden by ``fixed``."""
    return tuple(fixed.get(a, (0, int(grid.shape[a]) - 1)) for a in range(3))


def _locate(ctx):
    if ctx.lane == "nonuniform":
        from rfx.nonuniform import position_to_index
        return lambda pos: position_to_index(ctx.grid, pos)
    return ctx.grid.position_to_index


def _error(exc):
    return f"{type(exc).__name__}: {exc}"


def _sources(sim, ctx):
    from rfx.realized_geometry import SourceGeometry
    rows = []
    for i, pe in enumerate(sim._ports):
        if pe.impedance != 0.0:
            continue
        try:
            cell = tuple(int(k) for k in field_index(ctx.grid, pe.position, pe.component))
            rows.append(SourceGeometry(f"port[{i}]", "soft_point", pe.component, (cell,)))
        except Exception as exc:  # a misplaced source refuses at run time
            rows.append(SourceGeometry(f"port[{i}]", "soft_point", pe.component, error=_error(exc)))
    if sim._tfsf is not None:
        rows.append(_tfsf_row(sim, ctx))
    return tuple(rows)


def _tfsf_row(sim, ctx):
    from rfx.realized_geometry import SourceGeometry
    from rfx.sources.tfsf import tfsf_from_declaration, tfsf_injection_planes
    entry, grid = sim._tfsf, ctx.grid
    try:
        if ctx.lane == "nonuniform":
            cfg, _ = tfsf_from_declaration(entry, grid, sim._freq_max)
            planes = tfsf_injection_planes(cfg)
        else:
            planes = sim.tfsf_box_indices()
        box = tuple(planes.get("xyz"[a]) for a in range(3))
        return SourceGeometry("tfsf", "tfsf", entry.polarization, index_box=box)
    except Exception as exc:
        return SourceGeometry("tfsf", "tfsf", entry.polarization, error=_error(exc))


def _probes(sim, ctx):
    from rfx.realized_geometry import ProbeGeometry
    grid, locate, rows = ctx.grid, _locate(ctx), []
    internal = getattr(sim, "_internal_probe_indices", None) or ()
    for i, pe in enumerate(sim._probes):
        if i in internal:
            continue  # a library witness probe (MSL / mixed extraction), not the user's
        try:
            index = tuple(int(k) for k in field_index(grid, pe.position, pe.component))
            rows.append(ProbeGeometry(f"probe[{i}]", "point", (pe.component,), index=index))
        except Exception as exc:
            rows.append(ProbeGeometry(f"probe[{i}]", "point", (pe.component,), error=_error(exc)))
    regions = getattr(sim, "_dft_plane_regions", {})
    for pe in sim._dft_planes:
        a = "xyz".index(pe.axis)
        try:
            pos = [0.0, 0.0, 0.0]
            pos[a] = pe.coordinate
            k = int(locate(tuple(pos))[a])
            fixed = {a: (k, k)}
            region = regions.get(pe.name)
            if region is not None:
                t = [b for b in range(3) if b != a]
                fixed[t[0]] = (int(region[0]), int(region[1]) - 1)
                fixed[t[1]] = (int(region[2]), int(region[3]) - 1)
            rows.append(ProbeGeometry(pe.name, "dft_plane", (pe.component,), index_box=_box(grid, fixed)))
        except Exception as exc:
            rows.append(ProbeGeometry(pe.name, "dft_plane", (pe.component,), error=_error(exc)))
    for pe in getattr(sim, "_flux_monitors", ()):
        rows.append(_flux_row(sim, ctx, pe, locate))
    if sim._ntff is not None:
        rows.append(_ntff_row(sim, ctx, locate))
    if getattr(sim, "_current_moments", None) is not None:
        rows.append(ProbeGeometry("current_moments", "current_moments", (), error=(
            "not represented: the block partition is built by the run's monitor")))
    return tuple(rows)


def _flux_row(sim, ctx, pe, locate):
    from rfx.realized_geometry import ProbeGeometry
    from rfx.probes.flux_region import resolve_flux_region
    a = "xyz".index(pe.axis)
    components = tuple(f"{f}{'xyz'[b]}" for f in "eh" for b in range(3) if b != a)
    try:
        pos = [0.0, 0.0, 0.0]
        pos[a] = pe.coordinate
        fixed = {a: (int(locate(tuple(pos))[a]),) * 2}
        region = resolve_flux_region(ctx.grid, pe, sim._domain, warn=False)
        if region is not None:
            k = int(region["normal_index"])
            fixed[a] = (k, k)
            t = [b for b in range(3) if b != a]
            for b, (lo, hi) in zip(t, region["cell_slices"]):
                fixed[b] = (int(lo), int(hi) - 1)
        return ProbeGeometry(pe.name, "flux", components, index_box=_box(ctx.grid, fixed))
    except Exception as exc:
        return ProbeGeometry(pe.name, "flux", components, error=_error(exc))


def _ntff_row(sim, ctx, locate):
    from rfx.realized_geometry import ProbeGeometry
    corner_lo, corner_hi, freqs = sim._ntff
    components = ("ex", "ey", "ez", "hx", "hy", "hz")
    try:
        from rfx.farfield import make_ntff_box
        b = make_ntff_box(ctx.grid, corner_lo, corner_hi, freqs)
        lo, hi = (b.i_lo, b.j_lo, b.k_lo), (b.i_hi, b.j_hi, b.k_hi)
        box = tuple((int(lo[a]), int(hi[a])) for a in range(3))
        return ProbeGeometry("ntff", "ntff", components, index_box=box)
    except Exception as exc:
        return ProbeGeometry("ntff", "ntff", components, error=_error(exc))


def placement_rows(sim, ctx):
    """``(sources, probes)`` rows for the record of ``ctx``'s grid."""
    return _sources(sim, ctx), _probes(sim, ctx)


def unrepresented(rows, why):
    """Rows of a lane that places them on its own grid (the refined region)."""
    from dataclasses import fields, replace
    blank = {"edges": (), "index": None, "index_box": None}
    return tuple(replace(r, error=f"not represented: {why}",
                         **{f.name: blank[f.name] for f in fields(r) if f.name in blank})
                 for r in rows)


def cells_per_wavelength(sim, sizes, domain, materials, pec_mask):
    """Interior cells per shortest wavelength at ``freq_max``, vacuum and medium.

    The cell is the largest interior (non-absorber) cell edge over the three
    axes; the medium uses the largest volume eps_r*mu_r over interior cells
    outside the PEC mask. ``medium`` is None when materials are traced.
    """
    from rfx.realized_geometry import CellsPerWavelength
    from rfx.core.jax_utils import is_tracer
    slices, widths = [], []
    for a, row in enumerate(domain):
        lo, n = row.padding[0], row.n_cells
        slices.append(slice(lo, lo + n) if n > 0 else slice(None))
        if n > 0:
            widths.append(float(np.max(sizes[a][lo:lo + n])))
    if not widths:
        return None
    f, cell = float(sim._freq_max), max(widths)
    vacuum = C0 / f / cell
    medium = eps_mu = None
    try:
        if not any(is_tracer(v) for v in (materials.eps_r, materials.mu_r, pec_mask)):
            region = tuple(slices)
            # Volume permittivity: a lumped capacitor stamp is not a medium.
            from rfx.model.materials import volume_eps_mu
            product = np.asarray(volume_eps_mu(materials), np.float64)
            product = np.broadcast_to(product, np.shape(materials.eps_r))[region]
            keep = (np.ones(product.shape, bool) if pec_mask is None
                    else ~np.asarray(pec_mask, bool)[region])
            if keep.any():
                eps_mu = float(product[keep].max())
                medium = float(C0 / (f * np.sqrt(eps_mu)) / cell)
    except Exception:  # an unreadable material operand leaves medium unset
        medium = eps_mu = None
    return CellsPerWavelength(vacuum, medium, cell, f, eps_mu)
