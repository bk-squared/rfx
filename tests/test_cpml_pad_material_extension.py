"""Locks for CPML pad material/pole extension (issue #627).

``rfx/api/_compile.py``'s ``_assemble_materials`` (mirrored on the
non-uniform mesh by ``rfx/runners/nonuniform.py``'s ``assemble_materials_nu``,
#582) extends the interior-edge ``eps_r``/``sigma``/``mu_r`` slice outward
into the CPML padding so guided modes see an impedance-matched absorber.
#627's review of that mirror found two gaps BOTH assemblers inherited from
the (pre-existing) uniform path, now fixed once in the shared
``rfx.geometry.rasterize_grid.extend_cpml_pad_materials``:

(a) For a ``Box`` whose hi face lands on (or past) the domain's last
    interior node, ``rfx.geometry.csg.Box``'s deliberately half-open
    ``[lo, hi)`` volume rasterization drops exactly that node from the
    box's own mask, so the naive interior-edge column for a hi-face pad
    read vacuum even though the structure's real material sits one column
    further in. Measured pre-fix on the #582 fixture: x-lo pad eps_r=4.0,
    x-hi pad eps_r=1.0, for a slab spanning the full x extent.
(b) Debye/Lorentz dispersion-pole masks were never extended into the pad
    at all (only the static eps_r/sigma/mu_r were), so a dispersive
    edge-touching material was impedance-matched at DC but not across the
    band.

The fix is bounded to exactly one column inward on the hi-face fallback
(the rasterizer's per-box shortfall there is deterministically one node)
so a genuine multi-cell vacuum buffer between an interior structure and
the CPML pad — the overwhelmingly common case — is untouched; that
invariant is locked here too (test 4), since an earlier, rejected design
(an unbounded backward scan for "the last non-vacuum column") would have
silently bridged it.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.geometry.csg import Box
from rfx.materials.debye import DebyePole
from rfx.runners.nonuniform import build_nonuniform_grid, assemble_materials_nu

NA, NB, NZ = 45, 39, 4
DX = 1e-3
F0 = 3e9


def _build_uniform(*, dispersive: bool):
    sim = Simulation(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary="cpml", cpml_layers=8)
    if dispersive:
        sim.add_material("slab", eps_r=4.0,
                          debye_poles=[DebyePole(delta_eps=1.0, tau=1e-10)])
    else:
        sim.add_material("slab", eps_r=4.0)
    # Domain-face-touching in x AND y — the #627a trigger.
    sim.add(Box((0.0, 0.0, 0.3 * DX), (NA * DX, NB * DX, 2.0 * DX)), material="slab")
    return sim


def _assemble_uniform(sim):
    grid = sim._build_grid()
    materials, debye_spec, lorentz_spec, pec_mask, *_ = sim._assemble_materials(grid)
    return materials, debye_spec, grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_z_lo


def _assemble_nu(sim):
    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, None,
        dx_profile=np.full(NA, DX), dy_profile=np.full(NB, DX),
    )
    materials, debye_spec, lorentz_spec, pec_mask = assemble_materials_nu(sim, grid)
    return materials, debye_spec, grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_z_lo


def test_hi_face_pad_matches_lo_face_for_domain_touching_box_uniform():
    """(#627a) x-hi pad must carry the slab's eps_r, not vacuum."""
    sim = _build_uniform(dispersive=False)
    materials, _, plx, phx, ply, plz = _assemble_uniform(sim)
    eps = np.asarray(materials.eps_r)
    j, k = ply + 1, plz + 1  # interior cell, well away from the y-edge artifact
    x_lo = float(eps[plx - 1, j, k])
    x_hi = float(eps[-phx, j, k])
    assert x_lo == 4.0, x_lo
    assert x_hi == x_lo, (
        f"x-hi pad eps_r={x_hi} != x-lo pad eps_r={x_lo}: hi-face pad is not "
        f"impedance-matched (vacuum copied instead of the slab's material)")


def test_hi_face_pad_matches_lo_face_for_domain_touching_box_nu():
    """(#627a) NU mirror of the uniform-path lock above."""
    sim = _build_uniform(dispersive=False)
    materials, _, plx, phx, ply, plz = _assemble_nu(sim)
    eps = np.asarray(materials.eps_r)
    j, k = ply + 1, plz + 1
    x_lo = float(eps[plx - 1, j, k])
    x_hi = float(eps[-phx, j, k])
    assert x_lo == 4.0, x_lo
    assert x_hi == x_lo, (
        f"NU x-hi pad eps_r={x_hi} != x-lo pad eps_r={x_lo}")


def test_dispersion_poles_extend_into_hi_face_pad_uniform():
    """(#627b) Debye poles must reach the x-hi pad, not stop at the interior
    edge. Pre-fix this was 0 unconditionally (poles were never extended at
    all, on EITHER face)."""
    sim = _build_uniform(dispersive=True)
    materials, debye_spec, plx, phx, ply, plz = _assemble_uniform(sim)
    assert debye_spec is not None
    poles, masks = debye_spec
    mask = np.asarray(masks[0])
    n_pad_x_hi = int(mask[-phx:, :, :].sum())
    n_pad_x_lo = int(mask[:plx, :, :].sum())
    assert n_pad_x_hi > 0, "no Debye pole cells reached the x-hi CPML pad"
    assert n_pad_x_hi == n_pad_x_lo, (
        f"x-hi pad pole cell count {n_pad_x_hi} != x-lo pad {n_pad_x_lo} "
        f"(both faces should be extended symmetrically for a slab spanning "
        f"the full x extent)")


def test_dispersion_poles_extend_into_hi_face_pad_nu():
    """(#627b) NU mirror of the uniform-path lock above."""
    sim = _build_uniform(dispersive=True)
    materials, debye_spec, plx, phx, ply, plz = _assemble_nu(sim)
    assert debye_spec is not None
    poles, masks = debye_spec
    mask = np.asarray(masks[0])
    n_pad_x_hi = int(mask[-phx:, :, :].sum())
    n_pad_x_lo = int(mask[:plx, :, :].sum())
    assert n_pad_x_hi > 0, "no Debye pole cells reached the NU x-hi CPML pad"
    assert n_pad_x_hi == n_pad_x_lo


def test_genuine_vacuum_buffer_before_cpml_is_not_bridged():
    """A structure that does NOT reach the domain edge (an ordinary interior
    box with several cells of air before the CPML pad — the overwhelmingly
    common simulation layout) must still see a plain-vacuum pad. This is the
    regression guard for the rejected "unbounded scan for the last
    non-vacuum column" design: that alternative would have smeared the
    interior structure's material across the intentional air gap into the
    absorber.
    """
    sim = Simulation(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary="cpml", cpml_layers=8)
    sim.add_material("slab", eps_r=4.0)
    # well inside the domain on every axis — at least 5 cells of vacuum
    # before any CPML pad on x/y, and centred on z
    sim.add(Box((10 * DX, 10 * DX, 0.3 * DX), (20 * DX, 20 * DX, 2.0 * DX)),
            material="slab")
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)
    eps = np.asarray(materials.eps_r)
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, phy = grid.pad_y_lo, grid.pad_y_hi
    plz = grid.pad_z_lo
    k = plz + 1
    # Sample transverse positions INSIDE the box's own extent (box spans
    # interior indices 10..19 on both x and y), not the domain midpoint —
    # a transverse position outside the box is vacuum under every design,
    # including the rejected unbounded-scan one, so it cannot distinguish
    # them. plx+15/ply+15 sit inside [10,20) and are the positions an
    # unbounded backward scan (from the OPPOSITE face's pad) would walk
    # through and incorrectly find the box's material.
    assert float(eps[-phx, ply + 15, k]) == 1.0, (
        "x-hi pad picked up material across a genuine multi-cell vacuum "
        "gap — the hi-face fallback must be bounded to the rasterizer's "
        "documented one-column shortfall, not an unbounded scan")
    assert float(eps[plx + 15, -phy, k]) == 1.0, (
        "y-hi pad picked up material across a genuine multi-cell vacuum gap")


def test_uniform_and_nu_assemblers_stay_byte_identical_after_the_fix():
    """Extends #582's verified byte-identity property to the #627 fix
    itself: both assemblers must still agree exactly, including the
    extended dispersion-pole masks.
    """
    for dispersive in (False, True):
        sim_u = _build_uniform(dispersive=dispersive)
        mat_u, debye_u, *_ = _assemble_uniform(sim_u)
        sim_n = _build_uniform(dispersive=dispersive)
        mat_n, debye_n, *_ = _assemble_nu(sim_n)

        assert np.array_equal(np.asarray(mat_u.eps_r), np.asarray(mat_n.eps_r))
        assert np.array_equal(np.asarray(mat_u.sigma), np.asarray(mat_n.sigma))
        assert np.array_equal(np.asarray(mat_u.mu_r), np.asarray(mat_n.mu_r))
        if debye_u is not None:
            _, masks_u = debye_u
            _, masks_n = debye_n
            assert np.array_equal(np.asarray(masks_u[0]), np.asarray(masks_n[0]))
