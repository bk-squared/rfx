"""The coax line's realized cross-section, with no FDTD.

Two properties the conductor realization has to have, both cheap enough to check
from the stamped arrays alone:

1. **The wall is sealed** — no path, diagonal steps included, from the
   dielectric annulus to the region outside the wall through cells the PEC edge
   realization has not shorted. A ring whose cells touch only at CORNERS is
   8-connected but leaves the free regions 4-DISCONNECTED, so a 4-connected
   count alone would call an open ring sealed; the test asks the 8-connected
   question, which is the one the Yee lattice answers (a corner contact does not
   kill the tangential E edge running through it).
2. **The line does not move with the mesh** — the realized outer radius of the
   dielectric stays within half a cell of the declared one at every cell size.
   Before the wall's inner radius was pinned to the declared ``b`` it was
   ``b - min(dx, (b-a)/2)``, so a dx ladder refined the GEOMETRY as well as the
   mesh: the annulus ran to 1680 um at 375 um cells and to 1897 um at 158 um
   cells, moving the line's characteristic impedance 40.7 -> 45.3 ohm against a
   declared 48.59.

Measured, not asserted in prose: `docs/design_notes/coax_conductor_realization.md`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    coaxial_tem_characteristic_impedance,
    stamp_coaxial_line,
)
from rfx.sources.sources import GaussianPulse

CENTRE = (0.004, 0.004)
# Three cell sizes spanning the lane's own recipe (">= about four annulus
# cells") and one deliberately below it, so the properties are checked where a
# user is warned off as well as where they are not.
ANNULUS = SMA_OUTER_RADIUS - SMA_PIN_RADIUS
CELL_SIZES_M = (ANNULUS / 2.0, ANNULUS / 4.0, ANNULUS / 6.0, ANNULUS / 9.0)


def _line(dx: float):
    sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.040), boundary="cpml",
                     dx=dx)
    sim.add_coaxial_port((CENTRE[0], CENTRE[1], 0.020), face="top", pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    grid = sim._build_grid()
    materials, _, _ = sim._build_materials(grid)
    materials, shell_inner, cells = stamp_coaxial_line(
        grid, materials, center_xy=CENTRE,
        z_lo_index=int(grid.pad_z_lo) + 4,
        z_hi_index=int(grid.shape[2]) - int(grid.pad_z_hi) - 2,
        pin_radius=SMA_PIN_RADIUS, outer_radius=SMA_OUTER_RADIUS)
    return grid, materials, shell_inner, np.asarray(cells)


def _radius(grid):
    dx = float(grid.dx)
    i = np.arange(int(grid.shape[0]))
    j = np.arange(int(grid.shape[1]))
    x = (i - int(grid.pad_x_lo)) * dx - CENTRE[0]
    y = (j - int(grid.pad_y_lo)) * dx - CENTRE[1]
    return np.hypot(x[:, None], y[None, :])


def _reaches_outside(free: np.ndarray, r: np.ndarray, start: np.ndarray,
                     beyond: np.ndarray) -> bool:
    """8-connected flood from ``start`` through ``free``; does it reach
    ``beyond``? Written as an explicit fill rather than imported, so the answer
    is this test's own arithmetic."""
    seeds = np.argwhere(free & start)
    if seeds.size == 0:
        return False
    seen = np.zeros(free.shape, dtype=bool)
    stack = [tuple(seeds[0])]
    seen[tuple(seeds[0])] = True
    nx, ny = free.shape
    while stack:
        ci, cj = stack.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < nx and 0 <= nj < ny and free[ni, nj] and not seen[ni, nj]:
                    seen[ni, nj] = True
                    stack.append((ni, nj))
    return bool((seen & beyond).any())


@pytest.mark.parametrize("dx", CELL_SIZES_M)
def test_the_realized_wall_seals_the_annulus_from_the_outside(dx):
    grid, _materials, _shell_inner, cells = _line(dx)
    edges = realized_pec_edge_masks(cells, sheets=(), wires=(),
                                    periodic=(False, False, False))
    k = int(grid.shape[2]) // 2
    mx, my, mz = (np.asarray(e)[:, :, k] for e in edges)
    free = ~(mx | my | mz)
    r = _radius(grid)
    annulus = (r > SMA_PIN_RADIUS) & (r < SMA_OUTER_RADIUS)
    outside = r > SMA_OUTER_RADIUS + 2.0 * float(grid.dx)
    assert annulus.any() and outside.any(), "the slice has no annulus or no exterior"
    assert not _reaches_outside(free, r, annulus, outside), (
        f"at dx = {dx*1e6:.2f} um an 8-connected path runs from the dielectric "
        "annulus to the region outside the wall through edges the realization "
        "did not short — the wall is not a closed screen")


def test_the_realized_line_does_not_move_with_the_cell_size():
    rows = []
    for dx in CELL_SIZES_M:
        grid, materials, shell_inner, cells = _line(dx)
        k = int(grid.shape[2]) // 2
        r = _radius(grid)
        eps = np.asarray(materials.eps_r)[:, :, k]
        fill = (eps > 1.0 + 1e-6) & ~cells[:, :, k]
        assert fill.any(), f"no dielectric cell at dx = {dx*1e6:.2f} um"
        rows.append((dx, float(r[fill].max()), float(shell_inner)))

    for dx, fill_max, shell_inner in rows:
        assert shell_inner == pytest.approx(SMA_OUTER_RADIUS, rel=0, abs=0), (
            f"the wall's inner radius is {shell_inner*1e6:.3f} um at "
            f"dx = {dx*1e6:.2f} um, not the declared "
            f"{SMA_OUTER_RADIUS*1e6:.3f} um")
        # Centre-sampled cells cannot land exactly on the declared radius; half
        # a cell is the most the rasterization may cost.
        assert abs(fill_max - SMA_OUTER_RADIUS) <= 0.5 * dx, (
            f"at dx = {dx*1e6:.2f} um the dielectric reaches "
            f"{fill_max*1e6:.2f} um, more than half a cell from the declared "
            f"{SMA_OUTER_RADIUS*1e6:.2f} um")

    spread = max(x[1] for x in rows) - min(x[1] for x in rows)
    assert spread <= 0.5 * max(CELL_SIZES_M), (
        f"the realized dielectric radius moves {spread*1e6:.2f} um across the "
        "cell sizes; the line's own cross-section is still a function of the mesh")


def test_the_wall_no_longer_contributes_to_the_impedance_s_mesh_dependence():
    """``Z_TEM`` from the DECLARED pin and the REALIZED dielectric outer radius.

    That term is the wall's own contribution, and it is what this change moves:
    measured 48.320 / 48.522 / 48.534 / 48.540 ohm at 2, 4, 6 and 9 annulus
    cells against a declared 48.591 — 0.6 % at the coarsest and 0.1 % at the
    rest. Before the wall's inner radius was pinned to ``b`` the same quantity
    ran 40.7 -> 45.3 ohm over that ladder.

    The pin's own rasterized radius is NOT in this figure and is NOT gated here.
    Feeding it in gives up to 16.5 % at 2 annulus cells and 10.5 % at 6, and
    that number is a property of the proxy rather than of the line: the measured
    ``Z0`` of this lane at 6 annulus cells, from the 25 and 100 ohm loads, is
    48.61 and 48.56 ohm, within 0.06 % of the declared value. An outermost cell
    CENTRE understates where the PEC edge realization actually puts the wall.
    Gating it would pin a bad estimator.
    """
    declared = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, float(PTFE_EPS_R))
    worst = 0.0
    for dx in CELL_SIZES_M:
        grid, materials, _shell_inner, cells = _line(dx)
        k = int(grid.shape[2]) // 2
        r = _radius(grid)
        eps = np.asarray(materials.eps_r)[:, :, k]
        fill = (eps > 1.0 + 1e-6) & ~cells[:, :, k]
        z = coaxial_tem_characteristic_impedance(
            SMA_PIN_RADIUS, float(r[fill].max()), float(PTFE_EPS_R))
        worst = max(worst, abs(z - declared) / declared)
    assert worst <= 0.01, (
        f"the realized dielectric outer radius implies an impedance up to "
        f"{worst*100:.2f} % from the declared {declared:.3f} ohm; the wall's "
        "inner radius is supposed to be the declared one at every cell size")


def test_a_wall_thinner_than_one_cell_is_refused_rather_than_silently_open():
    """The mutation for the seal: ask for a wall the mesh cannot rasterize and
    the stamper must refuse. Without this the thickness could quietly round to
    nothing and leave the ring open again."""
    grid = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.040), boundary="cpml",
                      dx=ANNULUS / 4.0)
    grid.add_coaxial_port((CENTRE[0], CENTRE[1], 0.020), face="top", pin_length=5e-3)
    g = grid._build_grid()
    materials, _, _ = grid._build_materials(g)
    with pytest.raises(ValueError, match="rasterizes to no cell"):
        stamp_coaxial_line(
            g, materials, center_xy=CENTRE, z_lo_index=int(g.pad_z_lo) + 4,
            z_hi_index=int(g.shape[2]) - int(g.pad_z_hi) - 2,
            pin_radius=SMA_PIN_RADIUS, outer_radius=SMA_OUTER_RADIUS,
            shell_thickness_m=1.0e-9)


def test_the_conductor_is_not_written_into_sigma():
    """The realization moved out of ``materials.sigma``; a stamper that put it
    back would make both mechanisms act at once and this test would catch it."""
    grid, materials, _shell_inner, cells = _line(ANNULUS / 4.0)
    sigma = np.asarray(materials.sigma)
    assert cells.any(), "the line realized no conductor cell"
    assert float(sigma[cells].max()) == 0.0, (
        "stamp_coaxial_line wrote a conductivity into the conductor cells; the "
        "conductor is realized as PEC edges by the caller and a sigma there "
        "would damp the plus-side edges a second time")
    eps = np.asarray(materials.eps_r)
    k = int(grid.shape[2]) // 2
    r = _radius(grid)
    fill = (eps[:, :, k] > 1.0 + 1e-6) & ~cells[:, :, k]
    assert np.all(r[fill] > SMA_PIN_RADIUS), "dielectric inside the pin radius"
    assert np.all(r[fill] <= SMA_OUTER_RADIUS + math.sqrt(2.0) * float(grid.dx)), (
        "dielectric beyond the declared outer radius by more than a cell diagonal")
