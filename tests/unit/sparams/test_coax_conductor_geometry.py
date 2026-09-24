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
import pathlib

import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                       / "scripts" / "diagnostics"))

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

# The staircase error oscillates on coarse meshes and only settles into its
# trend once the mesh resolves the conductor boundaries, so the order is
# fitted from this rung up. Measured exponent 0.986, residual 0.006.
FIT_FROM_CELLS = 9.0
ORDER_TOL = 0.15
FIT_RESIDUAL_MAX = 0.05


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
    Feeding it in gives up to 16.5 % at 2 annulus cells and 10.5 % at 6. An
    earlier version of this docstring dismissed that as a property of the proxy
    because the lane's reported ``Z0`` is within 0.06 % of declared; **that
    argument is withdrawn**. That number is built from the declared load and a
    conductivity carrying the same geometric factor as the line, so it cancels
    and cannot certify the geometry. The electrostatic witness in
    ``test_the_realized_cross_section_converges_toward_the_declared_impedance``
    puts the realized cross-section 20.4 % from the smooth value at 4 annulus
    cells, the same order as the proxy's figure. The proxy may still be crude;
    nothing here shows it.
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


def test_a_wall_on_a_lateral_absorbers_face_is_refused():
    """The coax lanes absorb on x and y (issue 1218), and a conductor on or in a
    CPML pad diverges, so the wall has to keep a whole cell of vacuum from every
    lateral absorber — measured on the REALIZED wall. The clamp keeps a cell in
    continuous arithmetic, but a conductor cell i shorts edges on nodes i AND
    i + 1: when the clamp binds on a high face the realized wall lands on that
    absorber's inner face. Here the line's axis is 3 mm from the x-hi pad face
    on 0.25 mm cells; the clamp thins the 1 mm wall to 0.695 mm, which leaves
    one cell in continuous terms and none once rasterized."""
    from rfx.sources.coaxial_port import SHELL_THICKNESS_M

    dx = 0.25e-3
    sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.012), boundary="cpml",
                     cpml_layers=8, dx=dx)
    g = sim._build_grid()
    materials, _, _ = sim._build_materials(g)
    cx = 0.005
    span_x = (int(g.shape[0]) - 1 - int(g.pad_x_lo) - int(g.pad_x_hi)) * dx
    room = (span_x - cx) - SMA_OUTER_RADIUS - dx          # the clamp's own arithmetic
    assert dx <= room < SHELL_THICKNESS_M, (
        "this case has to be one the continuous clamp alone accepts (a wall of at "
        f"least one cell fits: {room / dx:.2f} cells), or it tests the old refusal")
    with pytest.raises(ValueError, match=r"0 cell\(s\) from the x-hi absorber"):
        stamp_coaxial_line(
            g, materials, center_xy=(cx, 0.004), z_lo_index=int(g.pad_z_lo) + 4,
            z_hi_index=int(g.shape[2]) - int(g.pad_z_hi) - 2,
            pin_radius=SMA_PIN_RADIUS, outer_radius=SMA_OUTER_RADIUS)


@pytest.mark.parametrize("dx", CELL_SIZES_M[1:])
def test_the_battery_line_keeps_a_whole_cell_from_every_lateral_absorber(dx):
    """The coax battery's own line (8 x 8 mm board, axis centred) at 4, 6 and 9
    annulus cells, read from the realized edges rather than the stamper's cell
    arithmetic: every shorted edge keeps at least one whole cell to each lateral
    pad face."""
    grid, _materials, _shell_inner, cells = _line(dx)
    edges = realized_pec_edge_masks(cells, sheets=(), wires=())
    for axis, name in ((0, "x"), (1, "y")):
        n = int(grid.shape[axis])
        touched = np.zeros(n, dtype=bool)
        for comp, m in enumerate(edges):
            hit = np.asarray(m, dtype=bool).any(axis=tuple(t for t in range(3) if t != axis))
            touched |= hit
            if comp == axis:                  # an edge along the axis also reaches node i + 1
                touched[1:] |= hit[:-1]
        nodes = np.nonzero(touched)[0]
        lo_face = int(getattr(grid, f"pad_{name}_lo"))
        hi_face = n - 1 - int(getattr(grid, f"pad_{name}_hi"))
        assert nodes.min() - lo_face >= 1, (name, "lo", int(nodes.min()), lo_face)
        assert hi_face - nodes.max() >= 1, (name, "hi", int(nodes.max()), hi_face)


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


@pytest.mark.parametrize("dx", CELL_SIZES_M)
def test_the_tem_source_injects_over_the_annulus_the_stamper_actually_built(dx):
    """The source's outer injection radius IS the stamper's dielectric edge.

    ``build_coaxial_tem_plane_source_specs`` selects its cells with
    ``pin_radius <= r <= shell_inner_radius`` and normalises the 1/r mode with
    ``log(outer_radius / pin_radius)``. Those two have to describe the same
    annulus. They did not: the builder carried its own copy of the wall
    formula the stamper used to use (``b - min(dz, (b-a)/2)``), so once the
    wall moved out to ``b`` the builder kept injecting over ``[a, b-dx]`` while
    normalising over ``ln(b/a)`` -- one ring of dielectric excited by nothing,
    against a mode normalised as though it were.

    No solve: this compares the radius the builder resolves against the radius
    the stamper returned, at four cell sizes, plus the cells each admits.
    """
    from rfx.sources.coaxial_port import build_coaxial_tem_plane_source_specs

    grid, materials, shell_inner, cells = _line(dx)
    sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.040),
                     boundary="cpml", dx=dx)
    sim.add_coaxial_port((CENTRE[0], CENTRE[1], 0.020), face="top",
                         pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    port = sim._coaxial_ports[0]

    assert shell_inner == pytest.approx(float(port.outer_radius), abs=1e-15), (
        f"the stamper's dielectric edge {shell_inner*1e6:.3f} um is not the "
        f"declared outer radius {float(port.outer_radius)*1e6:.3f} um")

    spec = build_coaxial_tem_plane_source_specs(
        grid=grid, port=port, n_steps=8)
    n_default = int(spec.source_cell_count)
    spec_explicit = build_coaxial_tem_plane_source_specs(
        grid=grid, port=port, n_steps=8, shell_inner_radius=shell_inner)
    assert int(spec_explicit.source_cell_count) == n_default, (
        "passing the stamper's own shell_inner_radius changed the source cell "
        "count, so the builder's default disagrees with the stamper")

    # The stale formula is one cell tighter; assert it would have differed, so
    # this test cannot pass by both sides being wrong in the same way.
    stale = float(port.outer_radius) - min(
        float(grid.dx),
        0.5 * (float(port.outer_radius) - float(port.pin_radius)))
    spec_stale = build_coaxial_tem_plane_source_specs(
        grid=grid, port=port, n_steps=8, shell_inner_radius=stale)
    assert int(spec_stale.source_cell_count) < n_default, (
        f"the pre-fix radius {stale*1e6:.3f} um admits "
        f"{int(spec_stale.source_cell_count)} cells and the stamper's "
        f"{shell_inner*1e6:.3f} um admits {n_default}; if these are equal this "
        "test is not discriminating and the comparison above proves nothing")

    # Every cell the source injects on must be dielectric, not conductor.
    k = int(grid.shape[2]) // 2
    r = _radius(grid)
    admitted = (r >= float(port.pin_radius)) & (r <= shell_inner)
    assert not (admitted & cells[:, :, k]).any(), (
        "the source injects on a cell the stamper realized as conductor")


class _Captured(Exception):
    """Raised by the spy once it has the runner's arguments."""


def _capture_runner_args(monkeypatch, call):
    """Run *call* until it reaches ``rfx.simulation.run``, then stop.

    The lanes do ``from rfx.simulation import run as _run`` INSIDE the
    function, so the lookup happens at call time and patching the module
    attribute reaches it. The spy raises before stepping, which is what keeps
    this in the fast lane: no FDTD, and the assertions still see exactly the
    arrays the runner would have been given.
    """
    import rfx.simulation

    seen = {}

    def _spy(grid, materials, n_steps, **kw):
        seen["grid"] = grid
        seen["materials"] = materials
        seen["pec_edge_masks"] = kw.get("pec_edge_masks")
        raise _Captured

    monkeypatch.setattr(rfx.simulation, "run", _spy)
    with pytest.raises(_Captured):
        call()
    assert seen, "the lane never reached rfx.simulation.run"
    return seen


def _one_port_call(dx):
    sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.020),
                     boundary="cpml", dx=dx)
    sim.add_coaxial_port((CENTRE[0], CENTRE[1], 0.010), face="top",
                         pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    return lambda: sim.compute_coaxial_line_reflection(
        termination="matched", dut_impedance=50.0, n_steps=8,
        freqs=np.array([8e9]), probe_count=3)


def _two_port_call(dx):
    sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.012),
                     boundary="cpml", dx=dx)
    sim.add_coaxial_port((CENTRE[0], CENTRE[1], 0.006), face="top",
                         pin_length=5e-3,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    return lambda: sim.compute_coaxial_two_port(
        n_steps=8, freqs=np.array([8e9]), probe_count=3,
        probe_start_cells=4, probe_spacing_cells=2)


@pytest.mark.parametrize("lane,builder", [("one_port", _one_port_call),
                                          ("two_port", _two_port_call)])
def test_the_lane_hands_the_runner_pec_edges_and_no_pec_sigma(monkeypatch, lane,
                                                              builder):
    """The conductors reach ``run`` as PEC EDGE MASKS, with no PEC sigma left.

    This is the structural half of the oracle and it needs no solve, so it
    belongs in the fast lane where a regression shows up on every PR rather
    than weekly. It pins the two halves of the change that mutation (b) flips:
    revive the defect and ``pec_edge_masks`` goes back to ``None`` while
    ``materials.sigma`` carries ``PEC_SIGMA`` again, and both assertions below
    go red without any FDTD being run.

    The third assertion is the one that is not a restatement of the helper:
    it maps every shorted edge to its radius and requires that none of them
    lies strictly inside the dielectric annulus. A mask built from the wrong
    radii, or one that swallowed the PTFE, fails it even though
    ``realized_pec_edge_masks`` was called correctly.
    """
    from rfx.sources.coaxial_port import PEC_SIGMA

    dx = ANNULUS / 4.0
    seen = _capture_runner_args(monkeypatch, builder(dx))

    masks = seen["pec_edge_masks"]
    assert masks is not None, (
        f"the {lane} lane passed pec_edge_masks=None; the conductors are then "
        "realized by nothing at all")
    masks = tuple(np.asarray(m) for m in masks)
    assert len(masks) == 3, f"expected ex/ey/ez masks, got {len(masks)}"
    assert any(m.any() for m in masks), (
        "the edge masks are empty, so no conductor edge is shorted")

    sigma = np.asarray(seen["materials"].sigma)
    assert float(sigma.max()) < 0.5 * float(PEC_SIGMA), (
        f"the {lane} lane still stamps sigma >= PEC_SIGMA/2 (max "
        f"{float(sigma.max()):.3e}); a conductor realized BOTH ways damps its "
        "plus-side edges twice")

    # No shorted edge may sit strictly inside the PTFE annulus.
    grid = seen["grid"]
    r = _radius(grid)
    k = int(grid.shape[2]) // 2
    inside = (r > SMA_PIN_RADIUS + math.sqrt(2.0) * float(grid.dx)) & (
        r < SMA_OUTER_RADIUS - math.sqrt(2.0) * float(grid.dx))
    for name, m in zip(("ex", "ey", "ez"), masks):
        plane = m[:, :, k] if m.ndim == 3 else m
        bad = int((plane[:inside.shape[0], :inside.shape[1]] & inside).sum())
        assert bad == 0, (
            f"{bad} {name} edges are shorted strictly inside the dielectric "
            "annulus, more than a cell diagonal from either conductor")


def test_the_realized_cross_section_converges_toward_the_declared_impedance():
    """The impedance of the cross-section the lattice BUILDS, no solve.

    A coaxial line's characteristic impedance is set by the shape of its two
    conductors: ``Z0 = eta0 / (sqrt(eps_r) * G)`` with ``G = C/eps``, which for
    smooth circles is ``2 pi / ln(b/a)``. A staircased pair at four cells across
    its annulus genuinely does not have that impedance, and should not be
    expected to -- the physics sentence this change rests on says the staircase
    moves ``Z_TEM`` and leaves ``beta`` alone.

    **The error does not fall monotonically, and asserting that it does would be
    wrong.** Which cells the rasterizer claims changes in jumps as the mesh
    crosses the conductor boundaries, so the staircase error OSCILLATES about
    its trend. Measured |G - Gc|/Gc: 11.51 / 20.40 / 5.69 / 7.56 / 3.85 / 2.57 /
    1.74 % at 3.789 / 4 / 6 / 9 / 18 / 27 / 40 annulus cells -- 4 cells is worse
    than 3.789, and 9 is worse than 6.

    What IS true, and what this asserts, is the trend once the mesh resolves the
    boundary: fitted over the rungs from 9 cells up the error goes as
    ``N**-0.986``, first order, with a largest log-space residual of 0.006. The
    coarse rungs are recorded and not asserted, because an oscillation is not a
    failure and a bar that forbade it would be measuring the rasterizer's phase
    rather than the discretisation.

    It exists because the one-port number the oracle reports cannot see any of
    this. That figure is ``R_dut (1 - Gamma)/(1 + Gamma)`` with ``R_dut``
    declared, and the annular resistor's conductivity is built from
    ``ln(shell_inner / a)``; the load's realized resistance and the line's
    impedance carry the same discrete geometric factor and it cancels. The blind
    review showed it by re-rasterizing the coax half a cell off-node: the
    realized line moved 3.7 % and the reported number moved 0.002 %. This
    estimator shares nothing with that path.
    """
    static = pytest.importorskip(
        "coax_realized_impedance_static",
        reason="the static witness lives in scripts/diagnostics")
    all_rungs = (3.789288121451007, 4.0, 6.0, 9.0, 18.0, 27.0, 40.0)
    rec = static.measure(all_rungs)
    g_cont = rec["geometric_factor_continuum"]
    rows = [(r["rung_annulus_cells"],
             abs(r["geometric_factor"] - g_cont) / g_cont,
             r["z0_realized_ohm"]) for r in rec["rungs"]]
    print("[coax static] " + "  ".join(
        f"{n:.4g} cells: G err {e*100:.2f} %, Z0 {z:.2f} ohm" for n, e, z in rows))

    fine = [(n, e) for n, e, _ in rows if n >= FIT_FROM_CELLS]
    assert len(fine) >= 3, (
        f"the fit needs at least three rungs at or above {FIT_FROM_CELLS} "
        f"cells; got {[n for n, _ in fine]}")
    x = np.log(np.array([n for n, _ in fine], dtype=float))
    y = np.log(np.array([e for _, e in fine], dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    order = -float(slope)
    resid = float(np.max(np.abs(y - (slope * x + intercept))))

    assert abs(order - 1.0) <= ORDER_TOL, (
        f"the realized geometric factor converges as N**-{order:.3f} over the "
        f"rungs from {FIT_FROM_CELLS:.0f} cells up, not the first order a "
        f"staircased boundary gives ({1.0 - ORDER_TOL:.2f}-{1.0 + ORDER_TOL:.2f} "
        f"allowed). Errors: "
        + ", ".join(f"{n:.4g} cells {e*100:.2f} %" for n, e, _ in rows))
    assert resid <= FIT_RESIDUAL_MAX, (
        f"the fitted trend does not describe the fine rungs: largest log-space "
        f"residual {resid:.4f} over {FIT_RESIDUAL_MAX}. That is not a clean "
        f"power law, so the order above means little. Errors: "
        + ", ".join(f"{n:.4g} cells {e*100:.2f} %" for n, e, _ in rows))
    print(f"[coax static] order {order:.3f} over rungs >= {FIT_FROM_CELLS:.0f} "
          f"(residual {resid:.4f})")
