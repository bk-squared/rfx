"""A waveguide port injects at the cell it sits in, not at the boundary cell.

A waveguide port drives the guide through a one-sided TFSF boundary: an H
correction on the upstream half-cell and an E correction on the source plane.
Each divides by a cell size. Both used to take the grid's BOUNDARY cell, so a
port placed inside a refined band injected at the boundary cell's scale
instead of its own. On the WR-90 fixture of
``tests/unit/sparams/test_waveguide_nu_sparam.py`` -- 1.5 mm at the walls,
0.75 mm through the middle 40 mm -- a port in the band ran both corrections
at 0.5000 of the local metric. That is gap G13, and the program document's
repro R2 is the same number.

Every NU port test in the suite puts its ports in boundary-sized cells, where
the local cell IS the boundary cell, which is why none of them saw it.

What each half is entitled to comes from the interface's metric table
(``rfx/_grid_metric.py``), read at the plane the correction acts on:
the H correction differences two E nodes across one cell and takes the PRIMAL
width there; the E correction sits on a node and takes that node's DUAL
spacing. On a uniform axis the two collapse to ``grid.dx`` and nothing moves.

Which test here is the gate
---------------------------
``test_launched_amplitude_does_not_depend_on_the_local_cell`` is. It runs the
real scan twice and compares what the port actually launched, so it covers
the WIRING -- which number reaches ``apply_waveguide_port_h/e`` -- and not
just the helper's arithmetic.

The helper tests below are a secondary check and cannot replace it. They call
``_waveguide_port_axis_metrics`` and compare it against the same grid
accessors the helper itself calls, which is two values out of one source.
Measured: reviving the defect at the CALL SITES, spelled
``grid.boundary_cell("x", "lo")`` so the helper is still called and its result
still computed, leaves every helper test in this file green. Only the
end-to-end gate turns red.

Why the reference-plane voltage and not the probe one: the two blocks put the
probe plane at different transit times from the source, so ``v_probe_t``
mixes the launched amplitude with when the wave arrives. Under the same
mutation it reads 0.9876, which looks correct. ``v_ref_t`` reads 0.5322
against 1.0643 fixed -- a clean factor 2.0000.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.auto_config import smooth_grading
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.nonuniform import (
    _waveguide_port_axis_metrics,
    make_nonuniform_grid,
)

_A_WG, _B_WG = 0.02286, 0.01016
_DX_COARSE, _DX_FINE = 1.5e-3, 0.75e-3
_PEC_WALLS = {"y_lo", "y_hi", "z_lo", "z_hi"}


def _wr90_profile():
    """The sparam fixture's coarse | fine | coarse x profile."""
    raw = np.concatenate([
        np.full(int(round(0.030 / _DX_COARSE)), _DX_COARSE),
        np.full(int(round(0.040 / _DX_FINE)), _DX_FINE),
        np.full(int(round(0.030 / _DX_COARSE)), _DX_COARSE),
    ])
    return smooth_grading(raw, max_ratio=1.3)


def _graded_grid():
    return make_nonuniform_grid(
        (_A_WG, _B_WG), np.full(10, 1.016e-3), _DX_COARSE,
        cpml_layers=8, dx_profile=_wr90_profile(), pec_faces=_PEC_WALLS)


def _uniform_x_grid():
    return make_nonuniform_grid(
        (_A_WG, _B_WG), np.full(10, 1.016e-3), _DX_COARSE,
        cpml_layers=8, pec_faces=_PEC_WALLS)


def _port(direction, index):
    return SimpleNamespace(direction=direction, x_index=index)



# ---------------------------------------------------------------------------
# The gate: what the port actually launched, through the real scan
# ---------------------------------------------------------------------------

_GATE_FREQS = jnp.linspace(8.2e9, 11.5e9, 3)


def _y_graded_profile():
    """coarse | fine | coarse along the PROPAGATION axis.

    The port travels +y here, so the grading is on y and the port can be put
    in either block of the axis it propagates along. That is the arrangement
    the defect needs: a port whose own cell differs from the boundary cell.
    """
    raw = np.concatenate([
        np.full(20, _DX_COARSE),
        np.full(54, _DX_FINE),
        np.full(20, _DX_COARSE),
    ])
    return smooth_grading(raw, max_ratio=1.3)


_GATE_PROFILE = _y_graded_profile()
_GATE_LY = float(np.sum(_GATE_PROFILE))


def _peak_modal_voltage_at_reference(port_y: float) -> float:
    """Run the real NU scan and return peak |v_ref_t| for one port position."""
    sim = Simulation(
        freq_max=12e9, domain=(_A_WG, _GATE_LY, _B_WG), dx=_DX_COARSE,
        boundary=BoundarySpec(x=Boundary(lo="pec", hi="pec"),
                              y=Boundary(lo="cpml", hi="cpml"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=8, dy_profile=_GATE_PROFILE)
    sim.add_waveguide_port(
        port_y, direction="+y", mode=(1, 0), mode_type="TE",
        x_range=(0.0, _A_WG), z_range=(0.0, _B_WG),
        freqs=_GATE_FREQS, f0=10.0e9, bandwidth=0.5, name="p1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=260, compute_s_params=False)
    cfg = next(iter(result.waveguide_ports.values()))
    return float(np.max(np.abs(np.asarray(cfg.v_ref_t))))


def test_launched_amplitude_does_not_depend_on_the_local_cell():
    """A mode launched through a one-sided TFSF pair carries the amplitude the
    source asked for, whatever the cell under the port happens to be.

    Two runs of the same guide, same port, same reference-plane offset; only
    the block the port sits in differs. The launched amplitude is read at the
    reference plane, where both runs measure the same distance from their own
    source, so the comparison is of what was launched and not of when it
    arrived.

    Measured on this host: 1.0643 with each half reading its own plane's
    metric, 0.5322 with both reading the boundary cell -- a factor of exactly
    2.0000. The 6.4 % the correct case sits above unity is not the defect: it
    is the physical difference between launching a mode into 0.75 mm cells
    and into 1.5 mm ones, numerical dispersion and the discretized mode
    profile. The band admits that and rejects a factor of two with room.

    This is the test that covers the wiring. Every helper test below stays
    green when the defect is revived at the call sites.
    """
    coarse = _peak_modal_voltage_at_reference(0.015)
    fine = _peak_modal_voltage_at_reference(0.045)
    ratio = fine / coarse

    assert ratio == pytest.approx(1.0, abs=0.15), (
        f"the port in the {_DX_FINE * 1e3:.2f} mm block launched {ratio:.4f} "
        f"of what the same port launched in the {_DX_COARSE * 1e3:.2f} mm "
        f"block (coarse {coarse:.6e}, fine {fine:.6e}). A one-sided TFSF pair "
        "must not change the launched amplitude with the local cell size. "
        "About 0.5 means both corrections divided by the boundary cell "
        "instead of their own plane's, which is G13; about 2.0 means the "
        "reverse. Measured 1.0643 correct, 0.5322 with the defect revived."
    )

@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_a_port_in_a_boundary_sized_cell_still_gets_the_boundary_cell(direction):
    """The no-op pin. Every NU port test in the suite lives here.

    On a uniform x axis both metrics ARE ``grid.dx``, bit for bit, so this
    change cannot move a number on any existing port fixture. Equality, not
    a tolerance: the coefficient is ``dt/(MU_0*dx)``, and a last-bit change
    in ``dx`` is a last-bit change in every injected sample.
    """
    grid = _uniform_x_grid()
    dx_h, dx_e = _waveguide_port_axis_metrics(
        grid, _port(direction, grid.index_of("x", 0.015)))
    assert dx_h == grid.dx
    assert dx_e == grid.dx


@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_a_port_inside_the_fine_band_gets_the_fine_cell(direction):
    """G13, as a number: 0.5000 before, 1.0000 after.

    The ratio is the injected coefficient's, not the cell's. The coefficient
    is inversely proportional to the metric, so a port that took 1.5 mm where
    it should have taken 0.75 mm injected at 0.75/1.5 = 0.5000 of what the
    local cell asks for.
    """
    grid = _graded_grid()
    inside = grid.index_of("x", 0.045)
    dx_h, dx_e = _waveguide_port_axis_metrics(grid, _port(direction, inside))

    assert dx_h == pytest.approx(_DX_FINE, rel=1e-9)
    assert dx_e == pytest.approx(_DX_FINE, rel=1e-9)

    # what the old scalar read would have used, and what that did to the
    # coefficient the port actually injects with
    assert grid.dx == pytest.approx(_DX_COARSE, rel=1e-9)
    assert dx_h / grid.dx == pytest.approx(0.5, rel=1e-9)
    assert dx_e / grid.dx == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_a_port_in_the_coarse_block_is_unchanged(direction):
    """The same guide, the port outside the band: nothing to fix there."""
    grid = _graded_grid()
    outside = grid.index_of("x", 0.015)
    dx_h, dx_e = _waveguide_port_axis_metrics(grid, _port(direction, outside))
    assert dx_h == pytest.approx(_DX_COARSE, rel=1e-9)
    assert dx_e == pytest.approx(_DX_COARSE, rel=1e-9)


def test_each_half_reads_the_plane_it_acts_on():
    """The H and E corrections are half a cell apart and can differ.

    ``apply_waveguide_port_h`` lands on ``x_index - 1`` for a ``+`` port and
    ``x_index`` for a ``-`` one; ``apply_waveguide_port_e`` lands on
    ``x_index`` and ``x_index + 1``. Reading both at the port's nominal index
    would be right only where the neighbourhood is uniform, so this pins the
    offsets on a node where the two neighbours differ.
    """
    grid = _graded_grid()
    cells, duals = grid.cells("x"), grid.duals("x")

    # a node where the cell on one side differs from the cell on the other
    edge = next(i for i in range(1, grid.nx - 1) if cells[i] != cells[i - 1])

    fwd_h, fwd_e = _waveguide_port_axis_metrics(grid, _port("+x", edge))
    assert fwd_h == float(cells[edge - 1])
    assert fwd_e == float(duals[edge])

    rev_h, rev_e = _waveguide_port_axis_metrics(grid, _port("-x", edge))
    assert rev_h == float(cells[edge])
    assert rev_e == float(duals[edge + 1])

    assert fwd_h != rev_h, (
        "the fixture's chosen node has equal neighbours, so this test cannot "
        "see which plane each half reads"
    )


def test_the_helper_and_the_boundary_scalar_are_distinguishable():
    """The helper's own output differs from the boundary cell. NOT the gate.

    This says only that the two numbers are not equal on this fixture. It
    cannot see which one reaches the solver, so it stays green when the
    defect is revived at the call sites. The end-to-end gate above is what
    catches that; this is here to show the separation is real and large
    rather than a last-bit difference.
    """
    grid = _graded_grid()
    inside = grid.index_of("x", 0.045)
    good = _waveguide_port_axis_metrics(grid, _port("+x", inside))
    revived = (float(grid.dx), float(grid.dx))
    assert good != revived
    assert abs(good[0] - revived[0]) == pytest.approx(_DX_COARSE - _DX_FINE,
                                                      rel=1e-9)


def test_a_port_reads_its_own_propagation_axis():
    """``direction`` picks the axis; ``x_index`` is the plane along it.

    ``tests/unit/nonuniform/test_nonuniform_api.py`` drives a ``+y`` port on
    a non-uniform grid, so this is not hypothetical. Before the fix every
    port took the x boundary scalar whatever it propagated along; on a
    dz-only graded mesh that was the right number by coincidence, because dx
    and dy are equal there. This pins that the y port now reads y.
    """
    grid = make_nonuniform_grid(
        (_A_WG, _B_WG), np.full(10, 1.016e-3), _DX_COARSE,
        cpml_layers=8, dx_profile=_wr90_profile(), pec_faces=_PEC_WALLS)

    extent = {"x": grid.nx, "y": grid.ny, "z": grid.nz}
    for direction, axis in (("+x", "x"), ("+y", "y"), ("-z", "z")):
        n = extent[axis]
        idx = min(12, n - 2)          # the narrow axes are only a few cells
        dx_h, dx_e = _waveguide_port_axis_metrics(grid, _port(direction, idx))
        cells, duals = grid.cells(axis), grid.duals(axis)
        forward = direction.startswith("+")
        h_plane = idx - 1 if forward else idx
        e_plane = idx if forward else idx + 1
        assert dx_h == float(cells[max(0, min(h_plane, n - 1))])
        assert dx_e == float(duals[max(0, min(e_plane, n - 1))])

    # x is graded here and y is not, so reading the wrong axis is visible:
    # inside the band x gives 0.75 mm where y gives the 1.5 mm boundary cell
    x_in_band = _waveguide_port_axis_metrics(grid, _port("+x", 47))
    y_same_index = _waveguide_port_axis_metrics(grid, _port("+y", 47))
    assert x_in_band[0] == pytest.approx(_DX_FINE, rel=1e-9)
    assert y_same_index[0] == pytest.approx(_DX_COARSE, rel=1e-9)


def test_a_traced_axis_keeps_its_tracer():
    """#1190: the mesh-as-design-variable path must survive the host reads.

    A concrete axis is resolved to Python floats once, outside the scan. A
    traced one stays in-trace, the way the CPML z pair already does.
    """
    seen = {}

    def f(profile):
        grid = make_nonuniform_grid(
            (_A_WG, _B_WG), np.full(10, 1.016e-3), _DX_COARSE,
            cpml_layers=8, dx_profile=profile, pec_faces=_PEC_WALLS)
        dx_h, dx_e = _waveguide_port_axis_metrics(grid, _port("+x", 20))
        seen["floats"] = isinstance(dx_h, float) and isinstance(dx_e, float)
        return dx_h + dx_e

    traced = jnp.asarray(_wr90_profile(), dtype=jnp.float32)
    value, grad = jax.value_and_grad(f)(traced)
    assert seen["floats"] is False
    assert np.isfinite(float(value))
    assert float(np.abs(np.asarray(grad)).sum()) > 0.0
