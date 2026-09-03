"""The waveguide port solves its transverse mode on the GUIDE's cells.

``WaveguidePort.u_slice`` / ``.v_slice`` are NODE-index spans: both builders
report the physical aperture as ``(hi - lo - 1) * d``
(``rfx/api/_compile.py::_range_to_slice``,
``rfx/runners/nonuniform.py::_range_to_slice_nu``). Everything inside
``init_waveguide_port`` past the unpack is CELL-centred — ``u_widths`` are cell
widths, ``u_coords`` are cell centres, and
``_galerkin_stiffness_mass_1d`` puts one unknown per cell — so an ``N``-cell
guide must give an ``N``-unknown transverse operator.

Reading the node span as a cell count built the operator on ``N + 1`` cells and
put ``WaveguidePortConfig.f_cutoff`` — the cutoff ``_compute_beta`` uses for the
reference-plane rotation and ``_compute_mode_impedance`` uses for Z_TE — on a
guide one cell wider than the walls make. On WR-90 at ``dx = a/36`` that is
6.378 GHz instead of 6.555 GHz, a 2.7 % error that grows to 10 % at
``dx = a/9``.

Each check below fails on the pre-correction code:

1. the cell count and the width sum,
2. ``f_cutoff`` against the closed form for the ``N``-cell discrete Neumann
   operator, with the ``N + 1`` value asserted to be a DIFFERENT number so the
   comparison cannot pass by coincidence,
3. the stored profile shape against the plane-indexer window, so the aperture
   the solver templates and the aperture the extractor slices stay one object.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.waveguide_port import _plane_indexer

C0 = 299_792_458.0
A_WR90 = 0.02286
B_WR90 = 0.01016
FREQS = jnp.asarray([9.0e9, 10.0e9, 11.0e9])

# a/N with N in {9, 18, 36}: dx divides BOTH walls exactly (b = 4a/9), so the
# rasterized aperture equals the declared one and the invariant below is a
# statement about the port, not about grid snapping.
LADDER = ((9, 0.00254), (18, 0.00127), (36, 0.000635))


def _sim(dx: float, *, y_range=None, z_range=None) -> Simulation:
    sim = Simulation(
        freq_max=11.6e9,
        domain=(0.12192, A_WR90, B_WR90),
        dx=dx,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=10,
    )
    sim.add_waveguide_port(0.0127, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=FREQS, f0=10e9, bandwidth=0.5, name="left",
                           y_range=y_range, z_range=z_range)
    return sim


def _cfg(sim: Simulation):
    grid = sim._build_grid()
    return grid, sim._build_waveguide_port_config(
        sim._waveguide_ports[0], grid, FREQS, int(grid.num_timesteps(4.0)))


def _discrete_neumann_cutoff_hz(n_cells: int, dx: float) -> float:
    """First non-zero eigenvalue of the 1-D cell-centred Neumann operator on
    ``n_cells`` uniform cells: ``kc = (2/dx)·sin(π/(2N))``. This is the TE10
    cutoff of the guide the Yee grid actually propagates in — it tends to
    ``c/2a`` as ``N → ∞`` and sits BELOW it at any finite N."""
    return (2.0 / dx) * math.sin(math.pi / (2 * n_cells)) * C0 / (2.0 * math.pi)


@pytest.mark.parametrize("n_cells,dx", LADDER, ids=[f"a_over_{n}" for n, _ in LADDER])
def test_transverse_operator_is_built_on_the_guides_own_cells(n_cells, dx):
    grid, cfg = _cfg(_sim(dx))
    n_v = int(round(B_WR90 / dx))

    # The node span the builder produced, and the cell span the port kept.
    assert grid.ny == n_cells + 1, (grid.ny, n_cells)
    assert grid.nz == n_v + 1, (grid.nz, n_v)
    assert (cfg.u_hi - cfg.u_lo) == n_cells, (cfg.u_lo, cfg.u_hi, n_cells)
    assert (cfg.v_hi - cfg.v_lo) == n_v, (cfg.v_lo, cfg.v_hi, n_v)

    u_w = np.asarray(cfg.u_widths, dtype=np.float64)
    v_w = np.asarray(cfg.v_widths, dtype=np.float64)
    assert u_w.size == n_cells and v_w.size == n_v
    # The widths span the guide, not the guide plus a cell past the wall.
    assert float(u_w.sum()) == pytest.approx(cfg.a, rel=1e-6)
    assert float(v_w.sum()) == pytest.approx(cfg.b, rel=1e-6)


@pytest.mark.parametrize("n_cells,dx", LADDER, ids=[f"a_over_{n}" for n, _ in LADDER])
def test_port_cutoff_is_the_discrete_cutoff_of_the_n_cell_guide(n_cells, dx):
    _, cfg = _cfg(_sim(dx))
    fc_n = _discrete_neumann_cutoff_hz(n_cells, dx)
    fc_n_plus_1 = _discrete_neumann_cutoff_hz(n_cells + 1, dx)
    # The two candidates must be far apart, otherwise the comparison below
    # would pass whichever operator was built.
    assert abs(fc_n - fc_n_plus_1) / fc_n > 0.02, (fc_n, fc_n_plus_1)
    assert float(cfg.f_cutoff) == pytest.approx(fc_n, rel=1e-9), (
        f"N={n_cells}: port carries {cfg.f_cutoff:.6e} Hz; the N-cell guide is "
        f"{fc_n:.6e} Hz and an (N+1)-cell aperture would be {fc_n_plus_1:.6e} Hz")
    # It approaches c/2a from below, and never exceeds it.
    assert float(cfg.f_cutoff) < C0 / (2.0 * A_WR90)


@pytest.mark.parametrize("n_cells,dx", LADDER, ids=[f"a_over_{n}" for n, _ in LADDER])
def test_profile_shape_equals_the_plane_indexer_window(n_cells, dx):
    grid, cfg = _cfg(_sim(dx))
    idx = _plane_indexer(cfg)
    window = np.zeros(grid.shape)[idx].shape
    for name in ("ey_profile", "ez_profile", "hy_profile", "hz_profile"):
        assert tuple(np.asarray(getattr(cfg, name)).shape) == window, name
    assert tuple(np.asarray(cfg.aperture_dA).shape) == window


def test_explicit_full_width_range_gives_the_same_cell_count():
    """An explicit ``y_range=(0, a)`` resolves to the same node span as the
    default branch, so it must also solve on N cells — it was the obvious
    workaround for the (N+1) cutoff and never was one."""
    n_cells, dx = 18, 0.00127
    _, default = _cfg(_sim(dx))
    _, explicit = _cfg(_sim(dx, y_range=(0.0, A_WR90), z_range=(0.0, B_WR90)))
    assert (explicit.u_hi - explicit.u_lo) == n_cells
    assert float(explicit.f_cutoff) == pytest.approx(float(default.f_cutoff), rel=1e-12)
