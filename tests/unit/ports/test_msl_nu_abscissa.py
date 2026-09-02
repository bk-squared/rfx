"""Contract tests: MSL probe-abscissa + V/I cell-profile are graded-mesh aware.

Promotes the Gate-0 falsifier
(docs/research_notes/experiments/msl_nu_gate0/gate0_abscissa.py).

The fix makes ``msl_probe_x_coords{,_n}`` and the transverse V/I cell-profile
helper read the ``NonUniformGrid`` per-cell arrays (``dx_arr``/``dy_arr``/
``dz``) instead of the scalar BOUNDARY ``grid.dx``, WITHOUT flipping the
``compute_msl_s_matrix`` NU fence (the full NU MSL runner path is a separate
track). On a uniform ``Grid`` every path here is byte-identical to the legacy
scalar formula.

Scope: extractor-math NU-readiness only. ``compute_msl_s_matrix`` on a NU mesh
still raises ``NotImplementedError`` (the fence is intact); this is unit-level
coverage of the probe-placement + transverse-profile helpers, not an
end-to-end NU MSL S-parameter validation.
"""
from __future__ import annotations

import numpy as np

from rfx.nonuniform import make_nonuniform_grid, position_to_index
from rfx.sources.msl_port import (
    MSLPort,
    msl_probe_x_coords,
    msl_probe_x_coords_n,
    _msl_position_to_index,
    _axis_cell_size,
)
from rfx.api._sparams import _msl_cell_profile


_DX = 0.0005
_NX = 80
_NZ = 24
_DZ = 0.0005
_LY = 0.020
_PEC = {"y_lo", "y_hi", "z_lo", "z_hi"}


def _tent(n: int, dx: float, peak: float = 2.0) -> np.ndarray:
    """Symmetric graded interior: dx at both ends, peak*dx in the middle."""
    t = np.linspace(0.0, 1.0, n)
    prof = dx * (1.0 + (peak - 1.0) * (1.0 - np.abs(2.0 * t - 1.0)))
    prof[0] = dx
    prof[-1] = dx
    return prof


def _nu_grid(dx_profile: np.ndarray):
    return make_nonuniform_grid(
        domain_xy=(float(np.sum(dx_profile)), _LY),
        dz_profile=np.full(_NZ, _DZ),
        dx=float(dx_profile[0]),
        cpml_layers=8,
        dx_profile=dx_profile,
        pec_faces=_PEC,
        cpml_axes="x",
    )


def _uniform_grid():
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    s = Simulation(
        freq_max=10e9,
        domain=(_DX * _NX, _LY, _NZ * _DZ),
        dx=_DX,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    return s._build_grid()


def _port(Lx: float) -> MSLPort:
    return MSLPort(
        feed_x=0.10 * Lx, y_lo=0.008, y_hi=0.012,
        z_lo=0.0, z_hi=0.001, direction="+x", impedance=50.0,
    )


# ---------------------------------------------------------------------------
# 1. NU mesh: probe coords now RUN (no AttributeError) and are cumsum-correct
# ---------------------------------------------------------------------------

def test_probe_coords_n_runs_on_nu_and_is_cumsum_correct():
    prof = _tent(_NX, _DX)
    grid = _nu_grid(prof)
    port = _port(float(np.sum(prof)))

    xs = np.asarray(
        msl_probe_x_coords_n(grid, port, 6, n_offset_cells=5, n_spacing_cells=3)
    )
    assert xs.shape == (6,)
    assert np.all(np.isfinite(xs))
    assert np.all(np.diff(xs) > 0), "+x probes must be physically increasing"

    # Independent cumsum cell-edge reference at the same clamped indices.
    i_feed, _, _ = position_to_index(grid, (port.feed_x, port.y_lo, port.z_lo))
    pad, pad_hi = int(grid.pad_x_lo), int(grid.pad_x_hi)
    interior = np.asarray(grid.dx_arr, float)[pad : grid.nx - pad_hi]
    edges = np.insert(np.cumsum(interior), 0, 0.0)

    ref, legacy = [], []
    for n in range(6):
        clamped = max(0, min(i_feed + (5 + n * 3), grid.nx - 1))
        u = clamped - pad
        ref.append(edges[max(0, min(u, len(edges) - 1))])
        legacy.append(u * float(grid.dx))   # the OLD scalar-grid.dx abscissa

    np.testing.assert_allclose(xs, ref, rtol=0, atol=1e-12)
    # The fix is real: the cumsum abscissa departs from the legacy scalar one
    # by mm-scale on this 2x-graded mesh (Gate-0 measured up to 5.9 mm).
    assert np.max(np.abs(xs - np.asarray(legacy))) > 1e-3


def test_probe_coords_n_runs_on_nu_minus_x():
    """-x port: graded-mesh probes must be physically DECREASING and match an
    independent reverse-direction cumsum reference."""
    prof = _tent(_NX, _DX)
    grid = _nu_grid(prof)
    Lx = float(np.sum(prof))
    port = MSLPort(
        feed_x=0.90 * Lx, y_lo=0.008, y_hi=0.012,
        z_lo=0.0, z_hi=0.001, direction="-x", impedance=50.0,
    )
    xs = np.asarray(
        msl_probe_x_coords_n(grid, port, 6, n_offset_cells=5, n_spacing_cells=3)
    )
    assert xs.shape == (6,) and np.all(np.isfinite(xs))
    assert np.all(np.diff(xs) < 0), "-x probes must be physically decreasing"

    i_feed, _, _ = position_to_index(grid, (port.feed_x, port.y_lo, port.z_lo))
    pad, pad_hi = int(grid.pad_x_lo), int(grid.pad_x_hi)
    interior = np.asarray(grid.dx_arr, float)[pad : grid.nx - pad_hi]
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    ref = []
    for n in range(6):
        clamped = max(0, min(i_feed - (5 + n * 3), grid.nx - 1))
        u = clamped - pad
        ref.append(edges[max(0, min(u, len(edges) - 1))])
    np.testing.assert_allclose(xs, ref, rtol=0, atol=1e-12)


def test_cell_profile_nu_shape_mismatch_raises():
    """The NU branch is authoritative: a wrong-length request must RAISE,
    never silently fall back to a scalar boundary-dx fill."""
    import pytest

    prof = _tent(_NX, _DX)
    grid = _nu_grid(prof)
    with pytest.raises(ValueError, match="per-cell profile shape"):
        _msl_cell_profile(grid, "y", grid.ny + 5)


def test_probe_coords_3_runs_on_nu():
    prof = _tent(_NX, _DX)
    grid = _nu_grid(prof)
    port = _port(float(np.sum(prof)))
    xs = msl_probe_x_coords(grid, port, n_offset_cells=5, n_spacing_cells=3)
    assert len(xs) == 3 and all(np.isfinite(xs))
    assert xs[0] < xs[1] < xs[2]


# ---------------------------------------------------------------------------
# 2. Uniform Grid: byte-identical to the legacy scalar formula
# ---------------------------------------------------------------------------

def test_probe_coords_uniform_byte_identical():
    grid = _uniform_grid()
    port = _port(_DX * _NX)
    i_feed, _, _ = grid.position_to_index((port.feed_x, port.y_lo, port.z_lo))
    pad = int(getattr(grid, "pad_x_lo", 0))
    nx = grid.nx

    def legacy(target: int) -> float:
        c = max(0, min(target, nx - 1))
        return float((c - pad) * grid.dx)

    xs = msl_probe_x_coords_n(grid, port, 6, n_offset_cells=5, n_spacing_cells=3)
    assert list(xs) == [legacy(i_feed + (5 + n * 3)) for n in range(6)]

    x3 = msl_probe_x_coords(grid, port, n_offset_cells=5, n_spacing_cells=3)
    assert x3 == (legacy(i_feed + 5), legacy(i_feed + 8), legacy(i_feed + 11))


# ---------------------------------------------------------------------------
# 3. _msl_cell_profile: NU per-cell arrays vs uniform scalar fill
# ---------------------------------------------------------------------------

def test_cell_profile_nu_reads_per_cell_arrays():
    prof = _tent(_NX, _DX)
    grid = _nu_grid(prof)
    np.testing.assert_array_equal(
        _msl_cell_profile(grid, "y", grid.ny), np.asarray(grid.dy_arr, float))
    np.testing.assert_array_equal(
        _msl_cell_profile(grid, "z", grid.nz), np.asarray(grid.dz, float))
    dx = _msl_cell_profile(grid, "x", grid.nx)
    np.testing.assert_array_equal(dx, np.asarray(grid.dx_arr, float))
    assert np.ptp(dx) > 1e-9, "graded x profile must not be constant grid.dx"


def test_cell_profile_uniform_byte_identical():
    grid = _uniform_grid()
    for axis, n in (("x", grid.nx), ("y", grid.ny), ("z", grid.nz)):
        np.testing.assert_array_equal(
            _msl_cell_profile(grid, axis, n), np.full(n, float(grid.dx)))


# ---------------------------------------------------------------------------
# 3b. _axis_cell_size (SOURCE-side helper): NU per-cell vs uniform scalar
# ---------------------------------------------------------------------------

def test_axis_cell_size_nu_reads_per_cell():
    """setup_msl_port / make_msl_port_sources weight sigma + the Ez feed by
    _axis_cell_size; on a graded mesh it must return the per-cell size, not the
    scalar boundary grid.dx."""
    prof = _tent(_NX, _DX)
    grid = _nu_grid(prof)
    # x axis is graded — sizes across interior indices must vary
    sizes = [_axis_cell_size(grid, "x", i) for i in range(grid.pad_x_lo, grid.nx - grid.pad_x_hi)]
    assert np.ptp(sizes) > 1e-9, "graded x: _axis_cell_size must not be constant"
    # each equals the per-cell dx_arr value
    dx_arr = np.asarray(grid.dx_arr, float)
    for i in range(grid.pad_x_lo, grid.nx - grid.pad_x_hi):
        assert abs(_axis_cell_size(grid, "x", i) - dx_arr[i]) < 1e-12
    # y / z read their own per-cell arrays (not the x boundary cell)
    assert abs(_axis_cell_size(grid, "y", grid.ny // 2)
               - float(np.asarray(grid.dy_arr, float)[grid.ny // 2])) < 1e-12
    assert abs(_axis_cell_size(grid, "z", grid.nz // 2)
               - float(np.asarray(grid.dz, float)[grid.nz // 2])) < 1e-12


def test_axis_cell_size_uniform_byte_identical():
    """On a uniform Grid (no NonUniformGrid branch) _axis_cell_size returns the
    legacy scalar grid.dx for every axis."""
    grid = _uniform_grid()
    for axis in ("x", "y", "z"):
        assert _axis_cell_size(grid, axis, 5) == float(grid.dx)


# ---------------------------------------------------------------------------
# 4. _msl_position_to_index duck-types over both grid types
# ---------------------------------------------------------------------------

def test_position_to_index_ducktypes_both_grids():
    prof = _tent(_NX, _DX)
    nu = _nu_grid(prof)
    port = _port(float(np.sum(prof)))
    pos = (port.feed_x, port.y_lo, port.z_lo)
    assert _msl_position_to_index(nu, pos) == position_to_index(nu, pos)

    uni = _uniform_grid()
    assert _msl_position_to_index(uni, pos) == uni.position_to_index(pos)
