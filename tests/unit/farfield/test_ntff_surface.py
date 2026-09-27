"""Geometry, phase/scale, reference and persistence contracts of surface export."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import NTFFSurface, Simulation, export_ntff_surface
from rfx.core.yee import FDTDState
from rfx.farfield import NTFFBox, NTFFData, accumulate_ntff, init_ntff_data
from rfx.grid import Grid
from rfx.nonuniform import make_nonuniform_grid


def _grid(graded=False):
    if graded:
        return make_nonuniform_grid((0.012, 0.012), np.array([1, 2, 3, 2, 1, 1]) * 1e-3,
                                    1e-3, cpml_layers=2, pec_faces={"x_lo", "z_hi"},
                                    dx_profile=np.array([1, 2, 1, 3, 2, 1]) * 1e-3,
                                    dy_profile=np.array([1, 3, 2, 1, 2, 1]) * 1e-3)
    return Grid(3e9, (0.009, 0.011, 0.008), dx=1e-3, cpml_layers=3,
                pec_faces={"x_lo", "z_hi"},
                face_layers={"x_lo": 0, "x_hi": 3, "y_lo": 2, "y_hi": 1,
                             "z_lo": 1, "z_hi": 0})


def _record(graded=False):
    grid = _grid(graded)
    box = NTFFBox.from_grid(grid, i_lo=2, i_hi=5, j_lo=3, j_hi=6,
                           k_lo=2, k_hi=5, freqs=np.array([2.4e9, 3.6e9]))
    raw = init_ntff_data(box)
    arrays = []
    for i, array in enumerate(raw[:6]):
        seq = np.arange(array.size).reshape(array.shape) + 1 + 10 * i
        arrays.append((seq + 1j * (seq**2 + 3)) * 1e-10)
    data = NTFFData(*arrays, *[np.full_like(a, 0.7 + 0.9j) for a in arrays])
    surface = export_ntff_surface(data, box, grid, dt=0.7 * grid.dt, n_steps=137, step_start=5)
    return grid, box, data, surface


@pytest.mark.parametrize("graded", [False, True])
def test_physical_surface_geometry_and_currents(graded):
    grid, box, data, surface = _record(graded)
    # Independent physical nodes and local areas, including zero/asymmetric padding.
    bounds = [(2, 5), (3, 6), (2, 5)]
    for face in range(6):
        axis, side = divmod(face, 2)
        a, b = [v for v in range(3) if v != axis]
        lo, hi = surface.face_offsets[face:face + 2]
        points, areas = [], []
        for i in range(*bounds[a]):
            for j in range(*bounds[b]):
                p = np.zeros(3)
                p[axis] = grid.node_of(axis, bounds[axis][side])
                p[a] = (grid.node_of(a, i) + grid.node_of(a, i + 1)) / 2
                p[b] = (grid.node_of(b, j) + grid.node_of(b, j + 1)) / 2
                points.append(p)
                areas.append((grid.node_of(a, i + 1) - grid.node_of(a, i))
                             * (grid.node_of(b, j + 1) - grid.node_of(b, j)))
        np.testing.assert_allclose(surface.positions[lo:hi], points, rtol=2e-15, atol=1e-17)
        np.testing.assert_allclose(surface.areas[lo:hi], areas, rtol=3e-15)
        e, h = np.zeros((2, hi - lo, 3), complex), np.zeros((2, hi - lo, 3), complex)
        values = np.asarray(data[face]).reshape(2, -1, 4)
        e[..., a], e[..., b] = values[..., 0], values[..., 1]
        h[..., a], h[..., b] = values[..., 2], values[..., 3]
        np.testing.assert_array_equal(surface.E_t[:, lo:hi], e)
        np.testing.assert_array_equal(surface.H_t[:, lo:hi], h)
        n = np.eye(3)[axis] * (-1 if side == 0 else 1)
        np.testing.assert_array_equal(surface.J_s[:, lo:hi], np.cross(n, h))
        np.testing.assert_array_equal(surface.M_s[:, lo:hi], -np.cross(n, e))
    vector_area = surface.normals * surface.areas[:, None]
    np.testing.assert_allclose(vector_area.sum(axis=0), 0, atol=1e-19)
    volume = np.prod([grid.node_of(a, hi) - grid.node_of(a, lo) for a, (lo, hi) in enumerate(bounds)])
    np.testing.assert_allclose(surface.positions.T @ vector_area, volume * np.eye(3), atol=volume * 1e-14)
    assert surface.positions.dtype == np.float64
    assert surface.E_t.dtype == np.complex128
    assert surface.dt != grid.dt  # actual run timestep, not inferred from grid
    assert surface.step_start == 5


def test_complex_reference_subtraction_preserves_phase_and_zero():
    _, _, _, surface = _record()
    ref = replace(surface, E_t=surface.E_t * (0.2 + 0.4j), H_t=surface.H_t * (0.3 - 0.6j))
    result = surface.subtract_reference(ref)
    np.testing.assert_array_equal(result.E_t, surface.E_t - ref.E_t)
    np.testing.assert_allclose(result.J_s, surface.J_s - ref.J_s, rtol=1e-15)
    np.testing.assert_allclose(result.M_s, surface.M_s - ref.M_s, rtol=1e-15)
    assert result.reference_subtracted
    np.testing.assert_array_equal(surface.subtract_reference(surface).E_t, 0)
    with pytest.raises(ValueError, match="un-subtracted"):
        result.subtract_reference(ref)
    with pytest.raises(ValueError, match="un-subtracted"):
        surface.subtract_reference(result)


@pytest.mark.parametrize("name", ["freqs", "positions", "areas", "dt", "n_steps", "step_start"])
def test_incompatible_reference_is_refused(name):
    _, _, _, surface = _record()
    value = getattr(surface, name)
    changed = value + (1 if name in ("n_steps", "step_start") else 0.01 * np.max(value))
    with pytest.raises(ValueError, match=name):
        surface.subtract_reference(replace(surface, **{name: changed}))


def test_save_load_roundtrip_and_owns_data(tmp_path):
    _, _, data, surface = _record(True)
    before = surface.E_t.copy()
    data.x_lo[...] = 0
    np.testing.assert_array_equal(surface.E_t, before)
    with pytest.raises(ValueError):
        surface.E_t[0, 0, 0] = 1
    result = surface.subtract_reference(replace(surface, E_t=0 * surface.E_t, H_t=0 * surface.H_t))
    path = tmp_path / "surface.npz"
    result.save_npz(path)
    restored = NTFFSurface.load_npz(path)
    for name in result.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(restored, name), getattr(result, name))
    assert not restored.E_t.flags.writeable
    with np.load(path, allow_pickle=False) as f:
        fields = dict(f)
        assert "V*s/m" in f["units"].item()
    for name in ("schema", "convention", "units"):
        np.savez(path, **{**fields, name: np.array("unknown")})
        with pytest.raises(ValueError, match=name):
            NTFFSurface.load_npz(path)


@pytest.mark.parametrize("kwargs,match", [
    ({"dt": 0}, "dt"), ({"dt": np.nan}, "dt"), ({"dt": 1}, "Nyquist"),
    ({"n_steps": 0}, "n_steps"), ({"n_steps": 4.5}, "n_steps"),
    ({"step_start": -1}, "step_start"), ({"reference_subtracted": 1}, "boolean"),
    ({"freqs": [np.nan, 3e9]}, "finite"), ({"freqs": [0, 3e9]}, "Nyquist"),
    ({"face_offsets": np.arange(7) + 0.5}, "integers"),
])
def test_invalid_records_are_refused(kwargs, match):
    _, _, _, surface = _record()
    with pytest.raises(ValueError, match=match):
        replace(surface, **kwargs)


def test_invalid_geometry_or_layout_is_refused():
    grid, box, data, _ = _record(True)
    for changed, match in ((box._replace(face_centre=False), "face_centre"),
                           (box._replace(i_lo=0), "no room"),
                           (box._replace(w_x_lo=0.1), "collocation")):
        with pytest.raises(ValueError, match=match):
            export_ntff_surface(data, changed, grid, dt=grid.dt, n_steps=10)
    with pytest.raises(ValueError, match="complete face shape"):
        export_ntff_surface(data._replace(x_lo=data.x_lo[:, :1]), box, grid, dt=grid.dt, n_steps=10)


def test_actual_accumulator_keeps_yee_clock_and_raw_integral_units():
    grid = Grid(3e9, (0.006,) * 3, dx=1e-3, cpml_layers=0)
    f, dt, steps, start = 3e9, 1 / (3e9 * 16), 32, 7
    box = NTFFBox.from_grid(grid, i_lo=2, i_hi=4, j_lo=2, j_hi=4,
                           k_lo=2, k_hi=4, freqs=np.array([f]))
    phasors = np.array([1 + 2j, 3 - 2j, -1 + 0.5j, 0.1 - 0.2j, -0.3 + 0.7j, 0.2 + 0.1j])
    data = init_ntff_data(box)
    for step in range(start, start + steps):
        t = (step + np.array([1, 1, 1, 0.5, 0.5, 0.5])) * dt
        values = np.real(phasors * np.exp(2j * np.pi * f * t))
        state = FDTDState(**{name: jnp.full(grid.shape, v, dtype=jnp.float32)
                             for name, v in zip(("ex", "ey", "ez", "hx", "hy", "hz"), values)},
                          step=jnp.asarray(step))
        data = accumulate_ntff(data, state, box, dt, jnp.asarray(step))
    surface = export_ntff_surface(data, box, grid, dt=dt, n_steps=steps, step_start=start)
    # Real harmonic over integer cycles integrates to T/2 * complex phasor.
    # An extra half-step phase shift, division by T, or factor of two fails.
    scale = steps * dt / 2
    for field, ph in ((surface.E_t, phasors[:3]), (surface.H_t, phasors[3:])):
        expected = ph[None, :] * (1 - np.abs(surface.normals)) * scale
        np.testing.assert_allclose(field[0], expected, rtol=4e-6, atol=scale * 1e-7)


def test_subgrid_result_without_global_origin_is_refused():
    # A real caller path: a local fine grid would export z planes 6 mm low.
    # Eight steps test admission/dataflow only, not settled radiation accuracy.
    sim = Simulation(freq_max=5e9, domain=(0.03,) * 3, dx=3e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_source((0.015,) * 3, "ez", amplitude_kind="field")
    sim.add_refinement(z_range=(0.006, 0.024), ratio=2, validation="research")
    sim.add_ntff_box(corner_lo=(0.009, 0.009, 0.012),
                     corner_hi=(0.021, 0.021, 0.018), freqs=[3e9])
    result = sim.run(n_steps=8, skip_preflight=True)
    with pytest.raises(ValueError, match="subgrid.*global-origin"):
        export_ntff_surface(result.ntff_data, result.ntff_box, result.grid,
                            dt=result.dt, n_steps=8)
