"""Physical-origin phase of an off-centre Hertzian source (#1369).

These are component tests: exact dipole fields are supplied on the Huygens
surface, so no FDTD dispersion, pulse truncation or boundary reflection is
mixed into the phase-origin measurement. The grid still has real asymmetric
pads, including a PEC lower z face. Neither source nor look directions lie
on the box's symmetry planes. The graded case varies all three cell profiles.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.farfield import (
    ETA_0,
    NTFFBox,
    NTFFData,
    init_ntff_data,
    make_ntff_box,
    with_face_centre_collocation,
)
from rfx.measurement.farfield import (
    compute_far_field,
    compute_far_field_jax,
)
from rfx.grid import C0, Grid
from rfx.nonuniform import make_nonuniform_grid, position_to_index
from rfx.ntff_surface import export_ntff_surface
from rfx.rcs import compute_rcs_jax
from tests.unit.farfield.test_ntff_second_order_oracle import _dipole_fields


FREQ = 10e9
DX = 1e-3
SOURCE = np.array([13.1, 16.7, 18.3]) * 1e-3
THETA = np.array([0.3, 1.0, 1.4, 2.2])
PHI = np.array([0.21, 0.83, 1.61, 2.4])
LOWER = (0.004, 0.005, 0.006)
UPPER = (0.028, 0.027, 0.029)


def _grid(graded, *, zero_face="pec", symmetric=False):
    faces = {"x_lo": 4, "x_hi": 6, "y_lo": 5, "y_hi": 7,
             "z_lo": 8, "z_hi": 3}
    kwargs = dict(cpml_layers=8)
    if not symmetric:
        kwargs["face_layers"] = faces
        if zero_face == "zero_cpml":
            faces["z_lo"] = 0
        elif zero_face == "inactive_axis":
            kwargs["cpml_axes"] = "xy"
        else:
            kwargs[f"{zero_face}_faces"] = {"z_lo"}
    if not graded:
        return Grid(1.5 * FREQ, (0.032,) * 3, dx=DX, **kwargs)
    profiles = [DX * (1 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, 32) * n))
                for n in (1, 2, 3)]
    assert all(np.all((p >= 0.75e-3) & (p <= 1.25e-3)) and np.isclose(p.sum(), 0.032, rtol=0, atol=1e-14) and np.allclose(p[[0, -1]], DX, rtol=0, atol=1e-14) for p in profiles) and all(not np.array_equal(profiles[a], profiles[b]) for a, b in ((0, 1), (0, 2), (1, 2))), "The graded profiles must have 0.75 to 1.25 mm cells, span 32 mm, retain 1 mm boundary cells, and differ by axis."
    return make_nonuniform_grid((0.032, 0.032), profiles[2], DX,
                                dx_profile=profiles[0], dy_profile=profiles[1],
                                **kwargs)


REQUIRED_BOX_FIELDS = 7


def _box(grid, graded, *, hand_built=False):
    if not graded:
        box = make_ntff_box(grid, LOWER, UPPER, [FREQ])
    else:
        lo, hi = position_to_index(grid, LOWER), position_to_index(grid, UPPER)
        box = NTFFBox.from_grid(grid, i_lo=lo[0], i_hi=hi[0],
                               j_lo=lo[1], j_hi=hi[1], k_lo=lo[2], k_hi=hi[2],
                               freqs=np.array([FREQ]))
    if hand_built:
        box = NTFFBox(*box[:REQUIRED_BOX_FIELDS], face_centre=True)
    return box


def _dipole_record(grid, box):
    """The midpoint truncation-error scale budgets both tangential directions and both phases in a directional difference."""
    nodes, widths = [], []
    for axis in "xyz":
        cells = np.asarray(grid.cells(axis), dtype=np.float64)
        edges = np.r_[0.0, np.cumsum(cells)]
        nodes.append(edges - edges[getattr(grid, f"pad_{axis}_lo")])
        widths.append(cells)
    bounds = [(box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi)]
    arrays = []
    for normal in range(3):
        a, b = [axis for axis in range(3) if axis != normal]
        for face in bounds[normal]:
            lines = [np.array([nodes[normal][face]]) if axis == normal
                     else (nodes[axis][lo:hi] + nodes[axis][lo + 1:hi + 1]) / 2
                     for axis, (lo, hi) in enumerate(bounds)]
            points = np.stack(np.meshgrid(*lines, indexing="ij"), axis=-1)
            E, H = _dipole_fields(points.reshape(-1, 3), SOURCE[None, :],
                                  [[0, 0, 1]], [[0, 0, 0]], k=2 * np.pi * FREQ / C0)
            shape = (1, bounds[a][1] - bounds[a][0], bounds[b][1] - bounds[b][0], 4)
            arrays.append(np.stack([E[:, a], E[:, b], H[:, a], H[:, b]], axis=-1).reshape(shape))
    h = max(np.max(d[lo:hi]) for d, (lo, hi) in zip(widths, bounds))
    distance = min(min(SOURCE[a] - nodes[a][lo], nodes[a][hi] - SOURCE[a])
                   for a, (lo, hi) in enumerate(bounds))
    FLOAT32_PHASE_ERROR_RAD = 2e-5
    tolerance = 4 * h**2 / 24 * ((2 * np.pi * FREQ / C0)**2 + 12 / distance**2) + FLOAT32_PHASE_ERROR_RAD
    return NTFFData(*arrays), tolerance


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("hand_built", [False, True], ids=["constructor", "hand-built"])
@pytest.mark.parametrize("transform", [compute_far_field, compute_far_field_jax], ids=["numpy", "jax"])
def test_off_centre_source_phase(graded, hand_built, transform):
    """The absolute phase reference uses a unit current moment."""
    grid = _grid(graded)
    box = _box(grid, graded, hand_built=hand_built)
    data, tolerance = _dipole_record(grid, box)
    ff = transform(data, box, grid, THETA, PHI)
    th, ph = np.meshgrid(THETA, PHI, indexing="ij")
    direction = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
    phase = (2 * np.pi * FREQ / C0) * (direction @ SOURCE)
    measured = np.angle(np.asarray(ff.E_theta[0]) / np.asarray(ff.E_theta[0, 0, 0]))
    analytic = np.angle(np.exp(1j * (phase - phase[0, 0])))
    residual = np.angle(np.exp(1j * (measured - analytic)))
    assert np.max(np.abs(residual)) < tolerance
    expected = 1j * ETA_0 * FREQ / (2 * C0) * np.sin(th) * np.exp(1j * phase)
    assert np.max(np.abs(np.angle(ff.E_theta[0] / expected))) < tolerance / 2
    print(f"\nphase {graded=} {hand_built=} {transform.__name__}: "
          f"measured_deg={np.degrees(measured).round(6).tolist()} "
          f"analytic_deg={np.degrees(analytic).round(6).tolist()} "
          f"max_error_deg={np.degrees(np.max(np.abs(residual))):.6g} "
          f"tolerance_deg={np.degrees(tolerance):.6g}")


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("zero_face", ["pec", "pmc", "zero_cpml", "inactive_axis"])
def test_ntff_constructors_store_realized_pads(graded, zero_face):
    grid = _grid(graded, zero_face=zero_face)
    box = _box(grid, graded)
    assert box.cpml_lo_z == 0
    assert box.cpml_lo_x == 4
    for axis in "xyz":
        for side in ("lo", "hi"):
            assert getattr(box, f"cpml_{side}_{axis}") == getattr(grid, f"pad_{axis}_{side}")


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("consumer", ["numpy", "jax", "rcs", "surface"])
@pytest.mark.parametrize("field,value", [("cpml_lo_z", 8), ("cpml_lo_x", 0), ("cpml_hi_y", 8)])
def test_conflicting_pad_metadata_is_rejected(graded, consumer, field, value):
    grid = _grid(graded)
    box = _box(grid, graded)._replace(**{field: value})
    data = init_ntff_data(box)
    with pytest.raises(ValueError, match=f"{field}=.*realized pad"):
        if consumer == "surface":
            export_ntff_surface(data, box, grid, dt=grid.dt, n_steps=10)
        elif consumer == "rcs":
            compute_rcs_jax(data, box, grid, THETA, PHI, e_inc_amplitude=1.0)
        else:
            transform = compute_far_field if consumer == "numpy" else compute_far_field_jax
            transform(data, box, grid, THETA, PHI)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("transform", [compute_far_field, compute_far_field_jax])
def test_symmetric_origin_is_bit_identical_to_scalar_grid(graded, transform):
    grid = _grid(graded, symmetric=True)
    box = _box(grid, graded)
    data, _ = _dipole_record(grid, box)
    attrs = {name: getattr(grid, name) for name in ("dx", "dy", "dz", "dx_arr", "dy_arr", "cpml_layers")
             if hasattr(grid, name)}
    legacy = SimpleNamespace(**attrs)
    hand = NTFFBox(*box[:REQUIRED_BOX_FIELDS], face_centre=True)
    actual = transform(data, box, grid, THETA, PHI)
    expected = transform(data, hand, legacy, THETA, PHI)
    np.testing.assert_array_equal(actual.E_theta, expected.E_theta)
    np.testing.assert_array_equal(actual.E_phi, expected.E_phi)


@pytest.mark.parametrize("graded", [False, True])
def test_surface_export_and_transform_share_physical_origin(graded):
    """The tolerance allows quantization between the transform’s float32 solver cell store and the export’s float64 coordinate spine."""
    grid = _grid(graded)
    box = _box(grid, graded)
    data, _ = _dipole_record(grid, box)
    surface = export_ntff_surface(data, box, grid, dt=grid.dt, n_steps=10)
    th, ph = np.meshgrid(THETA, PHI, indexing="ij")
    direction = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
    th_hat = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], axis=-1)
    ph_hat = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], axis=-1)
    k = 2 * np.pi * FREQ / C0
    phase = np.exp(1j * k * (direction @ surface.positions.T)) * surface.areas
    N, L = phase @ surface.J_s[0], phase @ surface.M_s[0]
    expected = -1j * k / (4 * np.pi) * np.sum(L * ph_hat + ETA_0 * N * th_hat, axis=-1)
    ff = compute_far_field(data, box, grid, THETA, PHI)
    np.testing.assert_allclose(ff.E_theta[0], expected, rtol=5 * np.finfo(np.float32).eps)


@pytest.mark.parametrize("graded", [False, True])
def test_margin_error_is_not_hidden_by_pad_conflict(graded):
    grid = _grid(graded)
    box = _box(grid, graded)._replace(i_lo=0, cpml_lo_x=0)
    with pytest.raises(ValueError, match="no room for the face-centre") as exc:
        with_face_centre_collocation(box, grid)
    assert "x: faces at index 0" in str(exc.value)
    assert "can carry a face anywhere in" in str(exc.value)


@pytest.mark.parametrize("transform", [compute_far_field, compute_far_field_jax])
def test_legacy_none_scalar_pad_means_unpadded(transform):
    """With no realized or explicit pads, a None scalar means no padding while an explicit per-face zero is preserved."""
    grid = _grid(False)
    box = NTFFBox(*_box(grid, False)[:REQUIRED_BOX_FIELDS], face_centre=True)
    data, _ = _dipole_record(grid, box)
    attrs = dict(dx=DX, face_layers={"x_lo": 0, "y_lo": 2})
    actual = transform(data, box, SimpleNamespace(**attrs, cpml_layers=None), THETA, PHI)
    expected = transform(data, box, SimpleNamespace(**attrs, cpml_layers=0), THETA, PHI)
    np.testing.assert_array_equal(actual.E_theta, expected.E_theta)
    np.testing.assert_array_equal(actual.E_phi, expected.E_phi)
