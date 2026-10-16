"""Frozen DESIGN.md E1–E4: declared interior deformation windows."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.model import NominalProfile, admit_graded_profiles, traced_cell_mask
from rfx.model.graded_admission import admit_nominal_profiles, admit_traced_windows
from rfx.model.graded_grid import make_nonuniform_grid


CELLS = np.array([1., 1.25, 0.75, 1.])
WINDOW = (1, 3)


def _grid(profile, declaration):
    return make_nonuniform_grid(
        (4., 3.), np.ones(3), 1., 2, dx_profile=profile,
        face_layers={"x_lo": 1, "x_hi": 2},
        nominal_profiles={"x": declaration} if declaration is not None else None)


@pytest.mark.parametrize("window", [
    1, [1, 3], (True, 3), (1, False), (np.int64(1), 3),
    (1., 3), (0, 1, 3), (-1, 2), (1, 5), (2, 2), (3, 2),
])
@pytest.mark.parametrize("entry", ["admission", "grid"])
def test_e1_window_refusals(window, entry):
    def objective(profile):
        declaration = {"x": NominalProfile(CELLS, window)}
        if entry == "admission":
            admit_graded_profiles((4., 3., 3.), profile, None, None,
                                  nominal_profiles=declaration)
        else:
            _grid(profile, declaration["x"])
        return jnp.sum(profile)

    with pytest.raises(ValueError, match="nominal_profiles.*x.*traced_cells"):
        jax.grad(objective)(jnp.asarray(CELLS))


@pytest.mark.parametrize("wrapped", [False, True])
def test_e1_width_admission_unchanged(wrapped):
    def objective(profile):
        value = NominalProfile(CELLS) if wrapped else CELLS.tolist()
        admitted = admit_nominal_profiles({"x": value}, (profile, None, None))
        assert admitted["x"].dtype == np.float64
        assert admitted["x"].tobytes() == CELLS.tobytes()
        assert admit_traced_windows({"x": value}) is None
        return jnp.sum(profile)

    jax.grad(objective)(jnp.asarray(CELLS))


def test_e2_readmission_preserves_widths_and_windows():
    def objective(profile):
        domain = (4., 4., 3.)
        declaration = {"x": NominalProfile(CELLS, WINDOW), "y": CELLS.tolist()}
        domain1, first = admit_graded_profiles(
            domain, profile, profile, None, nominal_profiles=declaration)
        domain2, second = admit_graded_profiles(
            domain1, profile, profile, None, nominal_profiles=first)
        assert domain1 == domain2 == domain
        assert isinstance(second["x"], NominalProfile)
        assert first["x"].traced_cells == second["x"].traced_cells == WINDOW
        np.testing.assert_array_equal(first["x"].cells, second["x"].cells)
        np.testing.assert_array_equal(first["y"], second["y"])
        assert second["x"].cells.dtype == second["y"].dtype == np.float64
        grid = _grid(profile, second["x"])
        assert grid.traced_cells == {"x": WINDOW}
        np.testing.assert_array_equal(grid.nominal_cells["x"], CELLS)
        return jnp.sum(grid.dx_arr)

    jax.grad(objective)(jnp.asarray(CELLS))


@pytest.mark.parametrize("window", [(1, 3), (0, 2), (2, 4), (0, 4)],
                         ids=["interior", "low_edge", "high_edge", "whole"])
def test_e3_mask(window):
    def objective(profile):
        grid = _grid(profile, NominalProfile(CELLS, window))
        interior = np.zeros(len(CELLS), dtype=bool)
        interior[window[0]:window[1]] = True
        expected = np.zeros(len(grid.dx_arr), dtype=bool)
        lo, hi = grid.pad_x_lo, grid.pad_x_hi
        expected[lo:lo + len(CELLS)] = interior
        expected[:lo] = interior[0]
        expected[lo + len(CELLS):lo + len(CELLS) + hi] = interior[-1]
        expected[-1] = interior[-1]
        mask = traced_cell_mask(grid, "x")
        assert mask.dtype == np.bool_
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(traced_cell_mask(grid, "x", padded=False), interior)
        assert traced_cell_mask(grid, "y") is None
        return jnp.sum(grid.dx_arr)

    jax.grad(objective)(jnp.asarray(CELLS))


def test_e3_no_window_or_untraced_axis():
    concrete = _grid(CELLS, None)
    assert traced_cell_mask(concrete, "x") is None
    assert traced_cell_mask(concrete._replace(traced_cells={"x": WINDOW}), "x") is None

    def objective(profile):
        for declaration in (None, CELLS, NominalProfile(CELLS)):
            grid = _grid(profile, declaration)
            assert grid.traced_cells is None
            assert traced_cell_mask(grid, "x") is None
        return jnp.sum(grid.dx_arr)

    jax.grad(objective)(jnp.asarray(CELLS))


def _width_scalar(profile):
    grid = _grid(profile, NominalProfile(CELLS, WINDOW))
    return jnp.sum(grid.dx_arr ** 2)


@pytest.mark.parametrize("compiled", [False, True], ids=["grad", "jit_grad"])
@pytest.mark.parametrize("case", ["inside", "outside", "nominal", "two_ulp"])
def test_e4_fixed_widths(case, compiled):
    widths = CELLS.astype(np.float32)
    if case == "inside":
        widths[1] += np.float32(1e-3)
        widths[2] -= np.float32(1e-3)
    elif case == "outside":
        widths[0] *= np.float32(1.001)
    elif case == "two_ulp":
        for _ in range(2):
            widths[0] = np.nextafter(widths[0], np.float32(np.inf))
    derivative = jax.grad(_width_scalar)
    if compiled:
        derivative = jax.jit(derivative)
    if case == "outside":
        # Compiled callbacks wrap the underlying ValueError in XlaRuntimeError.
        error = jax.errors.JaxRuntimeError if compiled else ValueError
        with pytest.raises(error, match=(
                r"traced x axis interior cell 0.*realized width.*nominal width"
                r".*window \(1, 3\)")):
            derivative(jnp.asarray(widths)).block_until_ready()
    else:
        actual = derivative(jnp.asarray(widths))
        # Pad multiplicities: low=1; high=2 plus the trailing bounding cell.
        np.testing.assert_array_equal(actual, 2 * widths * np.array([2, 1, 1, 4]))


def test_vmap_grad_carries_window():
    profiles = np.stack([CELLS, CELLS + np.array([0., 0.01, -0.01, 0.])])
    actual = jax.vmap(jax.grad(_width_scalar))(jnp.asarray(profiles, dtype=jnp.float32))
    expected = 2 * profiles.astype(np.float32) * np.array([2, 1, 1, 4])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("lane", ["run_distributed_nu", "fwd_distributed_nu",
                                  "vmap_sweep_batched"])
def test_nominal_profile_keeps_lane_refusal(lane):
    from rfx import Simulation
    from rfx.runners._admission import admit

    def objective(profile):
        sim = Simulation(domain=(4., 3., 3.), dx=1., freq_max=1e6,
                         cpml_layers=2, dx_profile=profile,
                         nominal_profiles={"x": NominalProfile(CELLS, WINDOW)})
        assert isinstance(sim._nominal_profiles["x"], NominalProfile)
        with pytest.raises(NotImplementedError, match="nominal_profiles="):
            admit(sim, lane)
        return jnp.sum(profile)

    jax.grad(objective)(jnp.asarray(CELLS))


# Two windows on both sides of a centred feed: the slots (or a patch's two
# opposite edges) move, the feed's own cells 3 and 4 stay at their nominal width.
CELLS8 = np.array([1., 1.25, 0.75, 1., 1., 0.75, 1.25, 1.])
WINDOWS = ((1, 3), (5, 7))


def _grid8(profile, windows):
    return make_nonuniform_grid(
        (8., 3.), np.ones(3), 1., 2, dx_profile=profile,
        face_layers={"x_lo": 1, "x_hi": 2},
        nominal_profiles={"x": NominalProfile(CELLS8, windows)})


@pytest.mark.parametrize("windows", [(), ((1, 3), (2,)), ((1, 3), (5, 9)),
                                     ((1, 3), (True, 5)), ((1, 3), 5), ((1, 3), (4, 4))])
def test_two_window_refusals(windows):
    def objective(profile):
        _grid8(profile, windows)
        return jnp.sum(profile)

    with pytest.raises(ValueError, match="nominal_profiles.*x.*traced_cells"):
        jax.grad(objective)(jnp.asarray(CELLS8))


def test_two_windows_mask_is_their_union():
    def objective(profile):
        grid = _grid8(profile, WINDOWS)
        assert grid.traced_cells == {"x": WINDOWS}
        expected = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(traced_cell_mask(grid, "x", padded=False), expected)
        return jnp.sum(grid.dx_arr)

    jax.grad(objective)(jnp.asarray(CELLS8))


def _width_scalar8(profile):
    return jnp.sum(_grid8(profile, WINDOWS).dx_arr ** 2)


@pytest.mark.parametrize("moved, refused", [((1, 6), None), ((3,), 3), ((4,), 4), ((0,), 0), ((7,), 7)],
                         ids=["both_slots", "feed_low", "feed_high", "below", "above"])
def test_two_windows_feed_cells_stay_fixed(moved, refused):
    widths = CELLS8.astype(np.float32)
    for cell in moved:
        widths[cell] *= np.float32(1.001)
    derivative = jax.grad(_width_scalar8)
    if refused is None:
        actual = derivative(jnp.asarray(widths))
        np.testing.assert_array_equal(actual, 2 * widths * np.array([2, 1, 1, 1, 1, 1, 1, 4]))
    else:
        with pytest.raises(ValueError, match=(
                rf"traced x axis interior cell {refused}.*window \(\(1, 3\), \(5, 7\)\)")):
            derivative(jnp.asarray(widths)).block_until_ready()
