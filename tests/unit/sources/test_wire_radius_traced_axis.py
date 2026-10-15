"""Radius stamps use nominal widths outside declared deformation windows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.core.yee import init_materials
from rfx.model import NominalProfile
from rfx.model.graded_grid import make_nonuniform_grid
from rfx.sources.sources import WirePort
from rfx.sources.wire_radius import stamp_wire_radius


D = .001
NOMINAL = D * np.array([1., .8, 1.2] + [1.] * 13)
CELL = (7, 7, 7)


def _grid(profiles, windows):
    return make_nonuniform_grid(
        (.016, .016), profiles.get("z", np.full(16, D)), D, 0,
        dx_profile=profiles.get("x", NOMINAL),
        dy_profile=profiles.get("y", NOMINAL),
        nominal_profiles={a: NominalProfile(NOMINAL, window)
                          for a, window in windows.items()})


def _ports(component):
    axis = {"ex": 0, "ez": 2}[component]
    ports = []
    for y in (.007, .010):
        start = [.007, y, .007]
        end = start.copy()
        end[axis] += .002
        ports.append(WirePort(tuple(start), tuple(end), component, radius=.0001))
    return ports


def _stamp_ports(grid, component):
    materials = init_materials(grid.shape)
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    transverse = 0 if axis == 1 else 1
    for pin in (7, 10):
        cells = []
        for axial in (7, 8):
            cell = list(CELL)
            cell[transverse] = pin
            cell[axis] = axial
            cells.append(tuple(cell))
        materials = stamp_wire_radius(grid, materials, component, .0001, cells)
    return materials


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_bit_identity_with_untraced_nominal_stamp(component):
    # ex/ey exercise two wires with x/y traced (axial + transverse);
    # ez additionally makes both traced axes transverse to each wire.
    expected = _stamp_ports(_grid({}, {}), component)

    def stamp(profile):
        grid = _grid({"x": profile, "y": profile}, {"x": (1, 3), "y": (1, 3)})
        return _stamp_ports(grid, component)

    # Move the widths BEFORE the pins; pin indices still use nominal nodes.
    moved = jnp.asarray(NOMINAL).at[1].add(.0001).at[2].add(-.0001)
    actual = jax.jit(stamp)(moved)
    for field in ("mu_r_wire", "eps_r_lumped", "eps_r"):
        for got, want in zip(jax.tree.leaves(getattr(actual, field)),
                             jax.tree.leaves(getattr(expected, field))):
            assert np.asarray(got).dtype == np.asarray(want).dtype
            assert np.asarray(got).shape == np.asarray(want).shape
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), field
        assert jax.tree.structure(getattr(actual, field)) == jax.tree.structure(
            getattr(expected, field))


def test_cpml_padded_nominal_widths_bit_identity():
    nominal = np.full(16, D)
    nominal[1:5] = .8 * D
    nominal[9:13] = 1.2 * D

    def stamp(profile, *, traced):
        grid = make_nonuniform_grid(
            (.016, .016), np.full(16, D), D, cpml_layers=4,
            cpml_axes="x", dx_profile=profile,
            nominal_profiles={"x": NominalProfile(nominal, (1, 3))} if traced else None)
        assert grid.pad_x_lo == grid.pad_x_hi == 4
        i = 7 + grid.pad_x_lo
        # Padded stencil 9..12 maps to interior 5..8 (1 mm); omitting
        # padding reads interior 9..12 (1.2 mm), which is not square with y.
        np.testing.assert_array_equal(
            nominal[i - 2 - grid.pad_x_lo:i + 2 - grid.pad_x_lo], np.full(4, D))
        assert np.all(nominal[i - 2:i + 2] != D)
        return stamp_wire_radius(
            grid, init_materials(grid.shape), "ez", .0001, [(i, 7, 7)])

    expected = stamp(nominal, traced=False)
    moved = jnp.asarray(nominal).at[1].add(.0001).at[2].add(-.0001)
    actual = jax.jit(lambda profile: stamp(profile, traced=True))(moved)
    for field in ("mu_r_wire", "eps_r_lumped", "eps_r"):
        got, want = getattr(actual, field), getattr(expected, field)
        assert jax.tree.structure(got) == jax.tree.structure(want)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            assert np.asarray(a).shape == np.asarray(b).shape
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), field


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2, 3])
def test_transverse_stencil_and_margin(axis, offset):
    component = "ez" if axis != "z" else "ex"
    # i-2..i+1 is read, i-3 and i+2 are the guard cells.
    index = CELL[0] + offset - 1
    _check_window(axis, component, (index, index + 1), refuses=True)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_axial_stencil_and_margin(axis, offset):
    index = CELL[0] + offset
    _check_window(axis, "e" + axis, (index, index + 1), refuses=True)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("axial", [False, True])
@pytest.mark.parametrize("side", ["before", "after"])
def test_one_cell_beyond_margin_is_admitted(axis, axial, side):
    component = "e" + axis if axial else ("ez" if axis != "z" else "ex")
    offset = (-2 if side == "before" else 2) if axial else (
        -4 if side == "before" else 3)
    index = CELL[0] + offset
    _check_window(axis, component, (index, index + 1), refuses=False)


@pytest.mark.parametrize("window, refuses", [
    ((4, 5), True), ((3, 4), False),
], ids=["touches_margin", "ends_before_margin"])
def test_cpml_padded_traced_window(window, refuses):
    def objective(profile):
        grid = make_nonuniform_grid(
            (.016, .016), np.full(16, D), D, cpml_layers=4,
            cpml_axes="x", dx_profile=profile,
            nominal_profiles={"x": NominalProfile(NOMINAL, window)})
        assert grid.pad_x_lo == grid.pad_x_hi == 4
        cell = (7 + grid.pad_x_lo, 7, 7)
        # Compute the padded range directly from the declaration and grid pad.
        last_moving_cell = window[1] - 1 + grid.pad_x_lo
        margin_cell = cell[0] - 3
        assert last_moving_cell == margin_cell - (0 if refuses else 1)
        materials = init_materials(grid.shape)
        if refuses:
            with pytest.raises(NotImplementedError) as exc:
                stamp_wire_radius(grid, materials, "ez", .0001, [cell])
            message = str(exc.value)
            assert f"port cell {cell}, axis x" in message
            assert f"stencil cells {cell[0] - 2}..{cell[0] + 1} (padded)" in message
            assert f"declared traced range {window} (half-open interior)" in message
            return jnp.sum(profile)
        stamped = stamp_wire_radius(grid, materials, "ez", .0001, [cell])
        return jnp.sum(stamped.eps_r)

    assert np.all(np.isfinite(jax.grad(objective)(jnp.asarray(NOMINAL))))


def _check_window(axis, component, window, *, refuses):
    def objective(profile):
        grid = _grid({axis: profile}, {axis: window})
        materials = init_materials(grid.shape)
        stamped = stamp_wire_radius(grid, materials, component, .0001, [CELL])
        return jnp.sum(stamped.eps_r)

    if refuses:
        with pytest.raises(NotImplementedError) as exc:
            jax.grad(objective)(jnp.asarray(NOMINAL))
        message = str(exc.value)
        assert f"port cell {CELL}, axis {axis}" in message
        lo, hi = (7, 7) if component == "e" + axis else (5, 8)
        assert f"stencil cells {lo}..{hi}" in message
        assert f"declared traced range {window}" in message
        assert "off-square Hankel bound and AD-vs-FD work" in message
        assert "not implemented" in message
    else:
        assert np.all(np.isfinite(jax.grad(objective)(jnp.asarray(NOMINAL))))


@pytest.mark.parametrize("declaration", ["no_profile", "no_window"])
def test_traced_axis_requires_nominal_profile_and_window(declaration):
    def objective(profile):
        windows = {} if declaration == "no_profile" else {"x": None}
        grid = _grid({"x": profile}, windows)
        stamp_wire_radius(grid, init_materials(grid.shape), "ex", .0001, [CELL])
        return jnp.sum(profile)

    with pytest.raises(NotImplementedError, match=(
            "does not support traced mesh metrics; .*nominal profile with traced_cells")):
        jax.grad(objective)(jnp.asarray(NOMINAL))


def test_forward_width_gradient_is_finite():
    def objective(width):
        profile = jnp.asarray(NOMINAL).at[1].set(width)
        profile = profile.at[2].set(NOMINAL[1] + NOMINAL[2] - width)
        sim = Simulation(
            domain=(.016,) * 3, dx=D, freq_max=30e9,
            boundary="pec", cpml_layers=0,
            dx_profile=profile, dy_profile=profile,
            nominal_profiles={a: NominalProfile(NOMINAL, (1, 3)) for a in "xy"})
        for port in _ports("ex"):
            sim.add_port(port.start, "ex", extent=.002, radius=.0001,
                         waveform=GaussianPulse(f0=15e9))
        sim.add_probe((.008, .008, .007), "ex")
        result = sim.forward(n_steps=24, skip_preflight=True)
        return jnp.sum(result.time_series ** 2)

    derivative = jax.grad(objective)(jnp.asarray(NOMINAL[1]))
    assert np.isfinite(float(derivative))


# Two moving windows, one on each side of the pin (#1750). The radius stamp
# reads the union of the windows: the pin is admitted when both stay clear of
# its stencil and one-cell margin, and refused when either one reaches them.
CLEAR = ((1, 4), (10, 13))


@pytest.mark.parametrize("component", ["ex", "ez"])
def test_two_windows_clear_of_pin_bit_identity(component):
    expected = _stamp_ports(_grid({}, {}), component)

    def stamp(profile):
        grid = _grid({"x": profile}, {"x": CLEAR})
        assert grid.traced_cells == {"x": CLEAR}
        return _stamp_ports(grid, component)

    # Move cells in both windows; each window keeps its own width sum.
    moved = (jnp.asarray(NOMINAL).at[1].add(.0001).at[2].add(-.0001)
             .at[11].add(.0001).at[12].add(-.0001))
    actual = jax.jit(stamp)(moved)
    for field in ("mu_r_wire", "eps_r_lumped", "eps_r"):
        got, want = getattr(actual, field), getattr(expected, field)
        assert jax.tree.structure(got) == jax.tree.structure(want)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            assert np.asarray(a).shape == np.asarray(b).shape
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), field


# Pin cell 7 on x. Across the wire (ez) the stamp reads cells 5..8 with
# margin cells 4 and 9; along it (ex) it reads cell 7 with margin cells 6 and 8.
@pytest.mark.parametrize("component, windows, refuses", [
    ("ez", CLEAR, False),
    ("ez", ((1, 5), (10, 13)), True),
    ("ez", ((1, 4), (9, 13)), True),
    ("ex", ((1, 6), (9, 13)), False),
    ("ex", ((1, 7), (9, 13)), True),
    ("ex", ((1, 6), (8, 13)), True),
], ids=["across_clear", "across_low_on_margin", "across_high_on_margin",
        "along_clear", "along_low_on_margin", "along_high_on_margin"])
def test_two_windows_margin(component, windows, refuses):
    _check_window("x", component, windows, refuses=refuses)
