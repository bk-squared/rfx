"""Build-only checks of the conducting geometry at absorbing faces."""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.geometry.smoothing import continued_conductor_shape


def _sim(domain=(8., 8., 8.)):
    return Simulation(domain=domain, dx=1., freq_max=1e6,
                      boundary="cpml", cpml_layers=2)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("kind", ("volume", "sheet", "thin"))
def test_reached_face_is_filled(axis, side, kind):
    sim = _sim()
    lo, hi = [2., 2., 2.], [6., 6., 6.]
    (lo if side == 0 else hi)[axis] = 0. if side == 0 else 8.
    if kind != "volume":
        normal = (axis + 1) % 3
        lo[normal] = hi[normal] = 4.
    declared = Box(tuple(lo), tuple(hi))
    if kind == "thin":
        sim.add_thin_conductor(declared, sigma_bulk=5.8e7, thickness=0.01)
    else:
        sim.add(declared, material="pec")
    grid = sim._build_grid()
    sheets = []
    _, _, _, cells, shapes, _, _ = sim._assemble_materials(
        grid, pec_sheets=sheets, pec_wires=[])
    actual = np.asarray(cells if kind == "volume" else sheets[0].footprint)
    # Read solved cells/footprints, independent of the continuation helper.
    face = 2 if side == 0 else grid.shape[axis] - 3 - (kind == "volume")
    pad = range(2) if side == 0 else range(grid.shape[axis]-3, grid.shape[axis]-1)
    reference = np.take(actual, face, axis=axis)
    assert reference.any()
    for layer in pad:
        np.testing.assert_array_equal(np.take(actual, layer, axis=axis), reference)
    if kind != "volume":
        assert shapes[0].corner_lo[normal] == shapes[0].corner_hi[normal] == 4.
    assert declared.corner_lo == tuple(lo) and declared.corner_hi == tuple(hi)


def test_corner_pad_zero_and_fractional_declared_face():
    sim = _sim((535.43, 8., 8.))
    grid = sim._build_grid()
    grid.pad_z_lo = 0
    shape = Box((0., 0., 0.), (535.43, 4., 4.))
    solved = continued_conductor_shape(sim, grid, shape)
    assert solved.corner_lo[0] < -2. and solved.corner_lo[1] < -2.
    assert solved.corner_hi[0] > 538.
    assert solved.corner_lo[2] == 0.


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("cross_side", (0, 1))
def test_cylinder_axis_continues_and_cross_axis_is_reported(axis, side, cross_side):
    sim = _sim()
    grid = sim._build_grid()
    cross_axis = (axis + 1) % 3
    center = [4., 4., 4.]
    center[axis] = 3. if side == 0 else 5.
    center[cross_axis] = 1. if cross_side == 0 else 7.
    shape = Cylinder(tuple(center), radius=1., height=6., axis="xyz"[axis])
    findings = []
    solved = continued_conductor_shape(sim, grid, shape, unextendable=findings)
    assert solved.height > 6.
    assert solved.radius == 1.
    assert [(u.axis, u.side, u.conductor) for u in findings] == [
        (cross_axis, "hi" if cross_side else "lo", True)]
    # Unsupported thin geometry is reported even without ordinary entries.
    from rfx.geometry.smoothing import smoothed_shape_pairs, warn_unextendable_shapes
    sim.add_thin_conductor(Cylinder((4., 4., 4.), radius=4., height=0., axis="z"),
                           sigma_bulk=5.8e7, thickness=0.01)
    pairs, thin_findings = smoothed_shape_pairs(sim, grid)
    assert not pairs
    assert {(u.axis, u.side, u.collection) for u in thin_findings} == {
        (a, s, "thin_conductor") for a in (0, 1) for s in ("lo", "hi")}
    with pytest.warns(UserWarning, match="No continuation is applied") as caught:
        warn_unextendable_shapes(thin_findings)
    assert "eps_r" not in str(caught[0].message)


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y"))
def test_port_signal_stays_declared_and_ground_continues(direction):
    sim = _sim()
    sim._msl_ports = [SimpleNamespace(position=(4., 4., 2.), width=2.,
                                     height=1., direction=direction)]
    grid = sim._build_grid()
    signal = (Box((0., 3., 3.), (8., 5., 3.)) if direction[-1] == "x"
              else Box((3., 0., 3.), (5., 8., 3.)))
    ground = Box((0., 0., 2.), (8., 8., 2.))
    sim.add(signal, material="pec")
    sim.add(ground, material="pec")
    strip = continued_conductor_shape(sim, grid, signal)
    plane = continued_conductor_shape(sim, grid, ground)
    axis = "xy".index(direction[-1])
    if direction[0] == "+":
        assert strip.corner_lo[axis] == 0.
        assert plane.corner_lo[axis] < -2.
    else:
        assert strip.corner_hi[axis] == 8.
        assert plane.corner_hi[axis] > 10.
    assert strip is signal
    assert plane.corner_lo[2] == plane.corner_hi[2] == 2.


@pytest.mark.parametrize("axis", range(3))
def test_mirror_cells(axis):
    def build(lo, hi):
        sim = _sim()
        sim.add(Box(tuple(lo), tuple(hi)), material="pec")
        grid = sim._build_grid()
        mask = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[3]
        return np.asarray(mask)[:-1, :-1, :-1]
    lo, hi = [2., 2., 2.], [6., 6., 6.]
    lo[axis] = 0.
    actual = build(lo, hi)
    reflected_lo, reflected_hi = lo.copy(), hi.copy()
    reflected_lo[axis], reflected_hi[axis] = 8-hi[axis], 8-lo[axis]
    np.testing.assert_array_equal(np.flip(actual, axis), build(reflected_lo, reflected_hi))
    assert np.take(actual, 0, axis=axis).any()


@pytest.mark.parametrize("nu", (False, True))
@pytest.mark.parametrize("sheet", (False, True))
@pytest.mark.parametrize("kind", ("debye", "lorentz"))
def test_conductor_continues_but_pole_occupancy_stays_declared(nu, sheet, kind):
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import LorentzPole
    from rfx.geometry.smoothing import smoothed_shape_pairs
    from rfx.runners.nonuniform import assemble_materials_nu
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2,
                     **({"dz_profile": np.ones(8)} if nu else {}))
    pole = (DebyePole(delta_eps=1., tau=1e-10) if kind == "debye" else
            LorentzPole(omega_0=1e9, delta=1e7, kappa=1e18))
    sim.add_material("conducting_pole", sigma=1e7, **{kind+"_poles": [pole]})
    declared = Box((0., 2., 4. if sheet else 2.),
                   (8., 6., 4. if sheet else 6.))
    sim.add(declared, material="conducting_pole")
    grid = sim._build_nonuniform_grid() if nu else sim._build_grid()
    sheets = []
    result = (assemble_materials_nu(sim, grid, pec_sheets=sheets, pec_wires=[])
              if nu else sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=[]))
    mask = np.asarray(result[1 if kind == "debye" else 2][1][0])
    assert not mask[:2].any() and not mask[-3:-1].any()
    assert mask.sum() == (32 if sheet else 128)
    conductor = np.asarray(sheets[0].footprint if sheet else result[3])
    assert conductor[:2].any() and conductor[-3:-1].any()
    pairs, _ = smoothed_shape_pairs(sim, grid)
    assert pairs[0][0] is declared


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("kind", ("volume", "sheet", "thin"))
def test_occupied_layer_reaches_without_a_declared_face(axis, side, kind):
    sim = _sim()
    lo, hi = [2., 2., 2.], [6., 6., 6.]
    # The volume occupies the end cell although its bound misses the node.
    # Sheets use their closed rasterizer's node tolerance, not half a cell.
    inset = .1 if kind == "volume" else 5e-10
    (lo if side == 0 else hi)[axis] = inset if side == 0 else 8.-inset
    if kind != "volume":
        normal = (axis+1) % 3
        lo[normal] = hi[normal] = 4.
    shape = Box(tuple(lo), tuple(hi))
    if kind == "thin":
        sim.add_thin_conductor(shape, sigma_bulk=5.8e7, thickness=.01)
    else:
        sim.add(shape, material="pec")
    grid = sim._build_grid()
    sheets = []
    result = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=[])
    mask = np.asarray(result[3] if kind == "volume" else sheets[0].footprint)
    assert np.take(mask, 0 if side == 0 else -2, axis=axis).any()


@pytest.mark.parametrize("nu", (False, True))
@pytest.mark.parametrize("kind", ("wire", "lumped", "coax", "mixed"))
def test_port_terminals_hold_signal_on_both_faces(kind, nu):
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2,
                     **({"dz_profile": np.ones(8)} if nu else {}))
    signal = Box((0., 3., 4.), (8., 5., 4.))
    ground = Box((0., 0., 1.), (8., 8., 1.))
    sim.add(signal, material="pec")
    sim.add(ground, material="pec")
    if kind in ("wire", "mixed"):
        sim.add_port((4., 4., 1.), component="ez", extent=3.)
    elif kind == "lumped":
        sim.add_port((4., 4., 3.), component="ez")
    else:
        sim.add_coaxial_port((4., 4., 1.), face="bottom", pin_length=3.)
    if kind == "mixed":
        sim._msl_ports = [SimpleNamespace(position=(6., 4., 1.), width=2.,
                                         height=3., direction="-x")]
    grid = sim._build_nonuniform_grid() if nu else sim._build_grid()
    assert continued_conductor_shape(sim, grid, signal) is signal
    realized_ground = continued_conductor_shape(sim, grid, ground)
    assert realized_ground.corner_lo[0] < -2.
    assert realized_ground.corner_hi[0] > 10.


def test_one_traced_mesh_axis_keeps_every_conductor_face_declared():
    import jax
    import jax.numpy as jnp

    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2, dz_profile=np.ones(8))
    shape = Box((0., 0., 2.), (8., 8., 6.))
    sim.add(shape, material="pec")
    grid = sim._build_nonuniform_grid()
    from rfx.runners.nonuniform import assemble_materials_nu
    concrete = assemble_materials_nu(sim, grid, pec_sheets=[], pec_wires=[])[3]
    assert np.asarray(concrete)[0].any()
    assert np.asarray(concrete)[-2].any()

    @jax.jit
    def traced(dz):
        traced_grid = grid._replace(dz=dz)
        first = continued_conductor_shape(sim, traced_grid, shape)
        second = continued_conductor_shape(sim, traced_grid, shape)
        return jnp.asarray((first.corner_lo, first.corner_hi,
                            second.corner_lo, second.corner_hi))

    with pytest.warns(UserWarning, match="mesh axis z is traced") as caught:
        actual = np.asarray(traced(grid.dz))
    assert len(caught) == 1
    np.testing.assert_array_equal(actual, np.asarray([
        (0., 0., 2.), (8., 8., 6.), (0., 0., 2.), (8., 8., 6.)]))
