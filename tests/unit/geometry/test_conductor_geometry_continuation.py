"""Build-only checks of the conducting geometry at absorbing faces."""

import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.geometry.port_termination import held_conductor_entries, port_termination_references
from rfx.geometry.smoothing import continued_conductor_shape


def _sim(domain=(8., 8., 8.)):
    return Simulation(domain=domain, dx=1., freq_max=1e6,
                      boundary="cpml", cpml_layers=2)


@pytest.mark.parametrize("kind", ("wire", "lumped", "coax", "msl"))
@pytest.mark.parametrize("selection", ("identity", "equal", "index", "pair", "thin"))
def test_explicit_termination_resolves_registered_conductors(kind, selection):
    sim = _sim()
    ground = Box((0., 0., 1.), (8., 8., 1.))
    strip = Box((0., 3., 4.), (8., 5., 4.))
    sim.add(ground, material="pec")
    if selection == "thin":
        sim.add_thin_conductor(strip, sigma_bulk=5.8e7, thickness=.01)
    else:
        sim.add(strip, material="pec")
    value = {"identity": strip, "equal": Box(strip.corner_lo, strip.corner_hi),
             "index": 1, "pair": [ground, 1], "thin": strip}[selection]
    if kind == "msl":
        sim.add_msl_port((4., 4., 1.), width=2., height=3., terminates=value)
        port = sim._msl_ports[-1]
    elif kind == "coax":
        sim.add_coaxial_port((4., 4., 1.), face="bottom", pin_length=3., terminates=value)
        port = sim._coaxial_ports[-1]
    else:
        sim.add_port((4., 4., 1.), extent=3. if kind == "wire" else None, terminates=value)
        port = sim._ports[-1]
    expected = (sim._thin_conductors if selection == "thin" else
                sim._geometry if selection == "pair" else [sim._geometry[1]])
    assert len(port.terminates) == len(expected)
    assert all(ref.entry is entry for ref, entry in zip(port.terminates, expected))
    grid = sim._build_grid()
    assert continued_conductor_shape(sim, grid, strip) is strip
    solved_ground = continued_conductor_shape(sim, grid, ground)
    assert (solved_ground is ground) == (selection == "pair")


@pytest.mark.parametrize("kind", ("port", "coaxial_port", "msl_port"))
@pytest.mark.parametrize("value", (-1, 4, Box((1., 1., 1.), (2., 2., 2.))))
def test_unknown_termination_is_rejected_at_registration(kind, value):
    sim = _sim()
    sim.add(Box((0., 0., 0.), (8., 8., 1.)), material="pec")
    kwargs = dict(width=2., height=3.) if kind == "msl_port" else {}
    with pytest.raises(ValueError, match="add_"+kind+" at .*terminates="):
        getattr(sim, "add_"+kind)((4., 4., 1.), terminates=value, **kwargs)
    assert not sim._ports and not sim._msl_ports and not sim._coaxial_ports


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y"))
@pytest.mark.parametrize("nu", (False, True))
def test_msl_default_uses_only_its_exact_realized_signal_aperture(direction, nu):
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=2,
                     **({"dz_profile": np.ones(8)} if nu else {}))
    # Ground and a neighbouring trace lie one node from the declared aperture.
    ground = Box((0., 0., 3.), (8., 8., 3.))
    strip = (Box((0., 3., 4.), (8., 5., 4.)) if direction[-1] == "x"
             else Box((3., 0., 4.), (5., 8., 4.)))
    neighbour = (Box((0., 6., 4.), (8., 7., 4.)) if direction[-1] == "x"
                 else Box((6., 0., 4.), (7., 8., 4.)))
    for shape in (ground, strip, neighbour):
        sim.add(shape, material="pec")
    sim.add_msl_port((4.25, 4.25, 1.), width=2., height=3., direction=direction)
    grid = sim._build_realized_grid()
    def resolved():
        assert sim._msl_ports[-1].terminates is None
        return [ref.entry for ref in port_termination_references(
            sim, "_msl_ports", sim._msl_ports[-1], grid)]
    assert len(resolved()) == 1 and resolved()[0] is sim._geometry[1]
    sim.add_msl_port((4.25, 4.25, 1.), width=2., height=3.5, direction=direction)
    assert len(resolved()) == 1 and resolved()[0] is sim._geometry[1]
    sim.add_msl_port((4.25, 4.25, 1.), width=2., height=4., direction=direction)
    assert resolved() == []
    sim.add_msl_port((4.25, 4.25, 1.), width=2., height=3., direction=direction, terminates=())
    assert sim._msl_ports[-1].terminates == ()


@pytest.mark.parametrize("thin", (False, True))
def test_termination_references_follow_entries_through_deepcopy_and_removal(thin):
    import copy
    sim = _sim()
    strip = Box((0., 3., 4.), (8., 5., 4.))
    for _ in range(2):
        if thin:
            sim.add_thin_conductor(strip, sigma_bulk=5.8e7, thickness=.01)
        else:
            sim.add(strip, material="pec")
    sim.add_port((4., 4., 1.), extent=3., terminates=strip)
    copied = copy.deepcopy(sim)
    collection = "_thin_conductors" if thin else "_geometry"
    entries = getattr(copied, collection)
    grid = copied._build_realized_grid()
    held = held_conductor_entries(copied, grid)
    assert len(held) == 1 and held[0] is entries[0]
    assert held[0] is not getattr(sim, collection)[0]
    sheets = []
    copied._assemble_materials(grid, pec_sheets=sheets, pec_wires=[])
    assert not np.asarray(sheets[0].footprint)[:2].any()
    assert np.asarray(sheets[1].footprint)[:2].sum() == 6
    # The remaining equal declaration is a different entry and is not held.
    setattr(copied, collection, entries[1:])
    assert held_conductor_entries(copied, grid) == []


@pytest.mark.parametrize("kind", ("wire", "lumped", "coax"))
def test_unnamed_non_msl_port_ends_nothing(kind):
    sim = _sim()
    strip = Box((0., 3., 4.), (8., 5., 4.))
    sim.add(strip, material="pec")
    if kind == "coax":
        sim.add_coaxial_port((4., 4., 1.), face="bottom", pin_length=3.)
        port = sim._coaxial_ports[-1]
    else:
        sim.add_port((4., 4., 3.), extent=1. if kind == "wire" else None)
        port = sim._ports[-1]
    assert port.terminates == ()
    solved = continued_conductor_shape(sim, sim._build_grid(), strip)
    assert solved.corner_lo[0] < 0. and solved.corner_hi[0] > 8.


def test_exact_aperture_preserves_the_gap_between_parallel_wire_edges():
    from rfx.geometry.port_termination import lattice_intersects_aperture

    nodes = (np.arange(5., dtype=float),)*3
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2:4, 1:3, 2] = True
    wire = [(mask, (True, False, False))]
    between = (2.5, 1.5, 2.)
    on_wire = (2.5, 1., 2.)
    assert lattice_intersects_aperture(wire, nodes, on_wire, on_wire)
    assert not lattice_intersects_aperture(wire, nodes, between, between)
    # The same node footprint of a sheet fills its in-plane segments.
    sheet = [(mask, (False, False, False))]
    assert lattice_intersects_aperture(sheet, nodes, between, between)
    above = (2.5, 1.5, 2.5)
    assert not lattice_intersects_aperture(sheet, nodes, above, above)


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
        first_findings, second_findings = [], []
        first = continued_conductor_shape(sim, traced_grid, shape, unextendable=first_findings)
        second = continued_conductor_shape(sim, traced_grid, shape, unextendable=second_findings)
        assert len(first_findings) == len(second_findings) == 1
        from rfx.geometry.smoothing import warn_unextendable_shapes
        warn_unextendable_shapes(first_findings)
        return jnp.asarray((first.corner_lo, first.corner_hi,
                            second.corner_lo, second.corner_hi))

    with pytest.warns(UserWarning, match="mesh axis z is traced") as caught:
        actual = np.asarray(traced(grid.dz))
    assert len(caught) == 1
    assert not hasattr(sim, "_conductor_traced_warning")
    traced.clear_cache()
    with pytest.warns(UserWarning, match="mesh axis z is traced"):
        traced(grid.dz)
    np.testing.assert_array_equal(actual, np.asarray([
        (0., 0., 2.), (8., 8., 6.), (0., 0., 2.), (8., 8., 6.)]))
