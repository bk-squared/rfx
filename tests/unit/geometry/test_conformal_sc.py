"""Independent node geometry, SC incidence, operands, and main identity pins."""

from dataclasses import dataclass
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Sphere, Simulation, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.conformal import compute_conformal_fractions, clamp_conformal_weights
from rfx.geometry.conformal_cells import sc_clamp
from rfx.model.tensor_operands import finish_conformal_mu, form_reference_operands
from rfx.model.materials import with_components
from rfx.core.yee import init_materials


@dataclass
class RectGrid:
    shape: tuple = (5, 5, 5)
    widths: tuple = (.001, .002, .003)
    pad_x_lo: int = 1
    pad_y_lo: int = 2
    pad_z_lo: int = 1

    def cells(self, axis):
        return np.full(self.shape[axis], self.widths[axis], dtype=np.float64)


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('fraction', [.1, .25, .75, .9])
def test_plane_lengths_and_areas_on_every_component(axis, fraction):
    grid = RectGrid()
    pads = (1, 2, 1)
    plane = (2 - pads[axis] + fraction) * grid.widths[axis]
    lo, hi = [-1.] * 3, [1.] * 3
    hi[axis] = plane
    lengths, areas = compute_conformal_fractions(grid, [Box(tuple(lo), tuple(hi))], clamp=False)
    for component in range(3):
        expected_l = np.empty(grid.shape)
        expected_s = np.empty(grid.shape)
        for index in np.ndindex(grid.shape):
            node = (index[axis] - pads[axis]) * grid.widths[axis]
            fraction_open = np.clip((node + grid.widths[axis] - plane) / grid.widths[axis], 0, 1)
            expected_l[index] = fraction_open if component == axis else float(node >= plane)
            expected_s[index] = float(node >= plane) if component == axis else fraction_open
        np.testing.assert_allclose(lengths[component], expected_l, atol=3e-8, rtol=0)
        np.testing.assert_allclose(areas[component], expected_s, atol=3e-8, rtol=0)
        assert lengths[component].dtype == areas[component].dtype == jnp.float32


@pytest.mark.parametrize('normal', range(3))
def test_sphere_faces_against_circle_rectangle_integral(normal):
    grid = RectGrid(widths=(.001,) * 3, pad_x_lo=0, pad_y_lo=0, pad_z_lo=0)
    u, v = [a for a in range(3) if a != normal]
    center = [.002] * 3
    center[u] = .0024 - .2
    radius = .2
    _, areas = compute_conformal_fractions(grid, [Sphere(tuple(center), radius)], clamp=False)
    for n, j in ((1, 1), (2, 2), (3, 3)):
        index = [0] * 3
        index[normal], index[u], index[v] = n, 2, j
        r = np.sqrt(radius**2 - (n * .001 - center[normal])**2)
        a, b = j * .001 - center[v], (j + 1) * .001 - center[v]
        def primitive(t):
            return .5 * (t * np.sqrt(r*r - t*t) + r*r * np.arcsin(t / r))
        metal_area = (center[u] - .002) * .001 + primitive(b) - primitive(a)
        expected = 1 - metal_area / 1e-6
        assert 0 < expected < 1
        assert abs(float(areas[normal][tuple(index)]) - expected) < 1e-3


def test_sc_three_cell_incidence():
    lengths = [np.ones((3, 3, 3)) for _ in range(3)]
    lengths[0][1, 1, 1] = .17
    areas = [np.ones((3, 3, 3)) for _ in range(3)]
    areas[0][1, 1, 1], areas[1][1, 1, 1], areas[2][1, 1, 1] = .1, .3, 0.
    expected = [a.copy() for a in lengths]
    # Enumerate each face's four bounding E edges independently.
    for normal in range(3):
        transverse = [a for a in range(3) if a != normal]
        for face in np.ndindex((3, 3, 3)):
            for edge_axis in transverse:
                other = next(a for a in transverse if a != edge_axis)
                for offset in (0, 1):
                    edge = list(face)
                    edge[other] += offset
                    if edge[other] < 3:
                        edge = tuple(edge)
                        expected[edge_axis][edge] = min(expected[edge_axis][edge], 2 * areas[normal][face])
    for actual, want in zip(sc_clamp(lengths, areas), expected):
        np.testing.assert_array_equal(actual, want)


def test_union_is_taken_at_nodes_before_edge_roots():
    grid = RectGrid(widths=(1.,) * 3, pad_x_lo=0, pad_y_lo=0, pad_z_lo=0)
    shapes = [Box((-10.,) * 3, (.25, 10., 10.)),
              Box((.75, -10., -10.), (10.,) * 3)]
    lengths, _ = compute_conformal_fractions(grid, shapes, clamp=False)
    assert float(lengths[0][0, 1, 1]) == 0.  # Both union endpoints are metal.


@pytest.mark.parametrize('value', [0., .05, .2, float('nan')])
def test_nondefault_floor_refuses_without_conformal_shapes(value):
    with pytest.raises(NotImplementedError, match='SC clamp'):
        Simulation(20e9, (.01,) * 3, dx=.001).run(n_steps=1, conformal_min_weight=value)
    with pytest.raises(NotImplementedError, match='SC clamp'):
        clamp_conformal_weights(None, None, None, value)


def test_mu_faces_and_reference_wall_geometry():
    from rfx.grid import Grid
    grid = Grid(20e9, (.01,) * 3, dx=.001, cpml_layers=0)
    mats = with_components(init_materials(grid.shape), grid, periodic=(False,) * 3)
    wall = Box((-1.,) * 3, (.0027, 1., 1.))
    obstacle = Sphere((.007,) * 3, .0015)
    weights, _ = compute_conformal_fractions(grid, [wall, obstacle])
    eps, inv, reference = form_reference_operands(
        grid, mats, subpixel_smoothing=False, conformal_weights=weights, boundary_pec_shapes=[wall])
    assert inv is None
    assert float(reference[0][0][7, 7, 7]) == 1.
    assert float(weights[0][7, 7, 7]) == 0.
    finished = finish_conformal_mu(mats, reference[1])
    for normal in (1, 2):
        assert float(finished.components.mu_update[normal][2, 5, 5]) == pytest.approx(.3)
        assert float(finished.components.mu_update[normal][1, 5, 5]) == 1.
    assert float(eps[0][2, 5, 5]) == pytest.approx(1 / .3)


def identity_scene(aligned):
    boundary = (BoundarySpec(x=Boundary(lo='pec', hi='pec', conformal=True),
                             y='pec', z='pec') if aligned else 'cpml')
    sim = Simulation(20e9, (12/1024, 10/1024, 8/1024), dx=1/1024,
                     boundary=boundary, cpml_layers=3)
    sim.add_source((4/1024, 4/1024, 4/1024), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=10e9, bandwidth=1.0))
    sim.add_probe((6/1024, 5/1024, 4/1024), 'ez')
    return sim


# Float32 bit patterns captured on main e189d9152 before any production edit.
MAIN_TRACE_BITS = {'False': [0, 0, 0, 0, 917736138, 935342772, 945896509, 953098004, 958945492, 964726106, 968705154, 973532710, 976963100, 981368214, 984216514, 987733456, 990984609, 993686643, 996988164, 999616566, 1002021373, 1004887813, 1007454164, 1009439202, 1011745991, 1014403569, 1016230226, 1017948513, 1019876438, 1022020126, 1023895993, 1025184621, 1026577102, 1028066713, 1029643647, 1031294878, 1032401681, 1033276277, 1034159787, 1035040215], 'True': [0, 0, 0, 0, 917736138, 935342772, 945896509, 953098004, 958945492, 964726106, 968705154, 973534024, 976964823, 981368506, 984212455, 987715655, 990961361, 993634828, 996887208, 999527620, 1001876469, 1004664621, 1007289198, 1009204636, 1011421774, 1013967205, 1015942847, 1017576345, 1019402313, 1021423460, 1023524669, 1024727061, 1026018185, 1027389997, 1028831568, 1030328463, 1031831258, 1032608419, 1033384112, 1034147067]}


@pytest.mark.parametrize('aligned', [False, True])
def test_main_bit_identity(aligned):
    result = identity_scene(aligned).run(n_steps=40, conformal_pec=aligned, skip_preflight=True)
    actual = np.asarray(result.time_series).view(np.uint32).reshape(-1)
    np.testing.assert_array_equal(actual, MAIN_TRACE_BITS[str(aligned)])


@pytest.mark.parametrize('mode', ['3d', '2d_tez', '2d_tmz'])
def test_sphere_cpml_200_steps_finite_fields(mode):
    from rfx import simulation
    sim = Simulation(20e9, (.012,) * 3, dx=.001, boundary='cpml', cpml_layers=3, mode=mode)
    sim.add(Sphere((.006, .006, 0. if mode != '3d' else .006), .0023), material='pec')
    component = 'ez' if mode != '2d_tez' else 'ex'
    z = 0. if mode != '3d' else .006
    sim.add_source((.003, .004, z), component, amplitude_kind='field',
                   waveform=GaussianPulse(f0=10e9, bandwidth=1.))
    sim.add_probe((.008, .008, z), component)
    original, observed = simulation.run, []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result
    with patch.object(simulation, 'run', capture):
        result = sim.run(n_steps=200, conformal_pec=True, skip_preflight=True)
    assert np.isfinite(result.time_series).all()
    assert np.any(np.asarray(result.time_series) != 0)
    assert len(observed) == 1
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        assert np.isfinite(np.asarray(getattr(observed[0].state, component))).all()


def test_waveguide_boundary_conformal_scan_smoke():
    sim = Simulation(10e9, (.06, .0234, .0104), dx=.002, cpml_layers=4,
                     boundary=BoundarySpec(x='cpml', y=Boundary('pec', 'pec', conformal=True),
                                           z=Boundary('pec', 'pec', conformal=True)), snap='declared')
    sim.add_waveguide_port(.016, direction='+x', f0=8e9, freqs=[8e9], name='left')
    sim.add_waveguide_port(.044, direction='-x', f0=8e9, freqs=[8e9], name='right')
    result = sim.compute_waveguide_s_matrix(n_steps=200, normalize=True)
    assert np.isfinite(np.asarray(result.s_params)).all()


def test_collapsed_2d_geometry_repeats_the_node_slice():
    from rfx.grid import Grid
    from rfx import Cylinder
    grid = Grid(20e9, (.012, .012, .001), dx=.001, cpml_layers=0, mode='2d_tmz')
    sphere = Sphere((.006, .006, 0.), .0023)
    cylinder = Cylinder((.006, .006, 0.), .0023, 1., axis='z')
    for actual, expected in zip(compute_conformal_fractions(grid, [sphere]),
                                 compute_conformal_fractions(grid, [cylinder])):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('normalize,modes', [(True, 1), ('flux', 1), ('flux', 2), (False, 1), (False, 2)])
def test_waveguide_extractors_receive_e_and_h_operands(monkeypatch, normalize, modes):
    from rfx.stepping import uniform_loop
    sim = Simulation(25e9, (.06, .0234, .0104), dx=.002, cpml_layers=4,
                     boundary=BoundarySpec(x='cpml', y=Boundary('pec', 'pec', conformal=True),
                                           z=Boundary('pec', 'pec', conformal=True)), snap='declared')
    sim.add(Sphere((.030, .012, .006), .0025), material='pec')
    for position, direction, name in ((.016, '+x', 'left'), (.044, '-x', 'right')):
        sim.add_waveguide_port(position, direction=direction, f0=20e9, freqs=[20e9],
                               n_modes=modes, name=name)
    class Captured(Exception):
        pass
    def capture(grid, materials, n_steps, **kwargs):
        assert kwargs['aniso_eps'] is not None
        weights = kwargs['conformal_weights']
        i, _, _ = grid.position_to_index((.03, .0, .0))
        assert float(materials.components.mu_update[2][i, 11, 2]) == pytest.approx(.7)
        body = grid.position_to_index((.030, .012, .006))
        assert float(weights[0][body]) == (1. if normalize else 0.)
        raise Captured
    monkeypatch.setattr(uniform_loop, 'run', capture)
    with pytest.raises(Captured):
        sim.compute_waveguide_s_matrix(n_steps=200, normalize=normalize)


@pytest.mark.parametrize('positive,negative,expected', [(1., -2., 1/9), (2., -1., 8/9), (1., -1., .25)])
def test_marching_square_saddles(positive, negative, expected):
    from rfx.geometry.conformal_cells import face_fraction
    corners = [np.array([d]) for d in (positive, negative, positive, negative)]
    # Two disjoint corner triangles or a rectangle with two triangles removed.
    assert face_fraction(corners, (2., 3.))[0] == pytest.approx(expected)


def test_waveguide_reference_conformal_eps_is_the_port_medium():
    """A conformal wall cuts the reference's own medium, not vacuum (review of the #1651 fix)."""
    from rfx.core.yee import init_materials
    from rfx.model.tensor_operands import form_reference_operands
    grid = Simulation(10e9, (.02, .02, .02), dx=.002, boundary='pec')._build_grid()
    wall = Box((-1., .0125, -1.), (1., 1., 1.))
    eps, _, (lengths, _) = form_reference_operands(
        grid, init_materials(grid.shape), subpixel_smoothing=False, conformal_weights=(None,),
        boundary_pec_shapes=[wall], reference_eps=2.5)
    ex, ey = np.asarray(eps[0]), np.asarray(eps[1])
    ly = np.asarray(lengths[1])
    free = np.asarray(lengths[0]) == 1
    assert np.allclose(ex[free], 2.5)
    cut = (ly > 0) & (ly < 1)
    assert cut.any() and np.allclose(ey[cut], 2.5 / ly[cut])


def test_conformal_run_keeps_a_filament_and_refuses_a_body_holding_no_node():
    """A lattice filament reaches the step through its wire edges; a body with no grid node inside would vanish."""
    from rfx.geometry.csg import PolylineWire
    sim = Simulation(10e9, (.02, .02, .02), dx=.002, boundary='pec')
    sim.add(Sphere((.0101, .0099, .0102), .005), material='pec')
    sim.add(PolylineWire(((.004, .004, .004), (.004, .004, .016)), 0.), material='pec')
    sim.add_source((.016, .006, .007), 'ez', amplitude_kind='current')
    assert sim.run(n_steps=2, conformal_pec=True, skip_preflight=True).state is not None
    tiny = Simulation(10e9, (.02, .02, .02), dx=.002, boundary='pec')
    tiny.add(Sphere((.011, .011, .011), .0012), material='pec')  # 0.6 cell, centred in a cell: no node inside
    tiny.add_source((.004, .006, .007), 'ez', amplitude_kind='current')
    with pytest.raises(NotImplementedError, match='holds no grid node'):
        tiny.run(n_steps=2, conformal_pec=True, skip_preflight=True)


@pytest.mark.parametrize('periodic, box, refused', [
    ('', ((.004, .004, .006), (.008, .008, .0064)), True),     # 0.4 cell thick, no node inside
    ('', ((.004, .004, .006), (.008, .008, .007)), False),     # one cell thick, both faces on node planes
    ('xy', ((.0043, .004, .006), (.008, .008, .006)), True),   # sheet edge 0.3 cell off on a periodic axis
    ('xy', ((.004, .004, .006), (.008, .008, .006)), False),   # sheet on node planes, periodic
])
def test_a_box_holding_no_node_inside_is_admitted_only_on_node_planes(periodic, box, refused):
    """With no node strictly inside, the cut cells see nothing and the staircase faces remain: exact on node planes."""
    from rfx.grid import Grid
    grid = Grid(10e9, (.012, .012, .012), dx=.001, cpml_layers=0, periodic_axes=periodic)
    if refused:
        with pytest.raises(NotImplementedError, match='cell off the node lattice'):
            compute_conformal_fractions(grid, [Box(*box)])
    else:
        lengths, areas = compute_conformal_fractions(grid, [Box(*box)])
        assert all(float(jnp.min(a)) == 1. for a in lengths + areas)
