"""Face-normal ties in issue #2799: traced parity, mirrors, and bisectors.

Geometry and effective-tie counts are evaluated independently in float64.
The closed forms use declared face normals and one-dimensional fill only.
"""
import resource

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.smoothing import (
    _ContinuedSheetShape,
    compute_inv_eps_tensor_diag,
    compute_smoothed_eps,
    compute_smoothed_eps_nonuniform,
)
from tests._cell_table_capture import record_cell_table_capture
from tests._x64_compat import enable_x64

MM = 1e-3


@pytest.fixture(scope='module', autouse=True)
def cpu_seconds(request):
    start = resource.getrusage(resource.RUSAGE_SELF)
    with enable_x64(False):
        yield
    end = resource.getrusage(resource.RUSAGE_SELF)
    seconds = end.ru_utime + end.ru_stime - start.ru_utime - start.ru_stime
    with request.config.pluginmanager.get_plugin('capturemanager').global_and_fixture_disabled():
        request.config.pluginmanager.get_plugin('terminalreporter').write_line(
            f'normal-tie contract CPU seconds: {seconds:.6f}')


def _grid(graded):
    profiles = dict(dy_profile=np.full(4, 3 * MM), dz_profile=np.full(4, 3 * MM)) if graded else {}
    return Simulation(freq_max=10e9, domain=(6 * MM, 12 * MM, 12 * MM),
                      dx=(.25 if graded else .3) * MM, boundary='pec',
                      cpml_layers=0, **profiles).freeze_mesh()


def _shape(kind, cx=3 * MM):
    if kind == 'box':
        return Box((cx - 2.1 * MM, 3.9 * MM, 2.4 * MM),
                   (cx + 2.1 * MM, 8.1 * MM, 9.6 * MM))
    return Cylinder((cx, (6.15 if kind == 'uniform' else 6) * MM, 6 * MM),
                    2.1 * MM, 7.2 * MM, axis='z')


def _lines(grid, component):
    lines = []
    widths = []
    for axis in range(3):
        cells = np.asarray(grid.cells(axis), dtype=np.float64)
        nodes = np.array([grid.node_of(axis, i) for i in range(grid.shape[axis])], dtype=np.float64)
        lines.append(nodes + cells / 2 if component == axis else nodes)
        widths.append(cells if component == axis else np.r_[cells[0], (cells[:-1] + cells[1:]) / 2])
    return lines, widths


def _effective_ties(grid, shape, component, *, nodes=False):
    lines, widths = _lines(grid, None if nodes else component)
    points = np.meshgrid(*lines, indexing='ij', sparse=True)
    tau = min(16 * np.finfo(np.float32).eps * 2 * max(grid.domain),
              .01 * min(float(np.min(w)) for w in widths))
    if isinstance(shape, Cylinder):
        delta = [p - c for p, c in zip(points, shape.center)]
        radius = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
        dr = radius - shape.radius
        dh = abs(delta[2]) - shape.height / 2
        side = [np.divide(d, radius, out=np.zeros(np.broadcast_shapes(d.shape, radius.shape)),
                          where=radius != 0) for d in delta[:2]] + [np.zeros_like(dh)]
        cap = [np.zeros_like(dr), np.zeros_like(dr), np.sign(delta[2])]
        tie = abs(dr - dh) <= tau
        different = abs(side[component] ** 2 - cap[component] ** 2) > 1e-12
        normals = [a + b for a, b in zip(side, cap)]
        distance = np.sqrt(np.maximum(dr, 0) ** 2 + np.maximum(dh, 0) ** 2) + np.minimum(np.maximum(dr, dh), 0)
    else:
        center = [(lo + hi) / 2 for lo, hi in zip(shape.corner_lo, shape.corner_hi)]
        half = [(hi - lo) / 2 for lo, hi in zip(shape.corner_lo, shape.corner_hi)]
        delta = [p - c for p, c in zip(points, center)]
        distances = [abs(d) - h for d, h in zip(delta, half)]
        top = np.maximum.reduce(np.broadcast_arrays(*distances))
        tied = [r >= top - tau for r in distances]
        tie = sum(a.astype(int) for a in tied) >= 2
        different = tied[component] & (abs(delta[component]) > tau)
        normals = [np.where(t, np.sign(d), 0) for t, d in zip(tied, delta)]
        distance = np.sqrt(sum(np.maximum(r, 0) ** 2 for r in distances)) + np.minimum(top, 0)
    norm = np.sqrt(sum(n * n for n in normals))
    normals = [np.divide(n, norm, out=np.zeros(np.broadcast_shapes(n.shape, norm.shape)), where=norm != 0)
               for n in normals]
    cell = np.meshgrid(*widths, indexing='ij', sparse=True)
    length = np.sqrt(sum((n * w) ** 2 for n, w in zip(normals, cell)))
    fill = .5 - np.divide(distance, length, out=np.zeros_like(distance), where=length != 0)
    return np.broadcast_to(tie & different & (fill > 0) & (fill < 1), grid.shape)


def _mirror(values, grid, shape, label, *, nodes=False, require_ties=True):
    cy = shape.center[1] if isinstance(shape, Cylinder) else (shape.corner_lo[1] + shape.corner_hi[1]) / 2
    tie_pairs = 0
    for component, data in enumerate(values):
        y = _lines(grid, None if nodes else component)[0][1]
        pairs = [(i, int(np.argmin(abs(y - (2 * cy - v))))) for i, v in enumerate(y) if v < cy - 1e-12]
        pairs = [(i, j) for i, j in pairs if abs(y[j] - (2 * cy - y[i])) <= 1e-12]
        ties = _effective_ties(grid, shape, component, nodes=nodes)
        for i, j in pairs:
            tie_pairs += int(np.count_nonzero(ties[:, i, :] & ties[:, j, :]))
            np.testing.assert_allclose(data[:, i, :], data[:, j, :], rtol=0, atol=1e-5,
                                       err_msg=f'{label} mirror component={component} y={y[i], y[j]}')
    if require_ties:
        assert tie_pairs > 0, f'{label}: no effective tie mirror pairs'
    print(f'{label}: effective tie mirror pairs={tie_pairs}')


@pytest.fixture(scope='module', params=['graded', 'uniform', 'box'])
def scene(request):
    kind = request.param
    grid = _grid(kind != 'uniform')
    builder = compute_smoothed_eps if kind == 'uniform' else compute_smoothed_eps_nonuniform

    def build(cx):
        return jnp.stack(builder(grid, [(_shape(kind, cx), 4.)]))

    host = np.asarray(build(3 * MM))
    traced = np.asarray(jax.jit(build)(jnp.float32(3 * MM)))
    return kind, grid, _shape(kind), host, traced


def test_traced_matches_plain(scene):
    kind, _, _, host, traced = scene
    for component in range(3):
        error = float(np.max(abs(host[component] - traced[component])))
        print(f'{kind}: component={component} traced/plain max difference={error:.9g}')
        assert error <= 1e-5 * np.max(host), (kind, component, error, float(np.max(host)))


def test_plain_mirror_pairs(scene):
    kind, grid, shape, host, _ = scene
    _mirror(host, grid, shape, kind)


def test_effective_ties(scene):
    kind, grid, shape, _, _ = scene
    count = sum(int(np.count_nonzero(_effective_ties(grid, shape, c))) for c in range(3))
    print(f'{kind}: independent float64 effective ties={count}')
    assert count > 0


def test_bisector_closed_form(scene):
    kind, grid, _, host, _ = scene
    if kind == 'uniform':
        # Ey (3, 4.05, 2.4) mm is on both the side and the bottom cap.
        point, fill = (3 * MM, 4.05 * MM, 2.4 * MM), .5
    else:
        # Ey (3, 4.5, 3) mm lies 0.6 mm inside both faces, L=3 mm.
        point, fill = (3 * MM, 4.5 * MM, 3 * MM), .5 + .6 / 3
    faces = np.array([[0., -1., 0.], [0., 0., -1.]])
    normal = faces.sum(axis=0)
    normal /= np.linalg.norm(normal)
    n_squared = normal[1] ** 2
    expected = 1 / (n_squared * (fill / 4 + 1 - fill)
                    + (1 - n_squared) / (fill * 4 + 1 - fill))
    lines = _lines(grid, 1)[0]
    index = tuple(int(np.argmin(abs(a - p))) for a, p in zip(lines, point))
    assert all(abs(a[i] - p) <= 1e-12 for a, i, p in zip(lines, index, point))
    print(f'{kind}: closed form={expected:.12g}, actual={float(host[1][index]):.12g}')
    assert abs(float(host[1][index]) - expected) <= 1e-4, (kind, index, float(host[1][index]), expected)


@pytest.mark.parametrize('lane', ['kottke_pec', 'full_tensor', 'continued_sheet'])
def test_other_callers_host_mirrors(lane):
    grid = _grid(False)
    shape = _shape('uniform')
    if lane == 'kottke_pec':
        values = compute_inv_eps_tensor_diag(grid, pec_shapes=[shape])
    elif lane == 'full_tensor':
        # Nodes and their neighboring triplets exercise the node union caller.
        shape = Cylinder((3 * MM, 6 * MM, 6 * MM), 2.1 * MM, 7.2 * MM, axis='z')
        values = compute_node_tensor_eps(grid, [(shape, 4.)])
        _mirror([np.asarray(v) for v in values.nodes.xi[:3]], grid, shape, lane + '-nodes', nodes=True)
    else:
        # Continue a Cylinder through x-lo. The declared normal is evaluated
        # at the face outside the domain, then its x component is removed.
        shape = Cylinder((0., 6.15 * MM, 6 * MM), 2.1 * MM, 7.2 * MM, axis='z')
        grid = Simulation(freq_max=10e9, domain=(6 * MM, 12 * MM, 12 * MM),
                          dx=.3 * MM, cpml_layers=2).freeze_mesh()
        bounds = shape.bounding_box()
        continued = _ContinuedSheetShape(shape, (None,) * 3, bounds, ((0, 'lo', 0.),))
        values = compute_inv_eps_tensor_diag(grid, pec_shapes=[continued])
    _mirror([np.asarray(v) for v in values], grid, shape, lane, require_ties=lane != 'full_tensor')


@pytest.fixture(scope='module', params=[
    (mesh, kind, component) for mesh in ('uniform', 'graded')
    for kind in ('box', 'disc') for component in (0, 2)
], ids=lambda p: f'{p[0]}-{p[1]}-{("Ex", "Ey", "Ez")[p[2]]}')
def thin_midplane(request):
    """Cases (iv)-(vi): each tested component has its own centred thin body.

    Binary-exact nodes place the body's mid-plane exactly on the sample.
    Ex uses a node-centred z support; Ez uses its primal z cell.
    """
    mesh, kind, component = request.param
    h = 2. ** -11
    profiles = (dict(dx_profile=np.full(8, h), dy_profile=np.full(8, h),
                     dz_profile=np.array([1, 1, 1.5, .5, 1, 1.5, .5, 1]) * h)
                if mesh == 'graded' else {})
    grid = Simulation(domain=(8 * h,) * 3, dx=h, freq_max=12e9,
                      boundary='pec', cpml_layers=0, **profiles).freeze_mesh()
    builder = compute_smoothed_eps_nonuniform if mesh == 'graded' else compute_smoothed_eps
    index = (4, 3, 2 if mesh == 'graded' else 4)
    lines, widths = _lines(grid, component)
    point = tuple(a[i] for a, i in zip(lines, index))
    center = point[2]
    length = widths[2][index[2]]
    thickness = .4 * length

    def shape_at(z):
        if kind == 'box':
            return Box((-4 * h, -4 * h, z - thickness / 2),
                       (12 * h, 12 * h, z + thickness / 2))
        return Cylinder((4 * h, 4 * h, z), 3 * h, thickness, axis='z')

    shape = shape_at(center)
    lo, hi = ((shape.corner_lo[2], shape.corner_hi[2]) if kind == 'box'
              else (shape.center[2] - shape.height / 2, shape.center[2] + shape.height / 2))
    assert (lo + hi) / 2 == center == float(np.float32(center))
    # Independent overlap of the declared body with the centred support,
    # the same geometric expectation as test_thin_planar_body_fill.py.
    support_lo, support_hi = center - length / 2, center + length / 2
    p = max(0., min(hi, support_hi) - max(lo, support_lo)) / length
    expected = 1 / (p / 4 + 1 - p) if component == 2 else p * 4 + 1 - p
    with record_cell_table_capture(grid, [shape]) as captures:
        host = np.asarray(builder(grid, [(shape, 4.)]))
    traced = np.asarray(jax.jit(lambda z: jnp.stack(builder(grid, [(shape_at(z), 4.)])))(
        jnp.float32(center)))
    return dict(mesh=mesh, kind=kind, component=component, grid=grid, shape=shape,
                index=index, point=point, p=p, expected=expected, host=host,
                traced=traced, captures=captures)


def test_thin_midplane_closed_form(thin_midplane):
    """(iv), (v): normal harmonic mean and tangential arithmetic mean."""
    s = thin_midplane
    actual = float(s['host'][s['component']][s['index']])
    np.testing.assert_allclose(actual, s['expected'], rtol=1e-6,
                               err_msg=f"{s['mesh']} {s['kind']} point={s['point']} p={s['p']}")


def test_thin_midplane_traced_matches_plain(thin_midplane):
    s = thin_midplane
    for component in range(3):
        error = float(np.max(abs(s['host'][component] - s['traced'][component])))
        assert error <= 1e-5 * np.max(s['host']), (component, error)


def test_thin_midplane_mirror_pairs(thin_midplane):
    s = thin_midplane
    # Compare the whole arrays, and separately require mirrored samples on
    # the opposite-cap/face tie plane inside the body (not adjacent faces).
    _mirror(s['host'], s['grid'], s['shape'], f"thin-{s['mesh']}-{s['kind']}", require_ties=False)
    y = _lines(s['grid'], s['component'])[0][1]
    h = 2. ** -11
    pairs = [(i, j) for i, a in enumerate(y) for j, b in enumerate(y)
             if a < 4 * h and abs(a + b - 8 * h) <= 1e-12 and abs(a - 4 * h) <= h]
    assert len(pairs) > 0
    for i, j in pairs:
        x, _, z = s['index']
        np.testing.assert_allclose(s['host'][s['component'], x, i, z],
                                   s['host'][s['component'], x, j, z], rtol=0, atol=1e-5)


def test_thin_midplane_laminate_fill(thin_midplane):
    """(vi): captured fill is the body's own overlap, including at its centre."""
    s = thin_midplane
    fills = [c['fraction'][s['index']] for c in s['captures']
             if int(c['component']) == s['component']]
    assert len(fills) == 1
    np.testing.assert_allclose(fills[0], s['p'], rtol=1e-6,
                               err_msg=f"{s['mesh']} {s['kind']} point={s['point']}")
