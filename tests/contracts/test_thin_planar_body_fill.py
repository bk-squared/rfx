"""A planar dielectric body thinner than its support is averaged with both faces.

Scene: a 4 mm PEC box, background eps 1, and an eps 4 slab (a Box across x
and y, or a Cylinder disc on the z axis) 0.3 of its cell thick with both
faces inside one cell. A Yee sample's normal support is [-L/2, L/2]; where
one run of planar own bodies has both ends inside it, the sample's fill is
the run's own length over L and the outside medium is the declared content
on both sides. Closed forms are the laminate's over the support, formed here
from the declared faces and cell edges: normal 1/(sum p_i/eps_i), tangential
sum p_i*eps_i.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.csg import Cylinder
from rfx.geometry.smoothing import (
    compute_inv_eps_tensor_diag, compute_smoothed_eps, compute_smoothed_eps_nonuniform,
)
from rfx.grid import Grid
from tests._cell_table_capture import record_cell_table_capture

MM = 1e-3
EPS = 4.
# Uniform: 0.5 mm cells; the slab z 2.05-2.20 mm sits in cell [2.0, 2.5] (k = 4).
# Graded: dz 0.5, 0.5, 0.6, ... mm; the slab z 1.10-1.28 mm sits in cell [1.0, 1.6] (k = 2).
SCENES = {
    'uniform': dict(lo=2.05 * MM, hi=2.20 * MM, k=4, cell=.5 * MM, ez_cell=(2.0 * MM, 2.5 * MM),
                    node_support=(1.75 * MM, 2.25 * MM)),
    'graded': dict(lo=1.10 * MM, hi=1.28 * MM, k=2, cell=.6 * MM, ez_cell=(1.0 * MM, 1.6 * MM),
                   # the node at 1.0 mm has the centred support (0.5 + 0.6)/2 = 0.55 mm
                   node_support=(.725 * MM, 1.275 * MM)),
}
DZ_GRADED = np.array([.5, .5, .6, .4, .5, .6, .4, .5]) * MM


def _grid(mesh):
    profiles = (dict(dx_profile=np.full(8, .5 * MM), dy_profile=np.full(8, .5 * MM),
                     dz_profile=DZ_GRADED) if mesh == 'graded' else {})
    return Simulation(domain=(4 * MM,) * 3, dx=.5 * MM, freq_max=12e9, boundary='pec',
                      cpml_layers=0, **profiles).freeze_mesh()


def _build(mesh, grid, pairs):
    return (compute_smoothed_eps_nonuniform(grid, pairs) if mesh == 'graded'
            else compute_smoothed_eps(grid, pairs))


def _slab(shape, lo, hi):
    if shape == 'box':
        return Box((-.1, -.1, lo), (.1, .1, hi))
    # The disc's axis passes through the node at (2 mm, 2 mm); radius 1.5 mm.
    return Cylinder(center=(2 * MM, 2 * MM, (lo + hi) / 2), radius=1.5 * MM,
                    height=hi - lo, axis='z')


def _share(lo, hi, support):
    a, b = support
    return max(0., min(hi, b) - max(lo, a)) / (b - a)


def _normal(p, eps=EPS, out=1.):
    return 1 / (p / eps + (1 - p) / out)


def _tangential(p, eps=EPS, out=1.):
    return p * eps + (1 - p) * out


def _read(label, values):
    print(f'READ {label} ' + ' '.join(f'{v:.9g}' for v in values), file=sys.stderr, flush=True)


def _components(result, index):
    return [float(np.asarray(a)[index]) for a in result]


def _expected(mesh, lo, hi):
    scene = SCENES[mesh]
    p_z = _share(lo, hi, scene['ez_cell'])
    p_t = _share(lo, hi, scene['node_support'])
    return _tangential(p_t), _tangential(p_t), _normal(p_z)


@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
@pytest.mark.parametrize('shape', ['box', 'disc'])
@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_thin_slab_reads_its_own_length(mesh, shape, traced):
    """Ez of the slab's cell is 1/(p/4 + 1 - p), p = 0.3; Ex and Ey over the
    node's support are 4p + 1 - p (0.3 uniform; graded, one face in the
    centred 0.55 mm support, p = 0.175/0.55)."""
    scene = SCENES[mesh]
    grid = _grid(mesh)
    index = (4, 4, scene['k'])

    def evaluate(hi):
        return _build(mesh, grid, [(_slab(shape, scene['lo'], hi), EPS)])

    result = jax.jit(evaluate)(jnp.float32(scene['hi'])) if traced else evaluate(scene['hi'])
    values = _components(result, index)
    _read(f'thin-slab {mesh} {shape} {"traced" if traced else "host"} Ex Ey Ez', values)
    np.testing.assert_allclose(values, _expected(mesh, scene['lo'], scene['hi']),
                               rtol=2e-5 if traced else 1e-6)


def test_kottke_pec_dielectric_stage_reads_the_slab_length():
    scene = SCENES['uniform']
    grid = _grid('uniform')
    inverse = compute_inv_eps_tensor_diag(
        grid, dielectric_shapes=[(_slab('box', scene['lo'], scene['hi']), EPS)])
    values = [1 / v for v in _components(inverse, (4, 4, 4))]
    _read('kottke-pec uniform box host Ex Ey Ez', values)
    np.testing.assert_allclose(values, _expected('uniform', scene['lo'], scene['hi']), rtol=1e-6)


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_thin_slab_gradient_moves_with_its_own_length(mesh):
    """d Ez / d z_hi = (1 - 1/4) Ez^2 / L for f = (z_hi - z_lo)/L."""
    scene = SCENES[mesh]
    grid = _grid(mesh)

    def ez(hi):
        return _build(mesh, grid, [(_slab('box', scene['lo'], hi), EPS)])[2][4, 4, scene['k']]

    gradient = float(jax.grad(ez)(jnp.float32(scene['hi'])))
    expected = .75 * _normal(.3) ** 2 / scene['cell']
    _read(f'gradient {mesh} dEz/dz_hi per mm', [gradient * MM])
    np.testing.assert_allclose(gradient, expected, rtol=1e-4)


def _periodic_grid():
    return Grid(12e9, (4 * MM,) * 3, dx=.5 * MM, cpml_layers=0, periodic_axes='z')


def test_periodic_image_of_a_slab_reads_its_own_length():
    """The slab is declared at z 4.05-4.20 mm, past the 4 mm period; its image
    at 0.05-0.20 mm lies in the cell [0, 0.5] mm and in the seam node's
    support [-0.25, 0.25] mm."""
    grid = _periodic_grid()
    values = _components(compute_smoothed_eps(grid, [(_slab('box', 4.05 * MM, 4.20 * MM), EPS)]),
                         (4, 4, 0))
    _read('periodic image 4.05-4.20 Ex Ey Ez', values)
    np.testing.assert_allclose(values, [_tangential(.3), _tangential(.3), _normal(.3)], rtol=1e-6)


def test_slab_straddling_the_periodic_seam_reads_its_own_length():
    """Declared at z 3.95-4.10 mm: the seam node's support [-0.25, 0.25] mm
    holds the run [-0.05, 0.10] mm."""
    grid = _periodic_grid()
    values = _components(compute_smoothed_eps(grid, [(_slab('box', 3.95 * MM, 4.10 * MM), EPS)]),
                         (4, 4, 0))
    _read('periodic straddle 3.95-4.10 Ex Ey', values[:2])
    np.testing.assert_allclose(values[:2], [_tangential(.3)] * 2, rtol=1e-6)


# Where the slab's upper face leaves a support: uniform Ez (cell [2.0, 2.5])
# at 2.50 mm, uniform Ex/Ey (node 2.0, support [1.75, 2.25]) at 2.25 mm,
# graded Ez (cell [1.0, 1.6]) at 1.60 mm, graded Ex/Ey (node 1.0, support
# [0.725, 1.275]) at 1.275 mm.
SWITCHES = [('uniform', 2.50 * MM), ('uniform', 2.25 * MM),
            ('graded', 1.60 * MM), ('graded', 1.275 * MM)]


@pytest.mark.parametrize('mesh,switch', SWITCHES, ids=['uniform-ez', 'uniform-exy',
                                                     'graded-ez', 'graded-exy'])
def test_operands_continuous_where_the_far_face_leaves_the_support(mesh, switch):
    """Every operand moves by at most one float32 step of its peak when the
    far face crosses the support's end by 2e-12 m."""
    grid = _grid(mesh)
    lo = SCENES[mesh]['lo']
    below, above = (_build(mesh, grid, [(_slab('box', lo, switch + d), EPS)])
                    for d in (-1e-12, 1e-12))
    for name, a, b in zip(('Ex', 'Ey', 'Ez'), below, above):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        step = float(np.spacing(np.float32(max(np.max(np.abs(a)), np.max(np.abs(b))))))
        jump = float(np.max(np.abs(b - a)))
        _read(f'switch {mesh} {switch / MM:.3f} mm {name} max-jump step', [jump, step])
        assert jump <= step, (name, jump, step)


def test_realization_capture_reads_the_thin_fill():
    """The captured Ez fill at the slab's cell is 0.15/0.5 = 0.3."""
    scene = SCENES['uniform']
    grid = _grid('uniform')
    shape = _slab('box', scene['lo'], scene['hi'])
    with record_cell_table_capture(grid, [shape]) as calls:
        compute_smoothed_eps(grid, [(shape, EPS)])
    fills = [c['fraction'][4, 4, 4] for c in calls if int(c['component']) == 2]
    _read('captured Ez fill', fills)
    assert len(fills) == 1
    np.testing.assert_allclose(fills[0], _share(scene['lo'], scene['hi'], scene['ez_cell']),
                               rtol=1e-6)


@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_thin_slab_beside_another_medium_reads_both_sides(traced):
    """An eps 2 Box fills z 2.40-4.00 mm and is declared first, so the slab's
    pass writes the Ez sample of cell [2.0, 2.5] last. That cell holds air
    0.05 mm, slab 0.15, air 0.20 and eps 2 0.10: 1/((0.05 + 0.15/4 + 0.20 +
    0.10/2)/0.5)."""
    scene = SCENES['uniform']
    grid = _grid('uniform')
    far = (Box((-.1, -.1, 2.40 * MM), (.1, .1, 4 * MM)), 2.)

    def evaluate(hi):
        return compute_smoothed_eps(grid, [far, (_slab('box', scene['lo'], hi), EPS)])

    result = jax.jit(evaluate)(jnp.float32(scene['hi'])) if traced else evaluate(scene['hi'])
    value = float(np.asarray(result[2])[4, 4, 4])
    _read(f'beside eps 2 {"traced" if traced else "host"} Ez', [value])
    layers = [(.05, 1.), (.15, EPS), (.20, 1.), (.10, 2.)]
    expected = 1 / (sum(t / e for t, e in layers) / .5)
    np.testing.assert_allclose(value, expected, rtol=2e-5 if traced else 1e-6)
