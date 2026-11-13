"""A traced shape on a periodic axis is admitted when its declared midpoint lies in [0, L].

A traced shape gets only the periodic images -1, 0 and 1. A shape centred
outside the base period can leave its copy across the opposite seam unread by
the smoothing samples there, which then read background in place of the body.
The rule is the midpoint, not the error: it also refuses declarations that the
readers fill correctly, and the remedy is a shift by a whole period.

Expected outcomes come from the midpoint m = (lo + hi) / 2 computed here from
the declaration, not from the implementation. Admitted rows include a period
shorter than two of the largest cells, a one-cell period and shapes centred on
the high seam. A traced shape inside [0, L] on a mesh with a traced z profile
runs as before.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.smoothing import compute_smoothed_eps, compute_smoothed_eps_nonuniform
from rfx.sources import GaussianPulse

MM = 1e-3
KW = dict(cpml_layers=0, boundary=BoundarySpec(x='periodic', y='pec', z='pec'))


def _uniform(L):
    return Simulation(20e9, (L, 6 * MM, 6 * MM), dx=MM, **KW).freeze_mesh(), compute_smoothed_eps, 3 * MM


def _graded(L):
    """x cells 1 mm (the period), y and z cells 3 mm."""
    grid = Simulation(20e9, (L, 12 * MM, 12 * MM), dx=MM, dy_profile=np.full(4, 3 * MM),
                      dz_profile=np.full(4, 3 * MM), **KW)._build_nonuniform_grid()
    return grid, compute_smoothed_eps_nonuniform, 6 * MM


def _box(c):
    """A Box between the traced x bounds; off-lattice faces across y and z."""
    return lambda lo, hi: Box((lo, c - 1.63e-3, c - 1.39e-3), (hi, c + 1.83e-3, c + 2.07e-3))


def _sphere(c):
    return lambda lo, hi: Sphere(((lo + hi) / 2, c + 0.13e-3, c - 0.11e-3), (hi - lo) / 2)


def _smoothed(mesh, L, kind):
    grid, fn, c = (_uniform if mesh == 'uniform' else _graded)(L)
    make = kind(c)
    return jax.jit(lambda lo, hi: jnp.stack([jnp.asarray(a) for a in fn(grid, [(make(lo, hi), 4.)])]))


REFUSED = [
    # [-L, 2L] admitted these with eps off by 0.81 (Ey 1.000 for 1.810), 0.318, and
    # 2.800 for 3.610 (a 0.13 mm gap at the seam); the last two read correctly.
    ('uniform', 10 * MM, 16.77 * MM, 19.77 * MM),
    ('graded', 10 * MM, -9.77 * MM, -6.77 * MM),
    ('uniform', 10 * MM, 9.9 * MM, 19.77 * MM),
    ('uniform', 10 * MM, 11 * MM, 14 * MM),
    ('uniform', 10 * MM, -9.77 * MM, 0.1 * MM),
]


@pytest.mark.parametrize('mesh, L, lo, hi', REFUSED,
                         ids=['high-end', 'graded-low-end', 'seam-gap', 'whole-period-out', 'low-overlap'])
def test_traced_shape_with_its_midpoint_outside_the_period_is_refused(mesh, L, lo, hi):
    m = (lo + hi) / 2
    assert not 0 <= m <= L
    with pytest.raises(Exception) as caught:
        jax.block_until_ready(_smoothed(mesh, L, _box)(jnp.float32(lo), jnp.float32(hi)))
    text = str(caught.value)
    found = re.search(r"periodic axis 'x': traced fill bounds \[(\S+), (\S+)\] m have their midpoint "
                      r"m=(\S+) m outside \[0, L\]; L=(\S+) m\.", text)
    assert found, text
    np.testing.assert_allclose([float(v) for v in found.groups()], [lo, hi, m, L], rtol=1e-6)
    assert 'Shift the shape by a whole period so its midpoint lies in [0, L]' in text
    assert 'also refuses some declarations that read correctly' in text


ADMITTED = [
    ('graded', 3 * MM, _sphere, 0.3 * MM, 2.7 * MM),   # period under two of the largest cells
    ('graded', 3 * MM, _box, 0.1 * MM, 2.9 * MM),
    ('uniform', 1 * MM, _box, 0.0, 1 * MM),            # a one-cell period
    ('uniform', 2 * MM, _box, 1.25 * MM, 2.75 * MM),   # midpoint on the high seam
    ('uniform', 10 * MM, _box, 8.5 * MM, 11.5 * MM),   # midpoint on the high seam
]


@pytest.mark.parametrize('mesh, L, kind, lo, hi', ADMITTED,
                         ids=['graded-3mm-sphere', 'graded-3mm-box', 'one-cell-period', 'seam-2mm', 'seam-10mm'])
def test_traced_shape_with_its_midpoint_in_the_period_is_admitted(mesh, L, kind, lo, hi):
    assert 0 <= (lo + hi) / 2 <= L
    eps = np.asarray(jax.block_until_ready(_smoothed(mesh, L, kind)(jnp.float32(lo), jnp.float32(hi))))
    assert np.isfinite(eps).all()


L_MESH, R = 10 * MM, 1.4 * MM
NOMINAL_Z = np.full(6, MM)


def _forward_on_traced_z_mesh(hi, d):
    profile = jnp.asarray(NOMINAL_Z, jnp.float32).at[2:4].add(jnp.stack([d, -d]))
    sim = Simulation(20e9, (L_MESH, 6 * MM, 6 * MM), dx=MM, dz_profile=profile,
                     nominal_profiles={'z': NOMINAL_Z}, **KW)
    sim.add_material('d', eps_r=4.)
    sim.add(Sphere((hi - R, 3.13e-3, 2.87e-3), R), material='d')
    sim.add_source((5.5e-3, 2.5e-3, 2.5e-3), 'ez', waveform=GaussianPulse(f0=10e9, bandwidth=0.8))
    sim.add_probe((7.5e-3, 3.5e-3, 3.5e-3), 'ez')
    return sim.forward(n_steps=4, subpixel_smoothing=False).time_series


def test_traced_shape_inside_the_period_runs_on_a_traced_mesh_axis():
    """A traced Sphere inside [0, L] with a traced z profile, staircase."""
    series = np.asarray(jax.block_until_ready(
        jax.jit(_forward_on_traced_z_mesh)(jnp.float32(L_MESH - 0.23e-3), jnp.float32(0.))))
    assert series.shape[0] == 4
    assert np.isfinite(series).all()
