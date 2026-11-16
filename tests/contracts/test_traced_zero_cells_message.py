"""A traced dielectric shape that owns no cell is refused with its own message under jax.jit.

The empty-mask check of a traced shape runs as a host callback after tracing.
It never reads the shape there: a traced radius belongs to the enclosing trace.
A Box, Sphere or Cylinder hands its bounds to the check as operands, so the
message prints the drawn thickness; any other shape prints it as unavailable.
Eager refusals and admitted traced shapes are unchanged.

The scene: a 3 x 12 x 12 mm PEC box with 1 mm cells (graded: 3 mm cells along
y and z) and one dielectric shape whose radius is traced. A disc 0.2 mm thick
along x, centred on the x node at 1 mm, lies between the cell centres at 0.5
and 1.5 mm, so it owns no cell; 2 mm thick, it owns the centre at 1.5 mm.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Cylinder, GaussianPulse, Simulation
from rfx.errors import RfxRefusal
from rfx.geometry.csg import PolylineWire

THIN = 0.2e-3
REFUSED = 'rasterizes to ZERO cells'
PULSE = GaussianPulse(f0=12e9, bandwidth=0.8)


def _series(shape_of, graded=False):
    def run(radius):
        kw = dict(dy_profile=np.full(4, 3e-3), dz_profile=np.full(4, 3e-3)) if graded else {}
        sim = Simulation(20e9, (3e-3, 12e-3, 12e-3), dx=1e-3, cpml_layers=0, boundary='pec', **kw)
        sim.add_material('diel', eps_r=4.)
        sim.add(shape_of(radius), material='diel')
        sim.add_source((1.3e-3, 4.1e-3, 5.3e-3), 'ex', waveform=PULSE)
        sim.add_probe((2.2e-3, 7.7e-3, 6.9e-3), 'ex')
        return sim.forward(n_steps=20, skip_preflight=True).time_series
    return run


def _disc(thickness):
    return lambda radius: Cylinder((1.0e-3, 6.1e-3, 6.3e-3), radius, thickness, axis='x')


def _x_thickness(text):
    found = re.search(r'drawn thickness \(x, y, z\) = \(([^,]+),', text)
    assert found, text
    return float(found.group(1))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_traced_disc_owning_no_cell_is_refused_with_its_thickness(graded):
    with pytest.raises(Exception) as caught:
        jax.block_until_ready(jax.jit(_series(_disc(THIN), graded))(jnp.float32(4e-3)))
    text = str(caught.value)
    assert REFUSED in text and 'UnexpectedTracerError' not in text
    assert 'drawn thickness (x, y, z) = (0.0002' in text
    assert abs(_x_thickness(text) - THIN) <= 1e-9


def test_eager_disc_owning_no_cell_keeps_its_refusal():
    with pytest.raises(RfxRefusal, match=REFUSED) as caught:
        _series(_disc(THIN))(4e-3)
    assert abs(_x_thickness(str(caught.value)) - THIN) <= 1e-9


def test_traced_disc_owning_cells_runs():
    series = np.asarray(jax.block_until_ready(
        jax.jit(_series(lambda r: Cylinder((1.5e-3, 6.1e-3, 6.3e-3), r, 2e-3, axis='x')))(jnp.float32(4e-3))))
    assert series.shape == (20, 1) and np.isfinite(series).all()


def test_traced_wire_owning_no_cell_is_refused_with_bounds_unavailable():
    """A PolylineWire's bounds are not formed in the trace."""
    wire = _series(lambda r: PolylineWire(((1.0e-3, 6.1e-3, 1.0e-3), (1.0e-3, 6.1e-3, 11.0e-3)), r * .01))
    with pytest.raises(Exception) as caught:
        jax.block_until_ready(jax.jit(wire)(jnp.float32(4e-3)))
    text = str(caught.value)
    assert REFUSED in text and 'UnexpectedTracerError' not in text
    assert 'drawn thickness (x, y, z) = unavailable' in text


def test_traced_wire_owning_cells_runs():
    wire = _series(lambda r: PolylineWire(((1.5e-3, 4.5e-3, 1.0e-3), (1.5e-3, 4.5e-3, 11.0e-3)), r * .5))
    series = np.asarray(jax.block_until_ready(jax.jit(wire)(jnp.float32(4e-3))))
    assert series.shape == (20, 1) and np.isfinite(series).all()
