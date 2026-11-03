"""Bulk media on both sides of a shared face, without time stepping."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.smoothing import (
    compute_inv_eps_tensor_diag, compute_smoothed_eps, compute_smoothed_eps_nonuniform,
)


@pytest.mark.parametrize('path', ['uniform', 'graded', 'kottke_pec', 'nodes', 'graded_nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
@pytest.mark.parametrize('reverse', [False, True], ids=['AB', 'BA'])
def test_shared_face_operands(path, traced, reverse):
    graded = path.startswith('graded')
    profiles = dict(dx_profile=np.full(8, .0005), dy_profile=np.full(8, .0005),
                    dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    nodes = np.asarray(sim.realized_geometry().nodes[2])
    widths = np.r_[np.diff(nodes), np.diff(nodes)[-1]]
    # Different E components sample different z positions. Include half fill
    # and samples on either side of the shared face.
    faces = [nodes[4], nodes[4]+widths[4]/2, nodes[4]-.1*widths[4]]
    for face in faces:
        def evaluate(z):
            pairs = [(Box((-.1, -.1, -.1), (.1, .1, z)), 3.38),
                     (Box((-.1, -.1, z), (.1, .1, .1)), 10.2)]
            if reverse:
                pairs.reverse()
            if path.endswith('nodes'):
                return compute_node_tensor_eps(grid, pairs).nodes.xi[:3]
            if path == 'kottke_pec':
                return tuple(1/a for a in compute_inv_eps_tensor_diag(grid, dielectric_shapes=pairs))
            return (compute_smoothed_eps_nonuniform(grid, pairs) if graded
                    else compute_smoothed_eps(grid, pairs))
        coordinate = jnp.float32(face) if traced else np.float64(face)
        # Amendment 3: host SDF arithmetic is float64 (1e-6 relative);
        # traced float32 subtracts box coordinates before dividing by cell
        # length: epsilon32*|x|/L ~ 1.7e-6, with relative sensitivity ~ 1–2.
        # The amendment bounds the error at ~1e-5 and uses twice that bound.
        # Returned arrays alone do not identify the
        # arithmetic dtype: the host path also casts its output to float32.
        rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
        result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
        for axis, values in enumerate(result):
            is_node = path.endswith('nodes')
            position = nodes[4] + (widths[4]/2 if axis == 2 and not is_node else 0.)
            length = (widths[4] if axis == 2 and not is_node else
                      (widths[3]+widths[4])/2)
            f = np.clip(.5+(position-face)/length, 0., 1.)
            expected = 1/(f/10.2+(1-f)/3.38) if axis == 2 else f*10.2+(1-f)*3.38
            if is_node:
                expected = 1/expected
            np.testing.assert_allclose(np.asarray(values)[4, 4, 4], expected, rtol=rtol)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('traced', [False, True])
def test_mask_only_outside_is_clamped_primal_cell(graded, traced):
    profiles = dict(dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    class MaskOnly:
        def __init__(self, enabled):
            self.enabled = enabled
        def mask(self, grid):
            # The low boundary cell owns points clamped from below the grid.
            return jnp.broadcast_to((jnp.arange(grid.shape[2]) == 0)[None, None, :], grid.shape) & self.enabled
    def evaluate(enabled):
        shapes = [(MaskOnly(enabled), 10.2),
                  (Box((-.1, -.1, 0.), (.1, .1, .003)), 3.38)]
        return (compute_smoothed_eps_nonuniform(grid, shapes) if graded
                else compute_smoothed_eps(grid, shapes))
    result = jax.jit(evaluate)(True) if traced else evaluate(True)
    np.testing.assert_allclose(np.asarray(result[0])[4, 4, 0], (3.38+10.2)/2, rtol=1e-6)
