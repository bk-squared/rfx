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
        # length. Amendment 4 corrects |x|/L to ~100 for the +/-0.1 m
        # corners: one rounding gives ~6e-6 in fill, consistent with the
        # observed 7.5e-6 relative error. The 2e-5 bound is unchanged.
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


@pytest.mark.parametrize('path', ['uniform', 'nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_same_permittivity_mask_body_across_sdf_face(path, traced):
    """Review finding 1: the SDF surface lies inside a homogeneous eps=4 body."""
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()

    class MaskAbove:
        def mask(self, grid):
            return np.broadcast_to((np.arange(grid.shape[2]) >= 4)[None, None, :],
                                   grid.shape).copy()

    def evaluate(top):
        pairs = [(Box((-.1, -.1, -.1), (.1, .1, top)), 4.0), (MaskAbove(), 4.0)]
        if path == 'nodes':
            return tuple(1 / a for a in compute_node_tensor_eps(grid, pairs).nodes.xi[:3])
        return compute_smoothed_eps(grid, pairs)

    coordinate = jnp.float32(.0022) if traced else np.float64(.0022)
    # Host-f64 SDF arithmetic and traced-f32 arithmetic, per Amendment 4.
    rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
    result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
    for values in result:
        np.testing.assert_allclose(np.asarray(values)[4, 4, 4], 4.0, rtol=rtol)


@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_pec_face_reads_bulk_dielectric(traced):
    """Review finding 3: a PEC boundary cuts a cell containing eps=4.4."""
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()

    def evaluate(face):
        substrate = Box((-.1, -.1, -.1), (.1, .1, face))
        pec = Box((-.1, -.1, face), (.1, .1, .1))
        return compute_inv_eps_tensor_diag(
            grid, dielectric_shapes=[(substrate, 4.4)], pec_shapes=[pec])

    coordinate = jnp.float32(.0023) if traced else np.float64(.0023)
    rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
    result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
    # Ez is at 2.25 mm in the 2.0--2.5 mm cell; PEC starts at 2.3 mm.
    # Its geometric fraction is 0.4, leaving 0.6 of bulk dielectric.
    f_pec = (.0025 - .0023) / .0005
    expected = (1 - f_pec) / 4.4
    actual = float(np.asarray(result[2])[4, 4, 4])
    print(f'PEC_BULK_FACE traced={traced} inverse_eps={actual:.12g} expected={expected:.12g}')
    np.testing.assert_allclose(actual, expected, rtol=rtol)
