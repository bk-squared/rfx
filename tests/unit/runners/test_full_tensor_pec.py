"""#1632 Addendum 1: free-edge couplings and late-time PEC stability."""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, GaussianPulse, Simulation, Sphere
from rfx.model.smoothing_admission import FULL_TENSOR_MESSAGE


def slab_sim():
    sim = Simulation(freq_max=20e9, domain=(.012,) * 3, dx=.001,
                     boundary='pec', cpml_layers=0)
    sim.add_material('hi', eps_r=38.)
    sim.add(Box((0, 0, .002), (.012, .012, .004)), material='pec')
    sim.add(Sphere((.0061, .0058, .0052), .0029), material='hi')
    sim.add_source((.0031, .0042, .0083), 'ez',
                   waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='current')
    sim.add_probe((.0072, .0051, .0064), 'ex')
    sim.add_probe((.0062, .0061, .0049), 'ez')
    return sim


def test_internal_pec_operator_is_symmetric():
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.core.node_tensor import off_diagonal
    from rfx.core.yee import init_materials
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.grid import Grid
    from rfx.model.materials import with_components
    from rfx.model.node_pec import restrict_node_pec
    from rfx.model.tensor_operands import TensorOperands

    with jax.enable_x64():
        grid = Grid(domain=(.006, .006, .001), dx=.001, freq_max=10e9,
                    mode='2d_tez', cpml_axes='', periodic_axes='xy')
        interface = Cylinder((.0024, .0023, .0005), .0017, 1., 'z')
        eps = compute_node_tensor_eps(grid, [(interface, 38.)], field_dtype=jnp.float64)
        mask = Box((.002, 0, 0), (.003, .006, .001)).mask(grid)
        periodic = (True, True, True)
        edges = realized_pec_edge_masks(mask, periodic=periodic)
        materials = with_components(init_materials(grid.shape), grid, periodic=periodic,
                                    tensor=TensorOperands.of(eps))
        nodes = restrict_node_pec(materials, edges, ()).components.tensor.node_smoothing
        count = 3 * int(np.prod(grid.shape))
        def apply(vector):
            curls = tuple(vector.reshape((3,) + grid.shape))
            result = off_diagonal(curls, nodes, periodic=periodic)
            # The step's independently realized output PEC projection.
            return jnp.stack(tuple(jnp.where(edge, 0., value)
                                   for value, edge in zip(result, edges))).ravel()
        operator = np.asarray(jax.jit(jax.vmap(apply))(jnp.eye(count))).T
        maximum = np.max(np.abs(operator))
        assert maximum > 0
        relative = np.max(np.abs(operator - operator.T)) / maximum
        print(f'internal PEC symmetry ratio={relative:.17g}')
        assert relative <= 1e-12


def test_internal_pec_slab_does_not_grow():
    # Unfixed HEAD e68513da7 exceeds 10x the second tenth at step 15701.
    # 16000 steps took 6.20 s on the review Mac, including setup/compilation.
    result = slab_sim().run(n_steps=16000, skip_preflight=True, subpixel_smoothing='full_tensor')
    amplitude = np.max(np.abs(np.asarray(result.time_series)), axis=1)
    assert np.all(np.isfinite(amplitude))
    second, last = np.max(amplitude[1600:3200]), np.max(amplitude[14400:])
    print(f'slab second tenth={second}, last tenth={last}, ratio={last / second}')
    assert second > 0
    assert last <= 2 * second


@pytest.mark.parametrize(('entry', 'declared'), [('run', False), ('run', True), ('forward', True)],
                         ids=['run-keyword', 'run-boundary', 'forward-boundary'])
def test_full_tensor_conformal_refuses(entry, declared):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    sim = slab_sim()
    if declared:
        sim._boundary_spec = BoundarySpec(x=Boundary('pec', 'pec', conformal=True),
                                          y='pec', z='pec')
    options = {} if declared else dict(conformal_pec=True)
    with pytest.raises(NotImplementedError, match=re.escape(FULL_TENSOR_MESSAGE) + '.*conformal PEC'):
        getattr(sim, entry)(n_steps=1, skip_preflight=True,
                            subpixel_smoothing='full_tensor', **options)
