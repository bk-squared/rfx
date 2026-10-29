"""#1632 Addendum 1: realized precision, observable cross update and pad refusal."""

from rfx.geometry.rasterize_grid import unpadded_high_walls
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Cylinder, GaussianPulse, Simulation, Sphere


def test_mixed_tensor_uses_diagonal_permittivity_dtype(monkeypatch):
    import rfx.simulation as core
    import rfx.stepping.uniform_loop as loop
    observed = []
    make_step = loop.make_core_step
    def observe(ctx, *args, **kwargs):
        tensor = ctx.materials.components.tensor
        observed.append(tuple(value.dtype for value in tensor.eps))
        return make_step(ctx, *args, **kwargs)
    monkeypatch.setattr(loop, 'make_core_step', observe)
    monkeypatch.setattr(core, 'make_core_step', observe)
    for smoothing in (True, 'full_tensor'):
        sim = Simulation(freq_max=20e9, domain=(.012,) * 3, dx=.001,
                         boundary='cpml', cpml_layers=2, precision='mixed')
        sim.add_material('sphere', eps_r=4.)
        sim.add(Sphere((.00537, .00613, .00573), .0023), material='sphere')
        sim.add_source((.0031, .0042, .0043), 'ez',
                       waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='field')
        sim.add_probe((.0072, .0051, .0064), 'ez')
        result = sim.run(n_steps=200, skip_preflight=True, subpixel_smoothing=smoothing)
        assert np.all(np.isfinite(result.time_series))
        assert np.max(np.abs(result.time_series)) > 0
    print(f'mixed eps dtypes True/full_tensor={observed}')
    assert observed == [(jnp.dtype('float32'),) * 3] * 2


def test_one_e_update_observes_cross_coupling():
    from rfx.core.yee import init_materials, init_state
    from rfx.stepping.yee import update_e_aniso
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.geometry.smoothing import compute_smoothed_eps
    from rfx.grid import Grid
    from rfx.model.materials import with_components
    from rfx.model.tensor_operands import TensorOperands

    grid = Grid(domain=(.008,) * 3, dx=.001, freq_max=10e9, periodic_axes='xyz')
    sphere = Sphere((.0041, .0038, .0043), .0023)
    hz = jnp.broadcast_to(jnp.sin(jnp.arange(grid.shape[1], dtype=jnp.float32))[None, :, None], grid.shape)
    state = init_state(grid.shape)._replace(hz=hz)
    outputs = []
    for build in (compute_smoothed_eps, compute_node_tensor_eps):
        eps = build(grid, [(sphere, 4.)])
        materials = with_components(init_materials(grid.shape), grid, periodic=(True,) * 3,
                                    tensor=TensorOperands.of(eps))
        outputs.append(update_e_aniso(state, materials, *eps, grid.dt, grid.dx, periodic=(True,) * 3, high_wall=unpadded_high_walls(grid)))
    diagonal, full = outputs
    for component in ('ey', 'ez'):
        np.testing.assert_array_equal(getattr(diagonal, component), 0.)
        maximum = np.max(np.abs(getattr(full, component)))
        print(f'one E update {component} maximum={maximum}')
        assert maximum > 0


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_oblique_full_tensor_pad_refusal_is_specific(entry):
    sim = Simulation(freq_max=20e9, domain=(.024, .012, .012), dx=.001, cpml_layers=2)
    sim.add_material('rod', eps_r=4.)
    sim.add(Cylinder((.0121, .0058, .006), .0029, .1, 'z'), material='rod')
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, direction='+x', polarization='ez', angle_deg=30.)
    # The same realized material is invariant for diagonal smoothing.
    getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing=True)
    with pytest.raises(ValueError, match="full_tensor's diagonal is not invariant along a padded periodic axis"):
        getattr(sim, entry)(n_steps=1, skip_preflight=True, subpixel_smoothing='full_tensor')
