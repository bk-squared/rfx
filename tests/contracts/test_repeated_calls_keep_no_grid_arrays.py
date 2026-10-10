"""A repeated forward() or gradient call must not leave grid-sized arrays alive (rfx tracker 1606).

A sweep, a timing loop or an optimisation loop calls forward() many times in
one process. Each call may leave compiled programs in JAX's caches, but no
array of the lattice's size: before the fix a graded-mesh run with CPML left
nine per call (0.65 GiB per call at 19 M cells), held by per-call closures
that JAX keeps with a ``custom_jvp`` function.
"""
import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation

# An odd lattice no other test in the process is likely to share.
DOMAIN = (.011, .013, .009)


def _sim(lane, boundary):
    kw = {}
    if lane == 'graded':
        kw['dy_profile'] = np.array([.001] + [.0009, .0011] * 5 + [.001, .001])
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=.001, boundary=boundary, **kw)
    sim.add_source((.003, .005, .004), 'ez', amplitude_kind='field')
    sim.add_probe((.008, .007, .005), 'ez')
    return sim


def _grid_shape(lane, boundary):
    sim = _sim(lane, boundary)
    grid = sim._build_nonuniform_grid() if lane == 'graded' else sim._build_grid()
    return tuple(int(v) for v in grid.shape)


def _live(shape):
    """Live arrays of the lattice's size; a slab of a multi-device run keeps the two transverse lengths."""
    gc.collect()
    return sum(1 for a in jax.live_arrays() if a.ndim >= 3 and tuple(a.shape[-2:]) == shape[-2:])


def _counts(call, shape, repeats=3):
    counts = []
    for _ in range(repeats):
        call()
        counts.append(_live(shape))
    return counts


@pytest.mark.parametrize('lane, boundary', [('uniform', 'cpml'), ('graded', 'pec'), ('graded', 'cpml')])
def test_repeated_forward_keeps_no_grid_arrays(lane, boundary):
    shape = _grid_shape(lane, boundary)

    def call():
        result = _sim(lane, boundary).forward(n_steps=8, checkpoint=False, skip_preflight=True)
        result.time_series.block_until_ready()

    counts = _counts(call, shape)
    assert counts[1] == counts[0] and counts[2] == counts[0], counts


@pytest.mark.parametrize('lane, boundary', [('graded', 'pec'), ('uniform', 'cpml')])
def test_repeated_eager_gradient_keeps_no_grid_arrays(lane, boundary):
    """What ``optimize()`` does by default: an eager value_and_grad per iteration."""
    shape = _grid_shape(lane, boundary)
    occupancy = jnp.zeros(shape, jnp.float32).at[4:6, 4:6, 3:5].set(0.2)

    def loss(occ):
        series = _sim(lane, boundary).forward(
            n_steps=8, skip_preflight=True, pec_occupancy_override=occ).time_series
        return jnp.sum(series ** 2)

    def call():
        value, grad = jax.value_and_grad(loss)(occupancy)
        grad.block_until_ready()

    counts = _counts(call, shape)
    assert counts[1] == counts[0] and counts[2] == counts[0], counts


@pytest.mark.parametrize('lane, boundary', [('uniform', 'cpml'), ('graded', 'cpml')])
def test_repeated_run_keeps_no_grid_arrays(lane, boundary):
    shape = _grid_shape(lane, boundary)

    def call():
        result = _sim(lane, boundary).run(n_steps=8, skip_preflight=True)
        jax.block_until_ready(result.time_series)

    counts = _counts(call, shape)
    assert counts[1] == counts[0] and counts[2] == counts[0], counts


def test_repeated_two_device_forward_keeps_no_grid_arrays():
    if len(jax.devices()) < 2:
        pytest.skip('requires two devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)')
    shape = _grid_shape('graded', 'cpml')

    def call():
        result = _sim('graded', 'cpml').forward(
            n_steps=8, skip_preflight=True, distributed=True, devices=jax.devices()[:2])
        result.time_series.block_until_ready()

    counts = _counts(call, shape)
    assert counts[1] == counts[0] and counts[2] == counts[0], counts


def test_optimize_iterations_keep_no_grid_arrays():
    """The default (eager) optimize() loop: more iterations leave no more arrays than one."""
    from rfx.optimize import DesignRegion, optimize
    domain = (.011, .013, .009)

    def run(n_iters):
        sim = Simulation(freq_max=15e9, domain=domain, dx=.001, boundary='cpml')
        sim.add_port((.003, .005, .004), 'ez')
        sim.add_probe((.008, .007, .005), 'ez')
        region = DesignRegion(corner_lo=(.004, .004, .003), corner_hi=(.007, .008, .006), eps_range=(1.0, 4.4))
        optimize(sim, region, lambda r: -jnp.sum(r.time_series ** 2), n_iters=n_iters, lr=0.2,
                 n_steps=8, verbose=False, skip_preflight=True)
        grid = sim._build_grid()
        return tuple(int(v) for v in grid.shape)

    shape = run(1)
    one = _live(shape)
    run(3)
    assert _live(shape) == one, (one, _live(shape))


@pytest.mark.gpu_gate
def test_device_memory_is_flat_over_repeated_forward_calls_on_this_backend():
    """On a GPU: bytes in use after the second and third call equal those after the first, within one array."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        stats = None
    if not stats or stats.get('bytes_in_use') is None:
        pytest.skip('this backend reports no device memory statistics')
    domain = (.060, .060, .060)
    profile = np.array([.001] + [.0009, .0011] * 29 + [.001])

    def call():
        sim = Simulation(freq_max=15e9, domain=domain, dx=.001, boundary='cpml', dy_profile=profile)
        sim.add_source((.018, .027, .024), 'ez', amplitude_kind='field')
        sim.add_probe((.042, .033, .030), 'ez')
        result = sim.forward(n_steps=8, checkpoint=False, skip_preflight=True)
        result.time_series.block_until_ready()
        return int(np.prod(sim._build_nonuniform_grid().shape))

    used = []
    for _ in range(3):
        cells = call()
        gc.collect()
        used.append(jax.local_devices()[0].memory_stats()['bytes_in_use'])
    one_array = 4 * cells
    # Before the fix each call kept nine such arrays.
    assert used[1] - used[0] <= one_array and used[2] - used[0] <= one_array, (used, one_array)

