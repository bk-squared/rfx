"""Stored slab drives: independent products, owner rows, scans and outer jit."""
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from rfx import Box, DebyePole, Simulation
from rfx.model.electric_metrics import material_drive_scales
from rfx.model.source_coefficients import drive_table, weight_drive_columns
from rfx.runners import distributed_nu as runner
from rfx.stepping.rank import mesh_ranks


def _bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype == np.float32
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def _ulp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    peak = np.max(np.abs(b))
    error = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    result = float(error / np.spacing(np.float32(peak)))
    print(f'peak_ulp={result:.9g}; differing={np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))}')
    return result


def test_t1_literal_products():
    literal = np.array([[1.1, -3., .125], [-.7, 2.1, 4.]], np.float32)
    weights = np.array([.3, .7, .13], np.float32)
    meta = [(1, 1, 1, c) for c in ('ez', 'hx', 'ex')]
    expected = literal.copy()
    expected[:, 0] = np.multiply(literal[:, 0], weights[0], dtype=np.float32)
    expected[:, 2] = np.multiply(literal[:, 2], weights[2], dtype=np.float32)
    _bits(weight_drive_columns(jnp.asarray(literal), weights, meta), expected)
    # Four incident half-filled cells leave exactly 1/16 on an E edge.
    expected = np.array([[.0625, -3., .0078125], [-.125, 2., .25]], np.float32)
    samples = [jnp.array(v, jnp.float32) for v in ([1., -2.], [-3., 2.], [.125, 4.])]
    _bits(drive_table(samples, meta, 2, occupancy=jnp.full((3, 3, 3), .5)), expected)


def _scene(kind='field', devices=2, seam=False, debye=False):
    sim = Simulation(freq_max=15e9, domain=(.020, .016, .012), dx=.001,
                     boundary='pec', dy_profile=np.array([.001] + [.0009, .0011] * 7 + [.001]))
    grid = sim._build_nonuniform_grid()
    x = ((grid.nx + devices - 1) // devices) * .001 if seam else .010
    pos = (x, .007, .005)
    sim.add_source(pos, 'ez', amplitude_kind=kind)
    sim.add_probe(pos, 'ez')
    sim.add_probe((.014, .009, .006), 'ez')
    lo, hi = (8, 13) if x >= .008 else (5, 10)
    if kind == 'current':
        sim.add_material('dielectric', eps_r=2.2,
                         **({'debye_poles': [DebyePole(.5, 1e-11)]} if debye else {}))
        sim.add(Box((lo * .001, .004, .003), (hi * .001, .011, .008)), material='dielectric')
    eps = jnp.ones(grid.shape, jnp.float32).at[lo:hi, 4:11, 3:8].set(2.2)
    return sim, eps


def _forward(sim, eps, devices, steps=12, **kw):
    return sim.forward(n_steps=steps, checkpoint=False, skip_preflight=True,
                       **(dict(eps_override=eps) if eps is not None else {}),
                       **(dict(distributed=True, devices=jax.devices('cpu')[:devices])
                          if devices > 1 else {}), **kw).time_series


@pytest.mark.parametrize('devices', [2, 3])
@pytest.mark.parametrize('seam', [False, True])
@pytest.mark.parametrize('kind', ['field', 'current'])
def test_t2_single_device_series(kind, seam, devices):
    if len(jax.devices('cpu')) < devices:
        pytest.skip('requires three forced host devices')
    sim, eps = _scene(kind, devices, seam)
    eps = eps if kind == 'current' else None
    for steps in (1, 12):
        single = _forward(sim, eps, 1, steps)
        slabs = _forward(sim, eps, devices, steps)
        if kind == 'field':
            _bits(slabs, single)
        else:
            assert _ulp(slabs, single) <= 9


def test_t2_nonowner_series_are_zero(monkeypatch):
    """Inspect the actual scan table too: an injection mask can hide bad storage."""
    original = runner.lax.scan
    seen = []
    def scan(fn, init, xs, *args, **kwargs):
        if isinstance(init, dict) and 'fdtd' in init and xs[1].ndim == 3:
            jax.debug.callback(lambda table: seen.append(np.asarray(table)), xs[1])
        return original(fn, init, xs, *args, **kwargs)
    monkeypatch.setattr(runner, 'lax', SimpleNamespace(**{**vars(jax.lax), 'scan': scan}))
    sim, _ = _scene()
    np.asarray(_forward(sim, None, 2))
    jax.effects_barrier()
    assert len(seen) == 1
    # x=10 mm is inside rank 0's 11 owned rows (21 total x nodes).
    assert seen[0].shape == (12, 2, 1)
    assert np.any(seen[0][:, 0, 0] != 0)
    assert np.all(seen[0][:, 1, 0] == 0)


def test_t3_local_material_read():
    mesh = Mesh(np.array(jax.devices('cpu')[:2]), ('x',))
    ranks = mesh_ranks(mesh)
    eps = jnp.arange(8 * 3 * 3, dtype=jnp.float32).reshape(8, 3, 3) / 20 + 1
    sigma = jnp.full_like(eps, .02)
    drives = ((0, 0, (2, 1, 1), 'ez', 1e-9, 0.),
              (1, 0, (1, 1, 1), 'ey', 1e-9, 0.))
    call = partial(material_drive_scales, mesh=mesh, drives=drives, dt=1e-12, ranks=ranks)
    local = jax.jit(partial(call, reduce_devices=False))
    values = np.asarray(local(eps, sigma))
    replicated = np.asarray(jax.jit(call)(eps, sigma))
    _bits(values[[0, 1], [0, 1]], replicated)
    _bits(values[[1, 0], [0, 1]], np.zeros(2, np.float32))
    hlo = local.lower(eps, sigma).compile().as_text()
    import re
    assert not re.search(r'\s(?:all-reduce|all-gather|all-to-all|collective-permute)(?:-start|-done)?\(', hlo)


@pytest.mark.parametrize('steps', [12, 13])
def test_t4_warmup_checkpoint(steps):
    sim, eps = _scene('current')
    plain = _forward(sim, eps, 2, steps)
    _bits(_forward(sim, eps, 2, steps, n_warmup=2), plain)
    _bits(_forward(sim, eps, 2, steps, checkpoint_every=2), plain)
    _bits(_forward(sim, eps, 2, steps, n_warmup=2, checkpoint_every=2), plain)
    def loss(e, checkpoint):
        return jnp.sum(_forward(sim, e, 2, steps, checkpoint_every=checkpoint) ** 2)
    _bits(loss(eps, 2), loss(eps, None))
    g0 = jax.grad(partial(loss, checkpoint=None))(eps)
    g2 = jax.grad(partial(loss, checkpoint=2))(eps)
    # Same Class A relative-gradient bar as test_distributed_nu_kernel.py's
    # test_distributed_checkpoint_every_grad_matches_no_segment.
    from tests._distributed_nu_tolerances import assert_class_a_grad
    assert_class_a_grad(g0, g2, label='slab_drive_checkpoint')


@pytest.mark.parametrize('kind', ['field', 'current'])
@pytest.mark.parametrize('devices', [1, 2])
def test_t5_outer_jit(kind, devices):
    sim, eps = _scene(kind, debye=kind == 'current')
    if kind == 'field':
        call = lambda: _forward(sim, None, devices)
        plain, compiled = call(), jax.jit(call)()
    else:
        call = lambda e: _forward(sim, e, devices)
        plain, compiled = call(eps), jax.jit(call)(eps)
    if np.asarray(plain).tobytes() == np.asarray(compiled).tobytes():
        _bits(compiled, plain)
    else:
        assert _ulp(compiled, plain) <= 9
