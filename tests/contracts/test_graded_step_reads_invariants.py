"""W0: graded operands come from invariants and exchange hooks carry fields."""

from rfx.geometry.rasterize_grid import unpadded_high_walls
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.stepping.graded_setup import _build_nu_scan
from rfx.nonuniform import make_nonuniform_grid
from rfx.stepping import HookPoint
from rfx.stepping import graded


def _check_bits(actual, expected):
    left, right = jax.tree.leaves(actual), jax.tree.leaves(expected)
    assert len(left) == len(right)
    for got, want in zip(left, right, strict=True):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert got.tobytes() == want.tobytes()


def _live_check():
    with pytest.raises(AssertionError):
        _check_bits(jnp.array([1.]), jnp.array([0.]))


def _setup(monkeypatch, widths, branch='ordinary', probes=((2, 2, 2, 'ez'), (3, 2, 2, 'hy'))):
    grid = make_nonuniform_grid(
        (.005, .005), np.full(5, .001), .001, cpml_layers=0,
        dx_profile=np.array(widths), dy_profile=np.full(5, .001))
    contexts = []
    make_step = graded.make_graded_step
    materials = init_materials(grid.shape)
    options = {}
    if branch == 'debye':
        from rfx.materials.debye import DebyePole, init_debye
        options['debye'] = init_debye([DebyePole(delta_eps=2., tau=5e-12)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
    elif branch == 'lorentz':
        from rfx.materials.lorentz import init_lorentz, lorentz_pole
        # 20 GHz: the 30 GHz pole was above the check's bound (growth 1.34 per step
        # on the smallest cells); this test reads invariants only.
        options['lorentz'] = init_lorentz(
            [lorentz_pole(delta_eps=1.5, omega_0=2 * np.pi * 20e9, delta=1e10)], materials, grid.dt, high_wall=unpadded_high_walls(grid))
    elif branch == 'anisotropic':
        options['aniso_eps'] = tuple(jnp.full(grid.shape, value) for value in (1.5, 2., 2.5))

    def capture(ctx):
        contexts.append(ctx)
        return make_step(ctx)

    with monkeypatch.context() as patch:
        patch.setattr(graded, 'make_graded_step', capture)
        setup = _build_nu_scan(
            grid, materials, 40, **options,
            sources=[(2, 2, 2, 'ez', jnp.sin(jnp.arange(40) * .3))],
            probes=list(probes),
            pec_faces={f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')})
    return contexts[0], setup


def _record(step, setup, invariants):
    def observed(carry, xs):
        new, probes = step(carry, xs, invariants)
        return new, (tuple(getattr(new['fdtd'], c) for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')), probes)

    return jax.jit(lambda carry: jax.lax.scan(
        observed, carry, (jnp.arange(40), setup.src_waveforms)))(setup.carry_init)


@pytest.mark.parametrize('branch', ('ordinary', 'debye', 'lorentz', 'anisotropic'))
def test_t1_context_a_with_b_invariants_equals_b_bitwise(monkeypatch, branch):
    _live_check()
    ctx_a, a = _setup(monkeypatch, [.001, .0008, .0012, .001, .001], branch)
    ctx_b, b = _setup(monkeypatch, [.001, .0012, .0008, .001, .001], branch)
    assert ctx_a.dt == ctx_b.dt
    assert ctx_a.grid.shape == ctx_b.grid.shape
    for name in ('inv_dy', 'inv_dz', 'inv_dy_h', 'inv_dz_h'):
        _check_bits(getattr(ctx_a, name), getattr(ctx_b, name))
    assert not {'inv_dx', 'inv_dx_h', 'drives', 'prb_meta'} & {f.name for f in fields(ctx_a)}
    moved = ('inv_dx', 'inv_dx_h', 'dx_arr', 'drives', 'probe_table', 'materials')
    invariants = {**a.invariants, **{name: b.invariants[name] for name in moved}}
    _check_bits(_record(graded.make_graded_step(ctx_a), b, invariants),
                _record(graded.make_graded_step(ctx_b), b, b.invariants))


@pytest.mark.parametrize('component', ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))
def test_probe_table_negative_indices_match_python_indexing(component):
    from rfx.core.yee import init_state
    from rfx.measurement.projector import project

    shape = (5, 6, 7)
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1
    state = init_state(shape)._replace(**{component: values})
    nodes = [(-1, -2, -3), (-5, -6, -7), (0, 1, 2), (-1, -2, -3)]
    table = graded.graded_probe_table([(*node, component) for node in nodes], shape, dtype=state.ex.dtype)
    expected = jnp.stack([values[node] for node in nodes])
    _check_bits(jax.jit(lambda st: project(st, table))(state), expected)


@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('side', ('low', 'high'))
def test_probe_table_outside_python_bounds_names_probe(axis, side):
    shape = (5, 6, 7)
    node = [0, 0, 0]
    node[axis] = -shape[axis] - 1 if side == 'low' else shape[axis]
    with pytest.raises(IndexError, match=r'Probe 1 \(hy,'):
        graded.graded_probe_table([(0, 0, 0, 'ez'), (*node, 'hy')], shape, dtype=jnp.float32)


def test_t3_step_end_field_write_is_carried(monkeypatch):
    _live_check()
    ctx, setup = _setup(monkeypatch, [.001, .0008, .0012, .001, .001])
    xs = (jnp.int32(0), jnp.array([.7], dtype=jnp.float32))
    initial = setup.carry_init['fdtd']

    def write(frame):
        frame.st = frame.st._replace(ez=frame.st.ez.at[2, 2, 2].set(7.))

    step = graded.make_graded_step(ctx, hooks={HookPoint.STEP_END: (write,)})
    actual, _ = jax.jit(lambda carry: step(carry, xs, setup.invariants))(setup.carry_init)
    expected = initial._replace(ez=initial.ez.at[2, 2, 2].set(7.), step=initial.step + 1)
    _check_bits(actual['fdtd'], expected)


def test_t3_after_h_write_is_read_by_e_update(monkeypatch):
    _live_check()
    ctx, setup = _setup(monkeypatch, [.001, .0008, .0012, .001, .001])
    from rfx.stepping.yee import update_e_nu
    from rfx.boundaries.pec import apply_pec_faces

    def write(frame):
        frame.st = frame.st._replace(hy=frame.st.hy.at[2, 2, 2].add(.125))

    initial = setup.carry_init['fdtd']
    altered = initial._replace(hy=initial.hy.at[2, 2, 2].add(.125))
    expected = update_e_nu(altered, setup.invariants['materials'], ctx.dt,
                           setup.invariants['inv_dx'], ctx.inv_dy, ctx.inv_dz,
                           boundary=ctx.curl_boundary, high_wall=(False, False, False))
    expected = apply_pec_faces(expected, ctx._pec_faces_frozen)
    step = graded.make_graded_step(ctx, hooks={HookPoint.AFTER_H: (write,)})
    actual, _ = jax.jit(lambda carry: step(
        carry, (jnp.int32(0), jnp.zeros(1)), setup.invariants))(setup.carry_init)
    assert np.count_nonzero(np.asarray(expected.ez)) > 0
    _check_bits(actual['fdtd'], expected)


def _count(jaxpr, primitive):
    total = 0
    for eqn in jaxpr.eqns:
        total += eqn.primitive.name == primitive
        for value in eqn.params.values():
            for sub in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(sub, 'jaxpr', sub)
                if hasattr(inner, 'eqns'):
                    total += _count(inner, primitive)
    return total


def test_host_probe_table_adds_no_gather(monkeypatch):
    """One device keeps the pre-W0 probe read: static node indexing, not the projector's gather.

    The gather read changed the reverse-mode program, so gradients through the probe record moved
    in their last bits (W0 declaration, record of 2026-10-12, option A).
    """
    widths = [.001, .0008, .0012, .001, .001]
    with_probes = _setup(monkeypatch, widths)
    without = _setup(monkeypatch, widths, probes=())

    def gathers(ctx, setup):
        step = graded.make_graded_step(ctx)
        xs = (jnp.int32(0), setup.src_waveforms[0])
        return _count(jax.make_jaxpr(lambda carry: step(carry, xs, setup.invariants))(
            setup.carry_init).jaxpr, 'gather')

    assert gathers(*with_probes) == gathers(*without)
    # Live: the projector route reads with a gather, and the count sees it.
    monkeypatch.setattr(graded, '_host_probe_nodes', lambda table: None)
    assert gathers(*with_probes) > gathers(*without)


def test_probe_record_follows_declaration_order(monkeypatch):
    """Each record column is its declared probe, read from the state the step returns."""
    _live_check()
    probes = ((3, 2, 2, 'hy'), (2, 2, 2, 'ez'), (1, 3, 2, 'ex'), (2, 2, 2, 'ez'), (2, 1, 3, 'hx'))
    ctx, setup = _setup(monkeypatch, [.001, .0008, .0012, .001, .001], probes=probes)
    step = graded.make_graded_step(ctx)
    carry = setup.carry_init
    for n in range(6):   # a few steps, so every probed component is nonzero
        carry, record = jax.jit(lambda c, n=n: step(c, (jnp.int32(n), setup.src_waveforms[n]),
                                                    setup.invariants))(carry)
    state = carry['fdtd']
    expected = jnp.stack([getattr(state, c)[i, j, k] for i, j, k, c in probes])
    assert np.count_nonzero(np.asarray(expected)) == len(probes)
    _check_bits(record, expected)
