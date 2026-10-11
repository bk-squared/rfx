"""W0: graded operands come from invariants and exchange hooks carry fields."""
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.nonuniform import _build_nu_scan, make_nonuniform_grid
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


def _setup(monkeypatch, widths):
    grid = make_nonuniform_grid(
        (.005, .005), np.full(5, .001), .001, cpml_layers=0,
        dx_profile=np.array(widths), dy_profile=np.full(5, .001))
    contexts = []
    make_step = graded.make_graded_step

    def capture(ctx):
        contexts.append(ctx)
        return make_step(ctx)

    with monkeypatch.context() as patch:
        patch.setattr(graded, 'make_graded_step', capture)
        setup = _build_nu_scan(
            grid, init_materials(grid.shape), 40,
            sources=[(2, 2, 2, 'ez', jnp.sin(jnp.arange(40) * .3))],
            probes=[(2, 2, 2, 'ez'), (3, 2, 2, 'hy')],
            pec_faces={f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')})
    return contexts[0], setup


def _record(step, setup, invariants):
    def observed(carry, xs):
        new, probes = step(carry, xs, invariants)
        return new, (tuple(getattr(new['fdtd'], c) for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')), probes)

    return jax.jit(lambda carry: jax.lax.scan(
        observed, carry, (jnp.arange(40), setup.src_waveforms)))(setup.carry_init)


def test_t1_context_a_with_b_invariants_equals_b_bitwise(monkeypatch):
    _live_check()
    ctx_a, a = _setup(monkeypatch, [.001, .0008, .0012, .001, .001])
    ctx_b, b = _setup(monkeypatch, [.001, .0012, .0008, .001, .001])
    assert ctx_a.dt == ctx_b.dt
    assert ctx_a.grid.shape == ctx_b.grid.shape
    for name in ('inv_dy', 'inv_dz', 'inv_dy_h', 'inv_dz_h'):
        _check_bits(getattr(ctx_a, name), getattr(ctx_b, name))
    assert not {'inv_dx', 'inv_dx_h', 'drives', 'prb_meta'} & {f.name for f in fields(ctx_a)}
    moved = ('inv_dx', 'inv_dx_h', 'dx_arr', 'drives', 'probe_table', 'materials')
    invariants = {**a.invariants, **{name: b.invariants[name] for name in moved}}
    _check_bits(_record(graded.make_graded_step(ctx_a), b, invariants),
                _record(graded.make_graded_step(ctx_b), b, b.invariants))


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
    from rfx.core.yee import update_e_nu
    from rfx.boundaries.pec import apply_pec_faces

    def write(frame):
        frame.st = frame.st._replace(hy=frame.st.hy.at[2, 2, 2].add(.125))

    initial = setup.carry_init['fdtd']
    altered = initial._replace(hy=initial.hy.at[2, 2, 2].add(.125))
    expected = update_e_nu(altered, setup.invariants['materials'], ctx.dt,
                           setup.invariants['inv_dx'], ctx.inv_dy, ctx.inv_dz,
                           boundary=ctx.curl_boundary)
    expected = apply_pec_faces(expected, ctx._pec_faces_frozen)
    step = graded.make_graded_step(ctx, hooks={HookPoint.AFTER_H: (write,)})
    actual, _ = jax.jit(lambda carry: step(
        carry, (jnp.int32(0), jnp.zeros(1)), setup.invariants))(setup.carry_init)
    assert np.count_nonzero(np.asarray(expected.ez)) > 0
    _check_bits(actual['fdtd'], expected)
