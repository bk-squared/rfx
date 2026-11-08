"""Target selection and returned poles of the synthetic closed-box fixture."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.ringdown import _jax as rj
from rfx.measurement.ringdown_resolution import completion_poles
from tests._x64_compat import enable_x64
from tests.unit.sparams.test_ringdown_run import _box, FREQS


def test_pole_target_refuses_longer_record():
    with pytest.raises(ValueError, match='pole_target.*against="longer_record"'):
        rd.gradient_witness(1., 1., pole_target=10., against="longer_record")


def test_returned_mask_preserves_unresolved_pole(monkeypatch):
    def host_poles(identify_window, window, k_max, dt, freqs):
        _, cdt = rj.working_dtypes()
        poles = jnp.asarray([-1e7 + 2j * np.pi * 5e9,
                             -1e7 + 2j * np.pi * 12.4e9,
                             np.log(rj.INERT_LAMBDA) / dt], cdt)
        return (poles, jnp.asarray([1, 2, 0], jnp.int32), jnp.int32(rj.STATUS_OK),
                jnp.zeros((len(freqs), 3), cdt), jnp.int32(2))

    def completion(y, dt, freqs, n0, s0, mask, tail_arg, plain=None, **kw):
        # Keep the stubbed poles connected to the differentiated record.
        return plain, {"s": s0 + 0 * jnp.sum(y)}

    monkeypatch.setattr(rj, 'host_poles', host_poles)
    monkeypatch.setattr(rj, 'completion', completion)
    sim = _box('uniform')
    eps = jnp.full(sim._build_grid().shape, 2.2, jnp.float32)

    def loss(x):
        result = sim.forward(n_steps=600, port_s11_freqs=FREQS, eps_override=eps * x,
                             ringdown=rd.RingdownSpec(), skip_preflight=True).ringdown
        return result.pole(12.4e9)[0], result

    (value, result), grad = jax.value_and_grad(loss, has_aux=True)(1.)
    np.testing.assert_array_equal(result.pole_mask, [1, 2, 0])
    np.testing.assert_array_equal(result.pole_mask_long, [1, 2, 0])
    assert np.isnan(value) and np.isnan(grad)
    assert np.isnan(result.pole(12.4e9)).all()
    assert 'unresolved' in rd.pole_selection_reason(
        result.poles, result.pole_mask, 12.4e9, result._pole_window_s())


def test_not_completed_poles_keep_nan_gradient(monkeypatch):
    monkeypatch.setattr(rd, '_refuse_stencil_order', lambda sim, caller: None)
    sim = _box('uniform', stencil_order=4)
    eps = jnp.full(sim._build_grid().shape, 2.2, jnp.float32)

    def loss(x):
        result = sim.forward(n_steps=600, port_s11_freqs=FREQS, eps_override=eps * x,
                             ringdown=rd.RingdownSpec(), skip_preflight=True).ringdown
        return jnp.sum(result.poles.real), result

    (value, result), grad = jax.value_and_grad(loss, has_aux=True)(1.)
    assert np.all(np.asarray(result._status) == rj.STATUS_PRECONDITION)
    assert result.poles.shape == (1,)
    assert np.isnan(result.poles).all() and np.isnan(value)
    assert np.isnan(grad), f'precondition pole gradient must be NaN, got {grad}'


@pytest.mark.parametrize('frequencies,masks,reason', [
    ([10., 30.], [1, 1], None),
    ([30.], [1], 'none within delta'),
    ([10., 10.5], [1, 1], '2 candidates within delta'),
    ([10.], [2], 'none within delta; unresolved mask-2 pole within delta'),
    ([10., -10., 40.], [1, 1, 0], None),
    ([10.], [0], 'none within delta'),
    ([], [], 'none within delta'),
])
def test_select_pole_rule_and_nan_gradient(frequencies, masks, reason):
    with enable_x64():
        poles = jnp.asarray([-1 + 2j * np.pi * f for f in frequencies], dtype=jnp.complex128)
        mask = jnp.asarray(masks, dtype=jnp.int32)
        got = rd.select_pole(poles, mask, 10., 1.)
        assert rd.pole_selection_reason(poles, mask, 10., 1.) == reason
        if reason is None:
            np.testing.assert_allclose(got, (10., 1 / (10 * np.pi)))
            assert np.isfinite(jax.grad(lambda x: rd.select_pole(poles * x, mask, 10., 1.)[0])(1.))
        else:
            assert np.isnan(got).all()
            if poles.size:
                for i in (0, 1):
                    assert np.isnan(jax.grad(lambda x: rd.select_pole(poles * x, mask, 10., 1.)[i])(1.))
        failed = poles * jnp.nan
        assert rd.pole_selection_reason(failed, mask, 10., 1., status=1) == 'failed window: the host identification failed'
        assert np.isnan(rd.select_pole(failed, mask, 10., 1.)).all()


@pytest.mark.parametrize('window', ['main', 'long'])
@pytest.mark.parametrize('bad', ['none', 'multiple', 'unresolved', 'failed'])
def test_pole_witness_cannot_judge_nonunique_window(window, bad):
    p = jnp.asarray([-1 + 20j * np.pi, -1 + 21j * np.pi])
    masks = [jnp.asarray([1, 0], jnp.int32), jnp.asarray([1, 0], jnp.int32)]
    i = 0 if window == 'main' else 1
    masks[i] = jnp.asarray({'none': [0, 0], 'multiple': [1, 1],
                           'unresolved': [2, 0], 'failed': [0, 0]}[bad], jnp.int32)
    status = jnp.asarray([1 if k == i and bad == 'failed' else 0 for k in (0, 1)])
    plan = SimpleNamespace(n_steps=2, n_start=1, grid=SimpleNamespace(dt=1.),
                           source_ratio_long=0., spec=rd.RingdownSpec())
    result = rd.RingdownForwardResult(None, None, None, jnp.zeros((2, 1)), (), None,
        status, 0., jnp.zeros(2), jnp.zeros(2), SimpleNamespace(plan=plan),
        poles=p, poles_long=p, pole_mask=masks[0], pole_mask_long=masks[1])
    w = rd.gradient_witness(1., 1., ringdown=result, pole_target=10.)
    assert not w.judged and not w.ok and np.isnan(w.value)
    assert w.note.startswith('cannot judge: ' + window + ': ')
    print(w.note)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_returned_poles_equal_host_and_fail_with_identification(lane, monkeypatch):
    with enable_x64():
        sim = _box(lane, cells=3)
        models = []
        original = rd.identify

        def capture(*args, **kw):
            model = original(*args, **kw)
            if args[2] == 0:  # completion callback; exclude lazy report identification
                models.append(model)
            return model

        monkeypatch.setattr(rd, 'identify', capture)
        out = sim.forward(n_steps=600, port_s11_freqs=FREQS,
                          ringdown=rd.RingdownSpec(), skip_preflight=True).ringdown
        jax.block_until_ready(out.poles)
        assert len(models) == 2
        for p, mask, model in zip((out.poles, out.poles_long),
                                  (out.pole_mask, out.pole_mask_long), models):
            expected, complete = completion_poles(model, 600, out._ctx.plan.spec.unit_tol)
            np.testing.assert_array_equal(np.asarray(p)[np.asarray(mask) > 0], expected)
            np.testing.assert_array_equal(np.asarray(mask)[:len(expected)], np.where(complete, 1, 2))
            assert np.all(np.asarray(mask)[len(expected):] == 0)
            assert mask.dtype == jnp.int32
        rebuilt = jax.tree_util.tree_map(lambda a: a, out)
        np.testing.assert_array_equal(rebuilt.poles, out.poles)
        with pytest.raises(AttributeError):
            out.poles = out.poles

        def fail(*args, **kw):
            raise ValueError('forced synthetic closed-box identification failure')

        monkeypatch.setattr(rd, 'identify', fail)
        eps = jnp.full(out._ctx.grid.shape, 2.2, dtype=jnp.float32)

        def loss(x):
            result = sim.forward(n_steps=600, port_s11_freqs=FREQS, eps_override=eps * x,
                                 ringdown=rd.RingdownSpec(), skip_preflight=True).ringdown
            return result.pole(12.4e9)[0], result

        (value, failed), grad = jax.value_and_grad(loss, has_aux=True)(1.)
        assert np.isnan(value) and np.isnan(grad)
        assert np.isnan(failed.poles).all() and np.isnan(failed.poles_long).all()
        assert 'identification' in failed.report.failure
