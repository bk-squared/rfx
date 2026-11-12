"""Target selection and returned poles of the synthetic closed-box fixture."""
from copy import copy
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


def _longer_record_stub(n_steps, *, frequencies=(8e9, 8.02e9),
                        masks=(1, 0), inv_q=.001, status=0):
    plan = SimpleNamespace(n_steps=n_steps, n_start=n_steps // 2,
                           grid=SimpleNamespace(dt=1e-9))
    poles = jnp.asarray([(-inv_q * np.pi + 2j * np.pi) * f for f in frequencies])
    # The earlier-start window is deliberately different: only main is read.
    long_poles = jnp.asarray([-1e7 + 2j * np.pi * 11e9])
    return rd.RingdownForwardResult(
        None, None, None, jnp.zeros((n_steps, 1)), (), None,
        jnp.asarray([status, 1]), 0., jnp.asarray([7, 0]), jnp.asarray([3, 0]),
        SimpleNamespace(plan=plan), poles=poles, poles_long=long_poles,
        pole_mask=jnp.asarray(masks, jnp.int32), pole_mask_long=jnp.asarray([1], jnp.int32))


_LONGER_RECORD_REASONS = {
    'none': 'none within delta',
    'multiple': '2 candidates within delta',
    'unresolved': 'none within delta; unresolved mask-2 pole within delta',
    'failed': 'failed window: the host identification failed',
    'over_budget': ('failed window: the identification kept 7 poles, more than the '
                    '3 slots the traced completion reserves for this window '
                    "(min of the pencil's capacity and TRACED_POLE_BUDGET = 512)"),
}


@pytest.mark.parametrize('record', [0, 1], ids=['grad', 'grad_other'])
@pytest.mark.parametrize('bad', list(_LONGER_RECORD_REASONS))
@pytest.mark.parametrize('pole_target', [8e9, None], ids=['pole', 'without_target'])
def test_longer_record_p4_selection_matrix(record, bad, pole_target):
    with enable_x64():
        records = [_longer_record_stub(12), _longer_record_stub(24)]
        masks = {'none': (0, 0), 'multiple': (1, 1), 'unresolved': (2, 0),
                 'failed': (1, 0), 'over_budget': (1, 0)}[bad]
        # Failed status with otherwise selectable poles pins status-before-selection.
        records[record] = _longer_record_stub(
            (12, 24)[record], masks=masks,
            status={'failed': 1, 'over_budget': 2}.get(bad, 0))
        cannot_judge = pole_target is not None or bad in ('failed', 'over_budget')
        gradients = [1., 1.]
        if cannot_judge:
            gradients[record] = np.nan
        w = rd.gradient_witness(*gradients, against='longer_record',
                                ringdown=records, pole_target=pole_target)
    if cannot_judge:
        assert w.judged is False and w.ok is False and np.isnan(w.value)
        label = ("grad's record (12 ns)", "grad_other's record (24 ns)")[record]
        radius = (' (radius 166.7 MHz)', ' (radius 83.33 MHz)')[record]
        assert w.note == ('cannot judge: ' + label + ': ' + _LONGER_RECORD_REASONS[bad]
                          + (radius if pole_target is not None else ''))
    else:
        assert w.judged is True and w.ok is True and w.value == 0.
        assert w.note == "records: grad's record (12 ns); grad_other's record (24 ns); record ratio 2"


def test_longer_record_p4_selection_differs():
    with enable_x64():
        records = (_longer_record_stub(12),
                   _longer_record_stub(24, frequencies=(8.12e9,), masks=(1,), inv_q=.002))
        # 120 MHz separation lies between the 83.33 and 166.7 MHz radii.
        w = rd.gradient_witness(1., 1., against='longer_record',
                                ringdown=records, pole_target=8.06e9)
    assert w.judged is False and w.ok is False and np.isnan(w.value)
    assert w.note == 'cannot judge (selection differs: 8e+09 Hz vs 8.12e+09 Hz)'


@pytest.mark.parametrize('reverse', [False, True], ids=['forward', 'swapped'])
@pytest.mark.parametrize('bin_axis', [None, 0], ids=['scalar', 'bins'])
@pytest.mark.parametrize('pole_target', [8e9, None], ids=['pole', 'without_target'])
def test_longer_record_p4_near_distinct_judged(reverse, bin_axis, pole_target):
    with enable_x64():
        records = [_longer_record_stub(12),
                   _longer_record_stub(24, frequencies=(8.02e9,), masks=(1,), inv_q=.002)]
        labels = ["grad's record (12 ns)", "grad_other's record (24 ns)"]
        poles = ['f=8e+09 Hz, 1/Q=0.001', 'f=8.02e+09 Hz, 1/Q=0.002']
        ratio = '2'
        if reverse:
            records.reverse()
            labels = ["grad's record (24 ns)", "grad_other's record (12 ns)"]
            poles.reverse()
            ratio = '0.5'
        gradients = (1., 1.005) if bin_axis is None else (np.array([0., 1.]), np.array([0., 1.005]))
        w = rd.gradient_witness(*gradients, against='longer_record', ringdown=records,
                                pole_target=pole_target, bin_axis=bin_axis)
        baseline = rd.gradient_witness(*gradients, against='longer_record', bin_axis=bin_axis)
    assert w.judged is True and w.ok is True and w.value == baseline.value
    assert w.bar == baseline.bar and w.rule == baseline.rule
    if pole_target is None:
        expected = f'records: {labels[0]}; {labels[1]}; record ratio {ratio}'
    else:
        expected = (f'selected at pole_target=8e+09 Hz: {labels[0]}: {poles[0]}; '
                    f'{labels[1]}: {poles[1]}; record ratio {ratio}')
    if bin_axis is not None:
        expected += '; skipped 1 zero-reference bins'
    assert w.note == expected


@pytest.mark.parametrize('other', [2., np.nan], ids=['disagreement', 'nan_gradient'])
@pytest.mark.parametrize('pole_target', [8e9, None], ids=['pole', 'without_target'])
def test_longer_record_p4_judged_failure(other, pole_target):
    with enable_x64():
        w = rd.gradient_witness(1., other, against='longer_record', pole_target=pole_target,
                                ringdown=(_longer_record_stub(12), _longer_record_stub(24)))
    assert w.judged is True and w.ok is False
    if np.isnan(other):
        assert np.isnan(w.value)
    else:
        assert w.value == 1. and w.value > w.bar


@pytest.mark.parametrize('grad,other,match', [
    ({'a': 1.}, {'b': 1.}, 'different structures'),
    (np.ones(2), np.ones(3), 'gradient leaves of shapes'),
])
def test_longer_record_p4_gradient_checks_precede_selection(grad, other, match):
    records = (_longer_record_stub(12, status=1), _longer_record_stub(24))
    with pytest.raises(ValueError, match=match):
        rd.gradient_witness(grad, other, against='longer_record', ringdown=records, pole_target=8e9)


def test_longer_record_p5_missing_pair_message():
    with pytest.raises(ValueError) as exc:
        rd.gradient_witness(1., 1., against='longer_record', pole_target=8e9)
    assert str(exc.value) == ('pole_target with against="longer_record" needs '
                              "ringdown=(result of grad's record, result of grad_other's record)")


@pytest.mark.parametrize('bad', ['single_result', 'one', 'three', 'non_result'])
@pytest.mark.parametrize('pole_target', [8e9, None], ids=['pole', 'without_target'])
def test_longer_record_p5_pair_type(bad, pole_target):
    result = _longer_record_stub(12)
    pair = {'single_result': result, 'one': (result,), 'three': (result,) * 3,
            'non_result': (result, object())}[bad]
    # Pair refusal must precede the deliberately mismatched gradient trees.
    with pytest.raises(TypeError):
        rd.gradient_witness({'a': 1.}, {'b': 1.}, against='longer_record',
                            ringdown=pair, pole_target=pole_target)


@pytest.mark.parametrize('record', [0, 1], ids=['grad', 'grad_other'])
@pytest.mark.parametrize('pole_target', [8e9, None], ids=['pole', 'without_target'])
def test_longer_record_p5_batched(record, pole_target):
    records = [_longer_record_stub(12), _longer_record_stub(24)]
    records[record]._channels = jnp.zeros((2, (12, 24)[record], 1))
    with pytest.raises(ValueError) as exc:
        rd.gradient_witness(1., 1., against='longer_record', ringdown=records, pole_target=pole_target)
    assert str(exc.value) == (
        'RingdownForwardResult.pole reads one result; this one is batched '
        f'(its port channels have shape {(2, (12, 24)[record], 1)}, '
        'one record is (n_steps, channels)), as jax.vmap returns it. Take '
        'one element first: jax.tree_util.tree_map(lambda a: a[i], result).')


@pytest.mark.parametrize('record', [0, 1], ids=['grad', 'grad_other'])
@pytest.mark.parametrize('field', ['_status', '_poles', '_poles_long', '_pole_mask', '_pole_mask_long'])
@pytest.mark.parametrize('pole_target', [8e9, None], ids=['pole', 'without_target'])
def test_longer_record_p5_traced(record, field, pole_target):
    concrete = [_longer_record_stub(12), _longer_record_stub(24)]
    original = getattr(concrete[record], field)

    def inside(value):
        records = [copy(result) for result in concrete]
        setattr(records[record], field, value)
        return rd.gradient_witness(1., 1., against='longer_record',
                                   ringdown=records, pole_target=pole_target).value

    with pytest.raises(TypeError, match='outside jax.jit'):
        jax.jit(inside)(original)
