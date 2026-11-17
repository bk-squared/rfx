"""Second derivatives through the ring-down completion are NaN with a named warning; first derivatives are unchanged.

Physical anchor. The completion moves the identified poles of a record of
damped oscillations by one frozen Gauss-Newton step: zero in value, exact in
first derivative (variable projection at a zero-residual fit), but linear in
the record, so its second derivative lacks the term quadratic in the record's
first derivative. A second-order form of an objective that reads a completed
output is therefore NaN, with ``RingdownSecondDerivativeWarning``; the first
derivative and every derivative of an objective on the recorded spectrum alone
are bit for bit those of the plain Gauss-Newton step.

The record is the closed-form one of ``test_ringdown_completion_jax.py`` (four
modes whose frequency, Q and residues move with a parameter ``p``), here
``y(p) = y0 + p v`` with ``v`` the exact tangent. The reference is the plain
step monkeypatched in place of the custom-derivative one.
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import RingdownSecondDerivativeWarning
from rfx.measurement.ringdown_tail_jax import plain_dft
from rfx.ringdown import _jax as rj
from tests._x64_compat import enable_x64
from tests.unit.sparams.test_ringdown_completion_jax import (
    DT, FREQS, K_TEST, N_START, N_WINDOWED, _identify_window, _series,
)


def _plain_pole_step(Yw, A, w, mask_c):
    return rj.lstsq_masked(A, ((Yw - jax.lax.stop_gradient(Yw)) * w[None, :]).reshape(-1, 1),
                           mask_c)[:, 0]


@functools.lru_cache(maxsize=None)
def _record():
    y0, v = _series(N_WINDOWED)
    with enable_x64():
        s0, mask, status, tail_arg, _count = rj.host_poles(
            _identify_window, jnp.asarray(y0)[N_START:], K_TEST, DT, FREQS)
        return y0, v, (s0, mask, tail_arg), int(status)


def _y(p):
    y0, v, _poles, _status = _record()
    return jnp.asarray(y0) + p * jnp.asarray(v)


def _completed(y):
    _y0, _v, (s0, mask, tail_arg), _status = _record()
    return rj.completion(y, DT, FREQS, N_START, s0, mask, tail_arg)


def _parts(c):
    return jnp.stack([jnp.real(c), jnp.imag(c)])


def completed_vector(p):
    return _parts(_completed(_y(p)))


def _power(c):
    return jnp.real(c) ** 2 + jnp.imag(c) ** 2


def completed_objective(p):
    return jnp.sum(_power(_completed(_y(p))))


def in_record_vector(p):
    y = _y(p)
    _completed(y)
    return _parts(plain_dft(y, DT, FREQS) ** 2)


def in_record_objective(p):
    y = _y(p)
    _completed(y)
    return jnp.sum(_power(plain_dft(y, DT, FREQS)) ** 2)


def _p0():
    return jnp.float64(0.0)


def _first_order(form, objective, vector):
    p0 = _p0()
    if form == "grad":
        return jax.grad(objective)(p0)
    if form == "jit(grad)":
        return jax.jit(jax.grad(objective))(p0)
    if form == "vmap(grad)":
        return jax.vmap(jax.grad(objective))(jnp.asarray([0.0, 1.0e-3, -2.0e-3]))
    if form == "grad(checkpoint)":
        return jax.grad(jax.checkpoint(objective))(p0)
    if form == "jacrev":
        return jax.jacrev(vector)(p0)
    if form == "jacfwd":
        return jax.jacfwd(vector)(p0)
    raise ValueError(form)


FIRST_ORDER = ("grad", "jit(grad)", "vmap(grad)", "grad(checkpoint)", "jacrev", "jacfwd")


def _second_order(form, objective, vector):
    p0 = _p0()
    if form == "hessian":
        return jax.hessian(vector)(p0)
    if form == "jacfwd(jacrev)":
        return jax.jacfwd(jax.jacrev(vector))(p0)
    if form == "jacrev(jacfwd)":
        return jax.jacrev(jax.jacfwd(vector))(p0)
    if form == "grad(grad)":
        return jax.grad(jax.grad(objective))(p0)
    if form == "jvp(grad)":
        return jax.jvp(jax.grad(objective), (p0,), (jnp.float64(1.0),))
    if form == "hessian(jit)":
        return jax.hessian(jax.jit(vector))(p0)
    if form == "jit(hessian)":
        return jax.jit(jax.hessian(vector))(p0)
    raise ValueError(form)


SECOND_ORDER = ("hessian", "jacfwd(jacrev)", "jacrev(jacfwd)", "grad(grad)", "jvp(grad)",
                "hessian(jit)", "jit(hessian)")


def _leaves(out):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(jax.block_until_ready(out))]


def _ours(seen):
    return [w for w in seen if issubclass(w.category, RingdownSecondDerivativeWarning)]


def _plain_and_custom(compute, monkeypatch):
    """``compute()`` with the plain pole step and with the custom one, fresh traces each."""
    with enable_x64():
        with monkeypatch.context() as patch:
            patch.setattr(rj, "_pole_step", _plain_pole_step)
            jax.clear_caches()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RingdownSecondDerivativeWarning)
                plain = _leaves(compute())
        jax.clear_caches()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            custom = _leaves(compute())
    return plain, custom, _ours(seen)


def _same_bits(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and x.dtype == y.dtype
                                    and np.array_equal(x, y) for x, y in zip(a, b))


def _assert_completion_ok():
    """The NaN rows are not a failed completion: status OK, finite value and first derivative."""
    assert _record()[3] == rj.STATUS_OK
    with enable_x64():
        value = np.asarray(completed_vector(_p0()))
        first = np.asarray(jax.jacfwd(completed_vector)(_p0()))
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(first))


@pytest.mark.parametrize("form", FIRST_ORDER)
def test_first_derivative_of_the_completed_spectrum_is_the_plain_step_s(form, monkeypatch):
    plain, custom, ours = _plain_and_custom(
        lambda: _first_order(form, completed_objective, completed_vector), monkeypatch)
    assert all(np.all(np.isfinite(x)) for x in custom)
    assert _same_bits(plain, custom)
    assert not ours, [str(w.message) for w in ours]


@pytest.mark.parametrize("form", SECOND_ORDER)
def test_completed_second_derivative_is_nan(form, monkeypatch):
    _assert_completion_ok()
    with enable_x64():
        jax.clear_caches()
        with pytest.warns(RingdownSecondDerivativeWarning):
            out = _leaves(_second_order(form, completed_objective, completed_vector))
    if form == "jvp(grad)":
        primal, tangent = out
        assert np.isnan(tangent).all(), tangent
        plain, custom, _ours_seen = _plain_and_custom(
            lambda: jax.grad(completed_objective)(_p0()), monkeypatch)
        assert np.all(np.isfinite(primal)) and _same_bits(plain, [primal])
    else:
        assert all(np.isnan(x).all() for x in out), out


@pytest.mark.parametrize("form", SECOND_ORDER)
def test_in_record_second_derivative_is_the_plain_step_s(form, monkeypatch):
    plain, custom, _seen = _plain_and_custom(
        lambda: _second_order(form, in_record_objective, in_record_vector), monkeypatch)
    assert all(np.all(np.isfinite(x)) for x in plain)
    assert _same_bits(plain, custom)


def test_the_warning_filter_turns_a_completed_hessian_into_an_exception():
    with enable_x64(), warnings.catch_warnings():
        warnings.filterwarnings("error", category=RingdownSecondDerivativeWarning)
        jax.clear_caches()
        with pytest.raises(RingdownSecondDerivativeWarning):
            jax.hessian(completed_vector)(_p0())


def _scaled(p, q):
    return q * completed_objective(p)


def test_joint_hessian_with_a_scale_after_the_completion_is_nan_in_every_entry(monkeypatch):
    _assert_completion_ok()

    def compute():
        return jax.hessian(_scaled, argnums=(0, 1))(_p0(), jnp.float64(2.0))

    plain, custom, _seen = _plain_and_custom(compute, monkeypatch)
    names = ("pp", "pq", "qp", "qq")
    entries = {n: (float(a), float(b)) for n, a, b in zip(names, plain, custom)}
    print(f"\n[joint hessian, plain step / custom step] {entries}")
    assert all(np.isnan(x).all() for x in custom), entries


def test_value_tangent_in_a_second_order_form_is_the_first_derivative(monkeypatch):
    def compute():
        _primal, (d_value, _d_grad) = jax.jvp(jax.value_and_grad(completed_objective),
                                                   (_p0(),), (jnp.float64(1.0),))
        return d_value, jax.grad(completed_objective)(_p0())

    plain, custom, _seen = _plain_and_custom(compute, monkeypatch)
    print(f"\n[value tangent / grad] plain step {plain}, custom step {custom}")
    assert np.isfinite(custom[0]) and float(custom[0]) == pytest.approx(float(plain[1]), rel=1e-9)


def test_the_unbatched_mixed_derivative_through_the_value_is_the_plain_step_s(monkeypatch):
    def compute():
        return jax.grad(lambda p: jax.grad(_scaled, argnums=1)(p, jnp.float64(2.0)))(_p0())

    plain, custom, _seen = _plain_and_custom(compute, monkeypatch)
    print(f"\n[grad_p(grad_q)] plain step {plain}, custom step {custom}")
    assert np.all(np.isfinite(custom[0])) and _same_bits(plain, custom)
