"""Second derivatives through the ring-down completion are NaN with a named warning; first derivatives are unchanged.

Physical anchor. The completion moves the identified poles of a record of
damped oscillations by one frozen Gauss-Newton step: zero in value, exact in
first derivative (variable projection at a zero-residual fit), but linear in
the record, so its second derivative lacks the term quadratic in the record's
first derivative. An entry of a second-order form that needs the completed
spectrum's second derivative is therefore NaN, with
``RingdownSecondDerivativeWarning``; an entry that needs only its first
derivative is exact or NaN; the first derivative and every derivative of an
objective on the recorded spectrum alone are bit for bit those of the plain
Gauss-Newton step.

The record is the closed-form one of ``test_ringdown_completion_jax.py`` (four
modes whose frequency, Q and residues move with a parameter ``p``), here
``y(p) = y0 + p v`` with ``v`` the exact tangent, read at six of its bins.
The reference is the plain step monkeypatched in place of the
custom-derivative one, computed once per form before the custom values.
Every form here runs staged under one ``jax.jit``. The same rows as written
(eager), and the in-record and joint second-order rows, are in
``tests/unit/sparams/test_ringdown_second_derivative.py``.
"""

from __future__ import annotations

import functools
import sys
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

BINS = FREQS[::40]
Q0 = 2.0


def plain_pole_step(Yw, A, w, mask_c):
    """The frozen Gauss-Newton step written without a custom derivative."""
    return rj.lstsq_masked(A, ((Yw - jax.lax.stop_gradient(Yw)) * w[None, :]).reshape(-1, 1),
                           mask_c)[:, 0]


@functools.lru_cache(maxsize=None)
def _record():
    y0, v = _series(N_WINDOWED)
    with enable_x64():
        s0, mask, status, tail_arg, count = rj.host_poles(
            _identify_window, jnp.asarray(y0)[N_START:], K_TEST, DT, BINS)
        return y0, v, (s0, mask, tail_arg), int(status), int(count)


def _y(p):
    y0, v = _record()[:2]
    return jnp.asarray(y0) + p * jnp.asarray(v)


def _completed(y):
    s0, mask, tail_arg = _record()[2]
    return rj.completion(y, DT, BINS, N_START, s0, mask, tail_arg)


def _parts(c):
    return jnp.stack([jnp.real(c), jnp.imag(c)])


def _power(c):
    return jnp.real(c) ** 2 + jnp.imag(c) ** 2


def completed_vector(p):
    return _parts(_completed(_y(p)))


def completed_objective(p):
    return jnp.sum(_power(_completed(_y(p))))


def in_record_vector(p):
    y = _y(p)
    _completed(y)
    return _parts(plain_dft(y, DT, BINS) ** 2)


def in_record_objective(p):
    y = _y(p)
    _completed(y)
    return jnp.sum(_power(plain_dft(y, DT, BINS)) ** 2)


def scaled_objective(p, q):
    return q * completed_objective(p)


def _p0():
    return jnp.float64(0.0)


FIRST_ORDER = ("grad", "jit(grad)", "vmap(grad)", "grad(checkpoint)", "jacrev", "jacfwd",
               "jit(jacfwd)")
SECOND_ORDER = ("hessian", "jacfwd(jacrev)", "jacrev(jacfwd)", "grad(grad)", "jvp(grad)",
                "hessian(jit)", "jit(hessian)")


def first_order(form, objective, vector, p0, batch):
    """One first-order form of ``objective`` (scalar) or ``vector`` (real array) at ``p0``."""
    if form == "grad":
        return jax.grad(objective)(p0)
    if form == "jit(grad)":
        return jax.jit(jax.grad(objective))(p0)
    if form == "vmap(grad)":
        return jax.vmap(jax.grad(objective))(batch)
    if form == "grad(checkpoint)":
        return jax.grad(jax.checkpoint(objective))(p0)
    if form == "jacrev":
        return jax.jacrev(vector)(p0)
    if form == "jacfwd":
        return jax.jacfwd(vector)(p0)
    if form == "jit(jacfwd)":
        return jax.jit(jax.jacfwd(vector))(p0)
    raise ValueError(form)


def second_order(form, objective, vector, p0):
    """One second-order form; Hessian forms read the per-bin ``vector``."""
    if form == "hessian":
        return jax.hessian(vector)(p0)
    if form == "jacfwd(jacrev)":
        return jax.jacfwd(jax.jacrev(vector))(p0)
    if form == "jacrev(jacfwd)":
        return jax.jacrev(jax.jacfwd(vector))(p0)
    if form == "grad(grad)":
        return jax.grad(jax.grad(objective))(p0)
    if form == "jvp(grad)":
        return jax.jvp(jax.grad(objective), (p0,), (jnp.ones_like(p0),))
    if form == "hessian(jit)":
        return jax.hessian(jax.jit(vector))(p0)
    if form == "jit(hessian)":
        return jax.jit(jax.hessian(vector))(p0)
    raise ValueError(form)


def joint_hessian(objective, p0, q0):
    """``jax.hessian`` over (p, q): entries pp, pq, qp, qq."""
    h = jax.hessian(objective, argnums=(0, 1))(p0, q0)
    return [h[0][0], h[0][1], h[1][0], h[1][1]]


def leaves(out):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(jax.block_until_ready(out))]


def same_bits(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and x.dtype == y.dtype
                                    and np.array_equal(x, y) for x, y in zip(a, b))


def custom_values(compute):
    """``compute()`` with the custom step, and the ring-down warnings it emitted."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        out = leaves(compute())
    return out, [w for w in seen if issubclass(w.category, RingdownSecondDerivativeWarning)]


def plain_references(cases):
    """Each case computed with the plain step, then every compiled program dropped."""
    out = {}
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore", RingdownSecondDerivativeWarning)
        patch.setattr(rj, "_pole_step", plain_pole_step)
        jax.clear_caches()
        for key, compute in cases.items():
            out[key] = leaves(compute())
    jax.clear_caches()
    return out


def staged(compute):
    """``compute`` traced and compiled as one program, built afresh on each call."""
    return lambda: jax.jit(lambda: compute())()


def cases(staging):
    """The plain-step references every row compares with, each form under ``staging``."""
    _record()
    batch = jnp.asarray([0.0, 1.0e-3, -2.0e-3])
    out = {("first", f): functools.partial(first_order, f, completed_objective,
                                           completed_vector, _p0(), batch)
           for f in FIRST_ORDER}
    out["grad"] = lambda: jax.grad(completed_objective)(_p0())
    out.update({("in-record", f): functools.partial(second_order, f, in_record_objective,
                                                    in_record_vector, _p0())
                for f in SECOND_ORDER})
    out["joint"] = lambda: joint_hessian(scaled_objective, _p0(), jnp.float64(Q0))
    return {key: staging(compute) for key, compute in out.items()}


def as_written(compute):
    return compute


def references(staging, keys=None):
    with enable_x64():
        built = cases(staging)
        return plain_references({key: built[key] for key in (built if keys is None else keys)})


def custom(staging, key):
    with enable_x64():
        return custom_values(cases(staging)[key])


def assert_completion_ok():
    """A NaN row is not a failed completion: status OK, all poles, finite value and first derivative."""
    _y0, _v, _poles, status, count = _record()
    assert status == rj.STATUS_OK and count == 8
    with enable_x64():
        value = np.asarray(completed_vector(_p0()))
        first = np.asarray(jax.jacfwd(completed_vector)(_p0()))
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(first))


def check_first_derivative(staging, form, plain):
    result, ours = custom(staging, ("first", form))
    assert all(np.all(np.isfinite(x)) for x in result)
    assert same_bits(plain["first", form], result)
    assert not ours, [str(w.message) for w in ours]


def check_completed_second_derivative_is_nan(staging, form, plain):
    """The gradient of an objective summed over bins agrees across two traces.

    The #2166 bar is matching shape and dtype and max absolute error at most
    1e-4 times the nonzero reference gradient peak.
    """
    assert_completion_ok()
    with enable_x64(), pytest.warns(RingdownSecondDerivativeWarning):
        out = leaves(staging(functools.partial(second_order, form, completed_objective,
                                               completed_vector, _p0()))())
    if form == "jvp(grad)":
        primal, tangent = out
        assert np.isnan(tangent).all(), tangent
        assert np.all(np.isfinite(primal))
        g = plain["grad"][0]
        assert primal.shape == g.shape and primal.dtype == g.dtype
        peak = np.max(np.abs(g))
        assert peak > 0
        assert np.max(np.abs(primal - g)) <= 1e-4 * peak
    else:
        assert all(np.isnan(x).all() for x in out), out


def check_in_record_second_derivative(staging, form, plain):
    reference = plain["in-record", form]
    result, _ours = custom(staging, ("in-record", form))
    assert all(np.all(np.isfinite(x)) for x in reference)
    assert same_bits(reference, result)


def check_joint_hessian(staging, plain, label):
    """For q G(p): H[p,p] and H[p,q] NaN; H[q,p] = G'(p) and H[q,q] = 0 as the plain step's."""
    assert_completion_ok()
    reference = plain["joint"]
    pp, pq, qp, qq = custom(staging, "joint")[0]
    print(f"\n[{label} joint hessian of q G(p)] custom pp {pp} pq {pq} qp {qp} qq {qq}; plain "
          f"pp {reference[0]} pq {reference[1]} qp {reference[2]} qq {reference[3]}",
          file=sys.stderr)
    assert np.isnan(pp).all() and np.isnan(pq).all()
    assert np.all(np.isfinite(qp)) and same_bits([reference[2]], [qp])
    assert same_bits([reference[3]], [qq]) and float(qq) == 0.0


@pytest.fixture(scope="module")
def plain():
    return references(staged, [("first", f) for f in FIRST_ORDER] + ["grad"])


@pytest.mark.parametrize("form", FIRST_ORDER)
def test_staged_first_derivative_of_the_completed_spectrum_is_the_plain_step_s(form, plain):
    check_first_derivative(staged, form, plain)


@pytest.mark.parametrize("form", SECOND_ORDER)
def test_staged_completed_second_derivative_is_nan(form, plain):
    check_completed_second_derivative_is_nan(staged, form, plain)


def test_the_warning_filter_turns_a_completed_hessian_into_an_exception():
    _record()
    with enable_x64(), warnings.catch_warnings():
        warnings.filterwarnings("error", category=RingdownSecondDerivativeWarning)
        with pytest.raises(RingdownSecondDerivativeWarning):
            jax.jit(jax.hessian(completed_vector))(_p0())
