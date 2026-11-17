"""Second derivatives through ``forward(ringdown=...)``: the completed S'' is NaN, the in-record S'' and every first derivative are unchanged.

Physical anchor. The PEC box of ``test_ringdown_run.py`` filled with eps_r 2.2
rings in TM110 near 12.4 GHz; scaling the fill by ``p`` moves the resonance as
``p**-1/2``. The completion's poles move with ``p`` by one frozen Gauss-Newton
step, exact to first order, so a second derivative of the completed S lacks
the term quadratic in the record's first derivative and is NaN with
``RingdownSecondDerivativeWarning``. The in-record S (``ForwardResult.s_params``,
from the run's DFT accumulators) does not pass through the completion, so its
second derivative is bit for bit what the plain Gauss-Newton step gives, through
the time-step scan, its checkpointing and the host identification callback.
The probe-channel completion (the same box driven by a current source, two
field probes) is checked the same way. Records of 600 steps, every tenth bin
of 8-18 GHz.
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
from rfx.ringdown import RingdownSpec, _jax as rj, gradient_witness
from tests.contracts.test_ringdown_second_derivative import (
    FIRST_ORDER, SECOND_ORDER, custom_values, first_order, joint_hessian, leaves,
    plain_references, same_bits, second_order,
)
from tests.unit.sparams import test_ringdown_probe_channels as probe_scene
from tests.unit.sparams.test_ringdown_run import FREQS, _box

N = 600
BINS = FREQS[::10]
Q0 = 2.0


@functools.lru_cache(maxsize=None)
def _port_scene():
    sim = _box("uniform")
    return sim, jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)


@functools.lru_cache(maxsize=None)
def _probe_scene():
    sim = probe_scene.box()
    return sim, jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)


def run(p, checkpoint=None):
    sim, eps = _port_scene()
    return sim.forward(n_steps=N, skip_preflight=True, port_s11_freqs=BINS,
                       eps_override=eps * p, ringdown=RingdownSpec(), checkpoint=checkpoint)


def probe_run(p):
    sim, eps = _probe_scene()
    return sim.forward(n_steps=N, skip_preflight=True, eps_override=eps * p,
                       ringdown=probe_scene.spec(probe_freqs=BINS))


def _parts(c):
    return jnp.stack([jnp.real(c), jnp.imag(c)])


def _power(c):
    return jnp.real(c) ** 2 + jnp.imag(c) ** 2


def completed_vector(p):
    return _parts(run(p).ringdown.s_params)


def completed_objective(p):
    return jnp.sum(_power(run(p).ringdown.s_params))


def in_record_vector(p):
    return _parts(run(p).s_params)


def in_record_objective(p):
    return jnp.sum(_power(run(p).s_params))


def in_record_objective_without_checkpoint(p):
    return jnp.sum(_power(run(p, checkpoint=False).s_params))


def scaled_objective(p, q):
    return q * completed_objective(p)


def probe_vector(p):
    return _parts(probe_run(p).ringdown.probe_spectra)


def probe_objective(p):
    return jnp.sum(_power(probe_run(p).ringdown.probe_spectra))


def probe_in_record_objective(p):
    return jnp.sum(probe_run(p).time_series ** 2)


def _p0():
    return jnp.float32(1.0)


def _witness_inputs():
    def both(p):
        out = run(p).ringdown
        return (jnp.sum(_power(out.s_params)), jnp.sum(_power(out.s_params_long))), out

    _values, pullback, rd_out = jax.vjp(both, _p0(), has_aux=True)
    (g,) = pullback((jnp.float32(1.0), jnp.float32(0.0)))
    (g_long,) = pullback((jnp.float32(0.0), jnp.float32(1.0)))
    witness = gradient_witness(g, g_long, ringdown=rd_out)
    return [g, g_long, np.float64(witness.value)]


def _cases():
    batch = jnp.asarray([1.0, 1.001], jnp.float32)
    cases = {("first", f): functools.partial(first_order, f, completed_objective,
                                             completed_vector, _p0(), batch)
             for f in FIRST_ORDER}
    cases.update({("in-record", f): functools.partial(second_order, f, in_record_objective,
                                                      in_record_vector, _p0())
                  for f in SECOND_ORDER})
    cases["in-record grad(grad) without checkpoint"] = (
        lambda: jax.grad(jax.grad(in_record_objective_without_checkpoint))(_p0()))
    cases["gradient_witness"] = _witness_inputs
    cases["joint"] = lambda: joint_hessian(scaled_objective, _p0(), jnp.float32(Q0))
    cases["probe jit(grad)"] = lambda: jax.jit(jax.grad(probe_objective))(_p0())
    cases["probe in-record grad(grad)"] = (
        lambda: jax.grad(jax.grad(probe_in_record_objective))(_p0()))
    return cases


@pytest.fixture(scope="module")
def plain():
    return plain_references(_cases())


def _custom(key):
    return custom_values(_cases()[key])


def _assert_completion_ok(result, vector):
    """A NaN row is not a failed completion: status OK, finite value and first derivative."""
    assert np.all(np.asarray(result.ringdown._status) == rj.STATUS_OK)
    assert np.all(np.isfinite(np.asarray(vector(_p0()))))
    assert np.all(np.isfinite(np.asarray(jax.jit(jax.jacfwd(vector))(_p0()))))


def _constructs(fn, *args):
    """Primitives of ``jax.make_jaxpr(fn)`` that stage, loop, branch, remat or call back."""
    names = set()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            names.add(eqn.primitive.name)
            for value in eqn.params.values():
                for sub in (value if isinstance(value, (list, tuple)) else (value,)):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        walk(inner)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RingdownSecondDerivativeWarning)
        walk(jax.make_jaxpr(fn)(*args).jaxpr)
    keep = ("scan", "while", "cond", "pjit", "jit", "remat", "checkpoint", "custom", "callback")
    return sorted(n for n in names if any(k in n for k in keep))


@pytest.mark.parametrize("form", FIRST_ORDER)
def test_first_derivative_of_the_completed_s_is_the_plain_step_s(form, plain):
    custom, ours = _custom(("first", form))
    assert all(np.all(np.isfinite(x)) for x in custom)
    assert same_bits(plain["first", form], custom)
    assert not ours, [str(w.message) for w in ours]


def test_the_gradient_witness_reads_the_plain_step_s_gradients(plain):
    custom, ours = _custom("gradient_witness")
    assert all(np.all(np.isfinite(x)) for x in custom)
    assert same_bits(plain["gradient_witness"], custom)
    assert not ours, [str(w.message) for w in ours]


@pytest.mark.parametrize("form", SECOND_ORDER)
def test_in_record_second_derivative_is_the_plain_step_s(form, plain):
    reference = plain["in-record", form]
    custom, _ours = _custom(("in-record", form))
    assert all(np.all(np.isfinite(x)) for x in reference)
    assert same_bits(reference, custom)


def test_in_record_second_derivative_without_checkpoint_is_the_plain_step_s(plain):
    reference = plain["in-record grad(grad) without checkpoint"]
    custom, _ours = _custom("in-record grad(grad) without checkpoint")
    assert all(np.all(np.isfinite(x)) for x in reference)
    assert same_bits(reference, custom)


@pytest.mark.parametrize("form", ["hessian", "grad(grad)"])
def test_completed_second_derivative_is_nan(form):
    _assert_completion_ok(run(_p0()), completed_vector)
    if form == "hessian":
        print(f"\n[forward(ringdown=) hessian program] "
              f"{_constructs(jax.hessian(completed_vector), _p0())}", file=sys.stderr)
    with pytest.warns(RingdownSecondDerivativeWarning):
        out = leaves(second_order(form, completed_objective, completed_vector, _p0()))
    assert all(np.isnan(x).all() for x in out), out


def test_the_warning_filter_turns_a_completed_hessian_into_an_exception():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RingdownSecondDerivativeWarning)
        with pytest.raises(RingdownSecondDerivativeWarning):
            jax.hessian(completed_vector)(_p0())


def test_joint_hessian_needs_the_second_derivative_only_where_it_is_nan(plain):
    """For q G(p): H[p,p] and H[p,q] NaN; H[q,p] = G'(p) and H[q,q] = 0 as the plain step's."""
    _assert_completion_ok(run(_p0()), completed_vector)
    reference = plain["joint"]
    pp, pq, qp, qq = _custom("joint")[0]
    print(f"\n[forward joint hessian of q G(p)] custom pp {pp} pq {pq} qp {qp} qq {qq}; plain "
          f"pp {reference[0]} pq {reference[1]} qp {reference[2]} qq {reference[3]}",
          file=sys.stderr)
    assert np.isnan(pp).all() and np.isnan(pq).all()
    assert np.all(np.isfinite(qp)) and same_bits([reference[2]], [qp])
    assert same_bits([reference[3]], [qq]) and float(qq) == 0.0


def test_probe_first_derivative_is_the_plain_step_s(plain):
    custom, ours = _custom("probe jit(grad)")
    assert all(np.all(np.isfinite(x)) for x in custom)
    assert same_bits(plain["probe jit(grad)"], custom)
    assert not ours, [str(w.message) for w in ours]


def test_probe_completed_hessian_is_nan():
    _assert_completion_ok(probe_run(_p0()), probe_vector)
    with pytest.warns(RingdownSecondDerivativeWarning):
        out = leaves(jax.hessian(probe_vector)(_p0()))
    assert all(np.isnan(x).all() for x in out), out


def test_probe_in_record_second_derivative_is_the_plain_step_s(plain):
    reference = plain["probe in-record grad(grad)"]
    custom, _ours = _custom("probe in-record grad(grad)")
    assert all(np.all(np.isfinite(x)) for x in reference)
    assert same_bits(reference, custom)
