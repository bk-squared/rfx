"""Probe completion derivatives, failure factors and cross-trace checks (T5-T6).

Physical anchor: permittivity moves the PEC TM110 resonance as eps**(-1/2).
Estimated 4-CPU time: 70-110 s, dominated by compilation; not measured here.
T5 uses declared 900/1500-step records (ratio 5/3); T6 uses 1500 steps.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.ringdown import _jax as rj
from tests.unit.sparams.test_ringdown_probe_channels import box, spec, channel_error, PROBES


def evaluator(n=900):
    sim = box()
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)

    def evaluate(p):
        return sim.forward(n_steps=n, skip_preflight=True, ringdown=spec(),
                           eps_override=eps * p).ringdown
    return evaluate


def losses(res):
    # Each channel has its own loss; the fixed impedance factor prevents H
    # from disappearing in float32 and is independent of the fitted spectra.
    scale = jnp.asarray([1., 377.])
    return jnp.stack([jnp.sum(jnp.abs(x * scale) ** 2, axis=0)
                      for x in (res.probe_spectra, res.probe_spectra_long)])


def test_t5_gradient_and_long_window_twin():
    """Physical anchor: d(eps) shifts the same free-ring poles in short and longer records."""
    evaluate = evaluator()

    def objective(p):
        res = evaluate(p)
        return losses(res), res

    value, tangent, res = jax.jit(lambda p: jax.jvp(
        objective, (p,), (jnp.ones_like(p),), has_aux=True))(jnp.float32(1.))
    g = jax.jit(jax.grad(lambda p: jnp.sum(objective(p)[0][0])))(jnp.float32(1.))
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent)) and np.isfinite(g)
    assert float(g) == pytest.approx(float(jnp.sum(tangent[0])), rel=1e-4)
    witness = rd.gradient_witness(tangent[0], tangent[1], ringdown=res, bin_axis=0)
    assert witness.judged and witness.value <= 1e-2, witness
    longer = evaluator(1500)

    def longer_objective(p):
        result = longer(p)
        return losses(result)[0], result

    _, other, other_res = jax.jit(lambda p: jax.jvp(
        longer_objective, (p,), (jnp.ones_like(p),), has_aux=True))(jnp.float32(1.))
    comparison = rd.gradient_witness(tangent[0], other, against="longer_record",
                                      ringdown=(res, other_res), bin_axis=0)
    assert comparison.judged and comparison.value <= 1e-2, comparison


def test_t5b_identification_failure_nan_value_and_gradient(monkeypatch):
    """Physical anchor: without identified free-ring poles no infinite-record derivative is known."""
    def fail(*args, **kwargs):
        raise ValueError("forced probe identification failure")

    monkeypatch.setattr(rd, "identify", fail)
    evaluate = evaluator(600)

    def objective(p):
        res = evaluate(p)
        return jnp.sum(losses(res)), res

    (value, res), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.float32(1.))
    assert np.isnan(value) and np.isnan(gradient)
    assert np.isnan(np.asarray(res.probe_spectra)).all()
    assert np.isnan(np.asarray(res.probe_spectra_long)).all()
    assert np.all(np.asarray(res._status) == rj.STATUS_FAILED)
    assert "forced probe identification failure" in res.report.failure


def test_t5c_precondition_nan_value_and_gradient(monkeypatch):
    """Physical anchor: a solver-step mismatch invalidates both physical-clock completions."""
    def fail(*args, **kwargs):
        raise rd._NotCompleted("solver_dt", .857, 1., "forced solver step mismatch")

    monkeypatch.setattr(rd.RingdownForward, "_check_solver_step", staticmethod(fail))
    evaluate = evaluator(600)
    for window in ("probe_spectra", "probe_spectra_long"):
        def objective(p):
            res = evaluate(p)
            return jnp.real(jnp.sum(getattr(res, window))), res

        (value, res), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.float32(1.))
        assert np.isnan(value) and np.isnan(gradient)
        assert np.isnan(np.asarray(res.probe_spectra)).all()
        assert np.isnan(np.asarray(res.probe_spectra_long)).all()
        assert np.all(np.asarray(res._status) == rj.STATUS_PRECONDITION)
        assert "solver_dt" in res.report.failure


def test_t3_pec_traced_failure():
    """Physical anchor: tangential E on a PEC wall is zero at every step."""
    sim = box()
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)
    request = spec(identification_probes=(PROBES[0], ((6.5e-3, 7e-3, 0.0), "ex")))

    def objective(p):
        res = sim.forward(n_steps=600, ringdown=request, eps_override=eps * p,
                           skip_preflight=True).ringdown
        return jnp.real(jnp.sum(res.probe_spectra)), res

    (value, res), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.float32(1.))
    assert np.isnan(value) and np.isnan(gradient)
    assert np.all(np.asarray(res._status) == rj.STATUS_FAILED)
    assert "identification probe ex" in res.report.failure
    assert "records no field over the window" in res.report.failure


def test_t6_plain_jit_vmap():
    """Physical anchor: a summed 1500-step spectrum is independent of JAX staging and batching."""
    evaluate = evaluator(1500)
    scales = jnp.asarray([1., 1.01], dtype=jnp.float32)
    plain = [evaluate(p) for p in scales]
    compiled = jax.jit(evaluate)(scales[0])
    batched = jax.jit(jax.vmap(evaluate))(scales)
    assert batched.probe_spectra.shape == (2, 101, 2)
    assert np.all(channel_error(compiled.probe_spectra, np.asarray(plain[0].probe_spectra)) <= 1e-4)
    for i in range(2):
        assert np.all(channel_error(batched.probe_spectra[i], np.asarray(plain[i].probe_spectra)) <= 1e-4)
    children, ctx = compiled.tree_flatten()
    assert children[0] is None and children[1] is None
    assert children[14] is compiled.probe_spectra and children[15] is compiled.probe_spectra_long
    rebuilt = rd.RingdownForwardResult.tree_unflatten(ctx, children)
    assert np.array_equal(np.asarray(rebuilt.probe_spectra), np.asarray(compiled.probe_spectra))


def test_t3_traced_source_off_message():
    """Physical anchor: raw field amplitude is linear in the source, unlike normalized port S."""
    from rfx._traced_refusal import TracedStructureRefusal
    from types import SimpleNamespace
    entry = SimpleNamespace(is_soft_source=True, position=(.002, .002, .001), component="ez")
    with pytest.raises(TracedStructureRefusal, match="completed probe spectrum is linear in the drive: scale it outside the call"):
        rd._require_source_off_rows([entry], [None], "forward", probe_mode=True)
