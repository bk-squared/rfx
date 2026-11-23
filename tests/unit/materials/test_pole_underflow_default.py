"""Default-on SI checks and one host report for a batched reverse pass."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx import _traced_check, backend
from rfx.materials import lorentz
from rfx.model import dispersive_poles
from rfx.materials.pole_map import SampledPoleMap, map_terms


@pytest.fixture(autouse=True)
def default_check(monkeypatch):
    monkeypatch.delenv("RFX_POLE_UNDERFLOW_CHECK", raising=False)
    jax.clear_caches()
    yield
    jax.effects_barrier()
    jax.clear_caches()


def objective(site, small):
    """One scalar entry or eight occupied map cells.

    The squared coefficient makes the incoming cotangent depend on kappa in
    every vmap lane.
    """
    scale = 1e-7 if small else 1.

    def evaluate(kappa, lane_scale=scale):
        if site == "scalar":
            c = lorentz.pole_update_coeffs(0., 2e9, kappa, 1e-12)[2]
        else:
            shape = (2, 2, 2)
            pole = SampledPoleMap(0., np.full(shape, 2e9, np.float32),
                                  jnp.broadcast_to(kappa, shape), (), None, True)
            sizes = (np.array([1., 1.2], np.float32), np.ones(2), np.ones(2)) if site == "graded_map" else None
            _, _, terms = map_terms(pole, None, 1e-12, shape, None,
                                    periodic=(True,) * 3, high_wall=(False,) * 3,
                                    cell_sizes=sizes)
            c = terms[0]
        return jnp.sum(c * c) * (lane_scale / 1e-15)

    return evaluate


def observe(monkeypatch, fn, value):
    calls = []
    report = dispersive_poles._report_pole_underflow

    def spy(count):
        calls.append(int(count))
        report(count)

    monkeypatch.setattr(dispersive_poles, "_report_pole_underflow", spy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AccuracyWarning)
        result = jax.block_until_ready(fn(value))
        jax.effects_barrier()
    notices = [w.message for w in caught if issubclass(w.category, AccuracyWarning)]
    return result, calls, notices


@pytest.mark.parametrize("site", ["scalar", "map", "graded_map"])
@pytest.mark.parametrize("setting, enabled", [(None, True), ("0", False), ("1", True), ("other", True)])
def test_default_jaxpr(monkeypatch, site, setting, enabled):
    if setting is not None:
        monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", setting)
    program = str(jax.make_jaxpr(jax.grad(objective(site, True)))(jnp.float32(3e20)))
    assert ("custom_linear_solve" in program) is enabled


@pytest.mark.parametrize("setting", [None, "0", "1", "other"])
def test_no_callback_backend(monkeypatch, setting):
    """A backend without host callbacks: the default is off and an explicit 1 refuses.

    The simulated MPS capability of ``test_each_owner_refuses_traced_operands``
    serves both the default decision and the traced-check admission. The
    scalar scene isolates the SI guard from map normalization's separate
    RelativeStrength admission when a jaxpr is staged.
    """
    no_callbacks = backend.backend_class("mps")
    monkeypatch.setattr(backend, "backend_class", lambda: no_callbacks)
    monkeypatch.setattr(_traced_check, "backend_class", lambda: no_callbacks)
    if setting is not None:
        monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", setting)
    grad = jax.grad(objective("scalar", True))
    value = jnp.float32(3e20)
    if setting == "1":
        message = (
            "RFX_POLE_UNDERFLOW_CHECK: pole strength cotangent underflow: "
            "this backend (mps) cannot run a host check inside compiled code"
        )
        with pytest.raises(NotImplementedError) as caught:
            jax.make_jaxpr(grad)(value)
        assert str(caught.value) == message
        with pytest.raises(NotImplementedError) as caught:
            grad(value)
        assert str(caught.value) == message
    else:
        assert "custom_linear_solve" not in str(jax.make_jaxpr(grad)(value))
        result, calls, notices = observe(monkeypatch, grad, value)
        assert np.isfinite(result)
        assert calls == []
        assert notices == []


@pytest.mark.parametrize("setting", [None, "0", "1", "other"])
def test_gpu_without_cpu_device(monkeypatch, setting):
    gpu = backend.backend_class("gpu")
    monkeypatch.setattr(backend, "backend_class", lambda: gpu)
    monkeypatch.setattr(_traced_check, "backend_class", lambda: gpu)
    cpu_error = RuntimeError("Unknown backend cpu")

    def no_cpu_device():
        raise cpu_error

    monkeypatch.setattr(backend, "host_cpu_device", no_cpu_device)
    monkeypatch.setattr(_traced_check, "host_cpu_device", no_cpu_device)
    if setting is not None:
        monkeypatch.setenv("RFX_POLE_UNDERFLOW_CHECK", setting)
    grad = jax.grad(objective("scalar", True))
    value = jnp.float32(3e20)
    if setting == "1":
        for run in (jax.make_jaxpr(grad), grad):
            with pytest.raises(RuntimeError, match="JAX_PLATFORMS") as caught:
                run(value)
            assert caught.value.__cause__ is cpu_error
    else:
        assert "custom_linear_solve" not in str(jax.make_jaxpr(grad)(value))
        result, calls, notices = observe(monkeypatch, grad, value)
        assert np.isfinite(result)
        assert calls == []
        assert notices == []


@pytest.mark.parametrize("site, expected_count", [("scalar", 2), ("map", 16), ("graded_map", 16)])
def test_vmapped_grad_mixed_lanes(monkeypatch, site, expected_count):
    grad = jax.grad(objective(site, False))
    fn = jax.jit(lambda scales: jax.vmap(grad, in_axes=(None, 0))(jnp.float32(3e20), scales))
    scales = jnp.array([1e-7, 1., 1e-7, 1.], dtype=jnp.float32)
    _, calls, notices = observe(monkeypatch, fn, scales)
    diagnostic = f"host_counts={calls}, warning_counts={[w.count for w in notices]}"
    assert len(calls) == 1, diagnostic
    assert len(notices) == 1, diagnostic
    assert calls == [expected_count], diagnostic
    assert notices[0].count == expected_count, diagnostic


@pytest.mark.parametrize("site, entries", [("scalar", 1), ("map", 8), ("graded_map", 8)])
@pytest.mark.parametrize("batch", [4, 64])
@pytest.mark.parametrize("small", [False, True], ids=["normal", "underflow"])
def test_vmapped_grad_one_host_call(monkeypatch, site, entries, batch, small):
    fn = jax.jit(jax.vmap(jax.grad(objective(site, small))))
    _, calls, notices = observe(monkeypatch, fn, jnp.linspace(3e20, 4e20, batch, dtype=jnp.float32))
    assert len(calls) == 1, (
        f"batch={batch}, host_calls={len(calls)}, counts={calls}, "
        f"warnings={len(notices)}, warning_counts={[w.count for w in notices]}")
    assert calls == [entries * batch if small else 0]
    assert len(notices) == (1 if small else 0)
    if small:
        assert notices[0].count == entries * batch


@pytest.mark.parametrize("site, entries", [("scalar", 1), ("map", 8), ("graded_map", 8)])
@pytest.mark.parametrize("small", [False, True], ids=["normal", "underflow"])
def test_unbatched_host_calls(monkeypatch, site, entries, small):
    _, calls, notices = observe(monkeypatch, jax.jit(jax.grad(objective(site, small))), jnp.float32(3e20))
    assert len(calls) == (1 if small else 0), calls
    assert calls == ([entries] if small else [])
    assert len(notices) == (1 if small else 0)
    if small:
        assert notices[0].count == entries
