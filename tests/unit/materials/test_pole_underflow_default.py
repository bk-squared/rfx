"""Default-on SI checks and one host report for a batched reverse pass."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx.materials import lorentz
from rfx.materials.pole_map import SampledPoleMap, map_terms


@pytest.fixture(autouse=True)
def default_check(monkeypatch):
    monkeypatch.delenv("RFX_POLE_UNDERFLOW_CHECK", raising=False)
    jax.clear_caches()
    yield
    jax.effects_barrier()
    jax.clear_caches()


def objective(site, small):
    # One scalar entry or eight occupied map cells. The squared coefficient
    # makes the incoming cotangent depend on kappa in every vmap lane.
    scale = 1e-7 if small else 1.

    def evaluate(kappa):
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
        return jnp.sum(c * c) * (scale / 1e-15)

    return evaluate


def observe(monkeypatch, fn, value):
    calls = []
    report = lorentz._report_pole_underflow

    def spy(count):
        calls.append(int(count))
        report(count)

    monkeypatch.setattr(lorentz, "_report_pole_underflow", spy)
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
