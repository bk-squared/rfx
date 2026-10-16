"""Default compilation and first-call host-read fallback (issue 1772)."""

from __future__ import annotations

import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warnings

from rfx.topology import TopologyDesignRegion, topology_optimize

from rfx import Simulation
from rfx.optimize import DesignRegion, optimize

_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"
_COUNTER = {"on": False, "n": 0}
_LISTENING = []


def _listener(event, duration, **kwargs):
    if _COUNTER["on"] and event == _BACKEND_COMPILE:
        _COUNTER["n"] += 1


def _compiles(fn, *args):
    """Run ``fn(*args)`` to completion; return (result, backend compiles)."""
    if not _LISTENING:
        jax.monitoring.register_event_duration_secs_listener(_listener)
        _LISTENING.append(True)
    _COUNTER["n"] = 0
    _COUNTER["on"] = True
    try:
        out = jax.block_until_ready(fn(*args))
    finally:
        _COUNTER["on"] = False
    return out, _COUNTER["n"]


def _tiny_sim():
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.011, 0.0075, 0.0075), "ez")
    return sim


def _tiny_region():
    return DesignRegion(corner_lo=(0.006, 0.006, 0.006),
                        corner_hi=(0.010, 0.010, 0.009), eps_range=(1.0, 4.4))


def _energy(result):
    return -jnp.sum(result.time_series ** 2)


def _run(n_iters, objective=_energy, **kwargs):
    return optimize(_tiny_sim(), _tiny_region(), objective,
                    n_iters=n_iters, lr=0.2, n_steps=20, verbose=False,
                    skip_preflight=True, **kwargs)


def _difference(actual, eager):
    actual, eager = np.asarray(actual), np.asarray(eager)
    delta = np.max(np.abs(actual - eager))
    scale = np.max(np.abs(eager))
    print(f"max difference={delta:.12g}, eager peak={scale:.12g}", file=sys.stderr)
    assert delta <= 1e-4 * scale


def test_default_compiles_once():
    _run(1)
    _, n1 = _compiles(_run, 1)
    result, n4 = _compiles(_run, 4)
    print(f"T1 backend compiles: {n1}, {n4}", file=sys.stderr)
    assert n1 >= 1
    assert n4 == n1
    assert len(result.loss_history) == 4
    calls = []

    def objective(result):
        calls.append(None)
        return _energy(result)

    _run(4, objective)
    assert len(calls) == 1
    calls.clear()
    _run(4, objective, jit=False)
    assert len(calls) == 4


def test_jit_agrees_with_eager():
    actual, eager = _run(3), _run(3, jit=False)
    _difference(actual.loss_history, eager.loss_history)
    _difference(actual.latent, eager.latent)


def test_untraceable_warns_once_for_all_starts():
    def objective(result):
        return -jnp.sum(result.time_series ** 2) * float(jnp.asarray(2.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _run(3, objective, n_starts=2)
    fallback = [w for w in caught if "objective reads a traced value" in str(w.message)]
    assert len(fallback) == 1
    assert fallback[0].category is UserWarning
    assert "optimize: ConcretizationTypeError:" in str(fallback[0].message)
    assert "every iteration now recompiles the whole simulation" in str(fallback[0].message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eager = _run(3, objective, n_starts=2, jit=False)
    assert not [w for w in caught if "objective reads a traced value" in str(w.message)]
    np.testing.assert_array_equal(actual.loss_history, eager.loss_history)
    np.testing.assert_array_equal(actual.latent, eager.latent)


def test_non_host_error_propagates():
    def objective(result):
        # Eager AD also traces the result. A constant only traces under jit,
        # so catching Exception and retrying eagerly would wrongly succeed.
        if isinstance(jnp.asarray(2.0), jax.core.Tracer):
            raise ValueError("objective failure")
        return _energy(result)

    with pytest.raises(ValueError, match="objective failure"):
        _run(1, objective)


def _topology(n_iterations, **kwargs):
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    region = TopologyDesignRegion(
        corner_lo=(0.006, 0.006, 0.006), corner_hi=(0.009, 0.009, 0.009),
        material_bg="air", material_fg="fr4")
    return topology_optimize(sim, region, _energy, n_iterations=n_iterations,
                             learning_rate=0.05, beta_schedule=[(0, 1.0), (2, 4.0)],
                             verbose=False, skip_preflight=True, **kwargs)


@pytest.mark.skipif(not importlib.util.find_spec("optax"), reason="optax not installed")
def test_topology_dynamic_beta_compiles_once():
    _topology(4)
    _, n1 = _compiles(_topology, 1)
    actual, n4 = _compiles(_topology, 4)
    print(f"T5 backend compiles: {n1}, {n4}", file=sys.stderr)
    assert n1 >= 1
    assert n4 == n1
    assert actual.beta_history == [1.0, 1.0, 4.0, 4.0]
    eager = _topology(4, jit=False)
    _difference(actual.history, eager.history)


def test_checkpoint_segments_work_under_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _run(3, checkpoint_segments=4)
    assert not [w for w in caught if "objective reads a traced value" in str(w.message)]
    eager = _run(3, checkpoint_segments=4, jit=False)
    _difference(actual.loss_history, eager.loss_history)
    _difference(actual.latent, eager.latent)


def test_distributed_default_stays_eager_on_a_gpu_mesh(monkeypatch):
    """A multi-device run on GPUs refuses an enclosing jit, so ``distributed=True``
    keeps the eager loop by default; an explicit ``jit=True`` still refuses loudly."""
    if jax.device_count() < 2:
        pytest.skip("needs two devices")
    import rfx.stepping.rank as rank
    from tests.unit.autodiff.test_progressive_optimize import _make_nu_factory
    monkeypatch.setattr(rank, "_is_gpu_mesh", lambda mesh: True)
    region = DesignRegion(corner_lo=(0.003, 0.003, 0.004), corner_hi=(0.007, 0.007, 0.006),
                          eps_range=(1.0, 4.0))
    def run(**kw):
        return optimize(_make_nu_factory()(1e-3), region, lambda r: -jnp.sum(r.time_series ** 2),
                        n_iters=1, lr=0.05, n_steps=10, verbose=False, skip_preflight=True,
                        distributed=True, **kw)
    default, eager = run(), run(jit=False)
    np.testing.assert_array_equal(default.loss_history, eager.loss_history)
    with pytest.raises(NotImplementedError):
        run(jit=True)
