"""The 2-D ADI solver refuses a derivative with respect to eps or sigma (#1373).

A 2-D TMz box with a soft source, stepped by ADI: ``jax.grad`` of the probe
energy through ``eps_override`` or ``sigma_override`` returned NaN, while a
central difference of the same function gives a finite number (1.2e-5 and
-2.3e-6 at the model below) and the 3-D ADI and 2-D Yee solvers return
gradients that match their central differences. The value is right; only the
derivative of the 2-D ADI update is not. Until that is fixed the derivative is
refused when it is requested, before any time step, and the value -- eager
and under ``jax.jit`` -- runs as before.

Mutation: drop the refusal (pass the material arrays straight through) and
the two refusal cases return NaN instead of raising.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.api import _execute


def _sim(solver="adi", mode="2d_tmz"):
    dz = 0.01 if mode == "3d" else 1e-3
    extra = {"adi_cfl_factor": 1} if solver == "adi" else {}
    sim = Simulation(freq_max=1e10, domain=(0.01, 0.01, dz), dx=1e-3,
                     boundary="pec", solver=solver, mode=mode, **extra)
    z = 0.005 if mode == "3d" else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_source((0.005, 0.005, z), "ez",
                       waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((0.006, 0.005, z), "ez")
    return sim, sim._build_grid().shape


def _energy(sim, **override):
    return jnp.sum(sim.forward(n_steps=40, skip_preflight=True,
                               **override).time_series ** 2)


@pytest.mark.parametrize("which", ["eps", "sigma"])
def test_adi_2d_material_gradient_is_refused_before_stepping(which, monkeypatch):
    sim, shape = _sim()

    def stepped(*args, **kwargs):
        pytest.fail("the 2-D ADI stepper was entered before the refusal")

    monkeypatch.setattr(_execute, "run_adi_2d", stepped)
    if which == "eps":
        f = lambda s: _energy(sim, eps_override=jnp.ones(shape) * s)
        x0 = 1.0
    else:
        f = lambda s: _energy(sim, sigma_override=jnp.zeros(shape) + s)
        x0 = 0.0
    with pytest.raises(NotImplementedError, match="cannot differentiate"):
        jax.grad(f)(x0)


def test_adi_2d_value_runs_eager_and_under_jit():
    sim, shape = _sim()
    eps = jnp.ones(shape) * 1.5
    eager = float(_energy(sim, eps_override=eps))
    jitted = float(jax.jit(lambda e: _energy(sim, eps_override=e))(eps))
    assert np.isfinite(eager) and eager > 0
    np.testing.assert_allclose(jitted, eager, rtol=1e-4)


@pytest.mark.parametrize("solver,mode", [("adi", "3d"), ("yee", "2d_tmz")])
def test_other_solvers_still_differentiate(solver, mode):
    sim, shape = _sim(solver, mode)
    g = float(jax.grad(lambda s: _energy(sim, eps_override=jnp.ones(shape) * s))(1.0))
    assert np.isfinite(g) and g != 0.0
