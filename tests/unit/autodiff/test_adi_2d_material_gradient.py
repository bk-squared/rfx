"""The 2-D ADI eps/sigma derivatives are finite after the unit fix (#1357).

The SI coupling's reverse pass squared a denominator near 1e-23, producing
NaN at every objective scale. #1373 refused that derivative pending a fix;
the shared eps_r-unit coefficient helper now removes the cause.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation


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
def test_adi_2d_material_gradient_matches_finite_difference(which):
    sim, shape = _sim()
    if which == "eps":
        f = lambda s: _energy(sim, eps_override=jnp.ones(shape) * s)
        x0, h = 1.0, 1e-3
    else:
        f = lambda s: _energy(sim, sigma_override=jnp.zeros(shape) + s)
        # A 1e-4 step loses digits when subtracting the float32 energies.
        x0, h = 0.01, 1e-3
    grad = float(jax.grad(f)(x0))
    tangent = float(jax.jvp(f, (x0,), (1.0,))[1])
    fd = float((f(x0 + h) - f(x0 - h)) / (2 * h))
    assert np.isfinite(grad) and grad != 0.0
    np.testing.assert_allclose(grad, tangent, rtol=1e-4)
    np.testing.assert_allclose(grad, fd, rtol=3e-3)


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
