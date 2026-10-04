"""The 2-D ADI eps/sigma derivatives are finite after the unit fix (#1357).

The SI coupling's reverse pass squared a denominator near 1e-23, producing
NaN at every objective scale. #1450 refused that derivative (refs #1373);
the shared eps_r-unit coefficient helper now removes the cause.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from tests._x64_compat import enable_x64


def _sim(solver="adi", mode="2d_tmz", boundary="pec"):
    dz = 0.01 if mode == "3d" else 1e-3
    extra = {"adi_cfl_factor": 1} if solver == "adi" else {}
    sim = Simulation(freq_max=1e10, domain=(0.01, 0.01, dz), dx=1e-3,
                     boundary=boundary, solver=solver, mode=mode, **extra)
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


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.parametrize("which", ["eps", "sigma"])
def test_adi_2d_all_material_derivative_modes_match_main_fd(which, boundary):
    """Replace #1450's finiteness refusal in forward and reverse modes.

    Nu's exact 40-step model: eps at 1, sigma at 0. CPML uses the default
    ten layers, whose absorbing sigma is added after material overrides.
    The FD values below were measured on main 9b82b751 with h=1e-3; all
    base/perturbed forward traces were byte-identical to this branch.
    """
    # A scalar override is the homogeneous fill: ADI refuses a traced array
    # (#1373), and a uniform array carried the same derivative.
    sim, shape = _sim(boundary=boundary)
    if which == "eps":
        def f(s):
            return _energy(sim, eps_override=s)
        x0, h = 1.0, 1e-3
    else:
        def f(s):
            return _energy(sim, sigma_override=s)
        x0, h = 0.0, 1e-3
    main_fd = {
        ("pec", "eps"): 1.2000781680399086e-5,
        ("pec", "sigma"): -2.254637365695089e-6,
        ("cpml", "eps"): 3.00915398838697e-5,
        ("cpml", "sigma"): -2.3598657207912765e-5,
    }[boundary, which]
    x0 = jnp.float32(x0)
    # Keep forward modes explicit: reverse-mode finiteness alone did not
    # address the original jacfwd/jvp NaNs reported by #1450's reviewer.
    derivatives = {
        "jacfwd": jax.jacfwd(f)(x0),
        "jvp": jax.jvp(f, (x0,), (jnp.ones_like(x0),))[1],
        "grad": jax.grad(f)(x0),
        "vjp": jax.vjp(f, x0)[1](jnp.ones_like(f(x0)))[0],
        "vmap(grad)": jax.vmap(jax.grad(f))(jnp.stack([x0, x0])),
        "jit(grad)": jax.jit(jax.grad(f))(x0),
    }
    fd = float((f(x0 + h) - f(x0 - h)) / (2 * h))
    np.testing.assert_allclose(fd, main_fd, rtol=3e-3, atol=0)
    for name, derivative in derivatives.items():
        assert np.all(np.isfinite(derivative)), name
        np.testing.assert_allclose(derivative, main_fd, rtol=3e-3, atol=0,
                                   err_msg=name)


def test_adi_2d_value_runs_eager_and_under_jit():
    sim, shape = _sim()
    eps = jnp.float32(1.5)  # a scalar fill: ADI refuses a traced array (#1373)
    eager = float(_energy(sim, eps_override=eps))
    jitted = float(jax.jit(lambda e: _energy(sim, eps_override=e))(eps))
    assert np.isfinite(eager) and eager > 0
    np.testing.assert_allclose(jitted, eager, rtol=1e-4)


@pytest.mark.parametrize("solver,mode", [("adi", "3d"), ("yee", "2d_tmz")])
def test_other_solvers_still_differentiate(solver, mode):
    sim, shape = _sim(solver, mode)
    # ADI takes the homogeneous fill as a scalar; it refuses a traced array (#1373).
    fill = (lambda s: s) if solver == "adi" else (lambda s: jnp.ones(shape) * s)
    g = float(jax.grad(lambda s: _energy(sim, eps_override=fill(s)))(1.0))
    assert np.isfinite(g) and g != 0.0


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.parametrize("which", ["eps", "sigma"])
def test_adi_2d_bin_gradients_match_fd_and_a_twice_longer_record(which, boundary):
    """A settled lossy record: each material parameter, each physical bin.

    The 40-step tests above check a finite-record derivative; their source
    has not finished. This separate witness uses eps_r=1.5, sigma=0.2 S/m
    and 1024/2048 steps. Fixed 3, 5 and 7 GHz quadrature bins and a fixed
    integration weight make the two record lengths comparable. Float64
    SI-primal differences avoid cancellation of the small sigma response.
    Both comparisons retain the existing 3e-3 relative FD tolerance.
    """
    sim, shape = _sim(boundary=boundary)
    dt = float(sim._build_grid().dt)
    index = 0 if which == "eps" else 1
    point = np.array([1.5, 0.2], dtype=np.float32)

    def objective_for(n_steps):
        phase = jnp.exp(
            -2j * jnp.pi * jnp.arange(n_steps, dtype=jnp.float32)[:, None]
            * jnp.asarray(np.array([3e9, 5e9, 7e9]) * dt,
                          dtype=jnp.float32)[None, :])

        def objective(parameter):
            params = jnp.asarray(point, dtype=parameter.dtype).at[index].set(parameter)
            ts = sim.forward(
                n_steps=n_steps, skip_preflight=True,
                # Scalar fills: ADI refuses a traced array override (#1373).
                eps_override=params[0],
                sigma_override=params[1],
            ).time_series[:, 0]
            spectrum = (dt / 1e-9) * jnp.sum(ts[:, None] * phase, axis=0)
            return jnp.concatenate((jnp.sum(ts ** 2)[None],
                                    jnp.real(spectrum * jnp.conj(spectrum))))

        return objective

    short, long = objective_for(1024), objective_for(2048)
    parameter = jnp.asarray(point[index])
    gradient = np.asarray(jax.jacrev(short)(parameter))
    longer_gradient = np.asarray(jax.jacrev(long)(parameter))
    assert np.all(np.isfinite(gradient)) and np.all(gradient != 0)
    np.testing.assert_allclose(longer_gradient, gradient, rtol=3e-3, atol=0)

    with enable_x64():
        parameter64 = jnp.asarray(float(point[index]), dtype=jnp.float64)
        h = 1e-3 if which == "eps" else 1e-4
        fd = np.asarray((short(parameter64 + h) - short(parameter64 - h)) / (2 * h))
    np.testing.assert_allclose(gradient, fd, rtol=3e-3, atol=0)
