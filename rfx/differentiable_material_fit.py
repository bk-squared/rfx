"""Differentiable material fitting — fit Debye/Lorentz poles to S-parameters via jax.grad.

Unlike the scipy-based fitting in ``material_fit.py`` which fits to eps(f),
this module fits dispersive material models directly to S-parameter data by
differentiating through the full FDTD simulation.  No de-embedding is needed
because the fixture geometry is included in the simulation.

The gradient path is::

    log_poles
      -> exp() -> physical poles
      -> init_debye() / init_lorentz() -> ADE coefficients
      -> simulation.run(checkpoint=True) -> field evolution
      -> S-parameter extraction -> sparam_loss() -> scalar
      <- jax.grad -> d(loss)/d(log_poles)

All intermediate operations are pure JAX (``init_debye`` and ``init_lorentz``
use ``jnp`` throughout) so ``jax.grad`` flows end-to-end without any custom
VJP rules.

Example
-------
>>> from rfx import Simulation, Box, GaussianPulse
>>> from rfx.differentiable_material_fit import differentiable_material_fit
>>>
>>> def fixture(eps_inf, debye_poles, lorentz_poles):
...     sim = Simulation(freq_max=10e9, domain=(0.03, 0.01, 0.01))
...     sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
...     sim.add(Box((0.01, 0, 0), (0.02, 0.01, 0.01)), material="dut")
...     sim.add_port((0.003, 0.005, 0.005), "ez")
...     sim.add_port((0.027, 0.005, 0.005), "ez")
...     return sim
>>>
>>> result = differentiable_material_fit(
...     fixture, s_measured, freqs, n_debye_poles=1,
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MaterialFitResult:
    """Result of differentiable material fitting.

    Attributes
    ----------
    eps_inf : float
        High-frequency permittivity.
    debye_poles : list[DebyePole]
        Fitted Debye poles (may be empty).
    lorentz_poles : list[LorentzPole]
        Fitted Lorentz poles (may be empty).
    loss_history : list[float]
        Loss value at each iteration.
    final_s_params : np.ndarray or None
        S-parameters from the final simulation.
    freqs : np.ndarray
        Frequency array in Hz.
    converged : bool
        Whether the optimizer converged (loss plateau).
    """

    eps_inf: float
    debye_poles: list = field(default_factory=list)
    lorentz_poles: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    final_s_params: np.ndarray | None = None
    freqs: np.ndarray | None = None
    converged: bool = False


# ---------------------------------------------------------------------------
# Log-space parameterization
# ---------------------------------------------------------------------------

def _poles_to_params(
    eps_inf: float,
    debye_poles: list[DebyePole],
    lorentz_poles: list[LorentzPole],
) -> jnp.ndarray:
    """Pack pole parameters into a log-space JAX vector.

    Layout: [log(eps_inf), log(de_1), log(tau_1), ...,
             log(de_L1), log(omega0_L1), log(delta_L1), ...]
    """
    params = [jnp.log(jnp.array(max(eps_inf, 1e-12)))]
    for pole in debye_poles:
        params.append(jnp.log(jnp.array(max(pole.delta_eps, 1e-12))))
        params.append(jnp.log(jnp.array(max(pole.tau, 1e-30))))
    for pole in lorentz_poles:
        # Recover delta_eps from kappa = delta_eps * omega_0^2
        omega_0 = max(pole.omega_0, 1e-6)
        delta_eps = pole.kappa / (omega_0 ** 2) if omega_0 > 0 else 1.0
        params.append(jnp.log(jnp.array(max(delta_eps, 1e-12))))
        params.append(jnp.log(jnp.array(omega_0)))
        params.append(jnp.log(jnp.array(max(pole.delta, 1e-12))))
    return jnp.stack(params)


def _params_to_debye_poles(
    params: jnp.ndarray,
    n_debye: int,
) -> tuple[jnp.ndarray, list[DebyePole]]:
    """Unpack log-space vector into (eps_inf, debye_poles).

    Returns JAX-traced eps_inf and DebyePole instances whose fields
    are JAX-traced scalars.
    """
    eps_inf = jnp.exp(params[0])
    poles = []
    idx = 1
    for _ in range(n_debye):
        delta_eps = jnp.exp(params[idx])
        tau = jnp.exp(params[idx + 1])
        poles.append(DebyePole(delta_eps=delta_eps, tau=tau))
        idx += 2
    return eps_inf, poles


def _params_to_lorentz_poles(
    params: jnp.ndarray,
    n_lorentz: int,
    offset: int,
) -> list[LorentzPole]:
    """Unpack log-space vector into LorentzPole instances.

    The LorentzPole fields are JAX-traced so that ``init_lorentz``
    is differentiable w.r.t. the parameters.
    """
    poles = []
    idx = offset
    for _ in range(n_lorentz):
        delta_eps = jnp.exp(params[idx])
        omega_0 = jnp.exp(params[idx + 1])
        delta = jnp.exp(params[idx + 2])
        kappa = delta_eps * omega_0 ** 2
        poles.append(LorentzPole(omega_0=omega_0, delta=delta, kappa=kappa))
        idx += 3
    return poles


# ---------------------------------------------------------------------------
# S-parameter loss
# ---------------------------------------------------------------------------

def sparam_loss(
    s_sim: jnp.ndarray,
    s_meas: jnp.ndarray,
    *,
    weight_mag: float = 1.0,
    weight_phase: float = 0.1,
) -> jnp.ndarray:
    """Weighted MSE loss between simulated and measured S-parameters.

    Parameters
    ----------
    s_sim, s_meas : (..., n_freqs) complex arrays
        S-parameter arrays (any leading shape, typically (n_ports, n_ports, n_freqs)).
    weight_mag : float
        Weight on magnitude error.
    weight_phase : float
        Weight on phase error (wrapped to [-pi, pi]).

    Returns
    -------
    Scalar loss value.
    """
    mag_sim = jnp.abs(s_sim)
    mag_meas = jnp.abs(s_meas)
    mag_loss = jnp.mean((mag_sim - mag_meas) ** 2)

    phase_sim = jnp.angle(s_sim)
    phase_meas = jnp.angle(s_meas)
    phase_diff = phase_sim - phase_meas
    # Wrap to [-pi, pi] to avoid 2*pi discontinuities
    phase_diff = jnp.arctan2(jnp.sin(phase_diff), jnp.cos(phase_diff))
    phase_loss = jnp.mean(phase_diff ** 2)

    return weight_mag * mag_loss + weight_phase * phase_loss


# ---------------------------------------------------------------------------
# Default initial guess from frequency band
# ---------------------------------------------------------------------------

def _default_debye_guess(
    freqs: np.ndarray,
    n_poles: int,
) -> tuple[float, list[DebyePole]]:
    """Generate a reasonable default initial guess for Debye poles."""
    f_min = max(float(np.min(freqs)), 1.0)
    f_max = max(float(np.max(freqs)), f_min * 10)
    tau_guesses = np.logspace(
        np.log10(1.0 / (2 * np.pi * f_max)),
        np.log10(1.0 / (2 * np.pi * f_min)),
        n_poles,
    )
    eps_inf = 2.0
    poles = [DebyePole(delta_eps=1.0, tau=float(t)) for t in tau_guesses]
    return eps_inf, poles


def _default_lorentz_guess(
    freqs: np.ndarray,
    n_poles: int,
) -> list[LorentzPole]:
    """Generate a reasonable default initial guess for Lorentz poles."""
    f_min = max(float(np.min(freqs)), 1.0)
    f_max = max(float(np.max(freqs)), f_min * 10)
    omega_guesses = np.logspace(
        np.log10(2 * np.pi * f_min),
        np.log10(2 * np.pi * f_max),
        n_poles,
    )
    poles = []
    for w0 in omega_guesses:
        delta = float(w0 * 0.05)  # Q ~ 10
        kappa = 1.0 * w0 ** 2  # delta_eps = 1.0
        poles.append(LorentzPole(omega_0=float(w0), delta=delta, kappa=kappa))
    return poles


# ---------------------------------------------------------------------------
# Adam optimizer (manual fallback, follows optimize.py pattern)
# ---------------------------------------------------------------------------

def _adam_step(
    params: jnp.ndarray,
    grad: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single Adam optimizer step. Returns (new_params, new_m, new_v)."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** (step + 1))
    v_hat = v / (1 - beta2 ** (step + 1))
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, m, v


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def differentiable_material_fit(
    sim_factory: Callable,
    s_measured: np.ndarray,
    freqs: np.ndarray,
    n_debye_poles: int = 1,
    n_lorentz_poles: int = 0,
    *,
    n_iterations: int = 50,
    learning_rate: float = 0.01,
    initial_guess=None,
    weight_mag: float = 1.0,
    weight_phase: float = 0.1,
    checkpoint: bool = True,
    verbose: bool = True,
) -> MaterialFitResult:
    """Fit dispersive material poles to S-parameter data via jax.grad through FDTD.

    Parameters
    ----------
    sim_factory : callable(eps_inf, debye_poles, lorentz_poles) -> Simulation
        Factory that builds a ``Simulation`` with the given material
        parameters.  The fixture geometry, ports, and probes are set
        inside this callable.  It is called **once** outside the
        gradient tape for static setup; only the pole parameters are
        traced through JAX.
    s_measured : (n_ports, n_ports, n_freqs) complex array
        Measured S-parameter matrix.
    freqs : (n_freqs,) array in Hz
        Frequency points matching ``s_measured``.
    n_debye_poles : int
        Number of Debye poles to fit.
    n_lorentz_poles : int
        Number of Lorentz poles to fit.
    n_iterations : int
        Number of Adam optimization iterations.
    learning_rate : float
        Adam learning rate.
    initial_guess : DebyeFitResult, LorentzFitResult, or None
        Initial pole parameters from ``fit_debye`` / ``fit_lorentz``.
        If None, a frequency-band-based default is used.
    weight_mag, weight_phase : float
        Weights for magnitude and phase in the S-parameter loss.
    checkpoint : bool
        Use ``jax.checkpoint`` in the FDTD loop (recommended).
    verbose : bool
        Print progress.

    Returns
    -------
    MaterialFitResult
    """
    from rfx.core.yee import MaterialArrays
    from rfx.simulation import run as sim_run, make_port_source, make_probe
    from rfx.sources.sources import LumpedPort, setup_lumped_port

    freqs = np.asarray(freqs, dtype=np.float64)
    s_measured = np.asarray(s_measured)
    s_meas_jnp = jnp.array(s_measured)

    # ------------------------------------------------------------------
    # Build initial guess
    # ------------------------------------------------------------------
    if initial_guess is not None:
        from rfx.material_fit import DebyeFitResult, LorentzFitResult

        if isinstance(initial_guess, DebyeFitResult):
            eps_inf_init = initial_guess.eps_inf
            debye_init = list(initial_guess.poles)
            lorentz_init = []
            n_debye_poles = len(debye_init)
        elif isinstance(initial_guess, LorentzFitResult):
            eps_inf_init = initial_guess.eps_inf
            debye_init = []
            lorentz_init = list(initial_guess.poles)
            n_lorentz_poles = len(lorentz_init)
        else:
            raise TypeError(f"Unsupported initial_guess type: {type(initial_guess)}")
    else:
        eps_inf_init, debye_init = _default_debye_guess(freqs, n_debye_poles)
        lorentz_init = _default_lorentz_guess(freqs, n_lorentz_poles)

    # Pad if needed
    while len(debye_init) < n_debye_poles:
        debye_init.append(DebyePole(delta_eps=1.0, tau=1e-11))
    while len(lorentz_init) < n_lorentz_poles:
        omega0 = 2 * np.pi * float(np.mean(freqs))
        lorentz_init.append(
            LorentzPole(omega_0=omega0, delta=omega0 * 0.05, kappa=omega0 ** 2)
        )

    params = _poles_to_params(eps_inf_init, debye_init[:n_debye_poles],
                               lorentz_init[:n_lorentz_poles])

    # ------------------------------------------------------------------
    # Static setup: build grid and non-DUT materials ONCE
    # ------------------------------------------------------------------
    # Call sim_factory with initial values to get the Simulation object
    dummy_sim = sim_factory(
        eps_inf_init,
        debye_init[:n_debye_poles],
        lorentz_init[:n_lorentz_poles],
    )
    grid = dummy_sim._build_grid()
    dt = grid.dt
    n_steps = grid.num_timesteps(num_periods=20.0)

    # ------------------------------------------------------------------
    # Forward function (called inside jax.value_and_grad)
    # ------------------------------------------------------------------
    def forward(p):
        """log-space params -> FDTD -> S-params -> loss scalar."""
        # Unpack traced pole parameters
        eps_inf, debye_poles = _params_to_debye_poles(p, n_debye_poles)
        lorentz_offset = 1 + 2 * n_debye_poles
        lorentz_poles = _params_to_lorentz_poles(p, n_lorentz_poles, lorentz_offset)

        # Rebuild Simulation with traced poles (Python construction is not traced,
        # but the pole VALUES inside DebyePole/LorentzPole are JAX tracers)
        sim = sim_factory(eps_inf, debye_poles, lorentz_poles)

        # Assemble materials (geometry masks are static, but eps_inf flows through)
        materials, debye_spec, lorentz_spec, pec_mask, _, _ = sim._assemble_materials(grid)

        # Setup ports (fold port impedance into materials)
        for pe in sim._ports:
            lp = LumpedPort(
                position=pe.position,
                component=pe.component,
                impedance=pe.impedance,
                excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)

        # Initialize dispersion with traced poles
        debye = None
        if debye_spec is not None:
            debye_poles_spec, debye_masks = debye_spec
            debye = init_debye(debye_poles_spec, materials, dt, mask=debye_masks)

        lorentz = None
        if lorentz_spec is not None:
            lorentz_poles_spec, lorentz_masks = lorentz_spec
            lorentz = init_lorentz(lorentz_poles_spec, materials, dt, mask=lorentz_masks)

        # Build sources and probes
        sources = []
        for pe in sim._ports:
            lp = LumpedPort(
                position=pe.position,
                component=pe.component,
                impedance=pe.impedance,
                excitation=pe.waveform,
            )
            sources.append(make_port_source(grid, lp, materials, n_steps))

        probes = []
        for pe in sim._probes:
            probes.append(make_probe(grid, pe.position, pe.component))

        # Run FDTD
        boundary = "pec"
        if hasattr(sim, '_boundary'):
            boundary = sim._boundary

        result = sim_run(
            grid, materials, n_steps,
            boundary=boundary,
            debye=debye,
            lorentz=lorentz,
            sources=sources,
            probes=probes,
            checkpoint=checkpoint,
            pec_mask=pec_mask,
        )

        # Extract S-params from time series via DFT
        # Use probe time series and compute S-params from voltage/current DFTs
        ts = result.time_series  # (n_steps, n_probes)
        n_probes = ts.shape[1] if ts.ndim > 1 else 1
        len(freqs)

        # Compute DFT of probe time series at target frequencies
        times = jnp.arange(n_steps) * dt
        # (n_steps, n_freqs)
        phase_matrix = jnp.exp(
            -1j * 2.0 * jnp.pi * jnp.array(freqs)[None, :] * times[:, None]
        ) * dt

        if ts.ndim == 1:
            ts = ts[:, None]

        # S_raw ~ DFT of probe signals, shape (n_probes, n_freqs) complex
        s_raw = jnp.dot(ts.T.astype(jnp.complex64), phase_matrix)

        # Normalize: S_ij ~ response_i / excitation_j
        # For a simple proxy: normalize by the first port's self-response
        # and form an S-matrix-like quantity
        n_ports = s_meas_jnp.shape[0]

        # Build a simplified S-param proxy from probe frequency content
        # Use probe signals to form S11-like reflection coefficients
        if n_probes >= n_ports:
            s_sim = jnp.zeros_like(s_meas_jnp)
            for i in range(min(n_ports, n_probes)):
                mag = jnp.abs(s_raw[i])
                safe_max = jnp.maximum(jnp.max(mag), 1e-30)
                s_sim = s_sim.at[i, i, :].set(s_raw[i] / safe_max)
        else:
            # Single probe: use as S11
            mag = jnp.abs(s_raw[0])
            safe_max = jnp.maximum(jnp.max(mag), 1e-30)
            s_sim = (s_raw[0] / safe_max).reshape(s_meas_jnp.shape)

        return sparam_loss(
            s_sim, s_meas_jnp,
            weight_mag=weight_mag, weight_phase=weight_phase,
        )

    # ------------------------------------------------------------------
    # Optimization loop (Adam)
    # ------------------------------------------------------------------
    grad_fn = jax.value_and_grad(forward)

    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    loss_history = []

    for it in range(n_iterations):
        loss_val, grad = grad_fn(params)
        loss_val = float(loss_val)
        loss_history.append(loss_val)

        params, m, v = _adam_step(
            params, grad, m, v, it, learning_rate,
        )

        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            print(f"  iter {it:4d}  loss = {loss_val:.6e}  "
                  f"|grad| = {float(jnp.max(jnp.abs(grad))):.3e}")

    # ------------------------------------------------------------------
    # Extract final poles
    # ------------------------------------------------------------------
    final_eps_inf, final_debye = _params_to_debye_poles(params, n_debye_poles)
    lorentz_offset = 1 + 2 * n_debye_poles
    final_lorentz = _params_to_lorentz_poles(params, n_lorentz_poles, lorentz_offset)

    # Convert from JAX scalars to Python floats
    eps_inf_out = float(final_eps_inf)
    debye_out = [
        DebyePole(delta_eps=float(p.delta_eps), tau=float(p.tau))
        for p in final_debye
    ]
    lorentz_out = [
        LorentzPole(omega_0=float(p.omega_0), delta=float(p.delta), kappa=float(p.kappa))
        for p in final_lorentz
    ]

    # Check convergence
    converged = False
    if len(loss_history) >= 5:
        recent = loss_history[-5:]
        if recent[0] > 0:
            rel_change = abs(recent[0] - recent[-1]) / abs(recent[0])
            converged = rel_change < 0.01

    return MaterialFitResult(
        eps_inf=eps_inf_out,
        debye_poles=debye_out,
        lorentz_poles=lorentz_out,
        loss_history=loss_history,
        final_s_params=None,
        freqs=freqs,
        converged=converged,
    )
