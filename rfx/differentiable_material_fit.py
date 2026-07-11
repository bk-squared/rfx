"""Differentiable material fitting — fit Debye/Lorentz poles via jax.grad.

Unlike the scipy-based fitting in ``material_fit.py`` which fits to eps(f),
this module differentiates through the full FDTD simulation to fit
dispersive material models.  No de-embedding is needed because the fixture
geometry is included in the simulation.

ACCURACY CAVEAT — despite the name, the current loss does NOT fit true
S-parameters. The extraction self-normalizes each probe spectrum by its own
per-probe maximum magnitude (see ~L470-477), which discards magnitude
information; the loss therefore matches only the *shape* of a self-normalized
probe spectrum, not S-parameter magnitude and phase. The code calls this an
"S-param proxy" internally. A real S-parameter fit is planned (the
review-remediation plan, Stage 3.5a).

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
        S-parameters from the final simulation.  When the fit was run with
        ``fit_nuisance=True`` this is the NUISANCE-APPLIED model S11 (the
        quantity compared against the measurement), not the bare simulated S11.
    freqs : np.ndarray
        Frequency array in Hz.
    converged : bool
        Whether the optimizer converged (loss plateau).
    nuisance_alpha : float or None
        Fitted reflection-tracking magnitude ``alpha`` (linear).  ``None``
        unless the fit was run with ``fit_nuisance=True`` (issue #273).
    nuisance_phi : float or None
        Fitted reflection-tracking phase in radians, wrapped to ``(-pi, pi]``.
    nuisance_tau : float or None
        Fitted one-way reference-plane offset delay in SECONDS (de-scaled
        from the internal ``tau_hat`` optimization variable).
    uq : object or None
        :class:`rfx.calibration_identifiability.IdentifiabilityReport` when the
        fit was run with ``compute_uq=True``; ``None`` otherwise.  Typed
        ``object`` because ``calibration_identifiability`` imports from this
        module (a top-level back-import would be circular).
    """

    eps_inf: float
    debye_poles: list = field(default_factory=list)
    lorentz_poles: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    final_s_params: np.ndarray | None = None
    freqs: np.ndarray | None = None
    converged: bool = False
    nuisance_alpha: float | None = None
    nuisance_phi: float | None = None
    nuisance_tau: float | None = None
    uq: object | None = None


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
        materials, debye_spec, lorentz_spec, pec_mask, *_ = sim._assemble_materials(grid)

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


# ---------------------------------------------------------------------------
# Physically-scaled S11 calibration (AD-traceable, no self-normalization)
# ---------------------------------------------------------------------------

def _count_excited_ports(sim) -> int:
    """Number of ports the forward pass will actively drive.

    ``Simulation.forward(port_s11_freqs=...)`` injects a source at *every*
    port whose ``excite`` flag is True (all others act as matched loads), so a
    one-port S11 calibration requires exactly one excited port. Counts across
    every port collection that carries an ``excite`` flag (lumped/wire, MSL,
    coaxial, waveguide); collections whose entries have no ``excite`` attribute
    contribute nothing.
    """
    n = 0
    for attr in ("_ports", "_msl_ports", "_coaxial_ports", "_waveguide_ports"):
        for entry in getattr(sim, attr, ()) or ():
            if getattr(entry, "excite", False):
                n += 1
    return n


# ---------------------------------------------------------------------------
# One-port VNA nuisance model (issue #273, Stage 2)
# ---------------------------------------------------------------------------

def _nuisance_tau_scale(freqs) -> float:
    """Scale for the nuisance delay: ``tau = tau_hat * tau_scale`` (seconds).

    ``tau_scale = 1 / (4*pi*max(freqs))`` makes one unit of ``tau_hat`` equal
    to 1 rad of round-trip phase at the top of the band, so Adam's step size
    on ``tau_hat`` is balanced against the O(1) log-space material parameters
    (a raw tau in seconds, ~1e-12, would be invisible to a shared learning
    rate).  ``tau_hat`` is signed: the reference plane may sit on either side
    of the nominal port location.
    """
    return 1.0 / (4.0 * np.pi * float(np.max(np.asarray(freqs))))


def _apply_nuisance(s11, freqs_jnp, log_alpha, phi, tau_hat, tau_scale):
    """Apply the one-port VNA nuisance model to a simulated S11 (complex64).

    Model (issue #273, v1 scope)::

        S11_model(f) = alpha * exp(j*phi) * exp(-j*4*pi*f*tau) * S11_sim(f)

    ``alpha * exp(j*phi)`` is the reflection-tracking term ``e_r`` of the
    standard one-port error model with directivity ``e_d`` and source match
    ``e_s`` deliberately dropped; ``tau = tau_hat * tau_scale`` is the one-way
    reference-plane offset delay, doubled in the exponent for the round trip.
    Convention: measured = nuisance(true) — the transform is applied to the
    SIMULATED S11; the measured data is never touched.

    ``alpha`` arrives in log-space (positivity by construction, matching the
    log-space material parameters); ``phi`` is linear radians, unbounded on
    the tape (wrap only when reporting).  Constants are built as float32
    (``jnp.float32(...)``) so nothing promotes the complex64 core to float64.
    """
    alpha = jnp.exp(log_alpha)
    # 4*pi*tau_scale is precomputed in float32 (~1/f_max, well inside float32
    # range); freqs (~1e10) times it stays O(1), then scales by tau_hat.
    round_trip = (jnp.float32(4.0 * np.pi) * jnp.float32(tau_scale)) * freqs_jnp * tau_hat
    factor = alpha * jnp.exp(1j * (phi - round_trip))
    return (factor * s11).astype(jnp.complex64)


def calibrate_material_s11(
    sim_factory: Callable,
    s11_measured: np.ndarray,
    freqs: np.ndarray,
    n_debye_poles: int = 0,
    n_lorentz_poles: int = 0,
    *,
    n_iterations: int = 50,
    learning_rate: float = 0.05,
    num_periods: float = 20.0,
    initial_guess=None,
    weight_mag: float = 1.0,
    weight_phase: float = 0.1,
    fit_nuisance: bool = False,
    nuisance_init: tuple = (1.0, 0.0, 0.0),
    compute_uq: bool = False,
    noise_std: float | None = None,
    checkpoint: bool = True,
    verbose: bool = True,
) -> MaterialFitResult:
    """Fit dispersive material poles to **physically-scaled** S11 via jax.grad.

    Unlike :func:`differentiable_material_fit`, which builds a
    *self-normalized* probe-spectrum proxy (each probe spectrum is divided by
    its own peak magnitude, discarding magnitude information), this function
    uses the AD-traceable wave-decomposition S11 emitted directly by
    ``Simulation.forward(port_s11_freqs=...)``.  That S11 is a single-run
    ``b/a`` wave decomposition (``a = (-V + Z0·I)/(2√Z0)``,
    ``b = (-V - Z0·I)/(2√Z0)``) with true magnitude and phase — no two-run
    reference and no self-normalization — so the loss matches both magnitude
    and phase of the measured S11.

    The gradient path is::

        log_poles -> exp() -> physical poles
          -> DebyePole/LorentzPole (traced scalars)
          -> sim_factory(...) -> Simulation.forward(port_s11_freqs=..., checkpoint=True)
          -> ForwardResult.s_params (pure-jnp complex64 S11)
          -> sparam_loss() -> scalar
          <- jax.grad -> d(loss)/d(log_poles)

    Constraints (v1 scope): uniform single-device meshes only (no non-uniform
    mesh, no ``distributed=True``).  ``boundary="pec"`` is fine.  Diagonal S11
    only — this is an S11-based calibration track.  The fixture must have
    exactly one excited port (the S11 port); any other ports must be passive
    matched loads (``excite=False``), else the fit targets an *active*
    reflection coefficient rather than S11 (checked before the first step).

    Joint VNA-nuisance fitting (``fit_nuisance=True``, issue #273 Stage 2)
    ----------------------------------------------------------------------
    Real one-port measurements differ from the simulated S11 by residual VNA
    error terms.  With ``fit_nuisance=True`` the fit co-estimates a nuisance
    transform applied to the SIMULATED S11 (measured = nuisance(true))::

        S11_model(f) = alpha * exp(j*phi) * exp(-j*4*pi*f*tau) * S11_sim(f)

    v1 nuisance scope — EXACTLY reflection tracking plus delay:
    ``alpha * exp(j*phi)`` is the reflection-tracking term ``e_r`` of the
    standard one-port error model, and ``tau`` is the one-way reference-plane
    offset delay (round trip in the exponent).  Directivity ``e_d`` and source
    match ``e_s`` are DELIBERATELY omitted: they add 4 more real parameters a
    single broadband S11 trace rarely constrains.  Consequence: real data with
    poor directivity will alias ``e_d`` into the material fit — do not use v1
    on uncorrected raw data from a bad bridge.

    The three nuisance parameters are APPENDED to the log-space parameter
    vector as ``[log_alpha, phi, tau_hat]`` where ``tau_hat = tau / tau_scale``
    with ``tau_scale = 1/(4*pi*max(freqs))`` (one ``tau_hat`` unit = 1 rad of
    round-trip phase at band top, balancing Adam steps across parameters).

    RECOMMENDED: pass ``weight_phase=1.0`` (explicitly) when
    ``fit_nuisance=True`` — the phase carries essentially all of the tau/phi
    information and the default ``weight_phase=0.1`` down-weights it 10x
    (the default is kept unchanged for existing material-only callers).
    If the joint fit drifts into a confounded basin early (the material
    parameters chasing the nuisance phase ramp before phi/tau adapt — Adam's
    per-parameter normalization amplifies their initially weak gradients),
    additionally up-weighting the magnitude channel (``weight_mag ~ 4``)
    anchors the material to the |S11| curve shape, which alpha can only scale
    uniformly (measured on the Stage-2 synthetic gates).
    Keep the true delay small relative to the band (|4*pi*f_max*tau| well
    below pi), else the phase term wraps and Adam can stall in a wrapped
    local minimum.  For real data with a large unknown offset, pre-estimate
    the delay from the measured group delay (slope of unwrapped phase) and
    fold it into ``nuisance_init`` — not implemented here.

    Parameters
    ----------
    sim_factory : callable(eps_inf, debye_poles, lorentz_poles) -> Simulation
        Factory that builds a ``Simulation`` (with at least one lumped/wire
        ``add_port(...)``) from the given material parameters.  Called inside
        the gradient tape with **traced** pole scalars, so any material built
        from these arguments stays on the ``jax.grad`` tape.
    s11_measured : (n_freqs,) or (..., n_freqs) complex array
        Measured (physically-scaled) S11.  Cast to complex64 internally to
        avoid dtype promotion against the complex64 forward S11.
    freqs : (n_freqs,) array in Hz
        Frequency points matching ``s11_measured``; passed to
        ``forward(port_s11_freqs=...)``.
    n_debye_poles, n_lorentz_poles : int
        Number of Debye / Lorentz poles to fit.  ``0/0`` fits ``eps_inf`` only.
    n_iterations : int
        Number of Adam optimization iterations.
    learning_rate : float
        Adam learning rate (log-space parameters).
    num_periods : float
        Number of periods at ``freq_max`` for the forward step count.
    initial_guess : DebyeFitResult, LorentzFitResult, or None
        Initial pole parameters.  If None, a frequency-band-based default is
        used.
    weight_mag, weight_phase : float
        Weights for magnitude and phase in :func:`sparam_loss`.  See the
        ``weight_phase=1.0`` recommendation above for ``fit_nuisance=True``.
    fit_nuisance : bool
        Co-estimate the VNA nuisance parameters (see above).  Default False —
        zero behavior change for existing material-only callers.
    nuisance_init : tuple
        Initial ``(alpha, phi_rad, tau_seconds)``.  The default
        ``(1.0, 0.0, 0.0)`` is the neutral (no-nuisance) starting point.
    compute_uq : bool
        After the fit, run :func:`rfx.calibration_identifiability.calibration_uq`
        at the fitted optimum and store the
        :class:`~rfx.calibration_identifiability.IdentifiabilityReport` in
        ``result.uq`` (Fisher / Cramér-Rao analysis, joint with the nuisance
        tail when ``fit_nuisance=True``).
    noise_std : float or None
        Assumed i.i.d. Gaussian noise std on Re/Im S11 for ``compute_uq``.
        ``None`` estimates ``sigma_hat`` from the residual at the optimum.
    checkpoint : bool
        Use gradient checkpointing in the FDTD scan (recommended).
    verbose : bool
        Print progress.

    Returns
    -------
    MaterialFitResult
        With ``final_s_params`` populated by the final physically-scaled S11
        (the self-normalized fitter leaves this ``None``).  When
        ``fit_nuisance=True``, ``final_s_params`` is the NUISANCE-APPLIED
        model S11 — the quantity actually compared against the measurement —
        and ``nuisance_alpha`` / ``nuisance_phi`` (wrapped to ``(-pi, pi]``) /
        ``nuisance_tau`` (seconds) carry the fitted nuisance estimates.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    freqs_jnp = jnp.asarray(freqs, dtype=jnp.float32)
    s11_meas_jnp = jnp.asarray(np.asarray(s11_measured), dtype=jnp.complex64)

    # ------------------------------------------------------------------
    # Build initial guess (reuses the log-space parameterization helpers)
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

    while len(debye_init) < n_debye_poles:
        debye_init.append(DebyePole(delta_eps=1.0, tau=1e-11))
    while len(lorentz_init) < n_lorentz_poles:
        omega0 = 2 * np.pi * float(np.mean(freqs))
        lorentz_init.append(
            LorentzPole(omega_0=omega0, delta=omega0 * 0.05, kappa=omega0 ** 2)
        )

    params = _poles_to_params(eps_inf_init, debye_init[:n_debye_poles],
                              lorentz_init[:n_lorentz_poles])

    lorentz_offset = 1 + 2 * n_debye_poles

    # Nuisance tail [log_alpha, phi, tau_hat] APPENDED after all pole params
    # (issue #273 Stage 2) — the front unpack helpers above are layout-stable.
    nuisance_offset = 1 + 2 * n_debye_poles + 3 * n_lorentz_poles
    tau_scale = _nuisance_tau_scale(freqs)
    if fit_nuisance:
        alpha_init, phi_init, tau_init = nuisance_init
        if not (np.isfinite(alpha_init) and alpha_init > 0):
            raise ValueError(
                f"nuisance_init alpha must be finite and positive, got {alpha_init!r}"
            )
        nuisance_tail = jnp.array(
            [np.log(float(alpha_init)), float(phi_init), float(tau_init) / tau_scale],
            dtype=params.dtype,
        )
        params = jnp.concatenate([params, nuisance_tail])

    # ------------------------------------------------------------------
    # One concrete pass OUTSIDE the AD tape (issue #273 corrections 2 & 3).
    # Build the fixture once with concrete initial parameters so we can:
    #   (2) guard the one-port S11 contract — forward() drives every
    #       excite=True port simultaneously, so >1 excited port silently
    #       fits an ACTIVE reflection coefficient instead of S11 and the fit
    #       converges to biased parameters; and
    #   (3) run preflight ONCE on concrete parameters so genuine setup
    #       mistakes (under-resolved mesh, geometry in CPML, probe in PEC)
    #       surface here instead of being silently absorbed into the fitted
    #       poles. The AD loop below keeps skip_preflight=True: preflight is
    #       not AD-traceable and must not run per iteration.
    # ------------------------------------------------------------------
    _eps_c, _debye_c = _params_to_debye_poles(params, n_debye_poles)
    _lorentz_c = _params_to_lorentz_poles(params, n_lorentz_poles, lorentz_offset)
    _probe_sim = sim_factory(_eps_c, _debye_c, _lorentz_c)

    _n_excited = _count_excited_ports(_probe_sim)
    if _n_excited != 1:
        raise ValueError(
            "calibrate_material_s11 fits a one-port reflection coefficient "
            f"(S11), but sim_factory built a simulation with {_n_excited} "
            "excited port(s). forward(port_s11_freqs=...) drives every "
            "excite=True port simultaneously, so the fitted quantity would be "
            "an active reflection coefficient, not S11, and the fit would "
            "converge to biased parameters. Leave exactly one port excite=True "
            "and mark the rest excite=False (passive matched loads)."
        )

    # Mirrors forward()'s own _auto_preflight; the AD loop skips it per step.
    _probe_sim._auto_preflight(skip=False, context="calibrate_material_s11")

    # ------------------------------------------------------------------
    # Forward: log-space params -> traced poles -> Simulation.forward -> S11 -> loss
    # ------------------------------------------------------------------
    def _s11_from_params(p):
        eps_inf, debye_poles = _params_to_debye_poles(p, n_debye_poles)
        lorentz_poles = _params_to_lorentz_poles(p, n_lorentz_poles, lorentz_offset)
        sim = sim_factory(eps_inf, debye_poles, lorentz_poles)
        result = sim.forward(
            port_s11_freqs=freqs_jnp,
            num_periods=num_periods,
            checkpoint=checkpoint,
            skip_preflight=True,
        )
        return result.s_params

    def _model_s11(p):
        """Model S11 compared against the measurement.

        With ``fit_nuisance=True`` this is the nuisance-applied simulated S11
        (measured = nuisance(true)); otherwise the bare simulated S11.
        """
        s11_sim = _s11_from_params(p)
        if fit_nuisance:
            s11_sim = _apply_nuisance(
                s11_sim, freqs_jnp,
                p[nuisance_offset],
                p[nuisance_offset + 1],
                p[nuisance_offset + 2],
                tau_scale,
            )
        return s11_sim

    def forward(p):
        return sparam_loss(
            _model_s11(p), s11_meas_jnp,
            weight_mag=weight_mag, weight_phase=weight_phase,
        )

    # ------------------------------------------------------------------
    # Optimization loop (Adam) — same pattern as differentiable_material_fit
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
    # Extract final poles and the final physically-scaled S11
    # ------------------------------------------------------------------
    final_eps_inf, final_debye = _params_to_debye_poles(params, n_debye_poles)
    final_lorentz = _params_to_lorentz_poles(params, n_lorentz_poles, lorentz_offset)

    eps_inf_out = float(final_eps_inf)
    debye_out = [
        DebyePole(delta_eps=float(p.delta_eps), tau=float(p.tau))
        for p in final_debye
    ]
    lorentz_out = [
        LorentzPole(omega_0=float(p.omega_0), delta=float(p.delta), kappa=float(p.kappa))
        for p in final_lorentz
    ]

    # Nuisance-applied when fit_nuisance=True (see MaterialFitResult docstring).
    final_s11 = np.asarray(_model_s11(params))

    nuisance_alpha = nuisance_phi = nuisance_tau = None
    if fit_nuisance:
        nuisance_alpha = float(np.exp(float(params[nuisance_offset])))
        # phi is unbounded on the tape; np.angle wraps the report to (-pi, pi].
        nuisance_phi = float(np.angle(np.exp(1j * float(params[nuisance_offset + 1]))))
        nuisance_tau = float(params[nuisance_offset + 2]) * tau_scale

    converged = False
    if len(loss_history) >= 5:
        recent = loss_history[-5:]
        if recent[0] > 0:
            rel_change = abs(recent[0] - recent[-1]) / abs(recent[0])
            converged = rel_change < 0.01

    result = MaterialFitResult(
        eps_inf=eps_inf_out,
        debye_poles=debye_out,
        lorentz_poles=lorentz_out,
        loss_history=loss_history,
        final_s_params=final_s11,
        freqs=freqs,
        converged=converged,
        nuisance_alpha=nuisance_alpha,
        nuisance_phi=nuisance_phi,
        nuisance_tau=nuisance_tau,
    )

    if compute_uq:
        # Lazy import: calibration_identifiability imports FROM this module at
        # top level, so a top-level back-import here would be circular.
        from rfx.calibration_identifiability import calibration_uq

        result.uq = calibration_uq(
            sim_factory,
            freqs,
            result,
            n_debye_poles=n_debye_poles,
            n_lorentz_poles=n_lorentz_poles,
            fit_nuisance=fit_nuisance,
            noise_std=noise_std,
            s11_measured=np.asarray(s11_measured),
            num_periods=num_periods,
            checkpoint=checkpoint,
        )

    return result
