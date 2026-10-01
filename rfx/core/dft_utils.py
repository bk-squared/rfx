"""Shared DFT utilities used by probes and waveguide ports."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def dft_window_weight(step: int, total_steps: int, window: str, alpha: float) -> jnp.ndarray:
    """Streaming DFT window weight for a given timestep.

    Parameters
    ----------
    step : int
        Current timestep index.
    total_steps : int
        Total number of timesteps in the simulation.
    window : str
        Window type: ``"rect"``, ``"hann"``, or ``"tukey"``.
    alpha : float
        Shape parameter for the Tukey window (ignored for others).

    Returns
    -------
    jnp.ndarray
        Scalar weight in [0, 1].
    """
    if total_steps <= 1 or window == "rect":
        return jnp.asarray(1.0, dtype=jnp.float32)

    n = jnp.asarray(step, dtype=jnp.float32)
    N = jnp.asarray(total_steps - 1, dtype=jnp.float32)
    x = n / jnp.maximum(N, 1.0)

    if window == "hann":
        return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x))

    if window == "tukey":
        a = float(alpha)
        if a <= 0.0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        if a >= 1.0:
            return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x))
        left = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * x / a - 1.0)))
        right = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * x / a - 2.0 / a + 1.0)))
        return jnp.where(
            x < a / 2.0,
            left,
            jnp.where(x <= 1.0 - a / 2.0, 1.0, right),
        )

    raise ValueError(f"dft_window must be 'rect', 'hann', or 'tukey', got {window!r}")


def half_step_current_phase(freqs: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Yee half-timestep phase correction for an H-derived port current DFT.

    In the leapfrog FDTD update the H update runs before the E update, so at
    the moment a port probe samples a step the electric field is ``E^{n+1}``
    while the magnetic field is ``H^{n+1/2}`` — the H field (and therefore any
    Ampere-loop current derived from it) lags the E field by exactly ``dt/2``.

    The V/I port DFT accumulates the E-derived voltage and the H-derived
    current against the SAME phase kernel ``exp(-j*2*pi*f*t)`` evaluated at
    ``t = step*dt``. That over-phases the current by half a timestep. This
    factor removes it by advancing the current sample's time argument by
    ``dt/2``:

        exp(-j*2*pi*f*t) * exp(+j*2*pi*f*(dt/2))  ==  exp(-j*2*pi*f*(t - dt/2))

    i.e. multiply the current sample's phase by ``exp(+j*pi*f*dt)``.

    Exactness: this is the exact linear-phase shift of the discrete DFT kernel;
    it is not a small-angle / first-order approximation and carries no
    geometry-, frequency-, or fixture-specific constant — ``dt`` is the runtime
    grid timestep and ``f`` the runtime probe frequency. It generalises to any
    board, any timestep, any frequency set. The physical assumption it encodes
    is only the leapfrog stagger itself (update_h then update_e leaves
    ``E=E^{n+1}``, ``H=H^{n+1/2}``); the sign (+dt/2) follows from that ordering.

    Provenance of the mechanism (be precise about this): the factor and its
    sign are established by DERIVATION from the leapfrog update order, plus a
    matched-port Ohm-law measurement (``arg(-V_port/(Z0 I))`` on the thru
    fixture: +0.0061/+0.0102/+0.0142 rad at 3/5/7 GHz, exactly linear in f,
    collapsing to a small NEGATIVE residual after the correction). They are
    NOT established by the "correct sign lowers the singular value, wrong sign
    raises it" experiment: sv_max is linear in the current phase near zero, so
    ANY small positive rotation of I — including a frequency-INDEPENDENT
    0.01 rad — lowers it. That test does not discriminate dt/2 from any other
    small positive rotation and must not be cited as confirmation. The same
    caveat applies to the reciprocity residual: for a geometrically symmetric
    two-port, S21 = S12 up to grid asymmetry regardless of any COMMON rotation
    of I, so a reciprocity improvement is an incidental move of a noise-level
    residual, not evidence for this factor.

    Applied to post-injection lumped and wire V/I in the production scan,
    and to owning-cell recordings in the distributed lumped scan driver.
    The pre-injection reference-voltage channel receives no H correction.

    Parameters
    ----------
    freqs : array of probe frequencies in Hz.
    dt : grid timestep in seconds (``grid.dt``), read at runtime.

    Returns
    -------
    complex array, same shape as ``freqs`` — multiply the current sample's DFT
    phase by this. Do NOT apply it to the voltage or incident-wave channels.
    """
    return jnp.exp(1j * jnp.pi * jnp.asarray(freqs) * dt)


def port_dft_phase(step, freqs, dt):
    """Production port stamp: float32(step)*dt, complex64 phase times dt.

    Accepts a scalar step in the scan or broadcast arrays for host replay.
    Preserve the scan's float32 time even when x64 is enabled.
    """
    phase_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    t = jnp.asarray(step, dtype=jnp.float32) * dt
    return jnp.exp(-1j * 2.0 * jnp.pi
                   * jnp.asarray(freqs).astype(phase_dtype)
                   * t.astype(phase_dtype)).astype(jnp.complex64) * dt
