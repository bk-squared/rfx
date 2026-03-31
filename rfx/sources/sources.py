"""FDTD source implementations.

All source functions are pure: they take state + parameters, return new state.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import EPS_0


@dataclass(frozen=True)
class GaussianPulse:
    """Differentiated Gaussian pulse with center frequency f0.

    s(t) = -2 * ((t - t0) / tau) * exp(-((t - t0) / tau)^2)

    Parameters
    ----------
    f0 : float
        Center frequency (Hz).
    bandwidth : float
        Fractional bandwidth (0–1). Default 0.5.
    amplitude : float
        Peak amplitude (V/m). Default 1.0.
    """

    f0: float
    bandwidth: float = 0.5
    amplitude: float = 1.0

    @property
    def tau(self) -> float:
        """Pulse width parameter."""
        return 1.0 / (self.f0 * self.bandwidth * math.pi)

    @property
    def t0(self) -> float:
        """Pulse delay (3*tau ensures smooth onset)."""
        return 3.0 * self.tau

    def __call__(self, t: float) -> float:
        """Evaluate pulse at time t."""
        arg = (t - self.t0) / self.tau
        return self.amplitude * (-2.0 * arg) * jnp.exp(-(arg**2))


def add_point_source(
    state,
    grid: Grid,
    position: tuple[float, float, float],
    component: str,
    value: float,
) -> object:
    """Add a soft point source (additive) to a field component.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    position : (x, y, z) in meters
    component : "ex", "ey", or "ez"
    value : field value to add
    """
    idx = grid.position_to_index(position)
    i, j, k = idx

    field = getattr(state, component)
    field = field.at[i, j, k].add(value)
    return state._replace(**{component: field})


def add_lumped_port(
    state,
    grid: Grid,
    position: tuple[float, float, float],
    component: str,
    voltage: float,
    impedance: float = 50.0,
) -> object:
    """Apply lumped port excitation (voltage source with internal impedance).

    V = V_src - I * Z_0
    Translates to E-field update: E += (V_src / dx) - (Z_0 * J) term.

    For Stage 2 — simplified version for Stage 1 acts as hard source.
    """
    idx = grid.position_to_index(position)
    i, j, k = idx
    dx = grid.dx

    # Simple hard source for Stage 1
    e_value = voltage / dx
    field = getattr(state, component)
    field = field.at[i, j, k].set(e_value)
    return state._replace(**{component: field})


@dataclass(frozen=True)
class LumpedPort:
    """Lumped port: voltage source with internal impedance for S-parameter simulation.

    Models a coaxial or waveguide feed as a 1-cell voltage source with
    series impedance Z0. The port impedance is modeled as equivalent
    conductivity sigma_port = 1/(Z0*dx) at the port cell.

    Parameters
    ----------
    position : (x, y, z) in meters
        Port center location.
    component : str
        E-field component driven by the port ("ex", "ey", or "ez").
    impedance : float
        Port reference impedance Z0 in ohms. Default 50 ohm.
    excitation : GaussianPulse
        Source waveform (returns volts).
    """

    position: tuple[float, float, float]
    component: str
    impedance: float
    excitation: GaussianPulse


def setup_lumped_port(grid: Grid, port: LumpedPort, materials) -> object:
    """Fold port impedance into material conductivity at the port cell.

    Call once before the time-stepping loop. The equivalent conductivity
    sigma_port = 1/(Z0*dx) is added to the existing sigma at the port
    cell so that update_e() naturally includes the port's resistive
    damping.

    Returns updated MaterialArrays.
    """
    idx = grid.position_to_index(port.position)
    sigma_port = 1.0 / (port.impedance * grid.dx)
    sigma = materials.sigma.at[idx[0], idx[1], idx[2]].add(sigma_port)
    return materials._replace(sigma=sigma)


def apply_lumped_port(state, grid: Grid, port: LumpedPort, t: float, materials) -> object:
    """Inject source voltage at the port cell. Call AFTER update_e().

    Since the port impedance is already folded into materials.sigma
    via setup_lumped_port(), this function only adds the source term:

        E[port] += Cb * V_src / dx

    where Cb matches the update_e() coefficient at the port cell
    (including the port impedance conductivity).
    """
    idx = grid.position_to_index(port.position)
    i, j, k = idx
    dx = grid.dx
    dt = grid.dt

    eps = float(materials.eps_r[i, j, k]) * EPS_0
    sigma = float(materials.sigma[i, j, k])  # includes sigma_port

    loss = sigma * dt / (2.0 * eps)
    cb = (dt / eps) / (1.0 + loss)

    v_src = port.excitation(t)

    field = getattr(state, port.component)
    field = field.at[i, j, k].add(cb * v_src / dx)
    return state._replace(**{port.component: field})


@dataclass(frozen=True)
class CWSource:
    """Continuous-wave sinusoidal source with smooth ramp-up.

    Parameters
    ----------
    f0 : float
        Frequency in Hz.
    amplitude : float
        Peak amplitude.
    ramp_steps : int
        Number of source-frequency cycles over which to apply a cosine
        taper onset.  0 means instant full-amplitude.
    """

    f0: float
    amplitude: float = 1.0
    ramp_steps: int = 50

    def __call__(self, t: float) -> float:
        """Evaluate CW source at time *t* (seconds)."""
        if self.ramp_steps <= 0:
            envelope = 1.0
        else:
            n_cycles = t * self.f0
            progress = jnp.clip(n_cycles / self.ramp_steps, 0.0, 1.0)
            envelope = 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
        return self.amplitude * jnp.sin(2.0 * jnp.pi * self.f0 * t) * envelope


@dataclass(frozen=True)
class CustomWaveform:
    """User-defined waveform wrapper.

    Parameters
    ----------
    func : callable
        Function ``f(t: float) -> float`` returning the source amplitude
        at time *t*.
    """

    func: object  # callable; use ``object`` for frozen dataclass compat

    def __call__(self, t: float) -> float:
        """Evaluate the user-defined waveform at time *t*."""
        return self.func(t)
