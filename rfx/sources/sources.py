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


@dataclass(frozen=True)
class ModulatedGaussian:
    """Gaussian-envelope modulated carrier (Meep-style source).

    s(t) = amplitude * sin(2π·f0·t) * exp(-((t - t0) / tau)^2)

    Key property: ∫s(t)dt = 0 exactly (carrier oscillation is symmetric).
    This eliminates DC accumulation on PEC surfaces, unlike the
    differentiated Gaussian which has a tiny DC residual (~1e-4).

    Parameters
    ----------
    f0 : float
        Center/carrier frequency (Hz).
    bandwidth : float
        Fractional bandwidth (0–1). Default 0.5.
    amplitude : float
        Peak amplitude. Default 1.0.
    """

    f0: float
    bandwidth: float = 0.5
    amplitude: float = 1.0

    @property
    def tau(self) -> float:
        return 1.0 / (self.f0 * self.bandwidth * math.pi)

    @property
    def t0(self) -> float:
        return 3.0 * self.tau

    def __call__(self, t: float) -> float:
        envelope = jnp.exp(-((t - self.t0) / self.tau) ** 2)
        carrier = jnp.sin(2.0 * jnp.pi * self.f0 * t)
        return self.amplitude * carrier * envelope


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
class WirePort:
    """Multi-cell lumped port spanning multiple grid cells along one axis.

    Models a vertical probe feed (e.g., coaxial probe through substrate)
    as a distributed voltage source with impedance split across N cells.

    Parameters
    ----------
    start : (x, y, z) in meters — one end of the wire
    end : (x, y, z) in meters — other end (must be axis-aligned with start)
    component : str — E-field component ("ex", "ey", or "ez")
    impedance : float — total port impedance in ohms (default 50)
    excitation : callable — source waveform
    """

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    component: str
    impedance: float = 50.0
    excitation: object = None  # GaussianPulse or similar


def _wire_port_cells(grid, port):
    """Return list of (i, j, k) cell indices along the wire."""
    import numpy as np
    s = np.array(port.start)
    e = np.array(port.end)
    diff = e - s
    nonzero = np.abs(diff) > 1e-10
    if np.sum(nonzero) != 1:
        raise ValueError("WirePort start and end must be axis-aligned")
    axis = int(np.argmax(nonzero))

    idx_s = grid.position_to_index(tuple(s))
    idx_e = grid.position_to_index(tuple(e))

    lo = min(idx_s[axis], idx_e[axis])
    hi = max(idx_s[axis], idx_e[axis])

    cells = []
    for a in range(lo, hi + 1):
        cell = list(idx_s)
        cell[axis] = a
        cells.append(tuple(cell))
    return cells


def setup_wire_port(grid, port, materials):
    """Distribute port impedance across all wire cells.

    For N cells in series with total impedance Z0:
      R_cell = Z0 / N
      sigma_cell = 1/(R_cell * dx) = N / (Z0 * dx)
    """
    cells = _wire_port_cells(grid, port)
    n_cells = max(len(cells), 1)
    sigma_per_cell = n_cells / (port.impedance * grid.dx)

    sigma = materials.sigma
    for cell in cells:
        sigma = sigma.at[cell[0], cell[1], cell[2]].add(sigma_per_cell)
    return materials._replace(sigma=sigma)


def apply_wire_port(state, grid, port, t, materials):
    """Inject source voltage distributed across all wire cells.

    Each cell gets V_src / N_cells, with Cb computed from local materials.
    """
    cells = _wire_port_cells(grid, port)
    n_cells = max(len(cells), 1)
    v_src = port.excitation(t) / n_cells
    dx = grid.dx
    dt = grid.dt

    field = getattr(state, port.component)
    for cell in cells:
        i, j, k = cell
        eps = float(materials.eps_r[i, j, k]) * EPS_0
        sigma = float(materials.sigma[i, j, k])
        loss = sigma * dt / (2.0 * eps)
        cb = (dt / eps) / (1.0 + loss)
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
