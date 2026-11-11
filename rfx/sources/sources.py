"""FDTD source implementations.

Field applicators return new states. Port setup also records its own load
stamp on the build-local descriptor for subsequent source assembly.
"""

from __future__ import annotations

from rfx._internal._grid_metric import field_index
from dataclasses import dataclass, field

import math

import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import EPS_0
from rfx.model.lumped_stamp import (
    _PORT_AXIS,
    _nu_cell_arrays,
    _axis_cell_sizes,
    _axis_dual_sizes,
    port_dual_transverse,
    port_sigma,
    _stamp_lumped,
    stamp_lumped_sigma,
    stamp_lumped_eps,
    port_d_parallel,
    setup_lumped_port,
    _wire_port_cells,
    wire_port_edge_span,
    _axis_node_position,
    _wire_port_live_cells,
    setup_wire_port,
)


def _validate_dg_cutoff(cutoff):
    try:
        cutoff = float(cutoff)
    except TypeError:  # traced: checked where it was concrete
        return
    if not cutoff >= 2.5:
        raise ValueError(
            f"GaussianPulse cutoff={cutoff!r} is below 2.5: the start would "
            "jump by 2c*exp(-c^2) of the amplitude and making it charge-free "
            "changes the pulse by more than 2 % (1.84e-2 at c = 2.5). Use "
            "cutoff >= 2.5 (default 3).")


def charge_free_dg(t, tau, cutoff, *, _arg_timing=None):
    """Unit-amplitude differentiated Gaussian with a charge-free onset.

    The onset and near-start series are shared by GaussianPulse and TF/SF.
    Values and slopes start continuously at zero; the time integral is zero.
    """
    _validate_dg_cutoff(cutoff)
    u = t / tau
    c = cutoff
    # Both terms are exactly 0 in float32 beyond 50 tau (exp(-2500)); the
    # clamps keep u**3 and arg**2 from overflowing there (inf * 0 = NaN in
    # the value and its gradient, e.g. for a tau of 1e-20 s), and change
    # nothing, value or gradient, for u < 50.
    uc = jnp.clip(u, -50.0, 50.0)
    # Preserve GaussianPulse's independent width evaluations in its reverse
    # f0 derivative. Sharing their value would change the rounding of that sum.
    t0, arg_tau = (cutoff * tau, tau) if _arg_timing is None else _arg_timing
    arg = jnp.clip((t - t0) / arg_tau, -50.0, 50.0)
    onset = jnp.exp(-c**2 - uc**2) * (2*c + (4*c**2 - 2)*uc - 4*c*uc**2 - 4*c**2*uc**3)
    value = -2.0 * arg * jnp.exp(-(arg**2)) - onset
    # For z = 2cu < 3 the two terms above nearly cancel (each about 2c*exp(-c^2)).
    # There the same pulse is exp(-c^2 - u^2) * (2c*R1(z) - 2u*R2(z)), z = 2cu,
    # with R1, R2 = exp(z) less its Taylor terms through z^1, z^2, summed as series
    # so every evaluation path keeps float32's relative accuracy.
    z = jnp.clip(2*c*u, 0.0, 3.0)
    term, r1, r2 = z**2 / 2, 0.0, 0.0
    for k in range(2, 22):
        r1 = r1 + term
        r2 = r2 + (term if k >= 3 else 0.0)
        term = term * z / (k + 1)
    near = jnp.exp(-c**2 - uc**2) * (2*c*r1 - 2*uc*r2)
    value = jnp.where(2*c*u < 3.0, near, value)
    return jnp.where(t >= 0, value, 0.0)


@dataclass(frozen=True)
class GaussianPulse:
    """Differentiated Gaussian pulse with center frequency f0, starting at t=0.

    s(t) = -2 * x * exp(-x^2) - exp(-c^2) * exp(-u^2) * (2c + (4c^2 - 2) u - 4c u^2 - 4c^2 u^3)
    for t >= 0 and 0 before, with x = (t - t0)/tau, u = t/tau, c = cutoff.

    The first term is the differentiated Gaussian. Cut at t=0 alone it would
    start with a jump of 2c*exp(-c^2) (7.4e-4 at c=3) and have the time
    integral -tau*exp(-c^2), which a soft current source deposits as a static
    charge whose field never decays. The second term is
    tau * d/dt[exp(-c^2) * phi(t)] with phi = exp(-u^2)(1 + 2c u + 2c^2 u^2),
    phi(0) = 1, phi(inf) = 0: it makes the integral exactly zero and the
    start continuous in value and slope. The added term's largest magnitude,
    near t = 0.36 tau, is 1.627e-3 of the amplitude at c = 3 and 1.837e-2 at
    c = 2.5. Below c = 2.5 it exceeds 2 % (the uncorrected start would jump by
    2c*exp(-c^2), 9.7e-3 at c = 2.5), so a smaller cutoff is refused.

    Parameters
    ----------
    f0 : float
        Center frequency (Hz).
    bandwidth : float
        Fractional bandwidth (0–1). Default 0.5.
    amplitude : float
        Peak amplitude (V/m). Default 1.0.
    cutoff : float
        Number of tau between t=0 and the pulse peak: t0 = cutoff * tau.
        Default 3.0; at least 2.5.
    """

    f0: float
    bandwidth: float = 0.5
    amplitude: float = 1.0
    cutoff: float = 3.0

    def __post_init__(self):
        _validate_dg_cutoff(self.cutoff)

    @property
    def tau(self) -> float:
        """Pulse width parameter."""
        return 1.0 / (self.f0 * self.bandwidth * math.pi)

    @property
    def t0(self) -> float:
        """Pulse delay: the peak lies cutoff*tau after the start at t=0."""
        return self.cutoff * self.tau

    def __call__(self, t: float) -> float:
        """Evaluate pulse at time t (zero before t=0)."""
        return self.amplitude * charge_free_dg(
            t, self.tau, self.cutoff, _arg_timing=(self.t0, self.tau))


@dataclass(frozen=True)
class ModulatedGaussian:
    """Gaussian-envelope modulated carrier (Meep-style source).

    s(t) = amplitude * sin(2π·f0·t) * exp(-((t - t0) / tau)^2)

    The carrier-centered spectrum generally reduces low-frequency/DC content
    relative to a differentiated Gaussian. A finite sampled pulse does not in
    general have an exactly zero integral, so DC-sensitive calculations should
    inspect the sampled spectrum and fields explicitly.

    DC suppression is set by the envelope width: the spectrum is a Gaussian
    centered at f0, and its value at DC relative to the carrier peak is
    e^(-1/bandwidth^2) — about 0.5 at bandwidth=1.2 but 0.018 at
    bandwidth=0.5. Wide fractional bandwidths therefore keep substantial DC
    drive; when the net deposited charge of a soft source matters (e.g. the
    ``until_decay`` interior-energy stop), use bandwidth <= 0.4
    (relative DC 4.0e-4 in one measured case — below the
    run() warning's 1e-3 threshold, which bandwidth=0.5 at ~2.6e-2 still
    exceeds) or a :class:`GaussianPulse` with a higher ``cutoff``.

    Parameters
    ----------
    f0 : float
        Center/carrier frequency (Hz).
    bandwidth : float
        Fractional bandwidth (0–1). Default 0.5.
    amplitude : float
        Peak amplitude. Default 1.0.
    cutoff : float
        Number of tau before the envelope peak. Controls how gradually
        the source turns on. Default 5.0 (Meep convention: source starts
        at t=0 and peaks at t = cutoff * tau).  Use 3.0 for a faster
        onset with a slightly non-zero initial amplitude.
    """

    f0: float
    bandwidth: float = 0.5
    amplitude: float = 1.0
    cutoff: float = 5.0

    @property
    def tau(self) -> float:
        return 1.0 / (self.f0 * self.bandwidth * math.pi)

    @property
    def t0(self) -> float:
        return self.cutoff * self.tau

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
    idx = field_index(grid, position, component)
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
    idx = field_index(grid, position, component)
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
    allow_in_absorber: bool = False
    # Build-local record of the actual stamp and unit-voltage shape. Excluded
    # from descriptor identity; setup populates it before source assembly.
    _drive_stamps: dict = field(default_factory=dict, init=False, repr=False, compare=False)


def apply_lumped_port(state, grid: Grid, port: LumpedPort, t: float, materials) -> object:
    """Inject source voltage at the port cell. Call AFTER update_e().

    E[port] += Cb * V_src / (R * transverse_dual_area)
    """
    from rfx.sources.port_drive import stamped_drive, port_drive_waveform

    idx = field_index(grid, port.position, port.component,
                      allow_in_absorber=port.allow_in_absorber)
    i, j, k = idx
    increment = port_drive_waveform(
        grid, idx, port.component, port.excitation, None, materials,
        sigma_port=stamped_drive(port, idx)[0],
        unit_field=stamped_drive(port, idx)[1], time=t)

    field = getattr(state, port.component)
    field = field.at[i, j, k].add(increment)
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
    radius : float or None — opt-in thin-probe radius in metres; None keeps
        the historical mesh-sized filament. See ``Simulation.add_port``.
    """

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    component: str
    impedance: float = 50.0
    excitation: object = None  # GaussianPulse or similar
    radius: float | None = None
    allow_in_absorber: bool = False
    _drive_stamps: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        from rfx.sources.wire_radius import validate_radius
        validate_radius(self.radius)


def wire_port_from_entry(pe):
    """Build the axis-aligned wire declaration used by distributed setup/recording."""
    end = list(pe.position)
    end[{"ex": 0, "ey": 1, "ez": 2}[pe.component]] += pe.extent
    return WirePort(start=pe.position, end=tuple(end), component=pe.component,
                    impedance=pe.impedance, excitation=pe.waveform, radius=pe.radius,
                    allow_in_absorber=getattr(pe, "allow_in_absorber", False))


def apply_wire_port(state, grid, port, t, materials, pec_edge_masks=None):
    """Inject source voltage distributed across the LIVE wire cells.

    Each live cell gets V_src / n_live, with Cb computed from local
    materials.  Dead extent cells (inside PEC) receive no
    injection: otherwise they accumulate phantom EMF on an edge whose
    only discharge path was the port sigma folded at that cell.
    """
    from rfx.sources.port_drive import stamped_drive, port_drive_waveform

    cells, live_flags, n_live = _wire_port_live_cells(grid, port, pec_edge_masks)

    field = getattr(state, port.component)
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        i, j, k = cell
        increment = port_drive_waveform(
            grid, cell, port.component, port.excitation, None, materials,
            sigma_port=stamped_drive(port, cell)[0],
            unit_field=stamped_drive(port, cell)[1], time=t)
        field = field.at[i, j, k].add(increment)

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
