"""FDTD source implementations.

All source functions are pure: they take state + parameters, return new state.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import EPS_0


def _axis_cell_sizes(grid, k_index: int = 0) -> tuple[float, float, float]:
    """Return (dx, dy, dz) cell sizes, handling both Grid and NonUniformGrid.

    For uniform Grid: dx = dy = dz = grid.dx.
    For NonUniformGrid: dx, dy are uniform; dz comes from grid.dz[k_index].
    """
    dx = float(grid.dx)
    dy = float(getattr(grid, 'dy', dx))
    dz_arr = getattr(grid, 'dz', None)
    dz = float(dz_arr[k_index]) if dz_arr is not None else dx
    return dx, dy, dz


def port_sigma(grid, position_ijk: tuple[int, int, int],
               component: str, impedance: float) -> float:
    """Compute lumped port equivalent conductivity for anisotropic cells.

    σ = d_parallel / (Z0 · d_perp1 · d_perp2)

    This ensures P_dissipated = V²/Z0 regardless of cell aspect ratio.
    For cubic cells (dx=dy=dz=d), reduces to 1/(Z0·d).
    """
    dx, dy, dz = _axis_cell_sizes(grid, k_index=position_ijk[2])
    sizes = [dx, dy, dz]
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    d_par = sizes[axis]
    d_perp = [sizes[i] for i in range(3) if i != axis]
    return d_par / (impedance * d_perp[0] * d_perp[1])


def port_d_parallel(grid, position_ijk: tuple[int, int, int],
                    component: str) -> float:
    """Return the cell size along the port's E-field direction."""
    dx, dy, dz = _axis_cell_sizes(grid, k_index=position_ijk[2])
    return [dx, dy, dz][{"ex": 0, "ey": 1, "ez": 2}[component]]


@dataclass(frozen=True)
class GaussianPulse:
    """Differentiated Gaussian pulse with center frequency f0.

    s(t) = -2 * ((t - t0) / tau) * exp(-((t - t0) / tau)^2)

    The onset truncation at t=0 leaves a nonzero time integral: over the
    sampled table, S = sum s(t_n)*dt ~ -amplitude * tau * exp(-cutoff^2).
    A soft current source deposits exactly that integral as a net static
    end-charge q = S/((1+loss)*dz_src) (issue #388), so the deposited DC
    scales as e^(-cutoff^2): raising ``cutoff`` from 3.0 to 4.5 cuts the
    deposited charge by e^(4.5^2 - 3^2) ~ 7.7e4 — about 5 decades.
    Use a higher cutoff when a static source remnant matters (e.g. the
    ``until_decay`` interior-energy stop on absorbing boundaries).

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
        Default 3.0 (historical value; waveform values are unchanged at
        the default). Higher values buy an e^(-cutoff^2) reduction of the
        deposited-DC integral at the cost of a longer onset (issue #388).
    """

    f0: float
    bandwidth: float = 0.5
    amplitude: float = 1.0
    cutoff: float = 3.0

    @property
    def tau(self) -> float:
        """Pulse width parameter."""
        return 1.0 / (self.f0 * self.bandwidth * math.pi)

    @property
    def t0(self) -> float:
        """Pulse delay (cutoff*tau ensures smooth onset)."""
        return self.cutoff * self.tau

    def __call__(self, t: float) -> float:
        """Evaluate pulse at time t."""
        arg = (t - self.t0) / self.tau
        return self.amplitude * (-2.0 * arg) * jnp.exp(-(arg**2))


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
    ``until_decay`` interior-energy stop, issue #388), use bandwidth <= 0.5
    or a :class:`GaussianPulse` with a higher ``cutoff``.

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

    Uses the 3D formula σ = d_parallel / (Z0 · d_perp1 · d_perp2)
    which reduces to 1/(Z0·dx) for cubic cells.
    """
    idx = grid.position_to_index(port.position)
    sp = port_sigma(grid, idx, port.component, port.impedance)
    sigma = materials.sigma.at[idx[0], idx[1], idx[2]].add(sp)
    return materials._replace(sigma=sigma)


def apply_lumped_port(state, grid: Grid, port: LumpedPort, t: float, materials) -> object:
    """Inject source voltage at the port cell. Call AFTER update_e().

    E[port] += Cb * V_src / d_parallel
    """
    idx = grid.position_to_index(port.position)
    i, j, k = idx
    d_par = port_d_parallel(grid, idx, port.component)
    dt = grid.dt

    eps = float(materials.eps_r[i, j, k]) * EPS_0
    sigma = float(materials.sigma[i, j, k])

    loss = sigma * dt / (2.0 * eps)
    cb = (dt / eps) / (1.0 + loss)

    v_src = port.excitation(t)

    field = getattr(state, port.component)
    field = field.at[i, j, k].add(cb * v_src / d_par)
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


def _wire_port_live_cells(grid, port, pec_mask=None):
    """Split the wire cells into (cells, live_flags, n_live) — issue #318.

    A cell is *live* when the assembled-geometry PEC mask (geometry PEC +
    PEC thin conductors, read BEFORE the runner's port-cell clearing) is
    False there.  A dead cell — one whose extent lies inside a PEC
    conductor — carries essentially no port current (measured on the
    issue-#313 thru: |I_dead|/|I_mid| = 0.003-0.03), so its per-cell port
    resistor is bypassed and it must not be counted in the series
    impedance distribution or the drive/normalization cell counts.

    ``pec_mask=None`` treats every cell as live — identical to the
    historical behaviour, and the degenerate case n_live == n makes every
    caller's formula bit-identical to the pre-#318 one.

    The mask must be concrete (host-readable): the live split is a static
    geometry decision, made once at setup time, mirroring the
    reference-plane spec builder's concreteness requirement.

    Raises
    ------
    ValueError
        When every extent cell is inside PEC (n_live == 0): such a port
        has no live cell to terminate or drive.
    """
    import numpy as np

    cells = _wire_port_cells(grid, port)
    if pec_mask is None:
        return cells, [True] * len(cells), max(len(cells), 1)

    mask_np = np.asarray(pec_mask)
    live_flags = [not bool(mask_np[c[0], c[1], c[2]]) for c in cells]
    n_live = sum(live_flags)
    if n_live == 0:
        raise ValueError(
            f"WirePort {port.start} -> {port.end} ({port.component}): all "
            f"{len(cells)} extent cells land inside PEC geometry, so the "
            "port has no live cell to terminate or drive (issue #318). "
            "Shorten the extent or move the port so at least one cell "
            "center sits outside PEC."
        )
    return cells, live_flags, n_live


def setup_wire_port(grid, port, materials, pec_mask=None):
    """Distribute port impedance across the LIVE wire cells (issue #318).

    For n_live live cells in series with total impedance Z0:
      σ_cell = n_live · d_parallel / (Z0 · d_perp1 · d_perp2)
    On uniform grids this reduces to n_live / (Z0 · dx).

    Cells whose extent lies inside PEC (dead cells) get NO port sigma:
    their resistor is bypassed by the surrounding conductor, so counting
    them made the physical series termination Z0·(n_live/n) instead of Z0
    (issue #318; measured 33.3 ohm on the issue-#313 thru, n=3 with one
    dead cell).  With ``pec_mask=None`` — or no dead cells — this is
    bit-identical to the historical all-cells formula.
    """
    cells, live_flags, n_live = _wire_port_live_cells(grid, port, pec_mask)

    sigma = materials.sigma
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        sp = port_sigma(grid, cell, port.component, port.impedance) * n_live
        sigma = sigma.at[cell[0], cell[1], cell[2]].add(sp)
    return materials._replace(sigma=sigma)


def apply_wire_port(state, grid, port, t, materials, pec_mask=None):
    """Inject source voltage distributed across the LIVE wire cells.

    Each live cell gets V_src / n_live, with Cb computed from local
    materials.  Dead extent cells (inside PEC, issue #318) receive no
    injection — pre-#318 they accumulated phantom EMF on an edge whose
    only discharge path was the port sigma folded at that cell.
    """
    cells, live_flags, n_live = _wire_port_live_cells(grid, port, pec_mask)
    v_src = port.excitation(t) / n_live
    dt = grid.dt

    field = getattr(state, port.component)
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        i, j, k = cell
        d_par = port_d_parallel(grid, (i, j, k), port.component)
        eps = float(materials.eps_r[i, j, k]) * EPS_0
        sigma = float(materials.sigma[i, j, k])
        loss = sigma * dt / (2.0 * eps)
        cb = (dt / eps) / (1.0 + loss)
        field = field.at[i, j, k].add(cb * v_src / d_par)

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
