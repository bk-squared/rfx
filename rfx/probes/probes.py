"""Field monitors and DFT probes for frequency-domain extraction.

DFT probes accumulate frequency-domain fields during the time-domain
simulation, avoiding the need to store full time-series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from rfx.grid import Grid
from rfx.sources.sources import LumpedPort


@dataclass(frozen=True)
class FieldMonitor:
    """Records time-domain field at a single point."""

    position: tuple[float, float, float]
    component: str  # "ex", "ey", "ez", "hx", "hy", "hz"


def sample_field(state, grid: Grid, monitor: FieldMonitor) -> float:
    """Sample a field component at a monitor position."""
    idx = grid.position_to_index(monitor.position)
    field = getattr(state, monitor.component)
    return field[idx[0], idx[1], idx[2]]


class DFTProbe(NamedTuple):
    """Running DFT accumulator at a point for multiple frequencies."""

    # Complex accumulated field (n_freqs,)
    accumulator: jnp.ndarray
    # Frequency array (Hz)
    freqs: jnp.ndarray
    # Grid index
    index: tuple[int, int, int]
    # Field component name
    component: str
    # Optional streaming DFT window metadata
    total_steps: int
    window: str
    window_alpha: float


def init_dft_probe(
    grid: Grid,
    position: tuple[float, float, float],
    component: str,
    freqs: jnp.ndarray,
    dft_total_steps: int = 0,
    dft_window: str = "rect",
    dft_window_alpha: float = 0.25,
) -> DFTProbe:
    """Create a DFT probe at a point for given frequencies."""
    idx = grid.position_to_index(position)
    return DFTProbe(
        accumulator=jnp.zeros(len(freqs), dtype=jnp.complex64),
        freqs=freqs,
        index=idx,
        component=component,
        total_steps=int(dft_total_steps),
        window=dft_window,
        window_alpha=float(dft_window_alpha),
    )


def update_dft_probe(
    probe: DFTProbe,
    state,
    dt: float,
) -> DFTProbe:
    """Accumulate one timestep into the running DFT.

    X(f) += x(t) * exp(-j*2π*f*t) * dt
    """
    t = state.step * dt
    field = getattr(state, probe.component)
    value = field[probe.index[0], probe.index[1], probe.index[2]]

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_acc = probe.accumulator + value * phase * dt * weight

    return probe._replace(accumulator=new_acc)


# ---------------------------------------------------------------------------
# Port voltage / current extraction
# ---------------------------------------------------------------------------

def port_voltage(state, grid: Grid, port: LumpedPort) -> jnp.ndarray:
    """Voltage across the lumped port cell.

    V = -E_component * dx

    The sign convention follows the port orientation: positive voltage
    drives current in the +component direction.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    port : LumpedPort

    Returns
    -------
    float scalar
    """
    idx = grid.position_to_index(port.position)
    i, j, k = idx
    field = getattr(state, port.component)
    return -field[i, j, k] * grid.dx


def port_current(state, grid: Grid, port: LumpedPort) -> jnp.ndarray:
    """Current through the lumped port via Ampere's law loop integral.

    Integrates H around the port cell using the same curl convention as
    update_e() in yee.py (backward differences: H[i] - H[i-1]).

    For an Ez port at (i, j, k):
        I = (Hy[i,j,k] - Hy[i-1,j,k] - Hx[i,j,k] + Hx[i,j-1,k]) * dx

    For an Ex port at (i, j, k):
        I = (Hz[i,j,k] - Hz[i,j-1,k] - Hy[i,j,k] + Hy[i,j,k-1]) * dx

    For an Ey port at (i, j, k):
        I = (Hx[i,j,k] - Hx[i,j,k-1] - Hz[i,j,k] + Hz[i-1,j,k]) * dx

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    port : LumpedPort

    Returns
    -------
    float scalar
    """
    idx = grid.position_to_index(port.position)
    i, j, k = idx
    dx = grid.dx

    if port.component == "ez":
        # curl_z = (Hy[i,j,k] - Hy[i-1,j,k] - Hx[i,j,k] + Hx[i,j-1,k]) / dx
        i_loop = (
            state.hy[i, j, k] - state.hy[i - 1, j, k]
            - state.hx[i, j, k] + state.hx[i, j - 1, k]
        ) * dx
    elif port.component == "ex":
        # curl_x = (Hz[i,j,k] - Hz[i,j-1,k] - Hy[i,j,k] + Hy[i,j,k-1]) / dx
        i_loop = (
            state.hz[i, j, k] - state.hz[i, j - 1, k]
            - state.hy[i, j, k] + state.hy[i, j, k - 1]
        ) * dx
    elif port.component == "ey":
        # curl_y = (Hx[i,j,k] - Hx[i,j,k-1] - Hz[i,j,k] + Hz[i-1,j,k]) / dx
        i_loop = (
            state.hx[i, j, k] - state.hx[i, j, k - 1]
            - state.hz[i, j, k] + state.hz[i - 1, j, k]
        ) * dx
    else:
        raise ValueError(f"Unknown port component: {port.component!r}")

    return i_loop


# ---------------------------------------------------------------------------
# S-parameter DFT probe
# ---------------------------------------------------------------------------

class SParamProbe(NamedTuple):
    """Running DFT accumulator for S-parameter extraction at a lumped port.

    Accumulates port voltage V(f), port current I(f), and the incident
    source waveform V_inc(f) simultaneously so that S11 can be computed
    after the simulation without storing time series.

    Attributes
    ----------
    v_dft : (n_freqs,) complex
        Accumulated port voltage DFT.
    i_dft : (n_freqs,) complex
        Accumulated port current DFT.
    v_inc_dft : (n_freqs,) complex
        Accumulated incident voltage DFT (excitation waveform).
    freqs : (n_freqs,) float
        Frequency array in Hz.
    port_index : (i, j, k)
        Grid index of the port cell.
    component : str
        E-field component name.
    """

    v_dft: jnp.ndarray
    i_dft: jnp.ndarray
    v_inc_dft: jnp.ndarray
    freqs: jnp.ndarray
    port_index: tuple[int, int, int]
    component: str
    total_steps: int
    window: str
    window_alpha: float


def init_sparam_probe(
    grid: Grid,
    port: LumpedPort,
    freqs: jnp.ndarray,
    dft_total_steps: int = 0,
    dft_window: str = "rect",
    dft_window_alpha: float = 0.25,
) -> SParamProbe:
    """Create a zeroed SParamProbe for the given port and frequency array.

    Parameters
    ----------
    grid : Grid
    port : LumpedPort
    freqs : jnp.ndarray
        1-D array of frequencies (Hz) at which to evaluate S-parameters.

    Returns
    -------
    SParamProbe
    """
    idx = grid.position_to_index(port.position)
    n = len(freqs)
    zeros = jnp.zeros(n, dtype=jnp.complex64)
    return SParamProbe(
        v_dft=zeros,
        i_dft=zeros,
        v_inc_dft=zeros,
        freqs=freqs,
        port_index=idx,
        component=port.component,
        total_steps=int(dft_total_steps),
        window=dft_window,
        window_alpha=float(dft_window_alpha),
    )


def update_sparam_probe(
    probe: SParamProbe,
    state,
    grid: Grid,
    port: LumpedPort,
    dt: float,
) -> SParamProbe:
    """Accumulate one timestep of V, I, and V_inc into the DFT.

    Call this every timestep *after* update_e() and apply_lumped_port()
    have been applied so that the port cell fields reflect the driven
    state.

    X(f) += x(t) * exp(-j*2π*f*t) * dt

    Parameters
    ----------
    probe : SParamProbe
    state : FDTDState
        Current simulation state (after E and H updates).
    grid : Grid
    port : LumpedPort
    dt : float
        Timestep size in seconds.

    Returns
    -------
    SParamProbe with updated accumulators.
    """
    t = float(state.step) * dt

    v = port_voltage(state, grid, port)
    i = port_current(state, grid, port)
    v_inc = port.excitation(t)

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_v = probe.v_dft + v * phase * dt * weight
    new_i = probe.i_dft + i * phase * dt * weight
    new_vinc = probe.v_inc_dft + v_inc * phase * dt * weight

    return probe._replace(v_dft=new_v, i_dft=new_i, v_inc_dft=new_vinc)


def extract_s11(probe: SParamProbe, z0: float = 50.0) -> jnp.ndarray:
    """Compute S11 from accumulated DFT data.

    Uses the wave-decomposition definition:

        a = (V + Z0 * I) / (2 * sqrt(Z0))   # incident wave
        b = (V - Z0 * I) / (2 * sqrt(Z0))   # reflected wave
        S11 = b / a

    which simplifies to:

        S11 = (V - Z0 * I) / (V + Z0 * I)

    Alternatively, when the excitation is known, S11 is normalised
    against the incident DFT:

        S11 = (V - Z0 * I) / (2 * V_inc)

    Both forms are returned:  this function uses the wave-decomposition
    form (first equation) which is self-contained and does not require
    a separate incident simulation.

    Parameters
    ----------
    probe : SParamProbe
        Fully accumulated probe (simulation complete).
    z0 : float
        Reference impedance in ohms. Default 50.

    Returns
    -------
    s11 : (n_freqs,) complex array
    """
    denom = probe.v_dft + z0 * probe.i_dft
    # Guard against division by zero at DC or unexcited frequencies
    safe_denom = jnp.where(jnp.abs(denom) > 0.0, denom, jnp.ones_like(denom))
    s11 = (probe.v_dft - z0 * probe.i_dft) / safe_denom
    return s11


# ---------------------------------------------------------------------------
# DFT plane probe — frequency-domain field on a 2D plane
# ---------------------------------------------------------------------------

class DFTPlaneProbe(NamedTuple):
    """Running DFT accumulator for a 2D field plane.

    Accumulates X(f, y, z) = Σ x(t, y, z) · exp(-j2πft) · dt
    over the simulation, avoiding storage of full time series.

    Attributes
    ----------
    accumulator : (n_freqs, n1, n2) complex
        Accumulated DFT field on the plane.
    freqs : (n_freqs,) float
        Frequency array in Hz.
    component : str
        Field component ("ex", "ey", "ez", "hx", "hy", "hz").
    axis : int
        Normal axis (0=x, 1=y, 2=z).
    index : int
        Position along normal axis.
    """
    accumulator: jnp.ndarray
    freqs: jnp.ndarray
    component: str
    axis: int
    index: int
    total_steps: int
    window: str
    window_alpha: float


def init_dft_plane_probe(
    axis: int,
    index: int,
    component: str,
    freqs: jnp.ndarray,
    grid_shape: tuple[int, int, int],
    dft_total_steps: int = 0,
    dft_window: str = "rect",
    dft_window_alpha: float = 0.25,
) -> DFTPlaneProbe:
    """Create a DFT plane probe.

    Parameters
    ----------
    axis : int
        Normal axis (0=x → yz plane, 1=y → xz plane, 2=z → xy plane).
    index : int
        Grid index along normal axis.
    component : str
        Field component to monitor.
    freqs : (n_freqs,) array
        Frequencies in Hz.
    grid_shape : (nx, ny, nz)
    """
    if axis == 0:
        plane_shape = (grid_shape[1], grid_shape[2])
    elif axis == 1:
        plane_shape = (grid_shape[0], grid_shape[2])
    else:
        plane_shape = (grid_shape[0], grid_shape[1])

    nf = len(freqs)
    acc = jnp.zeros((nf,) + plane_shape, dtype=jnp.complex64)
    return DFTPlaneProbe(
        accumulator=acc, freqs=freqs,
        component=component, axis=axis, index=index,
        total_steps=int(dft_total_steps),
        window=dft_window,
        window_alpha=float(dft_window_alpha),
    )


def update_dft_plane_probe(
    probe: DFTPlaneProbe, state, dt: float,
) -> DFTPlaneProbe:
    """Accumulate one timestep into the plane DFT."""
    t = state.step * dt
    field = getattr(state, probe.component)

    if probe.axis == 0:
        plane = field[probe.index, :, :]
    elif probe.axis == 1:
        plane = field[:, probe.index, :]
    else:
        plane = field[:, :, probe.index]

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_acc = probe.accumulator + plane[None, :, :] * phase[:, None, None] * dt * weight
    return probe._replace(accumulator=new_acc)


from rfx.core.dft_utils import dft_window_weight as _dft_window_weight


# ---------------------------------------------------------------------------
# Wire port voltage / current extraction
# ---------------------------------------------------------------------------

def wire_port_voltage(state, grid, port) -> jnp.ndarray:
    """Voltage at the WirePort midpoint cell.

    Uses a single-cell measurement at the wire midpoint (same location
    as wire_port_current) for balanced V/I wave decomposition.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    port : WirePort

    Returns
    -------
    float scalar
    """
    from rfx.sources.sources import _wire_port_cells

    cells = _wire_port_cells(grid, port)
    mid = cells[len(cells) // 2]
    field = getattr(state, port.component)
    return -field[mid[0], mid[1], mid[2]] * grid.dx


def wire_port_current(state, grid, port) -> jnp.ndarray:
    """Current through a WirePort via Ampere's law at the midpoint cell.

    Uses H-field loop integral at the center cell of the wire,
    identical to the single-cell port_current() calculation.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    port : WirePort

    Returns
    -------
    float scalar
    """
    from rfx.sources.sources import _wire_port_cells

    cells = _wire_port_cells(grid, port)
    mid = cells[len(cells) // 2]
    i, j, k = mid
    dx = grid.dx

    if port.component == "ez":
        i_loop = (
            state.hy[i, j, k] - state.hy[i - 1, j, k]
            - state.hx[i, j, k] + state.hx[i, j - 1, k]
        ) * dx
    elif port.component == "ex":
        i_loop = (
            state.hz[i, j, k] - state.hz[i, j - 1, k]
            - state.hy[i, j, k] + state.hy[i, j, k - 1]
        ) * dx
    elif port.component == "ey":
        i_loop = (
            state.hx[i, j, k] - state.hx[i, j, k - 1]
            - state.hz[i, j, k] + state.hz[i - 1, j, k]
        ) * dx
    else:
        raise ValueError(f"Unknown port component: {port.component!r}")

    return i_loop


# ---------------------------------------------------------------------------
# Wire port S-parameter DFT probe
# ---------------------------------------------------------------------------

def init_wire_sparam_probe(
    grid,
    port,
    freqs: jnp.ndarray,
    dft_total_steps: int = 0,
    dft_window: str = "rect",
    dft_window_alpha: float = 0.25,
) -> SParamProbe:
    """Create a zeroed SParamProbe for a WirePort.

    The port_index is set to the midpoint cell of the wire.
    """
    from rfx.sources.sources import _wire_port_cells

    cells = _wire_port_cells(grid, port)
    mid_idx = tuple(cells[len(cells) // 2])
    n = len(freqs)
    zeros = jnp.zeros(n, dtype=jnp.complex64)
    return SParamProbe(
        v_dft=zeros,
        i_dft=zeros,
        v_inc_dft=zeros,
        freqs=freqs,
        port_index=mid_idx,
        component=port.component,
        total_steps=int(dft_total_steps),
        window=dft_window,
        window_alpha=float(dft_window_alpha),
    )


def update_wire_sparam_probe(
    probe: SParamProbe,
    state,
    grid,
    port,
    dt: float,
) -> SParamProbe:
    """Accumulate one timestep of V, I, and V_inc for a WirePort.

    Call this every timestep after update_e() and apply_wire_port().
    V and I are measured at the midpoint cell. V_inc uses the per-cell
    source voltage (V_src / N_cells) to match the single-cell measurement.
    """
    from rfx.sources.sources import _wire_port_cells

    t = float(state.step) * dt
    n_cells = max(len(_wire_port_cells(grid, port)), 1)

    v = wire_port_voltage(state, grid, port)
    i_val = wire_port_current(state, grid, port)
    v_inc = port.excitation(t) / n_cells

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_v = probe.v_dft + v * phase * dt * weight
    new_i = probe.i_dft + i_val * phase * dt * weight
    new_vinc = probe.v_inc_dft + v_inc * phase * dt * weight

    return probe._replace(v_dft=new_v, i_dft=new_i, v_inc_dft=new_vinc)


def extract_s_matrix(
    grid,
    materials,
    ports: list,
    freqs: jnp.ndarray,
    n_steps: int | None = None,
    *,
    boundary: str = "pec",
    cpml_axes: str = "xyz",
    debye_spec: tuple[list, list[jnp.ndarray]] | None = None,
    lorentz_spec: tuple[list, list[jnp.ndarray]] | None = None,
) -> jnp.ndarray:
    """Extract full N-port S-parameter matrix.

    Runs N simulations (one per excitation port).  All port impedances
    are active in every run so that port loading is consistent.

    Parameters
    ----------
    grid : Grid
    materials : MaterialArrays
        Base materials **without** port impedances folded in.
    ports : list of LumpedPort
    freqs : (n_freqs,) Hz
    n_steps : int or None
        Defaults to ``grid.num_timesteps(num_periods=30)``.
    boundary : "pec" or "cpml"
    cpml_axes : axes string for CPML (default "xyz")
    debye_spec, lorentz_spec : optional ``(poles, masks)`` tuples
        Used to rebuild dispersive coefficients after port loading is
        folded into ``materials``.

    Returns
    -------
    S : (n_ports, n_ports, n_freqs) complex array
        ``S[i, j, :]`` is S_{i+1, j+1} (response at port *i* when
        exciting port *j*).
    """
    import numpy as np
    from rfx.core.yee import init_state, update_h
    from rfx.boundaries.pec import apply_pec
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    from rfx.simulation import _update_e_with_optional_dispersion
    from rfx.sources.sources import setup_lumped_port, apply_lumped_port

    n_ports = len(ports)
    n_freqs = len(freqs)
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=30)

    dt, dx = grid.dt, grid.dx
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state_init = init_cpml(grid)

    # Fold ALL port impedances into materials (once)
    mats = materials
    for p in ports:
        mats = setup_lumped_port(grid, p, mats)

    debye = None
    if debye_spec is not None:
        debye_poles, debye_masks = debye_spec
        debye = init_debye(debye_poles, mats, dt, mask=debye_masks)

    lorentz = None
    if lorentz_spec is not None:
        lorentz_poles, lorentz_masks = lorentz_spec
        lorentz = init_lorentz(lorentz_poles, mats, dt, mask=lorentz_masks)

    S = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    for j in range(n_ports):
        state = init_state(grid.shape)
        sprobes = [init_sparam_probe(grid, p, freqs, dft_total_steps=n_steps) for p in ports]
        cpml_state = cpml_state_init if use_cpml else None
        debye_state = debye[1] if debye is not None else None
        lorentz_state = lorentz[1] if lorentz is not None else None

        for step in range(n_steps):
            t = step * dt
            state = update_h(state, mats, dt, dx)
            if use_cpml:
                state, cpml_state = apply_cpml_h(
                    state, cpml_params, cpml_state, grid, cpml_axes)

            state, debye_state, lorentz_state = _update_e_with_optional_dispersion(
                state,
                mats,
                dt,
                dx,
                debye=(debye[0], debye_state) if debye is not None else None,
                lorentz=(lorentz[0], lorentz_state) if lorentz is not None else None,
            )

            if use_cpml:
                state, cpml_state = apply_cpml_e(
                    state, cpml_params, cpml_state, grid, cpml_axes)
            state = apply_pec(state)

            # Excite only port j
            state = apply_lumped_port(state, grid, ports[j], t, mats)

            # Record V / I at all ports
            for i in range(n_ports):
                sprobes[i] = update_sparam_probe(
                    sprobes[i], state, grid, ports[i], dt)

        # Incident wave at excitation port j
        z0_j = ports[j].impedance
        a_j = (sprobes[j].v_dft + z0_j * sprobes[j].i_dft) / (
            2.0 * np.sqrt(z0_j))
        safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))

        # Response at each receiving port i
        for i in range(n_ports):
            z0_i = ports[i].impedance
            b_i = (sprobes[i].v_dft - z0_i * sprobes[i].i_dft) / (
                2.0 * np.sqrt(z0_i))
            S[i, j, :] = np.array(b_i / safe_a)

    return S


def extract_s11_normalised(probe: SParamProbe, z0: float = 50.0) -> jnp.ndarray:
    """Compute S11 normalised against the incident source DFT.

    Uses:

        S11 = (V - Z0 * I) / (2 * V_inc)

    This form requires that v_inc_dft has been accumulated and that a
    matched-load reference (or the excitation alone) was recorded.

    Parameters
    ----------
    probe : SParamProbe
    z0 : float

    Returns
    -------
    s11 : (n_freqs,) complex array
    """
    v_ref = 2.0 * probe.v_inc_dft
    safe_ref = jnp.where(jnp.abs(v_ref) > 0.0, v_ref, jnp.ones_like(v_ref))
    s11 = (probe.v_dft - z0 * probe.i_dft) / safe_ref
    return s11


def extract_s_matrix_wire(
    grid,
    materials,
    ports: list,
    freqs: jnp.ndarray,
    n_steps: int | None = None,
    *,
    boundary: str = "pec",
    cpml_axes: str = "xyz",
    debye_spec: tuple[list, list[jnp.ndarray]] | None = None,
    lorentz_spec: tuple[list, list[jnp.ndarray]] | None = None,
    pec_mask: object | None = None,
) -> jnp.ndarray:
    """Extract full N-port S-parameter matrix for WirePort objects.

    Runs N simulations (one per excitation port).  All port impedances
    are active in every run so that port loading is consistent.

    Parameters
    ----------
    grid : Grid
    materials : MaterialArrays
        Base materials **without** port impedances folded in.
    ports : list of WirePort
    freqs : (n_freqs,) Hz
    n_steps : int or None
        Defaults to ``grid.num_timesteps(num_periods=30)``.
    boundary : "pec" or "cpml"
    cpml_axes : axes string for CPML (default "xyz")
    debye_spec, lorentz_spec : optional dispersion specs

    Returns
    -------
    S : (n_ports, n_ports, n_freqs) complex array
    """
    import numpy as np
    from rfx.core.yee import init_state, update_h
    from rfx.boundaries.pec import apply_pec
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    from rfx.simulation import _update_e_with_optional_dispersion
    from rfx.sources.sources import setup_wire_port, apply_wire_port

    n_ports = len(ports)
    n_freqs = len(freqs)
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=30)

    dt, dx = grid.dt, grid.dx
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state_init = init_cpml(grid)

    # Fold ALL port impedances into materials (once)
    mats = materials
    for p in ports:
        mats = setup_wire_port(grid, p, mats)

    debye = None
    if debye_spec is not None:
        debye_poles, debye_masks = debye_spec
        debye = init_debye(debye_poles, mats, dt, mask=debye_masks)

    lorentz = None
    if lorentz_spec is not None:
        lorentz_poles, lorentz_masks = lorentz_spec
        lorentz = init_lorentz(lorentz_poles, mats, dt, mask=lorentz_masks)

    S = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    for j in range(n_ports):
        state = init_state(grid.shape)
        sprobes = [
            init_wire_sparam_probe(grid, p, freqs, dft_total_steps=n_steps)
            for p in ports
        ]
        cpml_state = cpml_state_init if use_cpml else None
        debye_state = debye[1] if debye is not None else None
        lorentz_state = lorentz[1] if lorentz is not None else None

        for step in range(n_steps):
            t = step * dt
            state = update_h(state, mats, dt, dx)
            if use_cpml:
                state, cpml_state = apply_cpml_h(
                    state, cpml_params, cpml_state, grid, cpml_axes)

            state, debye_state, lorentz_state = _update_e_with_optional_dispersion(
                state,
                mats,
                dt,
                dx,
                debye=(debye[0], debye_state) if debye is not None else None,
                lorentz=(lorentz[0], lorentz_state) if lorentz is not None else None,
            )

            if use_cpml:
                state, cpml_state = apply_cpml_e(
                    state, cpml_params, cpml_state, grid, cpml_axes)
            state = apply_pec(state)

            if pec_mask is not None:
                from rfx.boundaries.pec import apply_pec_mask
                state = apply_pec_mask(state, pec_mask)

            # Excite only port j
            state = apply_wire_port(state, grid, ports[j], t, mats)

            # Record V / I at all ports
            for i in range(n_ports):
                sprobes[i] = update_wire_sparam_probe(
                    sprobes[i], state, grid, ports[i], dt)

        # Wave decomposition at the midpoint cell.
        # Since V and I are measured at a single cell with per-cell
        # impedance Z0_cell = Z0/N, extract Z_in from the DFTs and
        # compute S11 against the total port impedance Z0.
        z0_j = ports[j].impedance
        safe_i = jnp.where(
            jnp.abs(sprobes[j].i_dft) > 0,
            sprobes[j].i_dft,
            jnp.ones_like(sprobes[j].i_dft) * 1e-30,
        )
        z_in_j = sprobes[j].v_dft / safe_i  # measured input impedance

        for i in range(n_ports):
            z0_i = ports[i].impedance
            if i == j:
                # S11: reflection from input impedance
                S[i, j, :] = np.array(
                    (z_in_j - z0_i) / (z_in_j + z0_i)
                )
            else:
                # Sij: use wave decomposition with per-cell Z for balancing
                n_cells_i = len(_wire_port_cells(grid, ports[i]))
                z0_cell_i = z0_i / max(n_cells_i, 1)
                b_i = (sprobes[i].v_dft - z0_cell_i * sprobes[i].i_dft) / (
                    2.0 * np.sqrt(z0_cell_i))
                n_cells_j = len(_wire_port_cells(grid, ports[j]))
                z0_cell_j = z0_j / max(n_cells_j, 1)
                a_j = (sprobes[j].v_dft + z0_cell_j * sprobes[j].i_dft) / (
                    2.0 * np.sqrt(z0_cell_j))
                safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))
                S[i, j, :] = np.array(b_i / safe_a)

    return S
