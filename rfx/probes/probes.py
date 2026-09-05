"""Field monitors and DFT probes for frequency-domain extraction.

DFT probes accumulate frequency-domain fields during the time-domain
simulation, avoiding the need to store full time-series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.core.dft_utils import dft_window_weight as _dft_window_weight
from rfx.core.dft_utils import half_step_current_phase as _half_step_current_phase
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
    # Accumulator dtype follows the x64 state (issue #477 residual): under
    # x64 the per-step update (field * complex phase) is complex128, and a
    # hardcoded complex64 accumulator breaks the scan-carry dtype contract.
    # With x64 off this is complex64 — bit-identical to the previous
    # behaviour. Same idiom as compute_msl_s_matrix's _complex_dtype.
    _cdtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    return DFTProbe(
        accumulator=jnp.zeros(len(freqs), dtype=_cdtype),
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


def _bwd_h(h, idx, axis, periodic=(False, False, False)):
    """``h`` one cell back along ``axis``, with the SAME out-of-domain
    convention the uniform E update uses.

    Uniform-lane twin of :func:`rfx.nonuniform._bwd_neighbor` (issue #689),
    and this is where the rule ships: ``rfx/__init__.py`` re-exports
    ``wire_port_current`` as the public ``rfx.wire_port_current``, and both
    extractors here sit behind ``extract_s_matrix`` / ``extract_s_matrix_wire``.

    ``update_e()`` differences H through ``_diff_bwd_o`` -> ``rfx.core.yee.
    _shift_bwd``, which pads explicitly with ZERO on a non-periodic axis. The
    raw ``state.h*[i - 1]`` these extractors used to spell is, at ``i == 0``,
    Python's negative index — H at the OPPOSITE face of the domain. A 4-term
    LOCAL Ampere loop must not read a cell it does not enclose. Measured at
    head on a (11,11,11) pec grid with dx = 0.2 mm and the port at index 0:
    perturbing that far-face H cell by 1e6 moved the reported current by
    exactly ±1e6·dx = ±200.0 A on all six (component, back-read axis)
    branches of EACH extractor — twelve in total (issue #692; the issue text
    says 1e6·dx², the loop weight is one factor of dx, not two).

    The wrap is KEPT, deliberately, on the two kinds of axis #689 measured it
    to be load-bearing on — an unconditional zero pad is a regression there:

    * ``h.shape[axis] == 1`` — the 2-D lane. ``rfx/simulation.py`` forces
      ``periodic[2] = True`` when ``grid.is_2d`` and ``nz == 1``; the wrap is
      what makes the single z cell its own neighbour. (``(0 - 1) % 1 == 0``,
      so this branch returns the cell itself, matching ``_bwd_neighbor``.)
    * ``periodic[axis]`` — cell ``0`` and cell ``n-1`` really are neighbours.
      Callers on a periodic lane must pass their run's flags.

    The default is the NON-periodic convention, and it is the right default
    for every shipped caller: preflight ``_validate_run_sparameter_request``
    (``rfx/api/_preflight.py``) refuses lumped/wire S-parameters under
    periodic axes (#206, ``NotImplementedError``), so no periodic run reaches
    these extractors today.
    """
    n = h.shape[axis]
    i = int(idx[axis])
    if n == 1 or periodic[axis]:
        back = list(idx)
        back[axis] = (i - 1) % n
        return h[tuple(back)]
    if i == 0:
        return jnp.zeros_like(h[idx])
    back = list(idx)
    back[axis] = i - 1
    return h[tuple(back)]


def port_current(state, grid: Grid, port: LumpedPort,
                 periodic=(False, False, False)) -> jnp.ndarray:
    """Current through the lumped port via Ampere's law loop integral.

    Integrates H around the port cell with the backward-difference curl
    convention ``update_e()`` in ``yee.py`` uses — INCLUDING its
    out-of-domain padding, which goes through ``_diff_bwd_o`` ->
    ``rfx.core.yee._shift_bwd`` and is an explicit ZERO, not a wrap. This
    docstring used to claim "the same curl convention as update_e()" while
    every backward read was a raw ``h[i-1]``; issue #689 falsified that
    claim on the non-uniform twin and #692 fixes it here. Each backward read
    now goes through :func:`_bwd_h`, which keeps the wrap only on a length-1
    (2-D) or genuinely periodic axis.

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
    periodic : (bool, bool, bool)
        Per-axis periodic flags for the run. Default is the non-periodic
        convention — see :func:`_bwd_h` for why that is right for every
        shipped caller.

    Returns
    -------
    float scalar
    """
    idx = tuple(grid.position_to_index(port.position))
    dx = grid.dx

    return _ampere_loop(state, idx, port.component, dx, periodic)


def _ampere_loop(state, idx, component, dx, periodic):
    """4-term discrete Ampere loop at ``idx`` on a uniform (cubic) grid.

    Single spelling shared by :func:`port_current` and
    :func:`wire_port_current` — they were two verbatim copies of the same
    six branches, and #692 had to fix the same wrap twice because of it.
    """
    if component == "ez":
        # curl_z = (Hy[i,j,k] - Hy[i-1,j,k] - Hx[i,j,k] + Hx[i,j-1,k]) / dx
        return (
            state.hy[idx] - _bwd_h(state.hy, idx, 0, periodic)
            - state.hx[idx] + _bwd_h(state.hx, idx, 1, periodic)
        ) * dx
    if component == "ex":
        # curl_x = (Hz[i,j,k] - Hz[i,j-1,k] - Hy[i,j,k] + Hy[i,j,k-1]) / dx
        return (
            state.hz[idx] - _bwd_h(state.hz, idx, 1, periodic)
            - state.hy[idx] + _bwd_h(state.hy, idx, 2, periodic)
        ) * dx
    if component == "ey":
        # curl_y = (Hx[i,j,k] - Hx[i,j,k-1] - Hz[i,j,k] + Hz[i-1,j,k]) / dx
        return (
            state.hx[idx] - _bwd_h(state.hx, idx, 2, periodic)
            - state.hz[idx] + _bwd_h(state.hz, idx, 0, periodic)
        ) * dx
    raise ValueError(f"Unknown port component: {component!r}")


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
    # Whole-port gap-voltage DFT (wire ports only, issue #764):
    # V_port = sum over LIVE wire cells of -E_c*dx. None for lumped ports
    # (single-cell V_port == v_dft there).
    v_port_dft: object = None
    # PRE-injection drive-sample reference DFT (wire ports only, issue
    # #683 x #764): the midpoint -E*dx sampled BEFORE apply_wire_port,
    # bit-identical to the historical v_dft channel.  Feeds only the
    # #308-calibrated off-diagonal incident wave and the byte-frozen
    # legacy diagonal in decompose_wire_s_matrix; the physical v/i/v_port
    # channels sample POST-injection (#683 verdict).  None for lumped.
    v_ref_dft: object = None


class PortVIReplayBundle(NamedTuple):
    """Raw V/I phasors captured alongside a production S-matrix.

    ``voltages`` and ``currents`` use shape
    ``(n_driven, n_ports, n_freqs)`` and follow the public
    ``rfx.port_vi_dump`` replay convention: voltage is positive into the DUT,
    current is positive into the DUT, and
    ``S[receiver_port, driven_port, frequency_index]``.
    """

    s_params: jnp.ndarray
    freqs: jnp.ndarray
    voltages: object
    currents: object
    port_impedances: object
    port_names: tuple[str, ...]
    driven_port_indices: tuple[int, ...]


class WirePortVIReplayBundle(NamedTuple):
    """Raw wire-port V/I phasors captured with a production S-matrix.

    Since issue #770 the production wire S-matrix is the whole-port wave
    frame on every genuinely driven column (diagonal AND off-diagonal —
    the per-cell off-diagonal frame survives only on the legacy
    ``v_port=None`` decomposer path).  The raw fields are stored in the
    FDTD-sign convention consumed by the independent wire replay
    diagnostic, not the generic ``rfx.port_vi_dump`` convention; the dump
    metadata's ``offdiag_frame`` tag tells the replay which off-diagonal
    frame the recorded production S used (absent = pre-#770 per-cell).
    """

    s_params: jnp.ndarray
    freqs: jnp.ndarray
    raw_voltages_fdt: object
    raw_currents: object
    port_impedances: object
    port_cell_counts: object
    port_names: tuple[str, ...]
    driven_port_indices: tuple[int, ...]
    # Whole-port gap-voltage phasors (issue #764): required to replay the
    # physical driven diagonal.  ``None`` marks a pre-#764 dump, whose
    # recorded production S-matrix used the legacy per-cell diagonal — the
    # replay diagnostic falls back to that convention for such dumps.
    raw_port_voltages_fdt: object = None
    # PRE-injection drive-sample reference phasors (issue #683 x #764):
    # required to replay the #308-calibrated off-diagonals and the
    # byte-frozen legacy diagonal now that ``raw_voltages_fdt`` holds the
    # physical POST-injection samples.  ``None`` marks a pre-#683 dump,
    # whose raw voltages ARE the reference — the replay diagnostic falls
    # back to them.
    raw_drive_ref_voltages_fdt: object = None


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

    Call this every timestep *after* update_e() / apply_pec() but
    **before** apply_lumped_port() so that the sampled port voltage
    reflects only the cavity/load response, not the source injection.
    Sampling after source injection contaminates V with the driving
    waveform, making the wave-decomposition S11 meaningless.

    CAVEAT (issue #683): the paragraph above is the #72 contract, and for
    WIRE ports it was REFUTED by measurement (2026-08-29) — the wire-port
    probes now sample the physical V/I/V_port AFTER injection (see
    ``update_wire_sparam_probe``) and keep the pre-injection sample only
    as the ``v_ref_dft`` calibration reference.  The lumped ordering here
    is deliberately unchanged pending its own decision run on a
    lumped-port known-load fixture; do not flip it just to match the wire
    family.

    X(f) += x(t) * exp(-j*2π*f*t) * dt

    Parameters
    ----------
    probe : SParamProbe
    state : FDTDState
        Current simulation state (after E update, before source).
    grid : Grid
    port : LumpedPort
    dt : float
        Timestep size in seconds.

    Returns
    -------
    SParamProbe with updated accumulators.
    """
    t = state.step * dt  # keep traceable: float(state.step) leaked the tracer

    v = port_voltage(state, grid, port)
    i = port_current(state, grid, port)
    v_inc = port.excitation(t)

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    # SCOPE NOTE (item B2, 2026-09-05): the Yee half-step current phase
    # `rfx.core.dft_utils.half_step_current_phase` is applied on the WIRE
    # lane only (``update_wire_sparam_probe`` and the wire blocks in
    # simulation.py / nonuniform.py), NOT here. Its derivation requires
    # E = E^{n+1} at the sample so that the E/H offset is exactly dt/2. On
    # the wire lane that holds because #683 moved the physical channels
    # post-injection. Here V is sampled PRE-injection at a DRIVEN cell (the
    # #72 contract kept above), and #683 measured that sample is "not any
    # field time level of the discrete update" — so the V/I time offset is
    # NOT established to be dt/2 and no correction is derivable without a
    # lumped known-load decision run. Applying the wire-lane factor here
    # would be exactly the align-first-decide-later move the #673/#672
    # ledger entry warns against; it also shifted every lumped |S11| bin
    # (max 2.3% on tests/fixtures/golden_forward_no_rlc_s11.npy) with no
    # lumped-side oracle behind it.
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_v = probe.v_dft + v * phase * dt * weight
    new_i = probe.i_dft + i * phase * dt * weight
    new_vinc = probe.v_inc_dft + v_inc * phase * dt * weight

    return probe._replace(v_dft=new_v, i_dft=new_i, v_inc_dft=new_vinc)


def extract_s11(probe: SParamProbe, z0: float = 50.0) -> jnp.ndarray:
    """Compute S11 from accumulated DFT data.

    Uses the input-impedance definition:

        Z_in = -V / I        (V = -E·dx follows FDTD sign convention)
        S11  = (Z_in - Z0) / (Z_in + Z0)

    which, after substitution, becomes:

        S11 = (V + Z0 · I) / (V - Z0 · I)

    The probe must be accumulated **before** source injection
    (see :func:`update_sparam_probe`) so that V reflects only the
    cavity/load response.

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
    denom = probe.v_dft - z0 * probe.i_dft
    # Guard against division by zero at DC or unexcited frequencies
    safe_denom = jnp.where(jnp.abs(denom) > 0.0, denom, jnp.ones_like(denom))
    s11 = (probe.v_dft + z0 * probe.i_dft) / safe_denom
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
    region : tuple[int, int, int, int] or None
        Optional static crop ``(lo1, hi1, lo2, hi2)`` in the two
        transverse plane axes. ``None`` retains the full plane.
    """
    accumulator: jnp.ndarray
    freqs: jnp.ndarray
    component: str
    axis: int
    index: int
    total_steps: int
    window: str
    window_alpha: float
    region: tuple[int, int, int, int] | None = None


def init_dft_plane_probe(
    axis: int,
    index: int,
    component: str,
    freqs: jnp.ndarray,
    grid_shape: tuple[int, int, int],
    dft_total_steps: int = 0,
    dft_window: str = "rect",
    dft_window_alpha: float = 0.25,
    region: tuple[int, int, int, int] | None = None,
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
    region : (lo1, hi1, lo2, hi2), optional
        Static half-open crop in the transverse plane axes.
    """
    if axis == 0:
        plane_shape = (grid_shape[1], grid_shape[2])
    elif axis == 1:
        plane_shape = (grid_shape[0], grid_shape[2])
    else:
        plane_shape = (grid_shape[0], grid_shape[1])

    if region is not None:
        lo1, hi1, lo2, hi2 = region
        if not (0 <= lo1 < hi1 <= plane_shape[0]
                and 0 <= lo2 < hi2 <= plane_shape[1]):
            raise ValueError(
                f"DFT plane region {region} is outside transverse shape "
                f"{plane_shape}."
            )
        plane_shape = (hi1 - lo1, hi2 - lo2)

    nf = len(freqs)
    # dtype follows the x64 state — see create_dft_probe (issue #477 residual).
    _cdtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    acc = jnp.zeros((nf,) + plane_shape, dtype=_cdtype)
    return DFTPlaneProbe(
        accumulator=acc, freqs=freqs,
        component=component, axis=axis, index=index,
        total_steps=int(dft_total_steps),
        window=dft_window,
        window_alpha=float(dft_window_alpha),
        region=region,
    )


def update_dft_plane_probe(
    probe: DFTPlaneProbe, state, dt: float,
) -> DFTPlaneProbe:
    """Accumulate one timestep into the plane DFT."""
    t = state.step * dt
    field = getattr(state, probe.component)

    region = probe.region
    if region is None:
        lo1 = lo2 = 0
        if probe.axis == 0:
            hi1, hi2 = field.shape[1], field.shape[2]
        elif probe.axis == 1:
            hi1, hi2 = field.shape[0], field.shape[2]
        else:
            hi1, hi2 = field.shape[0], field.shape[1]
    else:
        lo1, hi1, lo2, hi2 = region

    if probe.axis == 0:
        plane = field[probe.index, lo1:hi1, lo2:hi2]
    elif probe.axis == 1:
        plane = field[lo1:hi1, probe.index, lo2:hi2]
    else:
        plane = field[lo1:hi1, lo2:hi2, probe.index]

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_acc = probe.accumulator + plane[None, :, :] * phase[:, None, None] * dt * weight
    return probe._replace(accumulator=new_acc)


# ---------------------------------------------------------------------------
# Poynting Flux DFT Monitor (Meep flux-region equivalent)
# ---------------------------------------------------------------------------

class FluxMonitor(NamedTuple):
    """Running DFT accumulator for Poynting flux through a plane.

    Accumulates frequency-domain E and H tangential components on a
    (possibly finite-size) region of the plane to compute
    ∫ Re(E × H*) · n̂ dA at each frequency.

    For x-normal plane: flux = Re(Ey·Hz* - Ez·Hy*) integrated over y,z.
    For y-normal plane: flux = Re(Ez·Hx* - Ex·Hz*) integrated over x,z.
    For z-normal plane: flux = Re(Ex·Hy* - Ey·Hx*) integrated over x,y.
    """
    e1_dft: jnp.ndarray   # (n_freqs, n1, n2) complex — first tangential E
    e2_dft: jnp.ndarray   # (n_freqs, n1, n2) complex — second tangential E
    h1_dft: jnp.ndarray   # (n_freqs, n1, n2) complex — first tangential H
    h2_dft: jnp.ndarray   # (n_freqs, n1, n2) complex — second tangential H
    freqs: jnp.ndarray    # (n_freqs,) float
    axis: int             # normal axis (0=x, 1=y, 2=z)
    index: int            # grid index along normal axis
    dA: jnp.ndarray       # area weight: scalar (1,1) or (n1,n2) — d1*d2
                          # over the two tangential axes (PROBE-C1 fix:
                          # axis-aware; the old scalar dx*dx assumed a
                          # cubic cell).
    total_steps: int
    window: str
    window_alpha: float
    lo1: int = 0          # start index in first tangential dimension
    hi1: int = -1         # end index (exclusive); -1 = full extent
    lo2: int = 0          # start index in second tangential dimension
    hi2: int = -1         # end index (exclusive); -1 = full extent


# Tangential field component names for each normal axis
_FLUX_COMPONENTS = {
    0: ("ey", "ez", "hy", "hz"),  # x-normal: Ey, Ez, Hy, Hz
    1: ("ez", "ex", "hz", "hx"),  # y-normal: Ez, Ex, Hz, Hx
    2: ("ex", "ey", "hx", "hy"),  # z-normal: Ex, Ey, Hx, Hy
}


def init_flux_monitor(
    axis: int,
    index: int,
    freqs: jnp.ndarray,
    grid_shape: tuple[int, int, int],
    d1,
    d2,
    dft_total_steps: int = 0,
    dft_window: str = "rect",
    dft_window_alpha: float = 0.25,
    lo1: int = 0,
    hi1: int = -1,
    lo2: int = 0,
    hi2: int = -1,
) -> FluxMonitor:
    """Create a Poynting flux monitor on a (possibly finite-size) plane region.

    ``lo1/hi1`` and ``lo2/hi2`` restrict the DFT accumulation to a
    sub-region of the tangential plane.  -1 means "full extent".

    ``d1`` / ``d2`` are the cell sizes along the two tangential axes of
    the monitor plane (axis-1 / axis-2 of ``_FLUX_COMPONENTS``). Each may
    be a scalar (uniform mesh) or a per-cell 1-D array (non-uniform). The
    flux area element is ``dA = d1 * d2`` — for an x-normal plane that is
    ``dy * dz``, etc. (PROBE-C1 fix: the old API took a single scalar
    ``dx`` and assumed a cubic cell.)
    """
    if axis == 0:
        full1, full2 = grid_shape[1], grid_shape[2]
    elif axis == 1:
        full1, full2 = grid_shape[0], grid_shape[2]
    else:
        full1, full2 = grid_shape[0], grid_shape[1]

    if hi1 < 0:
        hi1 = full1
    if hi2 < 0:
        hi2 = full2

    n1 = hi1 - lo1
    n2 = hi2 - lo2
    nf = len(freqs)
    zeros = jnp.zeros((nf, n1, n2), dtype=jnp.complex128)

    # Axis-aware area element dA = d1*d2 over the (lo,hi) sub-region.
    # Scalars stay scalar (broadcast); per-cell arrays are sliced and
    # outer-multiplied to (n1, n2). jnp.reshape(scalar, (-1,1)) -> (1,1).
    w1 = jnp.asarray(d1, dtype=jnp.float32)
    w2 = jnp.asarray(d2, dtype=jnp.float32)
    if w1.ndim > 0:
        w1 = w1[lo1:hi1]
    if w2.ndim > 0:
        w2 = w2[lo2:hi2]
    dA = jnp.reshape(w1, (-1, 1)) * jnp.reshape(w2, (1, -1))

    return FluxMonitor(
        e1_dft=zeros, e2_dft=zeros,
        h1_dft=zeros, h2_dft=zeros,
        freqs=freqs, axis=axis, index=index,
        dA=dA,
        total_steps=int(dft_total_steps),
        window=dft_window,
        window_alpha=float(dft_window_alpha),
        lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2,
    )


def update_flux_monitor(
    mon: FluxMonitor, state, dt: float,
) -> FluxMonitor:
    """Accumulate one timestep of E and H tangential fields."""
    t = state.step * dt
    e1_name, e2_name, h1_name, h2_name = _FLUX_COMPONENTS[mon.axis]

    def _slice(field):
        if mon.axis == 0:
            return field[mon.index, :, :]
        elif mon.axis == 1:
            return field[:, mon.index, :]
        return field[:, :, mon.index]

    e1 = _slice(getattr(state, e1_name))
    e2 = _slice(getattr(state, e2_name))
    h1 = _slice(getattr(state, h1_name))
    h2 = _slice(getattr(state, h2_name))

    phase = jnp.exp(-1j * 2.0 * jnp.pi * mon.freqs * t)
    weight = _dft_window_weight(state.step, mon.total_steps, mon.window, mon.window_alpha)
    kernel = phase[:, None, None] * dt * weight

    return mon._replace(
        e1_dft=mon.e1_dft + e1[None, :, :] * kernel,
        e2_dft=mon.e2_dft + e2[None, :, :] * kernel,
        h1_dft=mon.h1_dft + h1[None, :, :] * kernel,
        h2_dft=mon.h2_dft + h2[None, :, :] * kernel,
    )


def flux_spectrum(mon: FluxMonitor, *, exact_f64: bool = False) -> jnp.ndarray:
    """Compute Poynting flux spectrum from accumulated DFT fields.

    Returns
    -------
    flux : (n_freqs,) float
        ∫ Re(E × H*) · n̂ dA at each frequency.
        Positive = power flowing in +axis direction.

    Notes
    -----
    With x64 disabled the accumulators are complex64, and per-cell E×H*
    products below the float32 minimum normal (~1.18e-38) are flushed to
    zero by XLA — a physically tiny-but-nonzero flux then returns EXACTLY
    0.0 at every frequency with healthy field accumulators (issue #304;
    single-cell fixed-amplitude sources radiate P ~ dx^6, so fine grids hit
    this readily). Eager calls detect that state with a float64 NumPy
    recompute of the identical sum and emit a UserWarning; under
    jit/grad the check is skipped entirely (tracer-safe, off the AD tape).

    ``exact_f64=True`` performs that float64 NumPy recompute AS the result
    (eager-only; raises under tracing): the per-cell products and the sum
    run in complex128 from the (healthy) float32 accumulators, which is
    the #304 warning's own sanctioned remedy. Absolute-scale energy
    consumers (the #589 flux-adjudication witness channel) use this path —
    partial subnormal flushing corrupts small-magnitude sums well before
    they reach exact zero, so the warning alone is not sufficient there.
    The default path is unchanged.
    """
    if exact_f64:
        import numpy as np
        import jax as _jax
        if isinstance(mon.e1_dft, _jax.core.Tracer):
            raise TypeError(
                "flux_spectrum(exact_f64=True) is eager-only — it is a "
                "NumPy float64 post-processing of concrete accumulators "
                "and cannot run under jit/grad."
            )
        e1 = np.asarray(mon.e1_dft, dtype=np.complex128)
        e2 = np.asarray(mon.e2_dft, dtype=np.complex128)
        h1 = np.asarray(mon.h1_dft, dtype=np.complex128)
        h2 = np.asarray(mon.h2_dft, dtype=np.complex128)
        integrand = e1 * np.conj(h2) - e2 * np.conj(h1)
        dA = np.asarray(mon.dA, dtype=np.float64)
        return np.real(np.sum(integrand * dA, axis=(-2, -1)))
    # Poynting: S_n = E1*H2* - E2*H1* (cyclic cross product).
    # mon.dA is the axis-aware area weight (scalar or (n1,n2)).
    integrand = mon.e1_dft * jnp.conj(mon.h2_dft) - mon.e2_dft * jnp.conj(mon.h1_dft)
    flux = jnp.real(jnp.sum(integrand * mon.dA, axis=(-2, -1)))
    _warn_if_flux_subnormal_flush(mon, flux)
    return flux


def _warn_if_flux_subnormal_flush(mon: FluxMonitor, flux) -> None:
    """Issue #304: eager-only detector for float32 subnormal-flushed flux.

    Warns when ``flux`` is exactly 0.0 at every frequency, the field
    accumulators are nonzero, AND a float64 recompute of the identical sum
    is nonzero — the decisive witness that the zeros are an underflow
    artefact, not physics. Never fires under tracing (imitates the
    ``warn_if_nonpassive_lumped_s11`` tracer-safety pattern) and never
    changes the returned value.
    """
    import jax as _jax
    try:
        if isinstance(flux, _jax.core.Tracer):
            return
    except Exception:
        return
    if flux.dtype != jnp.float32:
        return  # complex128 accumulators (scoped x64) don't flush this way
    import numpy as np
    f32 = np.asarray(flux)
    if f32.size == 0 or np.any(f32 != 0.0):
        return
    e1 = np.asarray(mon.e1_dft)
    e2 = np.asarray(mon.e2_dft)
    if not (np.any(e1 != 0) or np.any(e2 != 0)):
        return  # genuinely empty monitor — zero flux is the right answer
    h1 = np.asarray(mon.h1_dft, dtype=np.complex128)
    h2 = np.asarray(mon.h2_dft, dtype=np.complex128)
    dA = np.asarray(mon.dA, dtype=np.float64)
    f64 = np.real(np.sum(
        (e1.astype(np.complex128) * np.conj(h2)
         - e2.astype(np.complex128) * np.conj(h1)) * dA,
        axis=(-2, -1),
    ))
    if not np.any(f64 != 0.0):
        return  # float64 agrees the flux is zero — no artefact
    import warnings
    peak = float(np.max(np.abs(f64)))
    warnings.warn(
        f"flux_spectrum returned exactly 0.0 at all {f32.size} frequencies, "
        f"but the DFT accumulators are healthy and a float64 recompute of "
        f"the same sum gives nonzero flux (peak |flux| = {peak:.3e}). The "
        f"per-cell E x H* products underflowed the float32 minimum normal "
        f"(~1.18e-38) and were flushed to zero (issue #304). Remedies: "
        f"enable x64 in a scoped context for the flux computation, increase "
        f"the source amplitude, or recompute from the (healthy) "
        f"accumulators in float64 as done for this check.",
        UserWarning, stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Wire port voltage / current extraction
# ---------------------------------------------------------------------------

def _wire_port_live_mid(grid, port, pec_mask=None):
    """Midpoint cell of the LIVE wire run (issue #764).

    A dead extent cell (inside PEC, issue #318) carries essentially no port
    current (measured |I_dead|/|I_mid| = 0.003-0.03 on the #313 thru), so an
    all-extent midpoint landing on a dead cell read a quenched Ampere loop
    as the port current. With ``pec_mask=None`` — or no dead cells — this is
    bit-identical to the historical all-extent midpoint.
    """
    from rfx.sources.sources import _wire_port_live_cells

    cells, live_flags, _ = _wire_port_live_cells(grid, port, pec_mask)
    live = [c for c, l in zip(cells, live_flags) if l]
    return live[len(live) // 2]


def wire_port_voltage(state, grid, port, pec_mask=None) -> jnp.ndarray:
    """Voltage at the WirePort LIVE-run midpoint cell.

    Uses a single-cell measurement at the live-run midpoint (same location
    as wire_port_current) for balanced V/I wave decomposition. The midpoint
    is that of the LIVE run (issue #764; ``pec_mask=None`` or no dead cells
    is bit-identical to the historical all-extent midpoint).

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    port : WirePort
    pec_mask : optional assembled-geometry PEC mask (issue #318 live split)

    Returns
    -------
    float scalar
    """
    mid = _wire_port_live_mid(grid, port, pec_mask)
    field = getattr(state, port.component)
    return -field[mid[0], mid[1], mid[2]] * grid.dx


def wire_port_gap_voltage(state, grid, port, pec_mask=None) -> jnp.ndarray:
    """Whole-port gap voltage: V_port = sum over LIVE cells of -E_c*dx.

    The discrete line integral of E across the whole gap (issue #764). Only
    this SUM is KVL/Faraday-constrained on the staggered grid; a single cell
    is not (a PEC short forces sum V_c = 0 while the midpoint V stays
    finite). Dead extent cells (inside PEC, issue #318) are excluded.

    This is the driven-diagonal V of the whole-port reflection
    ``S_kk = (V_port - Z0*I)/(V_port + Z0*I)`` against the whole-port Z0
    that #318 physically realizes as the series termination.
    """
    from rfx.sources.sources import _wire_port_live_cells

    cells, live_flags, _ = _wire_port_live_cells(grid, port, pec_mask)
    field = getattr(state, port.component)
    v_port = 0.0
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        v_port = v_port - field[cell[0], cell[1], cell[2]] * grid.dx
    return v_port


def wire_port_current(state, grid, port,
                      periodic=(False, False, False),
                      pec_mask=None) -> jnp.ndarray:
    """Current through a WirePort via Ampere's law at the LIVE-run midpoint.

    Uses the H-field loop integral at the center cell of the LIVE wire run
    (issue #764; with no dead cells this is the historical all-extent
    midpoint), the same :func:`_ampere_loop` :func:`port_current` uses.
    Out-of-domain H is ZERO (#692) — see :func:`_bwd_h` for the rule and
    for the two axis kinds that keep the wrap.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    port : WirePort
    periodic : (bool, bool, bool)
        Per-axis periodic flags for the run; default non-periodic.
    pec_mask : optional assembled-geometry PEC mask (issue #318 live split)

    Returns
    -------
    float scalar
    """
    mid = _wire_port_live_mid(grid, port, pec_mask)
    dx = grid.dx

    return _ampere_loop(state, tuple(mid), port.component, dx, periodic)


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
    pec_mask=None,
) -> SParamProbe:
    """Create a zeroed SParamProbe for a WirePort.

    The port_index is set to the midpoint cell of the LIVE wire run
    (issue #764; identical to the all-extent midpoint with no dead cells).
    """
    mid_idx = tuple(_wire_port_live_mid(grid, port, pec_mask))
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
        v_port_dft=zeros,
        v_ref_dft=zeros,
    )


def update_wire_sparam_probe(
    probe: SParamProbe,
    state,
    grid,
    port,
    dt: float,
    n_live: int | None = None,
    pec_mask=None,
) -> SParamProbe:
    """Accumulate one timestep of V, I, V_inc, and V_port for a WirePort.

    Call this every timestep after update_e() / apply_pec() AND **after**
    apply_wire_port() (issue #683, decided by measurement 2026-08-29):
    the post-injection E is the true field level ``E^{n+1}`` of the
    discrete update, and only the post-injection V/I pair satisfies the
    known-load circuit law at an excited port (the previous contract —
    sample before injection, issue #72 — survives for LUMPED ports in
    ``update_sparam_probe`` pending their own decision run; the #683
    measurement was made on wire ports).  The PRE-injection drive-sample
    reference channel is accumulated separately by
    ``update_wire_drive_ref_probe`` (call it BEFORE apply_wire_port) —
    the #308/#313 off-diagonal calibration is pinned against that sample
    (see ``decompose_wire_s_matrix`` and
    docs/design_notes/issue683_decomposer_flip_predeclaration.md).

    V and I are measured at the LIVE-run midpoint cell; V_port is the
    whole-gap sum over live cells (issue #764).  V_inc uses the per-cell
    source voltage (V_src / n_live, matching ``apply_wire_port``'s
    live-cell injection — issue #318; ``n_live=None`` falls back to the
    all-cells count, the historical behaviour and the degenerate
    no-dead-cells case).
    """
    from rfx.sources.sources import _wire_port_cells

    t = state.step * dt  # keep traceable: float(state.step) leaked the tracer
    if n_live is None:
        n_live = max(len(_wire_port_cells(grid, port)), 1)

    v = wire_port_voltage(state, grid, port, pec_mask=pec_mask)
    i_val = wire_port_current(state, grid, port, pec_mask=pec_mask)
    v_port = wire_port_gap_voltage(state, grid, port, pec_mask=pec_mask)
    v_inc = port.excitation(t) / n_live

    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    # Yee half-step: H (hence the Ampere-loop current) is at n+1/2 while E
    # (voltage / gap voltage) is at n+1; advance the current sample by dt/2
    # so both DFT channels share the same reference time (rfx/core/dft_utils).
    i_phase = phase * _half_step_current_phase(probe.freqs, dt)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window, probe.window_alpha)
    new_v = probe.v_dft + v * phase * dt * weight
    new_i = probe.i_dft + i_val * i_phase * dt * weight
    new_vinc = probe.v_inc_dft + v_inc * phase * dt * weight
    old_vp = probe.v_port_dft if probe.v_port_dft is not None else 0.0
    new_vport = old_vp + v_port * phase * dt * weight

    return probe._replace(v_dft=new_v, i_dft=new_i, v_inc_dft=new_vinc,
                          v_port_dft=new_vport)


def update_wire_drive_ref_probe(
    probe: SParamProbe,
    state,
    grid,
    port,
    dt: float,
    pec_mask=None,
) -> SParamProbe:
    """Accumulate one timestep of the PRE-injection drive-sample reference.

    Call this after update_e() / apply_pec() but **before**
    apply_wire_port() — the historical (#72) sampling slot.  The channel
    is bit-identical to the pre-#683 ``v_dft`` and exists because the
    #308 receive-wave sign and Z0/n_cells normalization were calibrated
    against this sample (issue #683 x #764 decomposer recalibration; see
    ``decompose_wire_s_matrix``).  It feeds ONLY the off-diagonal
    incident-wave denominator and the byte-frozen legacy diagonal; the
    physical V/I/V_port channels are accumulated post-injection by
    ``update_wire_sparam_probe``.
    """
    t = state.step * dt
    v_ref = wire_port_voltage(state, grid, port, pec_mask=pec_mask)
    phase = jnp.exp(-1j * 2.0 * jnp.pi * probe.freqs * t)
    weight = _dft_window_weight(state.step, probe.total_steps, probe.window,
                                probe.window_alpha)
    old_ref = probe.v_ref_dft if probe.v_ref_dft is not None else 0.0
    return probe._replace(v_ref_dft=old_ref + v_ref * phase * dt * weight)


# ---------------------------------------------------------------------------
# Pure wave decomposers (single source of truth for the lumped/wire S-matrix)
# ---------------------------------------------------------------------------
#
# These extract the *post-processing* of the eager extractors into reusable,
# AD-friendly (jnp) module-level functions so that both the eager
# Python-loop extractor (extract_s_matrix / extract_s_matrix_wire) and the
# production-scan driver (rfx/probes/sparam_driver.py) decompose V/I phasors
# with ONE implementation.  The eager extractors call these so the eager path
# stays bit-identical (item-5 Stage 1, 2026-06-22).

def decompose_lumped_s_matrix(v, i, z0):
    """Lumped-port N-port S-matrix from accumulated V/I DFTs.

    Wave decomposition with the FDTD sign convention (``V = -E·dx``),
    role-selected per port (issue #308):

        a_j = (-V[j,j] + Z0[j]·I[j,j]) / (2·√Z0[j])    # incident at driven port j
        b_j = (-V[j,j] - Z0[j]·I[j,j]) / (2·√Z0[j])    # reflected at the DRIVE port
        b_i = (V[j,i] - Z0[i]·I[j,i]) / (2·√Z0[i])     # arriving at PASSIVE port i≠j
        S[i,j] = b_i / a_j

    Drive-port waves (``a_j`` and the diagonal ``b_j``) keep the
    ``extract_lumped_s11`` algebra unchanged.  At a passive receive port the
    historical ``(-V - Z0·I)`` channel structurally cancelled the arriving
    wave: the port-cell resistor law makes ``-V == +Z0_cell·I`` identically
    at a matched port, so a matched thru read |S21| near-null (verified,
    issue #308).  The receive channel is therefore the orthogonal
    combination ``±(V - Z0·I)``, and the overall sign is pinned
    EMPIRICALLY by the low-frequency falsifier on the canonical 2-port
    thru (2026-07-10): under ``(V - Z0·I)`` the measured S21(DC) -> +1
    (the DC thru limit); the first-cut opposite sign ``(-V + Z0·I)``
    measured S21(DC) -> -1 (the pi sat in the raw cross-port phasors,
    arg(V2/V1) ≈ pi - beta·L, from the source-driven cell field sense in
    ``V = -E·dx``).

    Sign convention: multiports whose ports share the same field component
    carry a single global receive-wave sign, pinned by that DC witness on
    the canonical thru.  A multiport mixing components (e.g. one ``ez``
    and one ``ey`` port) has no orientation input to relate the per-port
    voltage polarities, so its off-diagonal S entries remain defined only
    up to a ±1 (pi-phase) factor (fence unchanged).

    OPEN item: the port-based |S21| MAGNITUDE is deflated relative to the
    extractor-independent flux referee (flux-true |S21| 0.97-1.0 vs
    0.52-0.67 here on the canonical thru); the Phase-0 closed-box referee
    traced it to the port-cell wave definitions themselves (near-field
    dominated, do not conserve power) — see issue #313.  The opt-in
    ``add_port(reference_plane_cells=N)`` reference-plane path
    (``rfx.probes.refplane``) replaces the opted off-diagonals with line
    plane waves; THIS default port-cell path keeps the deflation and its
    committed regression locks unchanged.

    The safe-denominator guard replaces a zero incident wave by 1 (so
    S → 0 / 1 = 0 rather than NaN).  Mirrors ``extract_s_matrix`` exactly.

    Parameters
    ----------
    v, i : (n_ports, n_ports, n_freqs) complex
        ``v[j, i]`` / ``i[j, i]`` are the V/I DFT phasors at receive port *i*
        when driving port *j* (FDTD sign convention).
    z0 : (n_ports,) real
        Per-port reference impedance.

    Returns
    -------
    S : (n_ports, n_ports, n_freqs) complex
        ``S[i, j]`` is the response at receive port *i* when driving port *j*.
    """
    v = jnp.asarray(v)
    i = jnp.asarray(i)
    z0 = jnp.asarray(z0)
    n_ports = v.shape[0]
    n_freqs = v.shape[-1]
    S = jnp.zeros((n_ports, n_ports, n_freqs), dtype=jnp.complex64)
    for j in range(n_ports):
        z0_j = z0[j]
        a_j = (-v[j, j] + z0_j * i[j, j]) / (2.0 * jnp.sqrt(z0_j))
        safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))
        for ri in range(n_ports):
            z0_i = z0[ri]
            if ri == j:
                # Drive-port reflected wave — byte-frozen extract_lumped_s11
                # algebra (issue #308 changes receive ports only).
                b_i = (-v[j, ri] - z0_i * i[j, ri]) / (2.0 * jnp.sqrt(z0_i))
            else:
                # Passive receive port: orthogonal wave channel; sign pinned
                # by the DC falsifier (S21(DC) -> +1 on the canonical thru,
                # issue #308).
                b_i = (v[j, ri] - z0_i * i[j, ri]) / (2.0 * jnp.sqrt(z0_i))
            S = S.at[ri, j, :].set((b_i / safe_a).astype(jnp.complex64))
    return S


def decompose_wire_s_matrix(v, i, z0, port_cell_counts, v_port=None,
                            v_ref=None):
    """Wire-port N-port S-matrix from accumulated midpoint V/I DFTs.

    Diagonal entries (every one of which is a DRIVEN reading here — column
    *j* comes from the drive run of port *j*) use the whole-port driven
    reflection (issue #764) when the whole-port gap-voltage channel
    ``v_port`` is provided:

        Z_driven = +V_port[j,j] / I[j,j]
        S_jj = (Z_driven − Z0_j) / (Z_driven + Z0_j)
             = (V_port − Z0_j·I) / (V_port + Z0_j·I)

    with ``V_port = sum over LIVE wire cells of -E_c*dx`` (the whole-gap
    line integral — the only KVL-constrained V on the staggered grid) and
    Z0 the whole-port impedance #318 physically realizes as the series
    termination.  PROVENANCE for the re-pin of the previously "byte-frozen"
    diagonal: the old ``Z_in = -V_mid/I`` measured the per-cell Z0/n_live
    against the whole-port Z0 (the #313/#318 frame mismatch) and did not
    track the load at all — a matched 50 ohm load read S11 = +0.35426 and a
    PEC short +0.26780 (ledger 2026-06-21 port review).  The ``-V/I`` sense
    itself is the PASSIVE port-branch convention; at a driven port the
    passive-sign current direction flips between the port branch and the
    external branch, so the driven relation is ``V_port = +Z_L*I`` (#683
    circuit law ``|I| = |V_src|/(Z0+Z_L)``).

    ``v_port=None`` keeps the legacy per-cell decomposition byte-for-byte
    (per-cell diagonal AND per-cell off-diagonal) — used ONLY by (a) the
    #313 reference-plane decomposer, whose non-plane channels are
    documented byte-frozen legacy, and (b) replay of pre-#770 dumps whose
    recorded production S-matrix used the per-cell off-diagonal frame
    (pre-#764 dumps additionally used the legacy diagonal).

    SAMPLING-ORDER CONTRACT (issue #683 x #764, landed 2026-08-29; the
    derivation is docs/design_notes/issue683_decomposer_flip_predeclaration.md):
    V, I, and V_port are sampled POST-injection — the true field level
    ``E^{n+1}``, the only sample that satisfies the known-load circuit
    law at the driven port (#683 verdict) — so the driven diagonal above
    is a validated physical reading (Gamma_L-exact class).  ``v_ref`` is
    the PRE-injection midpoint drive sample (bit-identical to the
    pre-#683 ``v``), consumed ONLY by the LEGACY ``v_port=None`` path
    (its #308-calibrated off-diagonal incident wave and its per-cell
    diagonal); ``v_ref=None`` falls back to ``v``, which for pre-#683
    data IS the reference.

    OFF-DIAGONAL FRAME (issue #770, adjudicated 2026-08-29 —
    docs/design_notes/issue770_offdiag_adjudication_predeclaration.md +
    the results note): with ``v_port`` provided the off-diagonals are the
    WHOLE-PORT wave pair, frame-consistent with the #764 diagonal:

        a_j = (V_port[j,j] + Z0_j·I[j,j]) / (2·√Z0_j)   # incident, driven port j
        b_i = (V_port[j,i] - Z0_i·I[j,i]) / (2·√Z0_i)   # outgoing at PASSIVE port i≠j

    ``a_j`` is the physical incident wave: the measured #683 circuit law
    makes ``V_port + Z0·I = Z0·Î0`` a drive-only constant, so
    ``|a_j|²`` is the available power of the Norton drive ``(Î0, Z0)``.
    ``b_i`` is the whole-port completion of the #308 orthogonal receive
    selection (the direct combination ``V_port + Z0·I`` vanishes at the
    matched termination — it is the re-incident wave); the global
    receive sign (+1) was re-pinned by the same DC witness class as
    #308 (S21(DC) -> +1; measured dev -0.058/-0.115 rad at 0.5/1 GHz,
    flipped channel at ~pi).  Measured provenance for the frame change
    (#770 harness, canonical THRU, NU lane primary + uniform parity
    3.97e-6): whole-port |S21| = 0.9341-0.9954 across 3-7 GHz vs the
    external flux referee 0.971-0.997 / openEMS 0.973-1.034; per-bin
    power closure |S11|²+|S21|² = 0.956-0.991 (deficit 0.9-4.4%,
    matching the measured 0.2-4.0% flux closure gap); reciprocity
    max|S21-S12| = 2.67e-4 (28x tighter than the per-cell 7.53e-3
    class).  The per-cell #308/#313 frame measured a net-through
    fraction |S21|²/(1-|S11|²) of 0.31-0.37 on the same runs — REFUTED
    as physics by the pre-declared falsifier F-A2 (it remains what the
    ledger always called it: a regression-lock frame, now preserved
    only on the legacy path below).

    LEGACY ``v_port=None`` off-diagonal (byte-frozen): the per-cell
    #308 decomposition against the PRE-injection drive reference,

        a_j = (-V_ref[j,j] + Z0c_j·I[j,j]) / (2·√Z0c_j)
        b_i = (V[j,i] - Z0c_i·I[j,i]) / (2·√Z0c_i),   Z0c = Z0/n_cells

    with the #308 receive-channel history: at a passive receive port the
    historical ``(-V - Z0·I)`` channel structurally cancelled the
    arriving wave (the port-cell resistor law makes ``-V == +Z0_cell·I``
    identically at a matched port, so a matched thru read |S21|
    near-null); the orthogonal combination's sign was pinned by the
    2026-07-10 DC falsifier.  Composing the per-cell channel with the
    POST drive sample instead of ``v_ref`` detonates the THRU locks
    (raw-flip review: |S21 - S12| max 76.79 — ``-V + Z0c*I`` is a
    structural cancellation at a matched POST drive), which is why the
    legacy path keeps its own reference channel.

    Sign convention: multiports whose ports share the same field component
    carry a single global receive-wave sign, pinned by that DC witness on
    the canonical thru.  A multiport mixing components (e.g. one ``ez``
    and one ``ey`` port) has no orientation input to relate the per-port
    voltage polarities, so its off-diagonal S entries remain defined only
    up to a ±1 (pi-phase) factor (fence unchanged).

    The #313 |S21| deflation (flux-true 0.97-1.0 vs port-cell 0.52-0.67
    on the canonical thru) was a property of the PER-CELL frame; the
    whole-port frame closes it (measured above), so the deflation now
    lives only on the legacy ``v_port=None`` path and its byte-frozen
    consumers.  The opt-in ``add_port(reference_plane_cells=N)`` path
    (``rfx.probes.refplane``) remains the de-embedded line-plane-wave
    alternative and is untouched.

    Mirrors ``extract_s_matrix_wire`` line-for-line.

    Parameters
    ----------
    v, i : (n_ports, n_ports, n_freqs) complex
        ``v[j, i]`` / ``i[j, i]`` are the midpoint-cell V/I DFT phasors at
        receive port *i* when driving port *j* (FDTD sign convention).
    z0 : (n_ports,) real
        Per-port total reference impedance.
    port_cell_counts : (n_ports,) int
        Number of wire cells per port (for per-cell impedance normalization).
    v_port : (n_ports, n_ports, n_freqs) complex or None
        Whole-port gap-voltage DFT phasors (``v_port[j, i]`` at receive
        port *i* when driving port *j*): the driven diagonal feeds the
        #764 physical reflection and every entry feeds the #770
        whole-port off-diagonal waves.  None selects the byte-frozen
        legacy per-cell decomposition (diagonal AND off-diagonal — see
        above).
    v_ref : (n_ports, n_ports, n_freqs) complex or None
        PRE-injection drive-sample reference phasors (issue #683 x #764);
        consumed ONLY by the legacy ``v_port=None`` path (its per-cell
        diagonal and its #308-calibrated incident wave, both reading the
        drive diagonal ``v_ref[j, j]``).  None falls back to ``v``
        (pre-#683 data, where ``v`` IS the pre-injection sample).

    Returns
    -------
    S : (n_ports, n_ports, n_freqs) complex
        ``S[i, j]`` is the response at receive port *i* when driving port *j*.
    """
    v = jnp.asarray(v)
    i = jnp.asarray(i)
    z0 = jnp.asarray(z0)
    if v_port is not None:
        v_port = jnp.asarray(v_port)
    # Drive-sample reference (issue #683 x #764): pre-injection midpoint
    # drive sample; falls back to ``v`` for pre-#683 data.
    v_ref = v if v_ref is None else jnp.asarray(v_ref)
    n_ports = v.shape[0]
    n_freqs = v.shape[-1]
    S = jnp.zeros((n_ports, n_ports, n_freqs), dtype=jnp.complex64)
    for j in range(n_ports):
        z0_j = z0[j]
        safe_i = jnp.where(
            jnp.abs(i[j, j]) > 0,
            i[j, j],
            jnp.ones_like(i[j, j]) * 1e-30,
        )
        if v_port is None:
            # Legacy per-cell input impedance — byte-frozen against the
            # PRE-injection reference (#313 refplane diagonals, replay).
            z_in_j = -v_ref[j, j] / safe_i
        else:
            z_in_j = v_port[j, j] / safe_i  # driven whole-port Z (issue #764)
        for ri in range(n_ports):
            z0_i = z0[ri]
            if ri == j:
                S = S.at[ri, j, :].set(
                    ((z_in_j - z0_i) / (z_in_j + z0_i)).astype(jnp.complex64))
            elif v_port is not None:
                # Whole-port off-diagonal frame (issue #770, adjudicated
                # 2026-08-29 — see OFF-DIAGONAL FRAME above): outgoing
                # wave at the passive receive port against the physical
                # incident wave at the driven port.  The receive sign
                # (+1) is the DC-witness pin; the incident wave is the
                # measured drive-only constant Z0·I0 (#683 circuit law).
                b_i = (v_port[j, ri] - z0_i * i[j, ri]) / (
                    2.0 * jnp.sqrt(z0_i))
                a_j = (v_port[j, j] + z0_j * i[j, j]) / (
                    2.0 * jnp.sqrt(z0_j))
                safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))
                S = S.at[ri, j, :].set((b_i / safe_a).astype(jnp.complex64))
            else:
                # LEGACY per-cell off-diagonal (byte-frozen; #313 refplane
                # decomposer + pre-#770 dump replay only).
                n_cells_i = max(int(port_cell_counts[ri]), 1)
                z0_cell_i = z0_i / n_cells_i
                # Passive receive port: orthogonal wave channel (issue #308
                # — the old (-V - Z0·I) channel structurally cancelled the
                # arriving wave at a matched port; sign pinned by the DC
                # falsifier, S21(DC) -> +1 on the canonical thru).
                b_i = (v[j, ri] - z0_cell_i * i[j, ri]) / (
                    2.0 * jnp.sqrt(z0_cell_i))
                n_cells_j = max(int(port_cell_counts[j]), 1)
                z0_cell_j = z0_j / n_cells_j
                # Incident wave against the PRE-injection drive-sample
                # reference (issue #683 x #764 — see the SAMPLING-ORDER
                # CONTRACT above; the #308 sign and Z0c normalization are
                # calibrations pinned on this reference).
                a_j = (-v_ref[j, j] + z0_cell_j * i[j, j]) / (
                    2.0 * jnp.sqrt(z0_cell_j))
                safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))
                S = S.at[ri, j, :].set((b_i / safe_a).astype(jnp.complex64))
    return S


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
    pec_mask: object | None = None,
    return_vi_dump: bool = False,
) -> jnp.ndarray | PortVIReplayBundle:
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
    return_vi_dump : bool
        When True, return a :class:`PortVIReplayBundle` containing the
        production S-matrix plus raw V/I phasors in the public replay
        convention.

    Returns
    -------
    S or PortVIReplayBundle
        By default, ``S[i, j, :]`` is S_{i+1, j+1} (response at port *i*
        when exciting port *j*). With ``return_vi_dump=True``, the bundle also
        carries replayable raw phasors.
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

    # FDTD-sign V/I phasors per (drive j, receive i) for the shared decomposer.
    v_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    i_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    raw_v = (
        np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
        if return_vi_dump else None
    )
    raw_i = (
        np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
        if return_vi_dump else None
    )

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
                    state, cpml_params, cpml_state, grid, cpml_axes,
                    materials=mats)

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
                    state, cpml_params, cpml_state, grid, cpml_axes,
                    materials=mats)
            state = apply_pec(state)

            # Apply interior PEC mask (e.g. ground plane, scatterers
            # added via ``Box(..., material="pec")``).  Without this,
            # eval simulations driven from ``run(compute_s_params=True)``
            # for lumped ports lose the ground-plane / scatterer
            # geometry — antenna stops coupling and |S11| collapses
            # toward 0 dB across the band, producing a 9–10 dB
            # train/eval disconnect identical in shape to the prior
            # time-gating-heuristic bug (issue #72).
            if pec_mask is not None:
                from rfx.boundaries.pec import apply_pec_mask
                # #689: default (non-periodic) is correct — preflight
                # `_validate_run_sparameter_request` (rfx/api/_preflight.py)
                # refuses lumped/wire S-params under periodic axes (#206,
                # NotImplementedError), and
                # this eager re-run is non-periodic by construction.
                # 2-D safety comes from the length-1-axis guard.
                state = apply_pec_mask(state, pec_mask)

            # Record V / I at all ports BEFORE source injection so that
            # the sampled voltage reflects only the load/cavity response
            # and is not contaminated by the driving waveform.
            for i in range(n_ports):
                sprobes[i] = update_sparam_probe(
                    sprobes[i], state, grid, ports[i], dt)

            # Excite only port j
            state = apply_lumped_port(state, grid, ports[j], t, mats)

        # Collect per-(drive j, receive i) V/I DFT phasors for the shared
        # wave decomposer (``decompose_lumped_s_matrix``) — single source of
        # truth for the lumped decomposition shared with the production-scan
        # driver (item-5 Stage 1).  The dump schema stores ``-V`` (voltage
        # positive into the DUT); the FDTD-sign V is fed to the decomposer.
        for i in range(n_ports):
            v_all[j, i, :] = np.asarray(sprobes[i].v_dft, dtype=np.complex128)
            i_all[j, i, :] = np.asarray(sprobes[i].i_dft, dtype=np.complex128)
            if return_vi_dump:
                # ``port_voltage`` returns the FDTD-sign voltage used by the
                # legacy extractor (V = -E·dx).  The portable dump schema uses
                # voltage positive into the DUT, so store ``-V``.  With current
                # positive into the DUT, the drive-port replay formulas
                # a=(V+ZI)/2√Z, b=(V−ZI)/2√Z reproduce the production
                # DIAGONAL below exactly; since issue #308 the production
                # PASSIVE-port b-wave is role-selected — in the dump's
                # into-DUT convention b_recv=−(V+ZI)/2√Z (sign pinned by the
                # DC falsifier) — so an off-diagonal replay must apply the
                # same per-role convention.
                raw_v[j, i, :] = np.asarray(-sprobes[i].v_dft, dtype=np.complex128)
                raw_i[j, i, :] = np.asarray(sprobes[i].i_dft, dtype=np.complex128)

    z0_arr = np.asarray([p.impedance for p in ports], dtype=np.float64)
    S = np.asarray(
        decompose_lumped_s_matrix(v_all, i_all, z0_arr), dtype=np.complex64)

    if return_vi_dump:
        return PortVIReplayBundle(
            s_params=S,
            freqs=jnp.asarray(freqs),
            voltages=raw_v,
            currents=raw_i,
            port_impedances=np.asarray([p.impedance for p in ports], dtype=np.float64),
            port_names=tuple(f"port_{idx}" for idx in range(n_ports)),
            driven_port_indices=tuple(range(n_ports)),
        )

    return S


def extract_lumped_s11(
    v_dft: jnp.ndarray,
    i_dft: jnp.ndarray,
    z0: float = 50.0,
) -> jnp.ndarray:
    """S11 from accumulated V/I DFTs at a single-cell lumped port (issue #72).

    Wave decomposition with FDTD sign convention (V = -E·dx):

        a = (-V + Z0·I) / (2·√Z0)        # incident (into DUT)
        b = (-V - Z0·I) / (2·√Z0)        # reflected (out of DUT)
        S11 = b / a = (V + Z0·I) / (V - Z0·I)

    Equivalent to ``extract_s11`` but operates on raw DFT arrays produced
    by the JIT-integrated lumped-port path
    (``SimResult.lumped_port_sparams``), enabling AD-friendly objectives
    on ``Simulation.forward()`` without the time-gating heuristic of
    ``minimize_s11_at_freq``.

    Parameters
    ----------
    v_dft : (n_freqs,) complex
        Accumulated port voltage DFT (FDTD sign convention).
    i_dft : (n_freqs,) complex
        Accumulated port current DFT.
    z0 : float
        Reference impedance (ohms). Default 50.

    Returns
    -------
    s11 : (n_freqs,) complex
    """
    denom = v_dft - z0 * i_dft
    safe_denom = jnp.where(jnp.abs(denom) > 0.0, denom, jnp.ones_like(denom))
    return (v_dft + z0 * i_dft) / safe_denom


def warn_if_nonpassive_lumped_s11(s_params, freqs, *, extractor: str,
                                  passivity_tol: float = 0.10) -> None:
    """Tracer-safe passivity self-check for lumped/wire port S11 (issue #206 sib).

    ``run(compute_s_params=True)`` and ``forward(port_s11_freqs=...)`` for
    lumped/wire ports assemble S11 from the eager single-cell extractor, but —
    unlike ``compute_*_s_matrix`` (which calls
    :func:`rfx.api._sparams._warn_if_nonpassive_smatrix`) — they previously
    surfaced no passivity check. A passive one-port cannot have ``|S11| > 1``;
    a gross violation (measured up to ~1.98 on a high-eps closed cavity) means
    the extractor is unreliable at that frequency (single-cell curl-of-H
    current is ill-conditioned where the incident wave is weak, i.e. spectral
    band edges), NOT that the device is exotic. Surface it so an eager
    analysis / optimizer setup does not trust or chase those bins.

    ``s_params``: ``(n_freqs,)`` single port, or ``(n_ports, n_freqs)`` per-port
    diagonal S_ii. ``passivity_tol`` matches the repo standard (0.10) so mild
    float32 noise just above 1.0 is not flagged.

    Tracer-safe: under ``jax.grad`` / ``jax.jit`` ``s_params`` is an abstract
    tracer with no concrete value, so the check is skipped entirely — it never
    perturbs an AD/optimization run, only the eager research call.
    """
    import numpy as np
    import jax as _jax
    try:
        if isinstance(s_params, _jax.core.Tracer):
            return
    except Exception:
        pass
    try:
        mag = np.abs(np.asarray(s_params))
        f = np.asarray(freqs)
    except Exception:
        return
    if mag.size == 0:
        return
    nonfinite = ~np.isfinite(mag)
    over = mag > (1.0 + passivity_tol)
    if not (over.any() or nonfinite.any()):
        return
    worst = float(np.nanmax(mag)) if np.isfinite(mag).any() else float("nan")
    bad_f = []
    try:
        bad_cols = np.unique(np.where(over | nonfinite)[-1])
        bad_f = [f"{float(f[c]) / 1e9:.3f}" for c in bad_cols[:6] if c < f.size]
    except Exception:
        pass
    # Describe the actual trigger accurately: a non-finite bin is its own
    # failure mode (don't report a misleadingly-small finite max for it).
    if nonfinite.any():
        kind = f"non-finite (and max finite|S11|={worst:.3f})"
    else:
        kind = f"max|S11|={worst:.3f} > 1+{passivity_tol:g}"
    import warnings as _w
    _w.warn(
        f"{extractor}: lumped/wire port S11 is non-passive ({kind}) "
        f"at freq(GHz)={bad_f}"
        f"{' …' if len(bad_f) == 6 else ''}. A passive one-port cannot have "
        f"|S11|>1; the single-cell extractor is unreliable where the incident "
        f"wave is weak (spectral band edges). Do not trust or optimize against "
        f"these bins — restrict the band to where the source has energy, or use "
        f"run() for a passive analysis curve.",
        stacklevel=3,
    )


def extract_s11_normalised(probe: SParamProbe, z0: float = 50.0) -> jnp.ndarray:
    """Compute S11 normalised against the incident source DFT.

    Uses the reflected-wave definition with FDTD sign convention
    (V = -E·dx):

        b ∝ (-V - Z0 · I)
        S11 = -(V + Z0 · I) / (2 · V_inc)

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
    s11 = -(probe.v_dft + z0 * probe.i_dft) / safe_ref
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
    return_vi_dump: bool = False,
) -> jnp.ndarray | WirePortVIReplayBundle:
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
    return_vi_dump : bool
        When True, return a :class:`WirePortVIReplayBundle` containing the
        production S-matrix plus raw midpoint-cell V/I phasors for independent
        replay of the current wire-port convention.

    Returns
    -------
    S or WirePortVIReplayBundle
        By default, returns an S-matrix. With ``return_vi_dump=True``, the
        bundle also carries raw phasors and per-port wire cell counts.
    """
    import numpy as np
    from rfx.core.yee import init_state, update_h
    from rfx.boundaries.pec import apply_pec
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    from rfx.simulation import _update_e_with_optional_dispersion
    from rfx.sources.sources import (
        setup_wire_port,
        apply_wire_port,
        _wire_port_cells,
        _wire_port_live_cells,
    )

    n_ports = len(ports)
    n_freqs = len(freqs)
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=30)

    dt, dx = grid.dt, grid.dx
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state_init = init_cpml(grid)

    # Fold ALL port impedances into materials (once), over LIVE cells only
    # (issue #318 — dead extent cells inside PEC are excluded from the
    # sigma distribution, injection, and normalization counts below).
    mats = materials
    for p in ports:
        mats = setup_wire_port(grid, p, mats, pec_mask=pec_mask)

    debye = None
    if debye_spec is not None:
        debye_poles, debye_masks = debye_spec
        debye = init_debye(debye_poles, mats, dt, mask=debye_masks)

    lorentz = None
    if lorentz_spec is not None:
        lorentz_poles, lorentz_masks = lorentz_spec
        lorentz = init_lorentz(lorentz_poles, mats, dt, mask=lorentz_masks)

    # FDTD-sign midpoint V/I phasors per (drive j, receive i) for the
    # shared wire decomposer (``decompose_wire_s_matrix``), plus the
    # whole-port gap-voltage channel for the driven diagonal (issue #764).
    v_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    i_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    vp_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    vref_all = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    raw_v = (
        np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
        if return_vi_dump else None
    )
    raw_i = (
        np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
        if return_vi_dump else None
    )
    # Per-port LIVE cell counts (issue #318): feed the wave-decomposition
    # normalization Z0c = Z0/n_live so the per-cell resistor law
    # R_cell = Z0/n_live == Z0c stays exact (the #308 receive-channel
    # selection relies on that identity).  With no dead cells this is the
    # historical all-cells count.
    port_n_live = [
        _wire_port_live_cells(grid, p, pec_mask)[2] for p in ports
    ]
    port_cell_counts = np.asarray(port_n_live, dtype=np.int64)

    for j in range(n_ports):
        state = init_state(grid.shape)
        sprobes = [
            init_wire_sparam_probe(grid, p, freqs, dft_total_steps=n_steps,
                                   pec_mask=pec_mask)
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
                    state, cpml_params, cpml_state, grid, cpml_axes,
                    materials=mats)

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
                    state, cpml_params, cpml_state, grid, cpml_axes,
                    materials=mats)
            state = apply_pec(state)

            if pec_mask is not None:
                from rfx.boundaries.pec import apply_pec_mask
                # #689: default (non-periodic) is correct — preflight
                # `_validate_run_sparameter_request` (rfx/api/_preflight.py)
                # refuses lumped/wire S-params under periodic axes (#206,
                # NotImplementedError), and
                # this eager re-run is non-periodic by construction.
                # 2-D safety comes from the length-1-axis guard.
                state = apply_pec_mask(state, pec_mask)

            # Record the PRE-injection drive-sample REFERENCE channel at
            # the historical (#72) slot — the #308/#313 off-diagonal
            # calibration is pinned against this sample (issue #683 x
            # #764 decomposer recalibration; bit-identical to the
            # pre-#683 v_dft).
            for i in range(n_ports):
                sprobes[i] = update_wire_drive_ref_probe(
                    sprobes[i], state, grid, ports[i], dt,
                    pec_mask=pec_mask)

            # Excite only port j (live cells only, issue #318)
            state = apply_wire_port(state, grid, ports[j], t, mats,
                                    pec_mask=pec_mask)

            # Record the PHYSICAL V / I / V_port channels at all ports
            # AFTER source injection (issue #683, decided by measurement
            # 2026-08-29 — docs/design_notes/
            # issue683_sampling_order_decision_protocol.md section 9).
            # The pre-injection sample is not any field time level of the
            # discrete update and fails the known-load circuit law at the
            # driven port; the post-injection sample is the true E^{n+1}.
            # Passive ports (every i != j) read identically on either
            # side: the injection writes only port j's live cells, and I
            # reads H only.  This keeps the eager path in lockstep with
            # the production-scan driver (rfx/simulation.py wire block),
            # which flipped in the same commit —
            # test_sparam_driver_matches_eager pins that.
            for i in range(n_ports):
                sprobes[i] = update_wire_sparam_probe(
                    sprobes[i], state, grid, ports[i], dt,
                    n_live=port_n_live[i], pec_mask=pec_mask)

        # Collect per-(drive j, receive i) midpoint V/I DFT phasors for the
        # shared wire wave decomposer (``decompose_wire_s_matrix``) — single
        # source of truth shared with the production-scan driver.  Wire dump
        # stores the FDTD-sign V (no negation, unlike lumped).
        for i in range(n_ports):
            v_all[j, i, :] = np.asarray(sprobes[i].v_dft, dtype=np.complex128)
            i_all[j, i, :] = np.asarray(sprobes[i].i_dft, dtype=np.complex128)
            vp_all[j, i, :] = np.asarray(sprobes[i].v_port_dft,
                                         dtype=np.complex128)
            vref_all[j, i, :] = np.asarray(sprobes[i].v_ref_dft,
                                           dtype=np.complex128)
            if return_vi_dump:
                raw_v[j, i, :] = np.asarray(sprobes[i].v_dft, dtype=np.complex128)
                raw_i[j, i, :] = np.asarray(sprobes[i].i_dft, dtype=np.complex128)

    z0_arr = np.asarray([p.impedance for p in ports], dtype=np.float64)
    # Issue #764 diagonal + issue #770 off-diagonal: the whole-port
    # gap-voltage channel feeds BOTH (the frame-consistent whole-port
    # wave pair; per-cell #308 frame refuted by the #770 adjudication).
    # v_ref is carried for the legacy/replay path only.
    S = np.asarray(
        decompose_wire_s_matrix(v_all, i_all, z0_arr, port_cell_counts,
                                v_port=vp_all, v_ref=vref_all),
        dtype=np.complex64)

    if return_vi_dump:
        return WirePortVIReplayBundle(
            s_params=S,
            freqs=jnp.asarray(freqs),
            raw_voltages_fdt=raw_v,
            raw_currents=raw_i,
            port_impedances=np.asarray([p.impedance for p in ports], dtype=np.float64),
            port_cell_counts=port_cell_counts,
            port_names=tuple(f"wire_{idx}" for idx in range(n_ports)),
            driven_port_indices=tuple(range(n_ports)),
            raw_port_voltages_fdt=vp_all,
            raw_drive_ref_voltages_fdt=vref_all,
        )

    return S
