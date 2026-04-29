"""Compiled FDTD simulation runner.

Composes Yee updates, boundaries, sources, probes, and optional TFSF
plane-wave injection into a single JIT-compiled time loop via
``jax.lax.scan``. All subsystem selection (CPML, dispersion, TFSF) is
resolved at Python trace-time so the compiled function contains only
the needed code paths.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_e, update_e_aniso, update_h, EPS_0, _shift_bwd,
    precompute_coeffs, update_he_fast,
)
from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_occupancy


# ---------------------------------------------------------------------------
# Source / probe specifications
# ---------------------------------------------------------------------------

class SourceSpec(NamedTuple):
    """Precomputed point source for the compiled runner.

    waveform : (n_steps,) float array — precomputed values added to the
        field component at (i, j, k) each timestep.
    """
    i: int
    j: int
    k: int
    component: str
    waveform: jnp.ndarray


class ProbeSpec(NamedTuple):
    """Point probe that records a field component each timestep."""
    i: int
    j: int
    k: int
    component: str


class WirePortSParamSpec(NamedTuple):
    """Wire port S-param DFT specification for the compiled runner.

    Pre-computed metadata for inline V/I DFT accumulation inside
    jax.lax.scan, avoiding the separate Python-loop S-param extraction.

    mid_i, mid_j, mid_k: midpoint cell for V and I measurement.
    component: 'ez', 'ex', or 'ey'.
    freqs: (n_freqs,) frequency array.
    impedance: port reference impedance (Z0).
    """
    mid_i: int
    mid_j: int
    mid_k: int
    component: str
    freqs: jnp.ndarray
    impedance: float


class LumpedPortSParamSpec(NamedTuple):
    """Lumped port S-param DFT specification for the compiled runner.

    Pre-computed metadata for inline V/I DFT accumulation inside
    jax.lax.scan, providing an AD-compatible alternative to the
    Python-loop ``extract_s_matrix`` path for single-cell lumped ports.

    Attributes
    ----------
    i, j, k : int
        Lumped port cell index.
    component : str
        E-field component ("ex", "ey", or "ez").
    freqs : (n_freqs,) jnp.ndarray
        Frequencies (Hz) at which to accumulate V and I DFTs.
    impedance : float
        Port reference impedance Z0 (ohms).

    Notes
    -----
    V is sampled as ``-E·dx`` and I as the curl-H loop integral times
    ``dx`` at the port cell.  S11 is computed post-hoc via the wave
    decomposition ``a = (-V + Z0·I)/(2√Z0)``, ``b = (-V - Z0·I)/(2√Z0)``,
    ``S11 = b/a`` — exact, no time-gating heuristic.  Issue #72.
    """
    i: int
    j: int
    k: int
    component: str
    freqs: jnp.ndarray
    impedance: float


class SimResult(NamedTuple):
    """Compiled simulation output.

    time_series : (n_steps, n_probes) float array, or (n_steps, 0) if
        no probes were specified.
    ntff_data : NTFFData or None
        Accumulated near-to-far-field DFT data (if NTFF box was used).
    dft_planes : tuple[DFTPlaneProbe, ...] or None
        Final accumulated DFT plane probes.
    waveguide_ports : tuple[WaveguidePortConfig, ...] or None
        Final accumulated waveguide-port configs.
    wire_port_sparams : tuple | None
        Final V/I/V_inc DFT accumulators for wire port S-params.
    lumped_port_sparams : tuple | None
        Final V/I DFT accumulators for lumped port S-params (issue #72).
        Each entry is ``(LumpedPortSParamSpec, (v_dft, i_dft))``.
    snapshots : dict[str, ndarray] or None
        Field snapshots keyed by component name.
    ntff_box : NTFFBox or None
        NTFF box specification used for accumulation.
    grid : Grid or None
        Grid metadata needed by post-processing / objective helpers.
    """
    state: FDTDState | None
    time_series: jnp.ndarray
    ntff_data: object = None
    dft_planes: tuple | None = None
    flux_monitors: tuple | None = None
    waveguide_ports: tuple | None = None
    wire_port_sparams: tuple | None = None
    lumped_port_sparams: tuple | None = None
    snapshots: dict | None = None
    ntff_box: object = None
    grid: Grid | None = None


# ---------------------------------------------------------------------------
# Helpers to build source / probe specs
# ---------------------------------------------------------------------------

def make_source(grid: Grid, position, component, waveform_fn, n_steps):
    """Create a SourceSpec by precomputing a waveform function (raw E-field add).

    WARNING: raw field source causes DC accumulation on PEC surfaces.
    Prefer ``make_j_source`` for resonance detection.
    """
    idx = grid.position_to_index(position)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = jax.vmap(waveform_fn)(times)
    return SourceSpec(i=idx[0], j=idx[1], k=idx[2],
                      component=component, waveform=waveform)


def make_j_source(grid: Grid, position, component, waveform_fn, n_steps, materials):
    """Create a SourceSpec using current density injection (J source).

    Unlike ``make_source``, the waveform is Cb-normalized so that
    the source enters through the Ampere update equation:

    E += Cb * waveform(t)

    where Cb = dt / (eps * (1 + sigma*dt/2eps)).

    Note: this is a soft *field* source, not a current-density
    source.  Injected amplitude depends on local material (eps, sigma)
    but NOT on cell size.  Power coupling varies with resolution;
    use Harminv/FFT for mode extraction rather than absolute amplitude.

    Parameters
    ----------
    grid : Grid
    position : (x, y, z) in metres
    component : "ex", "ey", or "ez"
    waveform_fn : callable(t) -> value
    n_steps : int
    materials : MaterialArrays (for Cb computation at source cell)
    """
    idx = grid.position_to_index(position)
    i, j, k = idx
    eps = materials.eps_r[i, j, k] * EPS_0
    sigma = materials.sigma[i, j, k]
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)

    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    # Cb normalization: the source enters the update equation through
    # the Cb coefficient (dt/eps/(1+loss)). This ensures:
    # 1. No DC accumulation on PEC (Cb scales with dt)
    # 2. Proper coupling to cavity modes
    # Power scales as Cb²·dx³ — weaker at fine grids, but Harminv
    # with bandpass filtering reliably extracts modes at all resolutions.
    waveform = cb * jax.vmap(waveform_fn)(times)
    return SourceSpec(i=i, j=j, k=k,
                      component=component, waveform=waveform)


def make_port_source(grid: Grid, port, materials: MaterialArrays, n_steps):
    """Create a SourceSpec for a lumped port (Cb-corrected waveform).

    The port impedance must already be folded into *materials* via
    ``setup_lumped_port()``.
    """
    from rfx.sources.sources import port_d_parallel
    idx = grid.position_to_index(port.position)
    i, j, k = idx

    eps = materials.eps_r[i, j, k] * EPS_0
    sigma = materials.sigma[i, j, k]
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)

    d_par = port_d_parallel(grid, idx, port.component)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / d_par) * jax.vmap(port.excitation)(times)
    return SourceSpec(i=i, j=j, k=k,
                      component=port.component, waveform=waveform)


def make_wire_port_sources(grid, port, materials, n_steps):
    """Create a list of SourceSpec for a multi-cell WirePort.

    Each cell in the wire gets its own SourceSpec with the Cb-corrected
    waveform scaled by 1/N_cells.  The port impedance must already be
    folded into *materials* via ``setup_wire_port()``.

    Returns
    -------
    list[SourceSpec]
    """
    from rfx.sources.sources import _wire_port_cells

    cells = _wire_port_cells(grid, port)
    n_cells = max(len(cells), 1)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt

    from rfx.sources.sources import port_d_parallel

    specs = []
    for cell in cells:
        i, j, k = cell
        d_par = port_d_parallel(grid, (i, j, k), port.component)
        eps = materials.eps_r[i, j, k] * EPS_0
        sigma = materials.sigma[i, j, k]
        loss = sigma * grid.dt / (2.0 * eps)
        cb = (grid.dt / eps) / (1.0 + loss)
        waveform = (cb / d_par) * jax.vmap(port.excitation)(times) / n_cells
        specs.append(SourceSpec(i=i, j=j, k=k,
                                component=port.component, waveform=waveform))
    return specs


def make_probe(grid: Grid, position, component):
    """Create a ProbeSpec from a physical position."""
    idx = grid.position_to_index(position)
    return ProbeSpec(i=idx[0], j=idx[1], k=idx[2], component=component)


def _update_e_with_optional_dispersion(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    dx: float,
    *,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    periodic: tuple = (False, False, False),
    aniso_eps: tuple | None = None,
) -> tuple[FDTDState, object | None, object | None]:
    """Update E with standard, Debye, Lorentz, or mixed dispersion.

    Parameters
    ----------
    aniso_eps : (eps_ex, eps_ey, eps_ez) or None
        Per-component relative permittivity arrays for subpixel smoothing.
        Only used when no dispersion model is active.
    """
    if debye is None and lorentz is None:
        if aniso_eps is not None:
            eps_ex, eps_ey, eps_ez = aniso_eps
            return update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                                  dt, dx, periodic=periodic), None, None
        return update_e(state, materials, dt, dx, periodic=periodic), None, None

    if debye is not None and lorentz is None:
        from rfx.materials.debye import update_e_debye

        debye_coeffs, debye_state = debye
        new_state, new_debye = update_e_debye(
            state, debye_coeffs, debye_state, dt, dx, periodic=periodic)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        from rfx.materials.lorentz import update_e_lorentz

        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = update_e_lorentz(
            state, lorentz_coeffs, lorentz_state, dt, dx, periodic=periodic)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz update.
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz

    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # Explicit Lorentz polarization update first.
    px_l_new = (
        lorentz_coeffs.a * lorentz_state.px
        + lorentz_coeffs.b * lorentz_state.px_prev
        + lorentz_coeffs.c * ex_old[None]
    )
    py_l_new = (
        lorentz_coeffs.a * lorentz_state.py
        + lorentz_coeffs.b * lorentz_state.py_prev
        + lorentz_coeffs.c * ey_old[None]
    )
    pz_l_new = (
        lorentz_coeffs.a * lorentz_state.pz
        + lorentz_coeffs.b * lorentz_state.pz_prev
        + lorentz_coeffs.c * ez_old[None]
    )

    dpx_l = jnp.sum(px_l_new - lorentz_state.px, axis=0)
    dpy_l = jnp.sum(py_l_new - lorentz_state.py, axis=0)
    dpz_l = jnp.sum(pz_l_new - lorentz_state.pz, axis=0)

    beta_sum = jnp.sum(debye_coeffs.beta, axis=0)
    gamma_base = 1.0 / lorentz_coeffs.cc
    gamma_total = jnp.maximum(gamma_base + beta_sum, EPS_0 * 1e-10)
    numer_base = lorentz_coeffs.ca * gamma_base

    ca = (numer_base - beta_sum) / gamma_total
    cb = dt / gamma_total
    cc_debye = (1.0 - debye_coeffs.alpha) / gamma_total
    cc_lorentz = 1.0 / gamma_total

    ex_new = (
        ca * ex_old
        + cb * curl_x
        + jnp.sum(cc_debye * debye_state.px, axis=0)
        - cc_lorentz * dpx_l
    )
    ey_new = (
        ca * ey_old
        + cb * curl_y
        + jnp.sum(cc_debye * debye_state.py, axis=0)
        - cc_lorentz * dpy_l
    )
    ez_new = (
        ca * ez_old
        + cb * curl_z
        + jnp.sum(cc_debye * debye_state.pz, axis=0)
        - cc_lorentz * dpz_l
    )

    new_fdtd = state._replace(
        ex=ex_new,
        ey=ey_new,
        ez=ez_new,
        step=state.step + 1,
    )
    new_debye = DebyeState(
        px=debye_coeffs.alpha * debye_state.px + debye_coeffs.beta * (ex_new[None] + ex_old[None]),
        py=debye_coeffs.alpha * debye_state.py + debye_coeffs.beta * (ey_new[None] + ey_old[None]),
        pz=debye_coeffs.alpha * debye_state.pz + debye_coeffs.beta * (ez_new[None] + ez_old[None]),
    )
    new_lorentz = LorentzState(
        px=px_l_new,
        py=py_l_new,
        pz=pz_l_new,
        px_prev=lorentz_state.px,
        py_prev=lorentz_state.py,
        pz_prev=lorentz_state.pz,
    )

    return new_fdtd, new_debye, new_lorentz


# ---------------------------------------------------------------------------
# Compiled runner
# ---------------------------------------------------------------------------

class SnapshotSpec(NamedTuple):
    """Mid-simulation field snapshot specification.

    interval : int
        Record a snapshot every *interval* timesteps.
    components : tuple of str
        Field components to capture, e.g. ("ez",) or ("ex", "hy").
    slice_axis : int or None
        If not None, capture a 2-D slice at *slice_index* along this axis
        instead of the full 3-D field (saves memory).
    slice_index : int or None
        Index along *slice_axis*.
    """
    interval: int = 10
    components: tuple = ("ez",)
    slice_axis: int | None = None
    slice_index: int | None = None


def _nearest_divisor(n: int, target: int) -> int:
    """Return the divisor of `n` closest to `target` (≥1).

    Used by the segmented-checkpointing path (issue #73) to suggest a
    valid `checkpoint_segments` value when the user passes one that does
    not divide `n_steps` evenly.
    """
    target = max(1, int(target))
    best = 1
    best_diff = abs(target - 1)
    # Search divisors in [1, sqrt(n)] and their cofactors.
    k = 1
    while k * k <= n:
        if n % k == 0:
            for d in (k, n // k):
                diff = abs(d - target)
                if diff < best_diff:
                    best, best_diff = d, diff
        k += 1
    return best


def _suggest_checkpoint_segments(n_steps: int) -> int:
    """Auto pick K ≈ √n_steps that divides n_steps."""
    import math
    return _nearest_divisor(n_steps, max(1, int(math.isqrt(n_steps))))


def run(
    grid: Grid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    boundary: str = "pec",
    cpml_axes: str = "xyz",
    pec_axes: str | None = None,
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    tfsf: tuple | None = None,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    dft_planes: list | None = None,
    flux_monitors: list | None = None,
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    checkpoint_segments: int | None = None,
    aniso_eps: tuple | None = None,
    pec_mask: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
) -> SimResult:
    """Run a compiled FDTD simulation via ``jax.lax.scan``.

    Parameters
    ----------
    grid : Grid
    materials : MaterialArrays
    n_steps : int
    boundary : "pec", "cpml", or "upml"
    cpml_axes : axes string for CPML (default "xyz")
    pec_axes : axes string or None
        Axes on which to enforce PEC after each update. If None, uses
        all non-periodic axes.
    periodic : (bool, bool, bool) or None
        Per-axis periodic boundary flags. If None, defaults to
        ``(False, False, False)`` in 3D and ``(False, False, True)``
        in 2D modes.
    debye : (DebyeCoeffs, DebyeState) tuple, or None
    lorentz : (LorentzCoeffs, LorentzState) tuple, or None
    tfsf : (TFSFConfig, TFSFState) tuple, or None
        Optional total-field/scattered-field plane-wave source. When
        provided, the 1D auxiliary FDTD is interleaved with the 3D Yee
        updates using the Taflove leapfrog ordering.
    sources : list of SourceSpec (precomputed waveforms)
    probes : list of ProbeSpec (point time-series recorders)
    dft_planes : list of DFTPlaneProbe or None
        Optional running frequency-domain plane probes.
    waveguide_ports : list of WaveguidePortConfig or None
        Optional waveguide-port source/probe configs.
    ntff : NTFFBox or None
        If provided, accumulate near-to-far-field DFT on a Huygens box.
    snapshot : SnapshotSpec or None
        If provided, record field snapshots at regular intervals.
    checkpoint : bool
        If True, wrap the scan body with ``jax.checkpoint`` to
        trade compute for memory during reverse-mode AD.  Reduces
        backward-pass memory from O(n_steps) to O(1) per step.
    return_state : bool
        If False, do not expose the final FDTD state in the result.
        This shrinks the differentiable output surface for optimisation.

    Returns
    -------
    SimResult with final state, time series, and optional NTFF data.
    """
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []

    dt = grid.dt
    dx = grid.dx

    # ---- boundary configuration ----
    if periodic is None:
        periodic = (False, False, False)
    else:
        if len(periodic) != 3:
            raise ValueError(f"periodic must have length 3, got {periodic!r}")
        periodic = tuple(bool(v) for v in periodic)

    if grid.is_2d:
        periodic = (periodic[0], periodic[1], True)

    # Skip CPML / PEC on periodic axes.
    axis_names = ("x", "y", "z")
    for axis_name, is_periodic in zip(axis_names, periodic):
        if is_periodic:
            cpml_axes = cpml_axes.replace(axis_name, "")
    default_pec_axes = "".join(
        axis_name for axis_name, is_periodic in zip(axis_names, periodic)
        if not is_periodic
    )
    if pec_axes is None:
        pec_axes = default_pec_axes
    else:
        pec_axes = "".join(axis for axis in pec_axes if axis in default_pec_axes)

    # ---- per-face PEC from grid.pec_faces ----
    _pec_faces = getattr(grid, "pec_faces", None) or set()
    use_pec_faces = bool(_pec_faces)
    # Freeze the set into a tuple for use inside the JIT-traced body
    _pec_faces_frozen = frozenset(_pec_faces) if use_pec_faces else frozenset()

    # ---- per-face PMC from grid.pmc_faces (T7 Phase 2 PR3) ----
    _pmc_faces = getattr(grid, "pmc_faces", None) or set()
    use_pmc_faces = bool(_pmc_faces)
    _pmc_faces_frozen = frozenset(_pmc_faces) if use_pmc_faces else frozenset()

    # ---- subsystem flags (resolved at trace time) ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_upml = boundary == "upml" and grid.cpml_layers > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_tfsf = tfsf is not None
    use_ntff = ntff is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    use_snapshot = snapshot is not None
    use_pec_mask = pec_mask is not None
    use_pec_occupancy = pec_occupancy is not None
    use_conformal = conformal_weights is not None
    wire_port_sparams = wire_port_sparams or []
    use_wire_sparams = len(wire_port_sparams) > 0
    lumped_port_sparams = lumped_port_sparams or []
    use_lumped_sparams = len(lumped_port_sparams) > 0
    lumped_rlc = lumped_rlc or []
    use_lumped_rlc = len(lumped_rlc) > 0
    use_kerr = kerr_chi3 is not None

    if use_kerr:
        from rfx.materials.nonlinear import apply_kerr_ade

    # ---- fast-path: pre-baked coefficients with PEC folded in ----
    # Eligible when the scan body only needs H update + E update + PEC —
    # no CPML, no TFSF, no dispersion, no anisotropic eps, no PEC mask,
    # no conformal, no lumped RLC, no Kerr, and no periodic axes.
    #
    # The fast path bakes PEC boundary enforcement into the update
    # coefficients (zero ca/cb at boundary cells), eliminating 12
    # per-step scatter operations.  This is always beneficial on GPU
    # (scatter ops launch separate kernels) and beneficial on CPU for
    # grids above ~500K cells where the scatter ops cause cache eviction.
    _on_gpu = jax.default_backend() != "cpu"
    _fast_eligible = (
        not use_cpml
        and not use_upml
        and not use_tfsf
        and not use_debye
        and not use_lorentz
        and not use_pec_mask
        and not use_pec_occupancy
        and not use_conformal
        and not use_lumped_rlc
        and not use_kerr
        and aniso_eps is None
        and periodic == (False, False, False)
    )
    # On GPU the baked-PEC path eliminates expensive scatter-update
    # kernel launches — always beneficial.  On CPU, XLA fuses scatter
    # ops efficiently so the extra coefficient arrays hurt at small-to-
    # medium grids; enable only when explicitly requested via GPU backend.
    use_fast_he = _fast_eligible and _on_gpu
    if use_fast_he:
        _fast_coeffs = precompute_coeffs(materials, dt, dx, pec_axes=pec_axes)

    # ---- initialise states ----
    _field_dtype = field_dtype if field_dtype is not None else jnp.float32
    fdtd = init_state(grid.shape, field_dtype=_field_dtype)

    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
        carry_init["cpml"] = cpml_state
    elif use_upml:
        from rfx.boundaries.upml import init_upml, apply_upml_e, apply_upml_h
        upml_coeffs = init_upml(grid, materials, axes=cpml_axes,
                                aniso_eps=aniso_eps)

    if use_debye:
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry_init["lorentz"] = lorentz_state

    if use_tfsf:
        from rfx.sources.tfsf import (
            update_tfsf_1d_e,
            update_tfsf_1d_h,
            apply_tfsf_e,
            apply_tfsf_h,
            is_tfsf_2d,
        )

        tfsf_cfg, tfsf_state = tfsf
        carry_init["tfsf"] = tfsf_state

        # Detect 2D auxiliary grid (oblique incidence)
        _tfsf_is_2d = is_tfsf_2d(tfsf_cfg)
        if _tfsf_is_2d:
            from rfx.sources.tfsf_2d import update_tfsf_2d_h, update_tfsf_2d_e

    if use_ntff:
        from rfx.farfield import init_ntff_data, accumulate_ntff
        carry_init["ntff"] = init_ntff_data(ntff)

    if use_dft_planes:
        carry_init["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)
    if use_flux_monitors:
        from rfx.probes.probes import _FLUX_COMPONENTS as _FC
        # Pre-extract metadata for the scan body (avoids closure issues)
        # Tuple: (axis, index, freqs, comp_names, lo1, hi1, lo2, hi2,
        #         total_steps, window, window_alpha)
        flux_meta = tuple(
            (fm.axis, fm.index, fm.freqs, _FC[fm.axis],
             fm.lo1, fm.hi1, fm.lo2, fm.hi2,
             fm.total_steps, fm.window, fm.window_alpha) for fm in flux_monitors
        )
        carry_init["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft) for fm in flux_monitors
        )
    if use_waveguide_ports:
        carry_init["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_t,
                cfg.v_ref_t,
                cfg.i_probe_t,
                cfg.i_ref_t,
                cfg.v_inc_t,
                cfg.n_steps_recorded,
            )
            for cfg in waveguide_ports
        )
    if use_wire_sparams:
        # Initialize V, I, V_inc DFT accumulators per wire port
        carry_init["wire_sparam_accs"] = tuple(
            (
                jnp.zeros(len(wp.freqs), dtype=jnp.complex64),  # v_dft
                jnp.zeros(len(wp.freqs), dtype=jnp.complex64),  # i_dft
                jnp.zeros(len(wp.freqs), dtype=jnp.complex64),  # v_inc_dft
            )
            for wp in wire_port_sparams
        )
        wire_sparam_meta = tuple(wire_port_sparams)

    if use_lumped_sparams:
        # Initialize V, I DFT accumulators per lumped port (issue #72).
        # No V_inc accumulator is needed: the wave decomposition
        # ``a = (-V + Z0·I)/(2√Z0)`` is exact regardless of source pulse
        # shape (this is the whole point — no time-gating heuristic).
        carry_init["lumped_sparam_accs"] = tuple(
            (
                jnp.zeros(len(lp.freqs), dtype=jnp.complex64),  # v_dft
                jnp.zeros(len(lp.freqs), dtype=jnp.complex64),  # i_dft
            )
            for lp in lumped_port_sparams
        )
        lumped_sparam_meta = tuple(lumped_port_sparams)

    if use_lumped_rlc:
        from rfx.lumped import init_rlc_state, update_rlc_element
        carry_init["rlc_states"] = tuple(init_rlc_state() for _ in lumped_rlc)
        rlc_meta = tuple(lumped_rlc)  # list of RLCCellMeta

    # ---- precompute source waveform matrix (n_steps, n_sources) ----
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # Static source/probe metadata (captured by closure)
    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]
    dft_meta = tuple(
        (probe.component, probe.axis, probe.index, probe.freqs)
        for probe in dft_planes
    )
    waveguide_meta = tuple(waveguide_ports)

    # ---- snapshot setup ----
    if use_snapshot:
        snap_components = snapshot.components
        snap_slice_axis = snapshot.slice_axis
        snap_slice_index = snapshot.slice_index
    else:
        snap_components = ()
        snap_slice_axis = None
        snap_slice_index = None

    def _take_snapshot(st):
        """Extract snapshot fields from state."""
        snaps = []
        for comp in snap_components:
            field = getattr(st, comp)
            if snap_slice_axis is not None and snap_slice_index is not None:
                sl = [slice(None)] * 3
                sl[snap_slice_axis] = snap_slice_index
                field = field[tuple(sl)]
            snaps.append(field)
        return snaps

    # ---- scan body ----
    def step_fn(carry, xs):
        _step_idx, src_vals = xs
        st = carry["fdtd"]
        tfsf_h_state = None

        if use_fast_he:
            # Fast path: combined H+E update with PEC baked into
            # pre-computed coefficients — eliminates separate apply_pec(),
            # coefficient recomputation, and reduces XLA scatter ops.
            st = update_he_fast(st, _fast_coeffs)
        else:
            # H update
            if use_upml:
                st = apply_upml_h(st, upml_coeffs, periodic=periodic)
            else:
                st = update_h(st, materials, dt, dx, periodic=periodic)
            if use_tfsf:
                st = apply_tfsf_h(st, tfsf_cfg, carry["tfsf"], dx, dt)
            if use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_h as _apply_wg_h_early
                for cfg_meta in waveguide_meta:
                    st = _apply_wg_h_early(st, cfg_meta, _step_idx, dt, dx)
            if use_cpml:
                st, cpml_new = apply_cpml_h(
                    st, cpml_params, carry["cpml"], grid, cpml_axes,
                    materials=materials)
            if use_pmc_faces:
                from rfx.boundaries.pmc import apply_pmc_faces
                st = apply_pmc_faces(st, _pmc_faces_frozen)
            if use_tfsf:
                if _tfsf_is_2d:
                    tfsf_h_state = update_tfsf_2d_h(tfsf_cfg, carry["tfsf"], dx, dt)
                else:
                    tfsf_h_state = update_tfsf_1d_h(tfsf_cfg, carry["tfsf"], dx, dt)

            if use_upml:
                if use_debye or use_lorentz:
                    raise ValueError("boundary='upml' does not yet support dispersion")
                st = apply_upml_e(st, upml_coeffs, periodic=periodic)
                debye_new = None
                lorentz_new = None
            else:
                st, debye_new, lorentz_new = _update_e_with_optional_dispersion(
                    st,
                    materials,
                    dt,
                    dx,
                    debye=(debye_coeffs, carry["debye"]) if use_debye else None,
                    lorentz=(lorentz_coeffs, carry["lorentz"]) if use_lorentz else None,
                    periodic=periodic,
                    aniso_eps=aniso_eps,
                )

            # Kerr nonlinear ADE correction (after linear E-update)
            if use_kerr:
                st = apply_kerr_ade(st, kerr_chi3, dt)

            if use_tfsf:
                st = apply_tfsf_e(st, tfsf_cfg, tfsf_h_state, dx, dt)
            if use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e_early
                for cfg_meta in waveguide_meta:
                    st = _apply_wg_e_early(st, cfg_meta, _step_idx, dt, dx)
            if use_cpml:
                st, cpml_new = apply_cpml_e(
                    st, cpml_params, cpml_new, grid, cpml_axes,
                    materials=materials)

            if pec_axes:
                st = apply_pec(st, axes=pec_axes)
            if use_pec_faces:
                st = apply_pec_faces(st, _pec_faces_frozen)

            if use_conformal:
                from rfx.geometry.conformal import apply_conformal_pec
                st = apply_conformal_pec(st, conformal_weights[0], conformal_weights[1], conformal_weights[2])
            elif use_pec_mask:
                from rfx.boundaries.pec import apply_pec_mask
                st = apply_pec_mask(st, pec_mask)

            if use_pec_occupancy:
                st = apply_pec_occupancy(st, pec_occupancy)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        if use_lumped_rlc:
            new_rlc_states = []
            for rlc_st, meta in zip(carry["rlc_states"], rlc_meta):
                st, rlc_st_new = update_rlc_element(st, rlc_st, meta)
                new_rlc_states.append(rlc_st_new)

        t = _step_idx.astype(jnp.float32) * dt

        # Wire port S-param DFT accumulation BEFORE source injection so
        # that sampled V/I reflects only the load/cavity response.
        if use_wire_sparams:
            new_wire_accs = []
            for accs, wp_meta in zip(carry["wire_sparam_accs"], wire_sparam_meta):
                v_dft, i_dft, vinc_dft = accs
                mi, mj, mk = wp_meta.mid_i, wp_meta.mid_j, wp_meta.mid_k
                v = -getattr(st, wp_meta.component)[mi, mj, mk] * dx
                if wp_meta.component == "ez":
                    i_val = (st.hy[mi,mj,mk] - st.hy[mi-1,mj,mk]
                             - st.hx[mi,mj,mk] + st.hx[mi,mj-1,mk]) * dx
                elif wp_meta.component == "ex":
                    i_val = (st.hz[mi,mj,mk] - st.hz[mi,mj-1,mk]
                             - st.hy[mi,mj,mk] + st.hy[mi,mj,mk-1]) * dx
                else:
                    i_val = (st.hx[mi,mj,mk] - st.hx[mi,mj,mk-1]
                             - st.hz[mi,mj,mk] + st.hz[mi-1,mj,mk]) * dx
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase = jnp.exp(-1j * 2.0 * jnp.pi * wp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_wire_accs.append((
                    v_dft + v * phase,
                    i_dft + i_val * phase,
                    vinc_dft,
                ))

        # Lumped port S-param DFT accumulation BEFORE source injection
        # (issue #72).  Same wave-decomposition pattern as the wire-port
        # path but for single-cell lumped ports.
        if use_lumped_sparams:
            new_lumped_accs = []
            for accs, lp_meta in zip(carry["lumped_sparam_accs"], lumped_sparam_meta):
                v_dft_l, i_dft_l = accs
                li, lj, lk = lp_meta.i, lp_meta.j, lp_meta.k
                v_l = -getattr(st, lp_meta.component)[li, lj, lk] * dx
                if lp_meta.component == "ez":
                    i_val_l = (st.hy[li,lj,lk] - st.hy[li-1,lj,lk]
                               - st.hx[li,lj,lk] + st.hx[li,lj-1,lk]) * dx
                elif lp_meta.component == "ex":
                    i_val_l = (st.hz[li,lj,lk] - st.hz[li,lj-1,lk]
                               - st.hy[li,lj,lk] + st.hy[li,lj,lk-1]) * dx
                else:
                    i_val_l = (st.hx[li,lj,lk] - st.hx[li,lj,lk-1]
                               - st.hz[li,lj,lk] + st.hz[li-1,lj,lk]) * dx
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase_l = jnp.exp(-1j * 2.0 * jnp.pi * lp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_lumped_accs.append((
                    v_dft_l + v_l * phase_l,
                    i_dft_l + i_val_l * phase_l,
                ))

        # Soft sources — cast source value to field dtype to avoid
        # mixed-precision scatter warnings (float32 -> float16).
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s].astype(field.dtype))
            st = st._replace(**{sc: field})

        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_new = update_tfsf_2d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)
            else:
                tfsf_new = update_tfsf_1d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)

        if use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                update_waveguide_port_probe,
            )

            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(carry["waveguide_port_accs"], waveguide_meta):
                cfg = cfg_meta._replace(
                    v_probe_t=accs[0],
                    v_ref_t=accs[1],
                    i_probe_t=accs[2],
                    i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                # TFSF-style H and E corrections are applied earlier in
                # their respective Yee sub-steps (canonical TFSF slots).
                cfg_updated = update_waveguide_port_probe(cfg, st, dt, dx)
                new_waveguide_port_accs.append(
                    (
                        cfg_updated.v_probe_t,
                        cfg_updated.v_ref_t,
                        cfg_updated.i_probe_t,
                        cfg_updated.i_ref_t,
                        cfg_updated.v_inc_t,
                        cfg_updated.n_steps_recorded,
                    )
                )

        # Probe samples
        samples = [getattr(st, pc)[pi, pj, pk]
                   for pi, pj, pk, pc in prb_meta]
        probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

        # NTFF accumulation
        if use_ntff:
            ntff_new = accumulate_ntff(
                carry["ntff"], st, ntff, dt, _step_idx)

        if use_dft_planes:
            t_plane = st.step * dt
            new_dft_planes = []
            for acc, (component, axis, index, freqs) in zip(carry["dft_planes"], dft_meta):
                field = getattr(st, component)
                if axis == 0:
                    plane = field[index, :, :]
                elif axis == 1:
                    plane = field[:, index, :]
                else:
                    plane = field[:, :, index]
                phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t_plane)
                new_dft_planes.append(
                    acc + plane[None, :, :] * phase[:, None, None] * dt
                )

        if use_flux_monitors:
            from rfx.core.dft_utils import dft_window_weight as _dft_w
            t_flux = st.step * dt
            new_flux_accs = []
            for (e1_acc, e2_acc, h1_acc, h2_acc), (
                ax, idx, fqs, comp_names, _lo1, _hi1, _lo2, _hi2,
                _tot_steps, _win_name, _win_alpha,
            ) in zip(carry["flux_monitors"], flux_meta):
                e1n, e2n, h1n, h2n = comp_names
                # H-fields are offset by +dx/2 along the normal axis on
                # the Yee grid.  Average H at idx-1 and idx to co-locate
                # with E at idx, giving a correct Poynting cross-product.
                # Slice to the finite-size region [lo1:hi1, lo2:hi2].
                idx_m1 = max(idx - 1, 0)
                if ax == 0:
                    e1 = getattr(st, e1n)[idx, _lo1:_hi1, _lo2:_hi2]
                    e2 = getattr(st, e2n)[idx, _lo1:_hi1, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h1n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h2n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                elif ax == 1:
                    e1 = getattr(st, e1n)[_lo1:_hi1, idx, _lo2:_hi2]
                    e2 = getattr(st, e2n)[_lo1:_hi1, idx, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h1n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h2n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                else:
                    e1 = getattr(st, e1n)[_lo1:_hi1, _lo2:_hi2, idx]
                    e2 = getattr(st, e2n)[_lo1:_hi1, _lo2:_hi2, idx]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                t_f64 = t_flux.astype(jnp.float64) if hasattr(t_flux, 'astype') else jnp.float64(t_flux)
                fqs64 = fqs.astype(jnp.float64)
                # Streaming DFT window weight (rect=1.0 default; Tukey/Hann
                # suppress late-time contributions from CPML reflections).
                _w = _dft_w(st.step, _tot_steps, _win_name, _win_alpha).astype(jnp.float64)
                # E is at time t_flux = step*dt; H is at t_flux - dt/2
                phase_e = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * t_f64)
                phase_h = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * (t_f64 - jnp.float64(dt * 0.5)))
                kernel_e = (phase_e[:, None, None] * dt * _w).astype(jnp.complex128)
                kernel_h = (phase_h[:, None, None] * dt * _w).astype(jnp.complex128)
                new_flux_accs.append((
                    e1_acc + e1.astype(jnp.float64)[None, :, :] * kernel_e,
                    e2_acc + e2.astype(jnp.float64)[None, :, :] * kernel_e,
                    h1_acc + h1.astype(jnp.float64)[None, :, :] * kernel_h,
                    h2_acc + h2.astype(jnp.float64)[None, :, :] * kernel_h,
                ))

        # Snapshot output
        if use_snapshot:
            snap_fields = _take_snapshot(st)
            output = (probe_out, snap_fields)
        else:
            output = (probe_out,)

        # Rebuild carry
        new_carry: dict = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if use_debye:
            new_carry["debye"] = debye_new
        if use_lorentz:
            new_carry["lorentz"] = lorentz_new
        if use_tfsf:
            new_carry["tfsf"] = tfsf_new
        if use_ntff:
            new_carry["ntff"] = ntff_new
        if use_dft_planes:
            new_carry["dft_planes"] = tuple(new_dft_planes)
        if use_flux_monitors:
            new_carry["flux_monitors"] = tuple(new_flux_accs)
        if use_waveguide_ports:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if use_wire_sparams:
            new_carry["wire_sparam_accs"] = tuple(new_wire_accs)
        if use_lumped_sparams:
            new_carry["lumped_sparam_accs"] = tuple(new_lumped_accs)
        if use_lumped_rlc:
            new_carry["rlc_states"] = tuple(new_rlc_states)

        return new_carry, output

    # ---- run ----
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)

    if checkpoint_segments is None:
        # Legacy path: optional per-step rematerialisation only. The scan
        # itself still keeps every step's carry, so peak memory grows
        # linearly with n_steps.
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        final_carry, outputs = jax.lax.scan(body, carry_init, xs)
    else:
        # Segmented checkpointing (issue #73): split the n_steps scan into
        # K segments of size s and rematerialise each segment as a unit.
        # Forward keeps only K segment-boundary carries (instead of all
        # n_steps), so peak memory drops from O(n_steps · |carry|) to
        # O((K + s) · |carry|). With s ≈ √n_steps this is the standard
        # √n_steps-memory trade-off (≈2× compute for backward).
        #
        # Implementation notes:
        #   * `prevent_cse=False` on the segment-level checkpoint is
        #     required so the inner scan is not CSE-deduplicated across
        #     the outer scan (which would defeat the memory saving).
        #   * `n_steps` must be divisible by K. We require this rather
        #     than padding because the carry holds DFT accumulators
        #     (lumped/wire/waveguide port S-params) that would integrate
        #     over the padded zero-source steps and produce numerically
        #     different results from an unsegmented run with the same
        #     n_steps. Picking K as a divisor of n_steps preserves
        #     bit-exact equivalence for forward and gradient.
        K = int(checkpoint_segments)
        if K < 1:
            raise ValueError(
                f"checkpoint_segments must be ≥ 1, got {checkpoint_segments}")
        if n_steps % K != 0:
            raise ValueError(
                f"checkpoint_segments={K} does not divide n_steps={n_steps}; "
                f"pick a divisor (e.g. K={_nearest_divisor(n_steps, K)} for "
                f"≈ √n_steps memory). Padding is intentionally rejected "
                f"because it would shift DFT accumulator integration windows."
            )
        s = n_steps // K
        xs_segmented = jax.tree_util.tree_map(
            lambda a: a.reshape(K, s, *a.shape[1:]), xs)

        def segment_body(carry, seg_xs):
            new_carry, seg_outputs = jax.lax.scan(step_fn, carry, seg_xs)
            return new_carry, seg_outputs

        seg_body_ckpt = jax.checkpoint(
            segment_body, prevent_cse=False) if checkpoint else segment_body
        final_carry, seg_outputs = jax.lax.scan(
            seg_body_ckpt, carry_init, xs_segmented)
        # seg_outputs leaves: (K, s, ...).  Flatten back to (n_steps, ...).
        outputs = jax.tree_util.tree_map(
            lambda a: a.reshape(n_steps, *a.shape[2:]),
            seg_outputs)

    if use_snapshot:
        time_series = outputs[0]
        # outputs[1] is a list of arrays, each (n_steps, ...)
        snapshots = {comp: outputs[1][i]
                     for i, comp in enumerate(snap_components)}
    else:
        time_series = outputs[0]
        snapshots = None

    final_dft_planes = None
    if use_dft_planes:
        final_dft_planes = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final_carry["dft_planes"])
        )

    final_flux_monitors = None
    if use_flux_monitors:
        final_flux_monitors = tuple(
            fm._replace(e1_dft=accs[0], e2_dft=accs[1], h1_dft=accs[2], h2_dft=accs[3])
            for fm, accs in zip(flux_monitors, final_carry["flux_monitors"])
        )

    final_waveguide_ports = None
    if use_waveguide_ports:
        final_waveguide_ports = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(waveguide_meta, final_carry["waveguide_port_accs"])
        )

    final_wire_sparams = None
    if use_wire_sparams:
        final_wire_sparams = tuple(
            (wp_meta, accs)
            for wp_meta, accs in zip(wire_sparam_meta, final_carry["wire_sparam_accs"])
        )

    final_lumped_sparams = None
    if use_lumped_sparams:
        final_lumped_sparams = tuple(
            (lp_meta, accs)
            for lp_meta, accs in zip(lumped_sparam_meta, final_carry["lumped_sparam_accs"])
        )

    return SimResult(
        state=final_carry["fdtd"] if return_state else None,
        time_series=time_series,
        ntff_data=final_carry.get("ntff"),
        dft_planes=final_dft_planes,
        flux_monitors=final_flux_monitors,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        lumped_port_sparams=final_lumped_sparams,
        snapshots=snapshots,
        ntff_box=ntff,
        grid=grid,
    )


# ---------------------------------------------------------------------------
# Field-decay-based stopping criterion (Python loop + JIT step)
# ---------------------------------------------------------------------------

def run_until_decay(
    grid: Grid,
    materials: MaterialArrays,
    *,
    decay_by: float = 1e-3,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50_000,
    monitor_component: str = "ez",
    monitor_position: tuple[float, float, float] | None = None,
    boundary: str = "pec",
    cpml_axes: str = "xyz",
    pec_axes: str | None = None,
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    tfsf: tuple | None = None,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    dft_planes: list | None = None,
    flux_monitors: list | None = None,
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    aniso_eps: tuple | None = None,
    pec_mask: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
) -> SimResult:
    """Run simulation until field energy decays to *decay_by* of peak.

    Uses a Python loop calling a JIT-compiled single-step function so
    that dynamic termination is possible without ``jax.lax.while_loop``.

    Parameters
    ----------
    decay_by : float
        Stop when |field|^2 < decay_by * peak|field|^2.
    check_interval : int
        Check decay every N steps.
    min_steps : int
        Always run at least this many steps.
    max_steps : int
        Hard upper limit on steps.
    monitor_component : str
        Field component to monitor ("ez", "hy", etc.).
    monitor_position : tuple or None
        Physical position (x, y, z) to monitor. If None, use center of
        the domain.

    All other parameters are identical to :func:`run`.

    Returns
    -------
    SimResult
    """
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []

    dt = grid.dt
    dx = grid.dx

    # ---- boundary configuration (same logic as run()) ----
    if periodic is None:
        periodic = (False, False, False)
    else:
        if len(periodic) != 3:
            raise ValueError(f"periodic must have length 3, got {periodic!r}")
        periodic = tuple(bool(v) for v in periodic)

    if grid.is_2d:
        periodic = (periodic[0], periodic[1], True)

    axis_names = ("x", "y", "z")
    for axis_name, is_periodic in zip(axis_names, periodic):
        if is_periodic:
            cpml_axes = cpml_axes.replace(axis_name, "")
    default_pec_axes = "".join(
        axis_name for axis_name, is_periodic in zip(axis_names, periodic)
        if not is_periodic
    )
    if pec_axes is None:
        pec_axes = default_pec_axes
    else:
        pec_axes = "".join(axis for axis in pec_axes if axis in default_pec_axes)

    # ---- subsystem flags ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_upml = boundary == "upml" and grid.cpml_layers > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_tfsf = tfsf is not None
    # ---- per-face PEC from grid.pec_faces ----
    _pec_faces_decay = getattr(grid, "pec_faces", None) or set()
    use_pec_faces = bool(_pec_faces_decay)
    _pec_faces_frozen = frozenset(_pec_faces_decay) if use_pec_faces else frozenset()

    # ---- per-face PMC from grid.pmc_faces (T7 Phase 2 PR3) ----
    _pmc_faces_decay = getattr(grid, "pmc_faces", None) or set()
    use_pmc_faces = bool(_pmc_faces_decay)
    _pmc_faces_frozen = frozenset(_pmc_faces_decay) if use_pmc_faces else frozenset()

    use_ntff = ntff is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    use_pec_mask = pec_mask is not None
    use_pec_occupancy = pec_occupancy is not None
    use_conformal = conformal_weights is not None
    wire_port_sparams = wire_port_sparams or []
    use_wire_sparams = len(wire_port_sparams) > 0
    lumped_port_sparams = lumped_port_sparams or []
    use_lumped_sparams = len(lumped_port_sparams) > 0
    lumped_rlc = lumped_rlc or []
    use_lumped_rlc = len(lumped_rlc) > 0
    use_kerr_decay = kerr_chi3 is not None

    if use_kerr_decay:
        from rfx.materials.nonlinear import apply_kerr_ade as _apply_kerr_ade_decay

    # ---- initialise states ----
    _field_dtype = field_dtype if field_dtype is not None else jnp.float32
    fdtd = init_state(grid.shape, field_dtype=_field_dtype)
    carry: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
        carry["cpml"] = cpml_state
    elif use_upml:
        from rfx.boundaries.upml import init_upml, apply_upml_e, apply_upml_h
        upml_coeffs = init_upml(grid, materials, axes=cpml_axes,
                                aniso_eps=aniso_eps)

    if use_debye:
        debye_coeffs, debye_state = debye
        carry["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry["lorentz"] = lorentz_state

    if use_tfsf:
        from rfx.sources.tfsf import (
            update_tfsf_1d_e,
            update_tfsf_1d_h,
            apply_tfsf_e,
            apply_tfsf_h,
            is_tfsf_2d,
        )
        tfsf_cfg, tfsf_state = tfsf
        carry["tfsf"] = tfsf_state

        # Detect 2D auxiliary grid (oblique incidence)
        _tfsf_is_2d = is_tfsf_2d(tfsf_cfg)
        if _tfsf_is_2d:
            from rfx.sources.tfsf_2d import update_tfsf_2d_h, update_tfsf_2d_e

    if use_ntff:
        from rfx.farfield import init_ntff_data, accumulate_ntff
        carry["ntff"] = init_ntff_data(ntff)

    if use_dft_planes:
        carry["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)
    if use_waveguide_ports:
        carry["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_t,
                cfg.v_ref_t,
                cfg.i_probe_t,
                cfg.i_ref_t,
                cfg.v_inc_t,
                cfg.n_steps_recorded,
            )
            for cfg in waveguide_ports
        )
    if use_wire_sparams:
        # Initialize V, I, V_inc DFT accumulators per wire port
        carry["wire_sparam_accs"] = tuple(
            (
                jnp.zeros(len(wp.freqs), dtype=jnp.complex64),  # v_dft
                jnp.zeros(len(wp.freqs), dtype=jnp.complex64),  # i_dft
                jnp.zeros(len(wp.freqs), dtype=jnp.complex64),  # v_inc_dft
            )
            for wp in wire_port_sparams
        )
        wire_sparam_meta = tuple(wire_port_sparams)

    if use_lumped_sparams:
        # Initialize V, I DFT accumulators per lumped port (issue #72).
        carry["lumped_sparam_accs"] = tuple(
            (
                jnp.zeros(len(lp.freqs), dtype=jnp.complex64),  # v_dft
                jnp.zeros(len(lp.freqs), dtype=jnp.complex64),  # i_dft
            )
            for lp in lumped_port_sparams
        )
        lumped_sparam_meta = tuple(lumped_port_sparams)

    if use_flux_monitors:
        from rfx.probes.probes import _FLUX_COMPONENTS as _FC_decay
        flux_meta_decay = tuple(
            (fm.axis, fm.index, fm.freqs, _FC_decay[fm.axis],
             fm.lo1, fm.hi1, fm.lo2, fm.hi2) for fm in flux_monitors
        )
        carry["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft) for fm in flux_monitors
        )

    if use_lumped_rlc:
        from rfx.lumped import init_rlc_state, update_rlc_element
        carry["rlc_states"] = tuple(init_rlc_state() for _ in lumped_rlc)
        rlc_meta = tuple(lumped_rlc)  # list of RLCCellMeta

    # Static source/probe metadata
    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]
    dft_meta = tuple(
        (probe.component, probe.axis, probe.index, probe.freqs)
        for probe in dft_planes
    )
    waveguide_meta = tuple(waveguide_ports)

    # ---- monitor position ----
    if monitor_position is None:
        # Center of the physical domain
        cx = (grid.nx - 1) * dx / 2.0
        cy = (grid.ny - 1) * dx / 2.0
        cz = 0.0 if grid.is_2d else (grid.nz - 1) * dx / 2.0
        monitor_position = (cx, cy, cz)
    mon_idx = grid.position_to_index(monitor_position)

    # ---- JIT-compiled single step ----
    @jax.jit
    def _single_step(carry_in, step_idx, src_vals):
        st = carry_in["fdtd"]
        tfsf_h_state = None

        # H update
        if use_upml:
            st = apply_upml_h(st, upml_coeffs, periodic=periodic)
        else:
            st = update_h(st, materials, dt, dx, periodic=periodic)
        if use_tfsf:
            st = apply_tfsf_h(st, tfsf_cfg, carry_in["tfsf"], dx, dt)
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import apply_waveguide_port_h as _apply_wg_h_slow
            for cfg_meta in waveguide_meta:
                st = _apply_wg_h_slow(st, cfg_meta, step_idx, dt, dx)
        if use_cpml:
            st, cpml_new = apply_cpml_h(
                st, cpml_params, carry_in["cpml"], grid, cpml_axes,
                materials=materials)
        if use_pmc_faces:
            from rfx.boundaries.pmc import apply_pmc_faces
            st = apply_pmc_faces(st, _pmc_faces_frozen)
        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_h_state = update_tfsf_2d_h(tfsf_cfg, carry_in["tfsf"], dx, dt)
            else:
                tfsf_h_state = update_tfsf_1d_h(tfsf_cfg, carry_in["tfsf"], dx, dt)

        if use_upml:
            if use_debye or use_lorentz:
                raise ValueError("boundary='upml' does not yet support dispersion")
            st = apply_upml_e(st, upml_coeffs, periodic=periodic)
            debye_new = None
            lorentz_new = None
        else:
            st, debye_new, lorentz_new = _update_e_with_optional_dispersion(
                st, materials, dt, dx,
                debye=(debye_coeffs, carry_in["debye"]) if use_debye else None,
                lorentz=(lorentz_coeffs, carry_in["lorentz"]) if use_lorentz else None,
                periodic=periodic,
                aniso_eps=aniso_eps,
            )

        # Kerr nonlinear ADE correction (after linear E-update)
        if use_kerr_decay:
            st = _apply_kerr_ade_decay(st, kerr_chi3, dt)

        if use_tfsf:
            st = apply_tfsf_e(st, tfsf_cfg, tfsf_h_state, dx, dt)
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e_slow
            for cfg_meta in waveguide_meta:
                st = _apply_wg_e_slow(st, cfg_meta, step_idx, dt, dx)
        if use_cpml:
            st, cpml_new = apply_cpml_e(
                st, cpml_params, cpml_new, grid, cpml_axes,
                materials=materials)

        if pec_axes:
            st = apply_pec(st, axes=pec_axes)
        if use_pec_faces:
            st = apply_pec_faces(st, _pec_faces_frozen)

        if use_conformal:
            from rfx.geometry.conformal import apply_conformal_pec
            st = apply_conformal_pec(st, conformal_weights[0], conformal_weights[1], conformal_weights[2])
        elif use_pec_mask:
            from rfx.boundaries.pec import apply_pec_mask
            st = apply_pec_mask(st, pec_mask)

        if use_pec_occupancy:
            st = apply_pec_occupancy(st, pec_occupancy)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        if use_lumped_rlc:
            new_rlc_states = []
            for rlc_st, meta in zip(carry_in["rlc_states"], rlc_meta):
                st, rlc_st_new = update_rlc_element(st, rlc_st, meta)
                new_rlc_states.append(rlc_st_new)

        # Compute step time first; wire/lumped S-param DFT blocks below
        # need `t` and must accumulate BEFORE source injection per the
        # rfx/probes/probes.py update_sparam_probe docstring contract
        # ("sample after E-update/apply_pec but before apply_lumped_port
        # so V reflects only the cavity/load response, not the driving
        # waveform"). The Python-loop scan path (this file, around the
        # forward() Python-loop body) already enforces this. The JIT
        # scan path violated it via PR #72 ordering, producing 5–10 dB
        # train/eval |S11| disagreement on near-matched antennas where
        # source-injection contamination is large relative to V.
        t = step_idx.astype(jnp.float32) * dt

        # Wire port S-param DFT accumulation BEFORE source injection so
        # that sampled V/I reflects only the load/cavity response.
        if use_wire_sparams:
            new_wire_accs = []
            for accs, wp_meta in zip(carry_in["wire_sparam_accs"], wire_sparam_meta):
                v_dft, i_dft, vinc_dft = accs
                mi, mj, mk = wp_meta.mid_i, wp_meta.mid_j, wp_meta.mid_k
                v = -getattr(st, wp_meta.component)[mi, mj, mk] * dx
                if wp_meta.component == "ez":
                    i_val = (st.hy[mi,mj,mk] - st.hy[mi-1,mj,mk]
                             - st.hx[mi,mj,mk] + st.hx[mi,mj-1,mk]) * dx
                elif wp_meta.component == "ex":
                    i_val = (st.hz[mi,mj,mk] - st.hz[mi,mj-1,mk]
                             - st.hy[mi,mj,mk] + st.hy[mi,mj,mk-1]) * dx
                else:
                    i_val = (st.hx[mi,mj,mk] - st.hx[mi,mj,mk-1]
                             - st.hz[mi,mj,mk] + st.hz[mi-1,mj,mk]) * dx
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase = jnp.exp(-1j * 2.0 * jnp.pi * wp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_wire_accs.append((
                    v_dft + v * phase,
                    i_dft + i_val * phase,
                    vinc_dft,
                ))

        # Lumped port S-param DFT accumulation BEFORE source injection
        # (issue #72). Same wave-decomposition pattern as the wire-port
        # path; mirrors the Python-loop scan body ordering.
        if use_lumped_sparams:
            new_lumped_accs = []
            for accs, lp_meta in zip(carry_in["lumped_sparam_accs"], lumped_sparam_meta):
                v_dft_l, i_dft_l = accs
                li, lj, lk = lp_meta.i, lp_meta.j, lp_meta.k
                v_l = -getattr(st, lp_meta.component)[li, lj, lk] * dx
                if lp_meta.component == "ez":
                    i_val_l = (st.hy[li,lj,lk] - st.hy[li-1,lj,lk]
                               - st.hx[li,lj,lk] + st.hx[li,lj-1,lk]) * dx
                elif lp_meta.component == "ex":
                    i_val_l = (st.hz[li,lj,lk] - st.hz[li,lj-1,lk]
                               - st.hy[li,lj,lk] + st.hy[li,lj,lk-1]) * dx
                else:
                    i_val_l = (st.hx[li,lj,lk] - st.hx[li,lj,lk-1]
                               - st.hz[li,lj,lk] + st.hz[li-1,lj,lk]) * dx
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase_l = jnp.exp(-1j * 2.0 * jnp.pi * lp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_lumped_accs.append((
                    v_dft_l + v_l * phase_l,
                    i_dft_l + i_val_l * phase_l,
                ))

        # Soft sources — cast source value to field dtype to avoid
        # mixed-precision scatter warnings (float32 -> float16).
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s].astype(field.dtype))
            st = st._replace(**{sc: field})

        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_new = update_tfsf_2d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)
            else:
                tfsf_new = update_tfsf_1d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)

        if use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                update_waveguide_port_probe,
            )
            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(carry_in["waveguide_port_accs"], waveguide_meta):
                cfg = cfg_meta._replace(
                    v_probe_t=accs[0], v_ref_t=accs[1],
                    i_probe_t=accs[2], i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                # TFSF-style H/E corrections applied earlier in canonical
                # Yee sub-steps (see L1247-L1288 region).
                # NOTE: this samples `st` AFTER source injection above.
                # The same docstring-contract concern as wire/lumped
                # applies here, but waveguide-port is out of scope for
                # this fix (issue #29 OPEN tracks waveguide-port issues).
                cfg_updated = update_waveguide_port_probe(cfg, st, dt, dx)
                new_waveguide_port_accs.append((
                    cfg_updated.v_probe_t, cfg_updated.v_ref_t,
                    cfg_updated.i_probe_t, cfg_updated.i_ref_t,
                    cfg_updated.v_inc_t,
                    cfg_updated.n_steps_recorded,
                ))

        # Probe samples
        samples = [getattr(st, pc)[pi, pj, pk]
                   for pi, pj, pk, pc in prb_meta]
        probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

        # NTFF accumulation
        if use_ntff:
            ntff_new = accumulate_ntff(
                carry_in["ntff"], st, ntff, dt, step_idx)

        if use_dft_planes:
            t_plane = st.step * dt
            new_dft_planes = []
            for acc, (component, axis, index, freqs) in zip(carry_in["dft_planes"], dft_meta):
                field = getattr(st, component)
                if axis == 0:
                    plane = field[index, :, :]
                elif axis == 1:
                    plane = field[:, index, :]
                else:
                    plane = field[:, :, index]
                phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t_plane)
                new_dft_planes.append(
                    acc + plane[None, :, :] * phase[:, None, None] * dt
                )

        # Flux monitor DFT accumulation (co-located E/H, finite-size region)
        if use_flux_monitors:
            t_flux = st.step * dt
            new_flux_accs = []
            for (e1_acc, e2_acc, h1_acc, h2_acc), (ax, idx, fqs, comp_names, _lo1, _hi1, _lo2, _hi2) in zip(
                carry_in["flux_monitors"], flux_meta_decay
            ):
                e1n, e2n, h1n, h2n = comp_names
                idx_m1 = max(idx - 1, 0)
                if ax == 0:
                    e1 = getattr(st, e1n)[idx, _lo1:_hi1, _lo2:_hi2]
                    e2 = getattr(st, e2n)[idx, _lo1:_hi1, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h1n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h2n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                elif ax == 1:
                    e1 = getattr(st, e1n)[_lo1:_hi1, idx, _lo2:_hi2]
                    e2 = getattr(st, e2n)[_lo1:_hi1, idx, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h1n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h2n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                else:
                    e1 = getattr(st, e1n)[_lo1:_hi1, _lo2:_hi2, idx]
                    e2 = getattr(st, e2n)[_lo1:_hi1, _lo2:_hi2, idx]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                t_f64 = t_flux.astype(jnp.float64) if hasattr(t_flux, 'astype') else jnp.float64(t_flux)
                fqs64 = fqs.astype(jnp.float64)
                phase_e = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * t_f64)
                phase_h = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * (t_f64 - jnp.float64(dt * 0.5)))
                kernel_e = (phase_e[:, None, None] * dt).astype(jnp.complex128)
                kernel_h = (phase_h[:, None, None] * dt).astype(jnp.complex128)
                new_flux_accs.append((
                    e1_acc + e1.astype(jnp.float64)[None, :, :] * kernel_e,
                    e2_acc + e2.astype(jnp.float64)[None, :, :] * kernel_e,
                    h1_acc + h1.astype(jnp.float64)[None, :, :] * kernel_h,
                    h2_acc + h2.astype(jnp.float64)[None, :, :] * kernel_h,
                ))

        # Monitor field value
        monitor_val = getattr(st, monitor_component)[mon_idx[0], mon_idx[1], mon_idx[2]]

        # Rebuild carry
        new_carry: dict = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if use_debye:
            new_carry["debye"] = debye_new
        if use_lorentz:
            new_carry["lorentz"] = lorentz_new
        if use_tfsf:
            new_carry["tfsf"] = tfsf_new
        if use_ntff:
            new_carry["ntff"] = ntff_new
        if use_dft_planes:
            new_carry["dft_planes"] = tuple(new_dft_planes)
        if use_flux_monitors:
            new_carry["flux_monitors"] = tuple(new_flux_accs)
        if use_waveguide_ports:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if use_wire_sparams:
            new_carry["wire_sparam_accs"] = tuple(new_wire_accs)
        if use_lumped_sparams:
            new_carry["lumped_sparam_accs"] = tuple(new_lumped_accs)
        if use_lumped_rlc:
            new_carry["rlc_states"] = tuple(new_rlc_states)

        return new_carry, probe_out, monitor_val

    # ---- precompute source waveforms up to max_steps ----
    if sources:
        src_waveforms = jnp.stack([s.waveform[:max_steps] if s.waveform.shape[0] >= max_steps
                                   else jnp.pad(s.waveform, (0, max_steps - s.waveform.shape[0]))
                                   for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((max_steps, 0), dtype=jnp.float32)

    # ---- Python loop with decay check ----
    peak_sq = 0.0
    all_probes = []
    actual_steps = 0

    for step in range(max_steps):
        step_idx = jnp.array(step, dtype=jnp.int32)
        src_vals = src_waveforms[step]
        carry, probe_out, monitor_val = _single_step(carry, step_idx, src_vals)

        all_probes.append(probe_out)
        actual_steps = step + 1

        # Decay check
        val_sq = float(monitor_val) ** 2
        if val_sq > peak_sq:
            peak_sq = val_sq

        if actual_steps >= min_steps and step % check_interval == 0 and peak_sq > 0.0:
            if val_sq < decay_by * peak_sq:
                break

    # ---- assemble result ----
    time_series = jnp.stack(all_probes, axis=0)

    final_dft_planes = None
    if use_dft_planes:
        final_dft_planes = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, carry["dft_planes"])
        )

    final_waveguide_ports = None
    if use_waveguide_ports:
        final_waveguide_ports = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0], v_ref_t=accs[1],
                i_probe_t=accs[2], i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(waveguide_meta, carry["waveguide_port_accs"])
        )

    final_wire_sparams = None
    if use_wire_sparams:
        final_wire_sparams = tuple(
            (wp_meta, accs)
            for wp_meta, accs in zip(wire_sparam_meta, carry["wire_sparam_accs"])
        )

    final_lumped_sparams = None
    if use_lumped_sparams:
        final_lumped_sparams = tuple(
            (lp_meta, accs)
            for lp_meta, accs in zip(lumped_sparam_meta, carry["lumped_sparam_accs"])
        )

    final_flux_monitors = None
    if use_flux_monitors:
        final_flux_monitors = tuple(
            fm._replace(e1_dft=accs[0], e2_dft=accs[1], h1_dft=accs[2], h2_dft=accs[3])
            for fm, accs in zip(flux_monitors, carry["flux_monitors"])
        )

    return SimResult(
        state=carry["fdtd"] if return_state else None,
        time_series=time_series,
        ntff_data=carry.get("ntff"),
        dft_planes=final_dft_planes,
        flux_monitors=final_flux_monitors,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        lumped_port_sparams=final_lumped_sparams,
        snapshots=None,
        ntff_box=ntff,
        grid=grid,
    )
