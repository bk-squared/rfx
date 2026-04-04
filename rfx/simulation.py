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
)
from rfx.boundaries.pec import apply_pec


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
    snapshots : dict[str, ndarray] or None
        Field snapshots keyed by component name.
    """
    state: FDTDState
    time_series: jnp.ndarray
    ntff_data: object = None
    dft_planes: tuple | None = None
    waveguide_ports: tuple | None = None
    wire_port_sparams: tuple | None = None
    snapshots: dict | None = None


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

    Unlike ``make_source``, the waveform is Cb-normalized and
    volume-compensated so that:
    1. No DC accumulation on PEC surfaces (enters through update equation)
    2. Injected power is resolution-independent (scales with 1/dx²)
    3. Proper coupling to cavity modes regardless of grid spacing

    E += Cb * J_source  where J = waveform(t) / dx²

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
    idx = grid.position_to_index(port.position)
    i, j, k = idx

    eps = materials.eps_r[i, j, k] * EPS_0
    sigma = materials.sigma[i, j, k]
    loss = sigma * grid.dt / (2.0 * eps)
    cb = (grid.dt / eps) / (1.0 + loss)

    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / grid.dx) * jax.vmap(port.excitation)(times)
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

    specs = []
    for cell in cells:
        i, j, k = cell
        eps = materials.eps_r[i, j, k] * EPS_0
        sigma = materials.sigma[i, j, k]
        loss = sigma * grid.dt / (2.0 * eps)
        cb = (grid.dt / eps) / (1.0 + loss)
        waveform = (cb / grid.dx) * jax.vmap(port.excitation)(times) / n_cells
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
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    aniso_eps: tuple | None = None,
    pec_mask: object | None = None,
    wire_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
) -> SimResult:
    """Run a compiled FDTD simulation via ``jax.lax.scan``.

    Parameters
    ----------
    grid : Grid
    materials : MaterialArrays
    n_steps : int
    boundary : "pec" or "cpml"
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

    Returns
    -------
    SimResult with final state, time series, and optional NTFF data.
    """
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
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

    # ---- subsystem flags (resolved at trace time) ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_tfsf = tfsf is not None
    use_ntff = ntff is not None
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    use_snapshot = snapshot is not None
    use_pec_mask = pec_mask is not None
    wire_port_sparams = wire_port_sparams or []
    use_wire_sparams = len(wire_port_sparams) > 0
    lumped_rlc = lumped_rlc or []
    use_lumped_rlc = len(lumped_rlc) > 0

    # ---- initialise states ----
    fdtd = init_state(grid.shape)

    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
        carry_init["cpml"] = cpml_state

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
    if use_waveguide_ports:
        carry_init["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_dft,
                cfg.v_ref_dft,
                cfg.i_probe_dft,
                cfg.i_ref_dft,
                cfg.v_inc_dft,
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

        # H update
        st = update_h(st, materials, dt, dx, periodic=periodic)
        if use_tfsf:
            st = apply_tfsf_h(st, tfsf_cfg, carry["tfsf"], dx, dt)
        if use_cpml:
            st, cpml_new = apply_cpml_h(
                st, cpml_params, carry["cpml"], grid, cpml_axes)
        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_h_state = update_tfsf_2d_h(tfsf_cfg, carry["tfsf"], dx, dt)
            else:
                tfsf_h_state = update_tfsf_1d_h(tfsf_cfg, carry["tfsf"], dx, dt)

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

        if use_tfsf:
            st = apply_tfsf_e(st, tfsf_cfg, tfsf_h_state, dx, dt)
        if use_cpml:
            st, cpml_new = apply_cpml_e(
                st, cpml_params, cpml_new, grid, cpml_axes)

        if pec_axes:
            st = apply_pec(st, axes=pec_axes)

        if use_pec_mask:
            from rfx.boundaries.pec import apply_pec_mask
            st = apply_pec_mask(st, pec_mask)

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

        # Soft sources
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s])
            st = st._replace(**{sc: field})

        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_new = update_tfsf_2d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)
            else:
                tfsf_new = update_tfsf_1d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)

        if use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                inject_waveguide_port,
                update_waveguide_port_probe,
            )

            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(carry["waveguide_port_accs"], waveguide_meta):
                cfg = cfg_meta._replace(
                    v_probe_dft=accs[0],
                    v_ref_dft=accs[1],
                    i_probe_dft=accs[2],
                    i_ref_dft=accs[3],
                    v_inc_dft=accs[4],
                )
                st = inject_waveguide_port(st, cfg_meta, t, dt, dx)
                cfg_updated = update_waveguide_port_probe(cfg, st, dt, dx)
                new_waveguide_port_accs.append(
                    (
                        cfg_updated.v_probe_dft,
                        cfg_updated.v_ref_dft,
                        cfg_updated.i_probe_dft,
                        cfg_updated.i_ref_dft,
                        cfg_updated.v_inc_dft,
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
        if use_waveguide_ports:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if use_wire_sparams:
            new_carry["wire_sparam_accs"] = tuple(new_wire_accs)
        if use_lumped_rlc:
            new_carry["rlc_states"] = tuple(new_rlc_states)

        return new_carry, output

    # ---- run ----
    body = jax.checkpoint(step_fn) if checkpoint else step_fn
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    final_carry, outputs = jax.lax.scan(body, carry_init, xs)

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

    final_waveguide_ports = None
    if use_waveguide_ports:
        final_waveguide_ports = tuple(
            cfg_meta._replace(
                v_probe_dft=accs[0],
                v_ref_dft=accs[1],
                i_probe_dft=accs[2],
                i_ref_dft=accs[3],
                v_inc_dft=accs[4],
            )
            for cfg_meta, accs in zip(waveguide_meta, final_carry["waveguide_port_accs"])
        )

    final_wire_sparams = None
    if use_wire_sparams:
        final_wire_sparams = tuple(
            (wp_meta, accs)
            for wp_meta, accs in zip(wire_sparam_meta, final_carry["wire_sparam_accs"])
        )

    return SimResult(
        state=final_carry["fdtd"],
        time_series=time_series,
        ntff_data=final_carry.get("ntff"),
        dft_planes=final_dft_planes,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        snapshots=snapshots,
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
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    aniso_eps: tuple | None = None,
    pec_mask: object | None = None,
    wire_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
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
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_tfsf = tfsf is not None
    use_ntff = ntff is not None
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    use_pec_mask = pec_mask is not None
    wire_port_sparams = wire_port_sparams or []
    use_wire_sparams = len(wire_port_sparams) > 0
    lumped_rlc = lumped_rlc or []
    use_lumped_rlc = len(lumped_rlc) > 0

    # ---- initialise states ----
    fdtd = init_state(grid.shape)
    carry: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
        carry["cpml"] = cpml_state

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
                cfg.v_probe_dft,
                cfg.v_ref_dft,
                cfg.i_probe_dft,
                cfg.i_ref_dft,
                cfg.v_inc_dft,
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
        st = update_h(st, materials, dt, dx, periodic=periodic)
        if use_tfsf:
            st = apply_tfsf_h(st, tfsf_cfg, carry_in["tfsf"], dx, dt)
        if use_cpml:
            st, cpml_new = apply_cpml_h(
                st, cpml_params, carry_in["cpml"], grid, cpml_axes)
        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_h_state = update_tfsf_2d_h(tfsf_cfg, carry_in["tfsf"], dx, dt)
            else:
                tfsf_h_state = update_tfsf_1d_h(tfsf_cfg, carry_in["tfsf"], dx, dt)

        st, debye_new, lorentz_new = _update_e_with_optional_dispersion(
            st, materials, dt, dx,
            debye=(debye_coeffs, carry_in["debye"]) if use_debye else None,
            lorentz=(lorentz_coeffs, carry_in["lorentz"]) if use_lorentz else None,
            periodic=periodic,
        )

        if use_tfsf:
            st = apply_tfsf_e(st, tfsf_cfg, tfsf_h_state, dx, dt)
        if use_cpml:
            st, cpml_new = apply_cpml_e(
                st, cpml_params, cpml_new, grid, cpml_axes)

        if pec_axes:
            st = apply_pec(st, axes=pec_axes)

        if use_pec_mask:
            from rfx.boundaries.pec import apply_pec_mask
            st = apply_pec_mask(st, pec_mask)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        if use_lumped_rlc:
            new_rlc_states = []
            for rlc_st, meta in zip(carry_in["rlc_states"], rlc_meta):
                st, rlc_st_new = update_rlc_element(st, rlc_st, meta)
                new_rlc_states.append(rlc_st_new)

        # Soft sources
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s])
            st = st._replace(**{sc: field})

        t = step_idx.astype(jnp.float32) * dt

        if use_tfsf:
            if _tfsf_is_2d:
                tfsf_new = update_tfsf_2d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)
            else:
                tfsf_new = update_tfsf_1d_e(tfsf_cfg, tfsf_h_state, dx, dt, t)

        if use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                inject_waveguide_port,
                update_waveguide_port_probe,
            )
            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(carry_in["waveguide_port_accs"], waveguide_meta):
                cfg = cfg_meta._replace(
                    v_probe_dft=accs[0], v_ref_dft=accs[1],
                    i_probe_dft=accs[2], i_ref_dft=accs[3],
                    v_inc_dft=accs[4],
                )
                st = inject_waveguide_port(st, cfg_meta, t, dt, dx)
                cfg_updated = update_waveguide_port_probe(cfg, st, dt, dx)
                new_waveguide_port_accs.append((
                    cfg_updated.v_probe_dft, cfg_updated.v_ref_dft,
                    cfg_updated.i_probe_dft, cfg_updated.i_ref_dft,
                    cfg_updated.v_inc_dft,
                ))

        # Wire port S-param DFT accumulation (JIT-integrated)
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
        if use_waveguide_ports:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if use_wire_sparams:
            new_carry["wire_sparam_accs"] = tuple(new_wire_accs)
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
                v_probe_dft=accs[0], v_ref_dft=accs[1],
                i_probe_dft=accs[2], i_ref_dft=accs[3],
                v_inc_dft=accs[4],
            )
            for cfg_meta, accs in zip(waveguide_meta, carry["waveguide_port_accs"])
        )

    final_wire_sparams = None
    if use_wire_sparams:
        final_wire_sparams = tuple(
            (wp_meta, accs)
            for wp_meta, accs in zip(wire_sparam_meta, carry["wire_sparam_accs"])
        )

    return SimResult(
        state=carry["fdtd"],
        time_series=time_series,
        ntff_data=carry.get("ntff"),
        dft_planes=final_dft_planes,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        snapshots=None,
    )
