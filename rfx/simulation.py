"""Compiled FDTD simulation runner.

Composes Yee updates, boundaries, sources, probes, and optional TFSF
plane-wave injection into a single JIT-compiled time loop via
``jax.lax.scan``. All subsystem selection (CPML, dispersion, TFSF) is
resolved at Python trace-time so the compiled function contains only
the needed code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_e, update_e_aniso, update_e_aniso_inv, update_e_box, update_h,
    e_update_coeffs, edge_averaged_materials, component_e_materials,
    e_component_coeffs, cell_component_e_coeffs, EPS_0, MU_0, _shift_bwd,
    map_lumped,
    precompute_coeffs, update_he_fast,
)
from rfx.boundaries.pec import (
    apply_pec,  # noqa: F401 -- re-exported; tests import it from here (not used by the step since #1164)
    resolve_wall_faces,
    apply_pec_edges,
    apply_pec_faces,
    apply_pec_occupancy,
    apply_pec_occupancy_box,
    kottke_fenced_edge_masks,
    realized_pec_edge_masks,
)
from rfx.progress import (
    ProgressReporter, check_not_traced, scan_with_progress,
    validate_report_every,
)
from rfx.snapshots import (
    plan_snapshot_pieces, snapshot_axes, snapshot_extractor,
    validate_snapshot_spec,
)


# CFL derating for the (2,4) fourth-order-in-space stencil.  The wider
# staggered difference raises the maximum spatial wavenumber the scheme
# resolves, so the explicit-leapfrog stability bound tightens: the (2,4)
# stable timestep is ~0.857x the (2,2) Courant limit (the 1D bound is
# 6/7 ≈ 0.857; see the kernel header in rfx/core/yee.py).  Applied only
# when stencil_order == 4; order=2 keeps grid.dt unchanged (byte-identical).
_ORDER4_CFL_FACTOR = 0.857


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


class MagneticSourceSpec(NamedTuple):
    """Precomputed H-component current source (Schelkunoff M/J magnetic).

    Used for the Schelkunoff J+M one-sided MSL port launch. The waveform
    is precomputed to include the -dt/mu coefficient so injection is a
    plain add (same pattern as SourceSpec for E fields).

    component : "hx", "hy", or "hz"
    waveform  : (n_steps,) float array — values added to the H field
        component at (i, j, k) each timestep (after the H update).
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

    mid_i, mid_j, mid_k: midpoint cell for V and I measurement — the
        midpoint of the LIVE wire run (issue #764; identical to the
        historical all-extent midpoint when no extent cell is dead).
    component: 'ez', 'ex', or 'ey'.
    freqs: (n_freqs,) frequency array.
    impedance: port reference impedance (Z0).
    live_cells: static tuple of (i, j, k) LIVE wire cells (issue #764) for
        the whole-port gap-voltage accumulator
        V_port = sum over live cells of -E_c*dx.  Empty tuple = degenerate
        fallback to the single midpoint cell (pre-#764 constructors).
    excite: whether this port is genuinely driven in THIS pass.  The
        whole-port driven-diagonal formula
        S_kk = (V_port - Z0*I)/(V_port + Z0*I) applies ONLY when True;
        a passive port's diagonal keeps the legacy per-cell convention
        (a load-independent port-cell diagnostic, not S_jj — issue #764).
    """
    mid_i: int
    mid_j: int
    mid_k: int
    component: str
    freqs: jnp.ndarray
    impedance: float
    live_cells: tuple = ()
    excite: bool = True


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
    excite : bool
        Whether this port is genuinely driven in THIS pass.  A driven port
        reads its diagonal from the terminal V/I pair
        (:func:`rfx.probes.probes.driven_port_reflection`); a passive one
        keeps the load-independent port-branch reading
        (:func:`rfx.probes.probes.extract_lumped_s11`).  Mirrors
        :class:`WireSParamSpec.excite`.

    Notes
    -----
    V is sampled as ``-E·dx`` and I as the curl-H loop integral times
    ``dx`` at the port cell, both AFTER source injection, with the Yee
    half-step phase on I.  Issue #72; sampling slot and diagonal formula
    decided by scripts/diagnostics/lumped_port_known_load_line.py.
    """
    i: int
    j: int
    k: int
    component: str
    freqs: jnp.ndarray
    impedance: float
    excite: bool = True


class DesignBoxSpec(NamedTuple):
    """One static box whose E update runs on its own permittivity (#1179).

    The box is a design region: its permittivity is the quantity being
    differentiated. Handing it here instead of writing it into
    ``materials.eps_r`` keeps the grid-wide update coefficients constant, so
    reverse-mode AD stores box-shaped arrays per timestep instead of
    grid-shaped ones. The physics is unchanged — ``update_e_box`` recomputes
    exactly what ``update_e`` would have computed at those cells.

    bounds : (i0, i1, j0, j1, k0, k1)
        Half-open cell index bounds of the box, resolved from the declared
        corners against the grid the solve builds.
    eps_r : box-shaped array (usually a tracer)
        Relative permittivity at the box cells.
    sigma : box-shaped array, a 3-tuple of them, or None
        Conductivity at the box cells. ``None`` (default) takes the run's own
        ``materials.sigma`` slice, so a lossy background inside the box is
        carried exactly rather than silently dropped.

        A single array is the one conductivity every E component in the box
        sees, exactly as ``materials.sigma`` is read by ``update_e``. A
        3-tuple ``(sigma_x, sigma_y, sigma_z)`` of box-shaped arrays gives
        each component its OWN conductivity, and therefore its own Ca/Cb.
        What needs it: a conducting SHEET has current only along its two
        in-plane edges, so a design variable per in-plane edge must not
        also load the edge through the sheet. ``(s, s, s)`` reproduces the
        single-array result bit for bit.
    """
    bounds: tuple
    eps_r: Any
    sigma: Any = None


class _DesignBoxCoeffs(NamedTuple):
    """Resolved ``(Ca, Cb)`` for a design box, built once outside the scan.

    ``bounds`` is the WRITE window, not the declared box: since #1210 the
    design material reaches one cell past the box on the plus side of each
    transverse axis (:func:`_design_box_edge_coeffs`). ``ca`` and ``cb`` are
    3-tuples of window-shaped arrays, one per E component in x, y, z order —
    per-component because of the edge average, and because a design
    conductivity may itself be per-component (#1216).
    """
    bounds: tuple
    ca: Any
    cb: Any


class DesignOccupancySpec(NamedTuple):
    """One static box whose PEC occupancy is the design variable (#1183).

    The relaxed-conductor rule scales every E component by ``1 - M``, ``M``
    the noisy-OR of the component's incident cells' occupancy. With a traced
    WHOLE-GRID occupancy that multiply keeps three grid-sized primal E arrays
    per timestep; with the traced values confined to a box the arrays it
    keeps are window-sized. Same physics — the window is recomputed from the
    field before the grid-wide scaling, so it reproduces exactly what
    ``apply_pec_occupancy`` would have written with the design occupancy in
    the array.

    bounds : (i0, i1, j0, j1, k0, k1)
        Half-open cell index bounds of the box.
    occupancy : box-shaped array (usually a tracer)
        Occupancy in [0, 1] at the box cells; REPLACES the run's own static
        occupancy there. Outside the box the static array is carried.
    """
    bounds: tuple
    occupancy: Any


class _DesignOccupancyKeep(NamedTuple):
    """Resolved write window and ``1 - M`` factors, built once outside the scan."""
    write: tuple
    keep: tuple


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
        Each entry is ``(LumpedPortSParamSpec, (v_dft, i_dft, v_ref_dft))``;
        v/i are POST-injection (i with the Yee half-step phase) and
        v_ref is the PRE-injection drive sample.
    wire_refplane_sparams : tuple | None
        Final reference-plane V/I DFT accumulators for the opt-in wire
        S-matrix plane path (issue #313).  Each entry is
        ``(WireRefPlaneSpec, (v_dft, i_minus_dft, i_plus_dft))``; two
        entries per opted port (plane slots 0 and 1).
    snapshots : dict[str, ndarray] or None
        Field snapshots keyed by component name, each
        ``(n_steps // interval, ...)``; see ``SnapshotSpec``.
    ntff_box : NTFFBox or None
        NTFF box specification used for accumulation.
    grid : Grid or None
        Grid metadata needed by post-processing / objective helpers.
    snapshot_axes : dict[str, rfx.snapshots.SnapshotAxes] or None
        Per snapshot component: sample coordinates in metres, the slice
        plane, and the step count and time of each frame (#1259). ``None``
        when no snapshot was recorded.
    dt : float or None
        The time step the scan actually advanced by, in seconds. It is
        ``grid.dt`` except under ``stencil_order=4``, whose step is derated
        by ``_ORDER4_CFL_FACTOR``; the time series and every frame are
        sampled at this step, so time and frequency must be read with it,
        not with ``grid.dt``.
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
    wire_refplane_sparams: tuple | None = None
    snapshot_axes: dict | None = None
    dt: float | None = None


# ---------------------------------------------------------------------------
# Helpers to build source / probe specs
# ---------------------------------------------------------------------------

def _source_cell_cb(grid: Grid, idx, materials, component,
                    periodic=(False, False, False)):
    """Cb of the E update for ``component`` at the source cell.

    Single spelling of the coefficient shared by ``make_source`` and
    ``make_j_source``. Since #1210 it is the PER-COMPONENT coefficient:
    ``make_j_source`` exists to turn a current into the field increment the
    update itself would have produced, and the update multiplies each
    component by the mean of eps and sigma over the four cells its edge
    touches. Reading the owning cell here instead put 0.556x the declared
    current into an Ey source standing on an eps 1|9 step along y.
    ``materials.eps_r``/``sigma`` may be JAX tracers (forward/AD path) —
    ``cell_component_e_coeffs`` indexes four cells and never calls ``float()``.
    """
    return cell_component_e_coeffs(
        materials, idx, component, grid.dt, periodic)[1]


def _uniform_cell_volume(grid: Grid) -> float:
    """dV = dx*dy*dz with missing axes duck-typed to dx (engineering
    principle 3). A 2D grid is treated as ONE CELL DEEP (dz -> dx), so
    dV = dx**3 there too — the documented #571 convention
    (``rfx/api/_source_semantics.py``), not per-unit-length dx**2."""
    return grid.dx * getattr(grid, "dy", grid.dx) * getattr(grid, "dz", grid.dx)


def make_source(grid: Grid, position, component, waveform_fn, n_steps,
                materials=None, amplitude_kind=None):
    """Create a SourceSpec by precomputing a waveform function (raw E-field add).

    WARNING: raw field source causes DC accumulation on PEC surfaces.
    Prefer ``make_j_source`` for resonance detection.

    Native amplitude convention (issue #571): ``'field'`` — ``E += w``.
    ``amplitude_kind='current'`` rescales the waveform by ``Cb/dV`` via
    ``rfx.api._source_semantics.source_amplitude_scale`` and REQUIRES
    ``materials`` (for Cb at the source cell). ``amplitude_kind`` of
    None or ``'field'`` is bit-identical to the historical output (no
    multiply is applied at all). ``materials`` is accepted and unused in
    that legacy/native case.
    """
    from rfx.api._source_semantics import needs_scale, source_amplitude_scale
    idx = grid.position_to_index(position)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = jax.vmap(waveform_fn)(times)
    if needs_scale(amplitude_kind, "raw"):
        # only 'current' lands here
        if materials is None:
            raise ValueError(
                "make_source(amplitude_kind='current') needs materials "
                "for the Cb normalization at the source cell")
        scale = source_amplitude_scale(
            amplitude_kind, "raw",
            cb=_source_cell_cb(grid, idx, materials, component),
            dV=_uniform_cell_volume(grid))
        waveform = scale * waveform
    return SourceSpec(i=idx[0], j=idx[1], k=idx[2],
                      component=component, waveform=waveform)


def make_j_source(grid: Grid, position, component, waveform_fn, n_steps, materials,
                  amplitude_kind=None):
    """Create a SourceSpec using current density injection (J source).

    Unlike ``make_source``, the waveform is Cb-normalized so that
    the source enters through the Ampere update equation:

    E += Cb * waveform(t)

    where Cb = dt / (eps * (1 + sigma*dt/2eps)).

    Note (scoped to ``amplitude_kind=None``): this is a soft *field*
    source, not a current-density source. Injected amplitude depends on
    local material (eps, sigma) but NOT on cell size. Power coupling
    varies with resolution; use Harminv/FFT for mode extraction rather
    than absolute amplitude.

    Native amplitude convention (issue #571): ``'cb'`` — ``E += Cb*w`` —
    which is NEITHER named kind (the legacy open-uniform third contract).
    ``amplitude_kind='current'`` rescales by ``1/dV``, ``'field'`` by
    ``1/Cb``, both via ``rfx.api._source_semantics.source_amplitude_scale``.
    None is bit-identical to the historical output (no multiply applied).

    Parameters
    ----------
    grid : Grid
    position : (x, y, z) in metres
    component : "ex", "ey", or "ez"
    waveform_fn : callable(t) -> value
    n_steps : int
    materials : MaterialArrays (for Cb computation at source cell)
    amplitude_kind : 'field' | 'current' | None (issue #571, above)
    """
    from rfx.api._source_semantics import needs_scale, source_amplitude_scale
    idx = grid.position_to_index(position)
    i, j, k = idx
    cb = _source_cell_cb(grid, idx, materials, component)

    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    # Cb normalization: the source enters the update equation through
    # the Cb coefficient (dt/eps/(1+loss)). This ensures:
    # 1. No DC accumulation on PEC (Cb scales with dt)
    # 2. Proper coupling to cavity modes
    # Power scales as Cb²·dx³ — weaker at fine grids, but Harminv
    # with bandpass filtering reliably extracts modes at all resolutions.
    waveform = cb * jax.vmap(waveform_fn)(times)
    if needs_scale(amplitude_kind, "cb"):
        # Python-level dispatch (issue #571 dossier): the scale may be a
        # tracer ('field' -> 1/Cb with traced materials), so never gate
        # the multiply on its value.
        waveform = source_amplitude_scale(
            amplitude_kind, "cb", cb=cb,
            dV=_uniform_cell_volume(grid)) * waveform
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

    # #1210: the drive coefficient is the update's own per-component Cb.
    cb = cell_component_e_coeffs(
        materials, idx, port.component, grid.dt)[1]

    d_par = port_d_parallel(grid, idx, port.component)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / d_par) * jax.vmap(port.excitation)(times)
    return SourceSpec(i=i, j=j, k=k,
                      component=port.component, waveform=waveform)


def make_wire_port_sources(grid, port, materials, n_steps, pec_edge_masks=None):
    """Create a list of SourceSpec for a multi-cell WirePort.

    Each LIVE cell in the wire gets its own SourceSpec with the
    Cb-corrected waveform scaled by 1/n_live (issue #318: dead extent
    cells inside PEC get no source — pre-#318 they accumulated phantom
    EMF).  With ``pec_edge_masks=None`` (or no dead cells) this is the
    historical all-cells 1/N_cells behaviour.  The port impedance must
    already be folded into *materials* via ``setup_wire_port()``.

    Returns
    -------
    list[SourceSpec]
    """
    from rfx.sources.sources import _wire_port_live_cells

    cells, live_flags, n_live = _wire_port_live_cells(grid, port, pec_edge_masks)
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt

    from rfx.sources.sources import port_d_parallel

    specs = []
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        i, j, k = cell
        d_par = port_d_parallel(grid, (i, j, k), port.component)
        cb = cell_component_e_coeffs(      # #1210
            materials, (i, j, k), port.component, grid.dt)[1]
        waveform = (cb / d_par) * jax.vmap(port.excitation)(times) / n_live
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
    aniso_inv_eps: tuple | None = None,
    stencil_order: int = 2,
    bloch: tuple | None = None,
) -> tuple[FDTDState, object | None, object | None]:
    """Update E with standard, Debye, Lorentz, or mixed dispersion.

    Parameters
    ----------
    aniso_eps : (eps_ex, eps_ey, eps_ez) or None
        Per-component relative permittivity arrays for subpixel smoothing.
        Only used when no dispersion model is active.
    aniso_inv_eps : (inv_xx, inv_yy, inv_zz) or None
        Stage 2 unified path: per-component inverse-permittivity tensor
        diagonal. When set, takes precedence over ``aniso_eps`` and
        dispatches to ``update_e_aniso_inv``. Numerically stable in the
        PEC limit (inv = 0); see derivation §5.
    stencil_order : int
        2 (default, byte-identical) or 4 ((2,4) wide stencil). Only the
        plain vacuum/dielectric branch honours order=4 — the dispersion
        (debye/lorentz) and anisotropic (aniso_eps/aniso_inv_eps) branches
        are fenced against order=4 upstream in ``_build_step_setup`` and
        keep their 2nd-order kernels regardless.
    """
    if debye is None and lorentz is None:
        if aniso_inv_eps is not None:
            inv_xx, inv_yy, inv_zz = aniso_inv_eps
            return update_e_aniso_inv(state, materials, inv_xx, inv_yy, inv_zz,
                                      dt, dx, periodic=periodic), None, None
        if aniso_eps is not None:
            eps_ex, eps_ey, eps_ez = aniso_eps
            return update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez,
                                  dt, dx, periodic=periodic), None, None
        return update_e(state, materials, dt, dx, periodic=periodic,
                        stencil_order=stencil_order, bloch=bloch), None, None

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

    # Narrow every output back to the dtype of the carry it came from
    # (issue #656) — same policy as the single-model bodies in
    # rfx/materials/{debye,lorentz}.py, which carry the full rationale.
    _fdtype = state.ex.dtype
    _dpdtype = jnp.promote_types(debye_state.px.dtype, _fdtype)
    _lpdtype = jnp.promote_types(lorentz_state.px.dtype, _fdtype)

    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # Explicit Lorentz polarization update first (per component, #1260).
    from rfx.materials.debye import per_component
    from rfx.materials.lorentz import (
        lorentz_p_component, mixed_e_component_coeffs,
    )
    e_old = (ex_old, ey_old, ez_old)
    curls = (curl_x, curl_y, curl_z)
    p_l = (lorentz_state.px, lorentz_state.py, lorentz_state.pz)
    p_l_prev = (lorentz_state.px_prev, lorentz_state.py_prev,
                lorentz_state.pz_prev)
    p_d = (debye_state.px, debye_state.py, debye_state.pz)
    p_l_new = tuple(
        lorentz_p_component(lorentz_coeffs, c, e_old[c], p_l[c], p_l_prev[c],
                            _lpdtype)
        for c in range(3))
    px_l_new, py_l_new, pz_l_new = p_l_new

    e_new, p_d_new = [], []
    for c in range(3):
        dp_l = jnp.sum(p_l_new[c] - p_l[c], axis=0)
        ca, cb, cc_debye, cc_lorentz = mixed_e_component_coeffs(
            debye_coeffs, lorentz_coeffs, c, dt)
        e_c = (
            ca * e_old[c]
            + cb * curls[c]
            + jnp.sum(cc_debye * p_d[c], axis=0)
            - cc_lorentz * dp_l
        ).astype(_fdtype)
        e_new.append(e_c)
        beta_c = per_component(debye_coeffs.beta, "beta")[c]
        p_d_new.append((debye_coeffs.alpha * p_d[c]
                        + beta_c * (e_c[None] + e_old[c][None])
                        ).astype(_dpdtype))
    ex_new, ey_new, ez_new = e_new

    new_fdtd = state._replace(
        ex=ex_new,
        ey=ey_new,
        ez=ez_new,
        step=state.step + 1,
    )
    new_debye = DebyeState(px=p_d_new[0], py=p_d_new[1], pz=p_d_new[2])
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
        Record a frame every *interval* time steps; an integer >= 1, anything
        else is refused. Counting is 1-based over completed steps: a run of
        ``n_steps`` records ``n_steps // interval`` frames, frame ``k``
        (0-based) holding the fields after step ``(k + 1) * interval``, i.e.
        after steps ``interval, 2*interval, ...``. The last
        ``n_steps % interval`` steps record no frame. ``interval=1`` records
        every step. Memory scales with the frame count.
    components : tuple of str
        Field components to capture, e.g. ("ez",) or ("ex", "hy").
    slice_axis : int or None
        0, 1 or 2. If not None and *slice_index* is not None, capture a 2-D
        slice at *slice_index* along this axis instead of the full 3-D field
        (saves memory). With *slice_index* None the full field is recorded.
    slice_index : int or None
        PADDED lattice index along *slice_axis*: the same index space as
        ``grid.shape`` and ``grid.position_to_index``, CPML cells counted
        (``grid.position_to_index(pos)[axis]`` gives the index of a physical
        position).

    Where each recorded sample sits (metres) and when each frame was taken
    (seconds): ``Result.snapshot_axes`` after a run, or
    ``rfx.snapshot_axes(grid, spec, n_steps)`` before one (``rfx/snapshots.py``
    states the lattice layout and the E/H time convention).
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


# ---------------------------------------------------------------------------
# Shared setup helper (W6.2)
# ---------------------------------------------------------------------------
#
# ``run()`` and ``run_until_decay()`` share ~73% of their setup code: boundary
# resolution, subsystem flags, carry_init assembly, metadata extraction, and
# the _StepContext field population.  ``_build_step_setup`` centralises that
# shared work and returns a ``_SimSetup`` NamedTuple.  Each driver then
# applies its handful of driver-specific overrides before constructing the
# final ``_StepContext``.
#
# Intentional behavioural differences preserved:
#   * ``flux_meta_11`` (11-field, with window name/alpha) used by ``run()``;
#     ``flux_meta_8`` (8-field, no window) used by ``run_until_decay()``
#     (use_flux_window=False).  Unifying decay onto windowed DFT is a
#     deliberate future task.
#   * ``use_fast_he`` / ``fast_coeffs`` — ``run()`` only; decay path is a
#     Python loop and the GPU fast-path is not applicable.

class _SimSetup(NamedTuple):
    """Shared setup artefacts returned by ``_build_step_setup``.

    ``ctx_kwargs`` is a dict of ``_StepContext`` fields that are identical
    between the two drivers.  Each driver extends / overrides it with its
    own fields before passing to ``_StepContext(**ctx_kwargs, **overrides)``.
    """
    carry_init: dict
    ctx_kwargs: dict
    # metadata kept separate for carry-assembly post-loop in run_until_decay
    src_meta: list
    mag_src_meta: list
    prb_meta: list
    dft_meta: tuple
    waveguide_meta: tuple
    wire_sparam_meta: tuple
    lumped_sparam_meta: tuple
    rlc_meta: tuple
    # issue #313 opt-in reference-plane accumulators (empty when unused)
    wire_refplane_meta: tuple
    # 8-field tuple: (axis, index, freqs, comp_names, lo1, hi1, lo2, hi2)
    flux_meta_8: tuple
    # 11-field tuple adds: total_steps, window, window_alpha
    flux_meta_11: tuple
    # resolved physical constants (both drivers need these for waveform setup)
    dt: float
    dx: float
    periodic: tuple
    pec_axes: str
    pec_faces: frozenset


def resolve_periodic(grid, periodic):
    """The run's per-axis periodic flags, after the 2-D lane override (#689).

    One spelling, shared by :func:`_build_step_setup` and by any lane that
    must realize the PEC edge masks (§1.7) before calling ``run`` — those
    masks are only correct under the flags the step function will use.
    """
    if periodic is None:
        periodic = (False, False, False)
    else:
        if len(periodic) != 3:
            raise ValueError(f"periodic must have length 3, got {periodic!r}")
        periodic = tuple(bool(v) for v in periodic)
    if getattr(grid, "is_2d", False):
        periodic = (periodic[0], periodic[1], True)
    return periodic


def _design_box_bounds(bounds, grid):
    """``(bounds, realized cell counts)`` of a design box on this grid.

    One spelling for the permittivity box (#1179) and the occupancy box
    (#1183): the REALIZED cell counts are what a design array must match,
    and a box that is empty or leaves the grid is an error, not a clamp.
    """
    i0, i1, j0, j1, k0, k1 = (int(v) for v in bounds)
    bounds = (i0, i1, j0, j1, k0, k1)
    box_shape = (i1 - i0, j1 - j0, k1 - k0)
    if any(n <= 0 for n in box_shape):
        raise ValueError(
            f"design box {bounds} is empty (realized cell counts "
            f"{box_shape}); it must span at least one cell per axis.")
    for axis, (lo, hi, n) in enumerate(
            zip((i0, j0, k0), (i1, j1, k1), grid.shape)):
        if lo < 0 or hi > n:
            raise ValueError(
                f"design box {bounds} leaves the grid on axis "
                f"{'xyz'[axis]}: cells [{lo}, {hi}) against {n} cells.")
    return bounds, box_shape


def _resolve_design_occupancy(
    spec: "DesignOccupancySpec",
    *,
    grid,
    periodic,
    pec_occupancy,
    field_dtype,
) -> "_DesignOccupancyKeep":
    """Check an occupancy design box against this run, then build ``1 - M`` (#1183).

    Far fewer fences than :func:`_resolve_design_box`, and the reason is
    physical: the occupancy scaling is a pure multiply applied AFTER the E
    update and the absorber, and it reads no material array. Whatever the E
    update was — dispersive, anisotropic, fourth-order — the step arrives at
    this slot with the same field on both formulations, so the window redo
    reproduces the grid-wide call. What it does need is that no cell outside
    the window can see the design occupancy, which is the shift reach
    :func:`pec_occupancy_box_keep` owns (and why it refuses periodic axes).
    """
    from rfx.boundaries.pec import pec_occupancy_box_keep

    bounds, box_shape = _design_box_bounds(spec.bounds, grid)
    occ = jnp.asarray(spec.occupancy)
    if tuple(occ.shape) != box_shape:
        raise ValueError(
            f"design occupancy has shape {tuple(occ.shape)} but the design "
            f"box {bounds} realizes {box_shape} cells.")
    write, keep = pec_occupancy_box_keep(
        bounds, occ, shape=tuple(grid.shape), dtype=field_dtype,
        pec_occupancy=pec_occupancy, periodic=periodic)
    return _DesignOccupancyKeep(write=write, keep=keep)


def _design_box_window(bounds, shape):
    """``(write bounds, computation window, write slice in it, box slice)``.

    The same window arithmetic :func:`rfx.boundaries.pec.pec_occupancy_box_keep`
    does for a traced occupancy, for the same reason. A cell's permittivity
    reaches the E components on the edges incident to it, and those are the
    edges at the cell and at its PLUS neighbour along each transverse axis
    (:func:`edge_averaged_materials` averages over BACKWARD neighbours). So the
    design material moves the update one cell past the box on the plus side —
    that layer must be written. One cell of context on the MINUS side is
    computed and not written: its own backward neighbour lies outside the
    window, so its value is wrong, while every cell in the write window has
    all four of its incident cells inside.
    """
    i0, i1, j0, j1, k0, k1 = (int(v) for v in bounds)
    lo = (i0, j0, k0)
    hi = (i1, j1, k1)
    c_lo = tuple(max(v - 1, 0) for v in lo)
    c_hi = tuple(min(v + 1, n) for v, n in zip(hi, shape))
    win = tuple(slice(a, b) for a, b in zip(c_lo, c_hi))
    write_bounds = (lo[0], c_hi[0], lo[1], c_hi[1], lo[2], c_hi[2])
    inner = tuple(slice(a - c, b - c) for a, b, c in zip(lo, c_hi, c_lo))
    box_local = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, c_lo))
    return write_bounds, win, inner, box_local


def _design_box_edge_coeffs(bounds, eps_r_box, sigma_box, materials, dt, shape):
    """Per-component ``(Ca, Cb)`` over the design box's write window (#1210).

    The box works in EDGE (per-E-component) values, and what it is handed
    decides how each value is made:

    * a CELL quantity -- the design permittivity always, and a single design
      conductivity array -- is written into a COPY of the background over the
      computation window and turned into edge values by the same four-cell
      average as the rest of the grid (``component_e_materials``). The
      window's minus-side context supplies the neighbour cells outside the box
      on its faces, and its plus-side layer is written because a cell's value
      reaches the edges on its plus faces. The box lane then equals what
      ``update_e`` would do with the design values in ``materials``.
    * a 3-tuple ``(sigma_x, sigma_y, sigma_z)`` is ALREADY per edge (#1216,
      the sheet lane: one conductivity per Yee edge, component ``c`` at the
      same box index). It is taken as-is and NOT averaged again: it replaces
      the edge conductivity at the box indices, and every other edge in the
      window -- the plus-side layer included -- keeps the background edge
      value. A box-shaped edge array cannot address the plus-face edge layer
      a cell array reaches, which is why a single array ``s`` and the tuple of
      its own averages agree on the box's edges and not on that layer.
    """
    write_bounds, win, inner, box_local = _design_box_window(bounds, shape)
    eps_box = jnp.asarray(eps_r_box)
    eps = jnp.asarray(materials.eps_r)[win]
    # Promote the BACKGROUND to the design dtype, never the other way: a
    # traced float64 design permittivity cast down to the float32 background
    # would silently lose the precision the x64 AD lanes run for (#646).
    eps = eps.astype(jnp.promote_types(eps.dtype, eps_box.dtype))
    eps = eps.at[box_local].set(eps_box)

    per_edge = isinstance(sigma_box, (tuple, list))
    sig = jnp.asarray(materials.sigma)[win]
    if not per_edge:
        sig_box = jnp.asarray(sigma_box)
        sig = sig.astype(jnp.promote_types(sig.dtype, sig_box.dtype))
        sig = sig.at[box_local].set(sig_box)

    # A lumped stamp in the window's context layer is edge-owned, not a cell
    # volume, so it is removed before the average and added back at its cell,
    # on its own component — the same rule ``component_e_materials`` applies
    # grid-wide (#1210, #1236). The design box itself is fenced off port and
    # source cells.
    win_mats = MaterialArrays(
        eps_r=eps, sigma=sig, mu_r=None,
        eps_r_lumped=map_lumped(
            getattr(materials, "eps_r_lumped", None),
            lambda a: jnp.asarray(a)[win].astype(eps.dtype)),
        sigma_lumped=map_lumped(
            getattr(materials, "sigma_lumped", None),
            lambda a: jnp.asarray(a)[win].astype(sig.dtype)))
    eps_c, sig_c = component_e_materials(win_mats, (False, False, False))

    if per_edge:
        def _put(edge_bg, edge_box):
            edge_box = jnp.asarray(edge_box)
            edge_bg = edge_bg.astype(
                jnp.promote_types(edge_bg.dtype, edge_box.dtype))
            return edge_bg.at[box_local].set(edge_box)
        sig_c = tuple(_put(bg, v) for bg, v in zip(sig_c, sigma_box))

    pairs = [e_update_coeffs(e, s_, dt) for e, s_ in zip(eps_c, sig_c)]
    return (write_bounds,
            tuple(p[0][inner] for p in pairs),
            tuple(p[1][inner] for p in pairs))


def _resolve_design_box(
    spec: "DesignBoxSpec",
    *,
    grid,
    materials,
    dt,
    use_cpml: bool,
    use_upml: bool,
    cpml_axes: str,
    use_debye: bool,
    use_lorentz: bool,
    use_kerr: bool,
    aniso_eps,
    aniso_inv_eps,
    stencil_order: int,
    bloch,
    sheet_impedance,
    cell_metas,
    periodic=(False, False, False),
) -> "_DesignBoxCoeffs":
    """Check a design box against this run, then build its Ca/Cb (#1179).

    The box carries its own permittivity, so ``materials`` stays constant and
    the E update inside the box is redone from these coefficients. That is
    only the same physics while NOTHING ELSE in the step reads ``materials``
    at a design cell — and the checks below are how that is decided, in the
    one place where every resolved flag and every port/source cell is in
    hand. Everything unsupported raises; nothing is silently dropped, because
    a dropped design variable is a zero gradient that looks like a converged
    optimisation.

    Raises for: UPML, Debye/Lorentz dispersion, anisotropic / subpixel
    permittivity, Kerr, ``stencil_order=4``, the oblique-periodic Bloch path,
    a box that reaches into the CPML absorber, and a box holding a source,
    port, RLC element or surface-impedance sheet edge.
    """
    unsupported = []
    if use_upml:
        unsupported.append("boundary='upml'")
    if use_debye:
        unsupported.append("debye dispersion")
    if use_lorentz:
        unsupported.append("lorentz dispersion")
    if aniso_eps is not None:
        unsupported.append("aniso_eps (subpixel/anisotropic eps)")
    if aniso_inv_eps is not None:
        unsupported.append("aniso_inv_eps (Kottke inverse-eps tensor)")
    if use_kerr:
        unsupported.append("Kerr nonlinearity")
    if stencil_order != 2:
        unsupported.append(f"stencil_order={stencil_order}")
    if bloch is not None:
        unsupported.append("oblique-periodic Bloch path (#404)")
    if unsupported:
        raise NotImplementedError(
            "the design-box permittivity (#1179) rebuilds only the plain "
            "lossy Yee E update inside its box, so every path that computes "
            "E some other way is rejected rather than silently ignored. "
            "Unsupported feature(s) present: " + ", ".join(unsupported)
            + ". Use eps_override (the whole-grid traced permittivity) for "
            "these — it is exact on every path, at a grid-sized AD tape."
        )

    bounds, box_shape = _design_box_bounds(spec.bounds, grid)
    i0, i1, j0, j1, k0, k1 = bounds
    # #1210: the design material reaches one cell past the box on the plus
    # side of every axis, and the redo WRITES there. Every fence below is
    # therefore held against the write window, not the declared box — a
    # source, a sheet edge or an absorber cell in that layer is the same
    # collision the declared box was already fenced against.
    _write_bounds, _, _, _ = _design_box_window(bounds, tuple(grid.shape))
    w_i0, w_i1, w_j0, w_j1, w_k0, w_k1 = _write_bounds
    # #1210: the window is built with the non-periodic convention -- it clips
    # at the grid face where the grid-wide update would WRAP. That is only a
    # difference for a box that touches a periodic face, and there it is a
    # silent one, so it is refused rather than approximated. The occupancy
    # box's sibling (``pec_occupancy_box_keep``) refuses periodic axes for the
    # same reason.
    for _axis, _name in enumerate("xyz"):
        if not periodic[_axis]:
            continue
        _lo, _hi = bounds[2 * _axis], bounds[2 * _axis + 1]
        _n = grid.shape[_axis]
        if _lo == 0 or _hi == _n:
            raise NotImplementedError(
                f"a design box (#1179) flush against a PERIODIC {_name} face "
                f"is not supported: cells [{_lo}, {_hi}) on a {_n}-cell "
                f"periodic axis. An E component takes the mean of its four "
                f"incident cells (#1210), and across a periodic seam those "
                f"neighbours wrap -- the box's one-cell window does not. Move "
                f"the box off the {_name} faces, or use eps_override (the "
                f"whole-grid traced permittivity), which wraps correctly.")

    # The absorber. apply_cpml_e's psi correction is written at the [:n] /
    # [-n:] face slabs, and its coefficient is dt/(eps_r*EPS_0) from
    # ``materials`` -- the one array this design stops tracing. A box inside
    # an ABSORBING slab would therefore be absorbed with the BACKGROUND
    # permittivity while it is updated with the design one.
    #
    # Per FACE, not per axis. The kernel's window width
    # (``_axis_buffer_depths``) is the axis MAXIMUM of the two pads, floored
    # at 1, and it is applied to both faces -- but a face whose profile is
    # the all-no-op one (b=1, c=0, kappa=1) adds exactly zero however wide
    # that window is: psi stays at its zero init and the kappa term is
    # (1/1 - 1). A face gets that profile exactly when its allocated pad is
    # 0, i.e. PEC / PMC / periodic (``Grid._face_pad``), and an absorbing
    # face's own pad IS its active layer count (the invariant stated in
    # ``_axis_buffer_depths``'s docstring). Reading the axis maximum instead
    # refused the ordinary patch on a ground plane: x_lo PEC + x_hi CPML at
    # cpml_layers=5 gives axis depths (5, 1, 1), so a box sitting on the PEC
    # wall was rejected for entering a 5-layer absorber that face does not
    # have, and the y/z floor of 1 rejected any box touching a PEC face.
    #
    # A per-face pad is that proxy only on an axis the GRID says it absorbs
    # on. ``init_cpml`` builds a real profile on every axis ``cpml_axes`` of
    # the RUN names, and a grid built with a narrower ``cpml_axes`` allocates
    # no pad on the others -- so "pad 0" there means "no padding cells", not
    # "no absorber" (the trap ``_axis_buffer_depths`` documents, measured at
    # a 49 % energy change). On such an axis, and on a duck-typed grid with
    # no per-face pads, fall back to the axis window: nothing is known to be
    # a no-op there.
    if use_cpml or use_upml:
        from rfx.boundaries.cpml import _axis_buffer_depths
        depths = _axis_buffer_depths(grid, grid.cpml_layers)
        grid_axes = getattr(grid, "cpml_axes", cpml_axes)
        for axis, name in enumerate("xyz"):
            if name not in cpml_axes:
                continue  # no correction is written on this axis at all
            if name in grid_axes and hasattr(grid, f"pad_{name}_lo"):
                pad_lo = int(getattr(grid, f"pad_{name}_lo"))
                pad_hi = int(getattr(grid, f"pad_{name}_hi"))
            else:
                pad_lo = pad_hi = int(depths[axis])
            lo = _write_bounds[2 * axis]
            hi = _write_bounds[2 * axis + 1]
            n = grid.shape[axis]
            if lo < pad_lo or hi > n - pad_hi:
                raise ValueError(
                    f"design box cells [{bounds[2 * axis]}, "
                    f"{bounds[2 * axis + 1]}) on axis {name} reach "
                    f"into the CPML absorber ({pad_lo} layer(s) at {name}_lo "
                    f"and {pad_hi} at {name}_hi, on a {n}-cell axis; the box's "
                    f"update window is [{lo}, {hi}), one cell wider on the "
                    f"plus side since #1210). The "
                    f"absorber builds its own coefficient from the background "
                    f"permittivity, which the design box no longer carries. "
                    f"Move the box into cells [{pad_lo}, {n - pad_hi - 1}).")

    def _in_box(cell) -> bool:
        i, j, k = (int(v) for v in cell)
        return (w_i0 <= i < w_i1) and (w_j0 <= j < w_j1) and (w_k0 <= k < w_k1)

    # A source or port cell reads the material AT SETUP to turn a current
    # into a field increment (``_source_cell_cb``) or to fold an impedance
    # into sigma. Those reads see the background array, so a design cell
    # under one of them would drive with the wrong permittivity.
    for kind, cells in cell_metas:
        hits = [tuple(int(v) for v in c) for c in cells if _in_box(c)]
        if hits:
            raise ValueError(
                f"the design box {bounds} holds {kind} cell(s) {hits}. "
                f"That cell's drive/load coefficient is built from the "
                f"background permittivity before the time loop starts, so "
                f"the design permittivity would not reach it. Move the box "
                f"off the {kind} cells, or use eps_override.")

    sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))
    w_sl = (slice(w_i0, w_i1), slice(w_j0, w_j1), slice(w_k0, w_k1))
    if sheet_impedance is not None:
        for mask in (sheet_impedance.mask_ex, sheet_impedance.mask_ey,
                     sheet_impedance.mask_ez):
            if mask is not None and bool(jnp.any(jnp.asarray(mask)[w_sl])):
                raise ValueError(
                    f"a surface_impedance_f0 sheet (#677) has loaded edges "
                    f"inside the design box {bounds}. The sheet operator "
                    f"REPLACES the E update at those edges with coefficients "
                    f"built from the background permittivity, so it would "
                    f"overwrite the design update. Move the box off the "
                    f"sheet, or use eps_override.")

    eps_r = jnp.asarray(spec.eps_r)
    if tuple(eps_r.shape) != box_shape:
        raise ValueError(
            f"design permittivity has shape {tuple(eps_r.shape)} but the "
            f"design box {bounds} realizes {box_shape} cells.")
    sigma = materials.sigma[sl] if spec.sigma is None else spec.sigma
    if isinstance(sigma, (tuple, list)):
        # Per-component conductivity: one (Ca, Cb) pair per E component, in
        # x, y, z order. The permittivity stays shared — a design SHEET is
        # one material whose current is anisotropic, not three materials.
        if len(sigma) != 3:
            raise ValueError(
                f"a per-component design conductivity is a 3-tuple "
                f"(sigma_x, sigma_y, sigma_z); got {len(sigma)} entries.")
        for name, s in zip("xyz", sigma):
            s = jnp.asarray(s)
            if tuple(jnp.shape(s)) != box_shape:
                raise ValueError(
                    f"design conductivity sigma_{name} has shape "
                    f"{tuple(jnp.shape(s))} but the design box {bounds} "
                    f"realizes {box_shape} cells.")
        # The tuple is already per E edge (#1216) and is taken as-is; the
        # shared permittivity is still a cell quantity and is edge-averaged
        # over the window (#1210). See _design_box_edge_coeffs.
        write_bounds, ca, cb = _design_box_edge_coeffs(
            bounds, eps_r, tuple(jnp.asarray(s) for s in sigma), materials,
            dt, tuple(grid.shape))
        return _DesignBoxCoeffs(bounds=write_bounds, ca=ca, cb=cb)
    sigma = jnp.asarray(sigma)
    if tuple(jnp.shape(sigma)) != box_shape:
        raise ValueError(
            f"design conductivity has shape {tuple(jnp.shape(sigma))} but "
            f"the design box {bounds} realizes {box_shape} cells.")
    # #1210: per-component coefficients over the write window, the design
    # values laid into a copy of the background so the box-edge cells average
    # with the constant background exactly as the grid-wide update would.
    write_bounds, ca, cb = _design_box_edge_coeffs(
        bounds, eps_r, sigma, materials, dt, tuple(grid.shape))
    return _DesignBoxCoeffs(bounds=write_bounds, ca=ca, cb=cb)


def _build_step_setup(
    grid: "Grid",
    materials: "MaterialArrays",
    *,
    boundary: str,
    cpml_axes: str,
    pec_axes: "str | None",
    periodic: "tuple | None",
    debye: "tuple | None",
    lorentz: "tuple | None",
    tfsf: "tuple | None",
    sources: list,
    probes: list,
    dft_planes: list,
    flux_monitors: list,
    waveguide_ports: list,
    ntff: object,
    aniso_eps: "tuple | None",
    aniso_inv_eps: "tuple | None",
    aniso_inv_eps_smooth: bool,
    pec_mask: object,
    pec_sheets: object,
    pec_wires: object,
    pec_edge_masks: object,
    pec_occupancy: object,
    conformal_weights: "tuple | None",
    wire_port_sparams: list,
    lumped_port_sparams: list,
    lumped_rlc: list,
    kerr_chi3: "object | None",
    field_dtype: object,
    mag_sources: list,
    stencil_order: int = 2,
    wire_refplane_sparams: "list | None" = None,
    sheet_impedance: "object | None" = None,
    design_box: "DesignBoxSpec | None" = None,
    design_occupancy: "DesignOccupancySpec | None" = None,
) -> "_SimSetup":
    """Build the shared setup artefacts used by both ``run`` and ``run_until_decay``.

    Returns a ``_SimSetup`` NamedTuple. Each driver extends ``ctx_kwargs``
    with its own driver-specific ``_StepContext`` fields before constructing
    the context.
    """
    dt = grid.dt
    dx = grid.dx

    # ---- boundary configuration ----
    periodic = resolve_periodic(grid, periodic)

    # Skip CPML / PEC on periodic axes.
    axis_names = ("x", "y", "z")
    for axis_name, is_periodic in zip(axis_names, periodic):
        if is_periodic:
            cpml_axes = cpml_axes.replace(axis_name, "")
    # ---- the walls, per face, from the grid's declaration (#1164) ----
    # ``resolve_wall_faces`` is the one rule every entry point shares:
    # periodic -> none; a magnetic face is never an electric wall; a
    # declared PEC face is one; any other face is PEC-backed unless the
    # legacy ``pec_axes`` string withholds that default for its axis.
    # The scan applies these per face only; the axis-wide ``apply_pec``
    # is gone from the step (an axis wall over a PMC face shorted a port
    # on that plane through run(), and through forward() after #1194).
    _pec_faces_frozen, _pmc_faces_frozen = resolve_wall_faces(
        grid, periodic, pec_axes)
    use_pec_faces = bool(_pec_faces_frozen)
    use_pmc_faces = bool(_pmc_faces_frozen)
    # Axes with an electric wall on BOTH faces, for the GPU fast path's
    # coefficient bake (``precompute_coeffs``) and for callers reading it.
    pec_axes = "".join(
        a for a in "xyz"
        if f"{a}_lo" in _pec_faces_frozen and f"{a}_hi" in _pec_faces_frozen)

    # ---- subsystem flags (resolved at Python trace time) ----
    use_cpml = boundary == "cpml" and grid.cpml_layers > 0
    use_upml = boundary == "upml" and grid.cpml_layers > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_tfsf = tfsf is not None
    use_ntff = ntff is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    # ---- #931 §1.7: realize the PEC edge masks ONCE, here ----
    # ``pec_edge_masks`` may arrive pre-realized (and port-cleared) from a
    # lane that had to read them before the run; otherwise realize from the
    # volume cell mask plus the declared sheets/wires with THIS run's
    # periodic flags.
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    if pec_edge_masks is None and (
            pec_mask is not None or pec_sheets or pec_wires):
        pec_edge_masks = realized_pec_edge_masks(
            pec_mask, sheets=pec_sheets, wires=pec_wires, periodic=periodic)
    use_pec_edges = pec_edge_masks is not None
    use_pec_occupancy = pec_occupancy is not None
    # The soft lane carries sheets/wires as STATIC edge masks (§1.6).
    #
    # Realized from the declarations, then INTERSECTED with the masks this
    # call was handed.  Those arrived port-cleared (§1.9: a port releases
    # the one component it drives, at its own cells), and re-realizing from
    # the declaration alone puts the conductor back over the port — a
    # 50 ohm Ex port standing on a PEC sheet reads a False Ex entry in the
    # masks it was given and a True one in the reconstruction.  Measured
    # through ``pec_occupancy_override=zeros``, which §1.6 requires to be
    # the hard path exactly: an off-sheet Ez probe moved 5.7245574 ->
    # 12.5340872 on a 12 mm board at dx = 2 mm.  The identity has to hold
    # THROUGH the override entry point, so the clearing is carried here
    # rather than recomputed.
    pec_static_edge_masks = None
    if use_pec_occupancy and (pec_sheets or pec_wires):
        pec_static_edge_masks = realized_pec_edge_masks(
            None, sheets=pec_sheets, wires=pec_wires, periodic=periodic)
        if pec_edge_masks is not None:
            pec_static_edge_masks = tuple(
                s & m for s, m in zip(pec_static_edge_masks, pec_edge_masks))
    use_conformal = conformal_weights is not None
    # Stage 2: when aniso_inv_eps is set, the inverse-permittivity tensor
    # encodes both PEC behaviour and dielectric subpixel smoothing —
    # apply_conformal_pec is redundant and SKIPPED to avoid double-zeroing.
    use_aniso_inv = aniso_inv_eps is not None
    # ---- #931 §1.8 fence: Kottke Stage-2 owns its own volume ----
    # ``subpixel_smoothing="kottke_pec"`` builds the tensor from the SAME
    # ``pec_shapes`` that produced ``pec_mask``, and a partially filled edge
    # gets a FRACTIONAL inverse permittivity.  Applying the §1.2 ownership
    # rule on top hard-zeroes those edges and throws the subpixel model away.
    # Restrict the applied set to what the tensor froze, plus the sheets and
    # wires the tensor cannot see (they own no cell).  ``aniso_inv_eps_smooth``
    # marks the occupancy-derived tensor instead, where the static declaration
    # and the traced override are two different geometries and the
    # intersection would silently delete declared metal — that lane keeps the
    # realized edges.
    if use_pec_edges and use_aniso_inv and not aniso_inv_eps_smooth:
        pec_edge_masks = kottke_fenced_edge_masks(
            pec_edge_masks, aniso_inv_eps,
            sheets=pec_sheets, wires=pec_wires, periodic=periodic)
    use_wire_sparams = len(wire_port_sparams) > 0
    use_lumped_sparams = len(lumped_port_sparams) > 0
    wire_refplane_sparams = wire_refplane_sparams or []
    use_wire_refplanes = len(wire_refplane_sparams) > 0
    use_lumped_rlc = len(lumped_rlc) > 0
    use_kerr = kerr_chi3 is not None
    use_mag_sources = len(mag_sources) > 0

    # ---- #404 oblique-periodic complex Bloch path activation ----
    # An oblique (2D-aux) TFSF injects a tilted plane wave that a plain-periodic
    # REAL grid cannot sustain (the pre-#404 under-tilt). Drive the shared solver
    # on a complex Bloch-envelope path instead: fields carry the envelope P and a
    # per-axis phase rides the periodic roll; the physical field is
    # Re(P·exp(-j k_t·y)). Gated strictly on the 2D-aux discriminator, so normal
    # incidence and every non-TFSF run stay real float32 and byte-identical.
    # NOTE: gated strictly on the 2D-aux (Bloch) discriminator. Open-domain
    # oblique Method B (MethodBConfig, NOT TFSF2DConfig) intentionally bypasses
    # this lane: it stays real float32 and its NTFF/DFT/flux monitors read
    # physical fields, so the fail-loud below must NOT be widened to angle != 0.
    _oblique_bloch = False
    if use_tfsf:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d_early
        _oblique_bloch = _is_tfsf_2d_early(tfsf[0])
    if _oblique_bloch:
        # Frequency-domain monitors accumulate the complex envelope P, not the
        # physical spectrum — fail loud rather than return truncated garbage.
        _unsupported_ob = []
        if use_ntff:
            _unsupported_ob.append("NTFF box")
        if use_dft_planes:
            _unsupported_ob.append("DFT plane probe")
        if use_flux_monitors:
            _unsupported_ob.append("flux monitor")
        if _unsupported_ob:
            raise NotImplementedError(
                "Oblique (angle_deg != 0) TFSF uses the #404 complex Bloch path; "
                "frequency-domain monitors are not yet transform-aware on it. "
                "Unsupported: " + ", ".join(_unsupported_ob) + ". Use field "
                "snapshots / final state (returned as physical fields), or "
                "compute_rcs for open-domain oblique scattering."
            )

    # The far-field integral is over the SCATTERED field, which it only is
    # when the Huygens box encloses the whole injected region. Plain ints,
    # evaluated here at trace time.
    if use_tfsf and use_ntff:
        from rfx.farfield import require_box_encloses_injected_region
        from rfx.sources.tfsf import tfsf_injection_planes
        require_box_encloses_injected_region(
            ntff, tfsf_injection_planes(tfsf[0]), shape=grid.shape)

    # ---- (2,4) fourth-order-in-space stencil (PR-1b) ----
    # order=2 is the default and BYTE-IDENTICAL: dt and every kernel call are
    # untouched.  order=4 is reachable ONLY on the plain uniform-Cartesian
    # vacuum/dielectric path (pec/periodic boundary, default solver, no
    # dispersion / anisotropy / conformal-PEC / Kerr).  Any other sub-feature
    # under order=4 raises here — a silent 2nd-order result on an unsupported
    # path is a CRITICAL correctness failure.
    if stencil_order not in (2, 4):
        raise ValueError(f"stencil_order must be 2 or 4, got {stencil_order}")
    if stencil_order == 4:
        _unsupported = []
        if use_cpml:
            _unsupported.append("boundary='cpml'")
        if use_upml:
            _unsupported.append("boundary='upml'")
        if use_debye:
            _unsupported.append("debye dispersion")
        if use_lorentz:
            _unsupported.append("lorentz dispersion")
        if aniso_eps is not None:
            _unsupported.append("aniso_eps (subpixel/anisotropic eps)")
        if aniso_inv_eps is not None:
            _unsupported.append("aniso_inv_eps (Kottke inverse-eps tensor)")
        if use_conformal:
            _unsupported.append("conformal PEC")
        if use_kerr:
            _unsupported.append("Kerr nonlinearity")
        if _unsupported:
            raise NotImplementedError(
                "stencil_order=4 is only supported on the plain uniform "
                "Cartesian vacuum/dielectric path (pec/periodic boundary, "
                "default solver, no dispersion/anisotropy/conformal/Kerr). "
                "Unsupported feature(s) present: " + ", ".join(_unsupported)
                + ". Use stencil_order=2 (the default) for these."
            )
        # Derate the timestep for the (2,4) stability bound.
        dt = dt * _ORDER4_CFL_FACTOR

    # Lazily-imported helper callables (imported now so they can be passed
    # into _StepContext and survive JIT reuse across calls).
    apply_cpml_h = apply_cpml_e = None
    apply_upml_h = apply_upml_e = None
    cpml_params = None
    upml_coeffs = None
    apply_tfsf_h = apply_tfsf_e = None
    update_tfsf_1d_h = update_tfsf_1d_e = None
    update_tfsf_2d_h = update_tfsf_2d_e = None
    _tfsf_is_2d = False
    tfsf_cfg = None
    _bloch = None
    init_ntff_data_fn = accumulate_ntff_fn = None
    apply_kerr_ade = None
    update_rlc_element = None
    debye_coeffs = lorentz_coeffs = None

    # ---- initialise states ----
    _field_dtype = field_dtype if field_dtype is not None else jnp.float32
    if _oblique_bloch:
        # Complex envelope P; overrides mixed-precision (float16) for this path.
        _field_dtype = jnp.complex64
    fdtd = init_state(grid.shape, field_dtype=_field_dtype)
    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml
        from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
        # psi carry dtype follows the field dtype so the scan carry stays
        # dtype-consistent when fields are complex (#404 oblique path).
        cpml_params, cpml_state = init_cpml(grid, field_dtype=_field_dtype)
        carry_init["cpml"] = cpml_state
    elif use_upml:
        from rfx.boundaries.upml import init_upml
        from rfx.boundaries.upml import apply_upml_e, apply_upml_h
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
        _tfsf_is_2d = is_tfsf_2d(tfsf_cfg)
        if _tfsf_is_2d:
            from rfx.sources.tfsf_2d import (
                update_tfsf_2d_h, update_tfsf_2d_e, bloch_phase_tuple,
            )
            # Per-axis Bloch phase on the 3D periodic roll (transverse axis),
            # matching the 2D-aux grid's transform (#404).
            _bloch = bloch_phase_tuple(tfsf_cfg, grid.dx)

    if use_ntff:
        from rfx.farfield import init_ntff_data as _init_ntff_data
        from rfx.farfield import accumulate_ntff as _accumulate_ntff
        init_ntff_data_fn = _init_ntff_data
        accumulate_ntff_fn = _accumulate_ntff
        # NTFF accumulator dtype follows the field dtype with a float32 floor
        # (#646), mirroring the CPML psi carry above. accumulate_ntff pins its
        # phase arithmetic to whatever is allocated here, so the scan carry
        # closes under jax_enable_x64 instead of raising a dtype mismatch.
        carry_init["ntff"] = _init_ntff_data(ntff, field_dtype=_field_dtype)

    if use_dft_planes:
        carry_init["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)

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

    # Port V/I DFT accumulator dtype (wire + lumped).  ``v * phase`` inside the
    # scan body promotes to complex128 under a scoped ``jax_enable_x64`` (dt/dx
    # and the phase factor become float64/complex128), so a hardcoded complex64
    # accumulator trips the lax.scan carry-dtype contract there.  Deriving the
    # dtype from ``result_type(complex64, float64)`` keeps it complex64 with x64
    # OFF (byte-identical to the historical pin) and promotes to complex128 with
    # x64 ON — this is what makes forward(port_s11_freqs=...) differentiable
    # under scoped x64 (WP 4-E gate + the pre-existing eps_override AD lane).
    _sparam_acc_dtype = jnp.result_type(jnp.complex64, jnp.float64)

    wire_sparam_meta: tuple = ()
    if use_wire_sparams:
        # Initialize V, I, V_inc DFT accumulators per wire port.
        carry_init["wire_sparam_accs"] = tuple(
            (
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),  # i_dft
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),  # v_inc_dft
                # v_port_dft (issue #764): whole-port gap voltage
                # V_port = sum over LIVE cells of -E_c*dx.
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),
                # v_ref_dft (issue #683 x #764): PRE-injection drive-sample
                # reference for the #308-calibrated off-diagonal
                # decomposition (bit-identical to the historical v_dft).
                jnp.zeros(len(wp.freqs), dtype=_sparam_acc_dtype),
            )
            for wp in wire_port_sparams
        )
        wire_sparam_meta = tuple(wire_port_sparams)

    wire_refplane_meta: tuple = ()
    if use_wire_refplanes:
        # Initialize V, I(-dx/2), I(+dx/2) DFT accumulators per reference
        # plane (issue #313 opt-in; two planes per opted port).  Same
        # accumulator dtype and rect-DFT kernel as the port-cell channels.
        carry_init["wire_refplane_accs"] = tuple(
            (
                jnp.zeros(len(rp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(rp.freqs), dtype=_sparam_acc_dtype),  # i_minus
                jnp.zeros(len(rp.freqs), dtype=_sparam_acc_dtype),  # i_plus
            )
            for rp in wire_refplane_sparams
        )
        wire_refplane_meta = tuple(wire_refplane_sparams)

    lumped_sparam_meta: tuple = ()
    if use_lumped_sparams:
        # Initialize V, I DFT accumulators per lumped port (issue #72).
        # No V_inc accumulator needed: the wave decomposition
        # ``a = (-V + Z0·I)/(2√Z0)`` is exact regardless of source pulse shape.
        # The third channel is the PRE-injection drive sample (the
        # historical #72 v_dft, bit-for-bit), kept because the #308
        # off-diagonal incident wave is calibrated against it; the
        # physical V/I are accumulated post-injection.
        carry_init["lumped_sparam_accs"] = tuple(
            (
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # i_dft
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # v_ref_dft
            )
            for lp in lumped_port_sparams
        )
        lumped_sparam_meta = tuple(lumped_port_sparams)

    # Flux monitor carry + both meta flavours (8-field for decay, 11 for run).
    flux_meta_8: tuple = ()
    flux_meta_11: tuple = ()
    if use_flux_monitors:
        from rfx.probes.probes import _FLUX_COMPONENTS as _FC
        flux_meta_8 = tuple(
            (fm.axis, fm.index, fm.freqs, _FC[fm.axis],
             fm.lo1, fm.hi1, fm.lo2, fm.hi2)
            for fm in flux_monitors
        )
        flux_meta_11 = tuple(
            (fm.axis, fm.index, fm.freqs, _FC[fm.axis],
             fm.lo1, fm.hi1, fm.lo2, fm.hi2,
             fm.total_steps, fm.window, fm.window_alpha)
            for fm in flux_monitors
        )
        carry_init["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft) for fm in flux_monitors
        )

    rlc_meta: tuple = ()
    if use_lumped_rlc:
        from rfx.lumped import init_rlc_state, rlc_carry_dtype
        from rfx.lumped import update_rlc_element
        # Thread the ADE-carry dtype from the metas (WP 4-E).  Concrete run()
        # metas are all Python floats + float32 fields -> float32 (byte-identical
        # to the historical init_rlc_state() pin); a traced float64 component
        # value under scoped-x64 promotes the carry so the lax.scan carry
        # input/output dtypes agree.
        _rlc_dtype = rlc_carry_dtype(lumped_rlc, _field_dtype)
        carry_init["rlc_states"] = tuple(
            init_rlc_state(dtype=_rlc_dtype) for _ in lumped_rlc)
        rlc_meta = tuple(lumped_rlc)

    if use_kerr:
        from rfx.materials.nonlinear import apply_kerr_ade

    # ---- metadata tuples ----
    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    mag_src_meta = [(s.i, s.j, s.k, s.component) for s in mag_sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]
    dft_meta = tuple(
        (probe.component, probe.axis, probe.index, probe.freqs, probe.region)
        for probe in dft_planes
    )
    waveguide_meta = tuple(waveguide_ports)

    # ---- #1179 design box: fence, then build its coefficients once ----
    design_box_coeffs = None
    if design_box is not None:
        design_box_coeffs = _resolve_design_box(
            design_box,
            grid=grid,
            materials=materials,
            dt=dt,
            periodic=periodic,
            use_cpml=use_cpml,
            use_upml=use_upml,
            cpml_axes=cpml_axes,
            use_debye=use_debye,
            use_lorentz=use_lorentz,
            use_kerr=use_kerr,
            aniso_eps=aniso_eps,
            aniso_inv_eps=aniso_inv_eps,
            stencil_order=stencil_order,
            bloch=_bloch,
            sheet_impedance=sheet_impedance,
            cell_metas=(
                ("source", [(s.i, s.j, s.k) for s in sources]),
                ("magnetic source", [(s.i, s.j, s.k) for s in mag_sources]),
                ("lumped port", [(p.i, p.j, p.k) for p in lumped_sparam_meta]),
                ("wire port", [
                    cell
                    for p in wire_sparam_meta
                    for cell in (tuple(p.live_cells)
                                 or ((p.mid_i, p.mid_j, p.mid_k),))
                ]),
                ("lumped RLC element", [(m.i, m.j, m.k) for m in rlc_meta]),
            ),
        )

    # ---- #1183 design occupancy: fence, then build its 1 - M once ----
    design_occupancy_keep = None
    if design_occupancy is not None:
        design_occupancy_keep = _resolve_design_occupancy(
            design_occupancy,
            grid=grid,
            periodic=periodic,
            pec_occupancy=pec_occupancy,
            field_dtype=_field_dtype,
        )

    # ---- shared _StepContext keyword arguments ----
    # These are identical for both drivers.  Each driver extends this dict
    # with its own overrides before calling _StepContext(**ctx_kwargs).
    ctx_kwargs: dict = dict(
        grid=grid,
        materials=materials,
        dt=dt,
        dx=dx,
        periodic=periodic,
        pec_axes=pec_axes,
        stencil_order=stencil_order,
        use_upml=use_upml,
        use_cpml=use_cpml,
        use_tfsf=use_tfsf,
        use_debye=use_debye,
        use_lorentz=use_lorentz,
        use_ntff=use_ntff,
        use_dft_planes=use_dft_planes,
        use_flux_monitors=use_flux_monitors,
        use_waveguide_ports=use_waveguide_ports,
        use_pec_faces=use_pec_faces,
        use_pmc_faces=use_pmc_faces,
        use_aniso_inv=use_aniso_inv,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        use_pec_edges=use_pec_edges,
        use_pec_occupancy=use_pec_occupancy,
        use_conformal=use_conformal,
        use_wire_sparams=use_wire_sparams,
        use_lumped_sparams=use_lumped_sparams,
        use_wire_refplanes=use_wire_refplanes,
        use_lumped_rlc=use_lumped_rlc,
        use_kerr=use_kerr,
        use_mag_sources=use_mag_sources,
        use_sheet_impedance=sheet_impedance is not None,
        sheet_impedance=sheet_impedance,
        use_design_box=design_box_coeffs is not None,
        design_box=design_box_coeffs,
        use_design_occupancy=design_occupancy_keep is not None,
        design_occupancy=design_occupancy_keep,
        cpml_params=cpml_params,
        cpml_axes=cpml_axes,
        upml_coeffs=upml_coeffs,
        tfsf_cfg=tfsf_cfg,
        tfsf_is_2d=_tfsf_is_2d,
        bloch=_bloch,
        debye_coeffs=debye_coeffs,
        lorentz_coeffs=lorentz_coeffs,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        pec_edge_masks=pec_edge_masks,
        pec_static_edge_masks=pec_static_edge_masks,
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        kerr_chi3=kerr_chi3,
        ntff=ntff,
        pec_faces_frozen=_pec_faces_frozen,
        pmc_faces_frozen=_pmc_faces_frozen,
        src_meta=tuple(src_meta),
        mag_src_meta=tuple(mag_src_meta),
        prb_meta=tuple(prb_meta),
        dft_meta=dft_meta,
        waveguide_meta=waveguide_meta,
        wire_sparam_meta=wire_sparam_meta,
        lumped_sparam_meta=lumped_sparam_meta,
        wire_refplane_meta=wire_refplane_meta,
        rlc_meta=rlc_meta,
        apply_cpml_h=apply_cpml_h,
        apply_cpml_e=apply_cpml_e,
        apply_upml_h=apply_upml_h,
        apply_upml_e=apply_upml_e,
        apply_tfsf_h=apply_tfsf_h,
        apply_tfsf_e=apply_tfsf_e,
        update_tfsf_1d_h=update_tfsf_1d_h,
        update_tfsf_1d_e=update_tfsf_1d_e,
        update_tfsf_2d_h=update_tfsf_2d_h if _tfsf_is_2d else None,
        update_tfsf_2d_e=update_tfsf_2d_e if _tfsf_is_2d else None,
        init_ntff_data=init_ntff_data_fn,
        accumulate_ntff=accumulate_ntff_fn,
        apply_kerr_ade=apply_kerr_ade if use_kerr else None,
        update_rlc_element=update_rlc_element if use_lumped_rlc else None,
    )

    return _SimSetup(
        carry_init=carry_init,
        ctx_kwargs=ctx_kwargs,
        src_meta=src_meta,
        mag_src_meta=mag_src_meta,
        prb_meta=prb_meta,
        dft_meta=dft_meta,
        waveguide_meta=waveguide_meta,
        wire_sparam_meta=wire_sparam_meta,
        lumped_sparam_meta=lumped_sparam_meta,
        rlc_meta=rlc_meta,
        wire_refplane_meta=wire_refplane_meta,
        flux_meta_8=flux_meta_8,
        flux_meta_11=flux_meta_11,
        dt=dt,
        dx=dx,
        periodic=periodic,
        pec_axes=pec_axes,
        pec_faces=_pec_faces_frozen,
    )


# ---------------------------------------------------------------------------
# Shared Yee scan body (W6.1)
# ---------------------------------------------------------------------------
#
# ``run()`` (jax.lax.scan) and ``run_until_decay()`` (Python loop + jax.jit)
# ran two ~85%-identical copies of the per-step Yee kernel.  ``make_core_step``
# is the single source of truth.  Both call sites build a ``_StepContext`` from
# their own setup code, then:
#   * ``run()``           wraps ``core`` in a scan body that unpacks ``xs`` and
#                         assembles the scan output tuple (probe + snapshot).
#   * ``run_until_decay`` calls ``core`` directly inside its Python loop and
#                         reads ``extras["monitor_val"]`` for the decay check.
#
# Every free variable of the old closures is passed explicitly via the context
# (no capture of caller locals) so the builder is unit-testable.  Numerics,
# Yee sub-step ordering, and dtype casts are unchanged — this is pure code
# motion.  The two documented behavioural differences are parameterised:
#   * ``use_fast_he`` / ``fast_coeffs`` — GPU baked-PEC path (run() only today).
#   * ``use_flux_window`` + 11-field ``flux_meta`` — streaming DFT window for
#     flux monitors.  ``run_until_decay`` passes ``use_flux_window=False`` and
#     an 8-field ``flux_meta`` to keep its historical rect-window (no weight)
#     behaviour bit-identical.  Unifying the decay path onto windows is a
#     deliberate follow-up, not part of this refactor.


@dataclass(frozen=True)
class _StepContext:
    """Static + array context for the shared Yee step kernel (W6.1).

    Holds every free variable the per-step kernel needs: physical constants,
    the resolved ``use_*`` subsystem flags, pre-extracted metadata tuples, and
    the boundary/dispersion helper callables that each caller imports during
    setup.  Lazily-imported helpers (waveguide-port, PMC, conformal, pec-mask,
    NTFF accumulate, RLC update) are re-imported inside the kernel body exactly
    as the original closures did and are therefore NOT stored here.
    """

    # ---- physical constants / grid ----
    grid: Any
    materials: Any
    dt: Any
    dx: Any
    periodic: tuple
    pec_axes: str
    # (2,4) fourth-order-in-space stencil order: 2 (default, byte-identical)
    # or 4. Threaded into the H / E kernel calls in core_step.
    stencil_order: int

    # ---- subsystem flags ----
    use_fast_he: bool
    use_upml: bool
    use_cpml: bool
    use_tfsf: bool
    use_debye: bool
    use_lorentz: bool
    use_ntff: bool
    use_dft_planes: bool
    use_flux_monitors: bool
    use_waveguide_ports: bool
    use_pec_faces: bool
    use_pmc_faces: bool
    use_aniso_inv: bool
    aniso_inv_eps_smooth: bool
    use_pec_edges: bool
    use_pec_occupancy: bool
    use_conformal: bool
    use_wire_sparams: bool
    use_lumped_sparams: bool
    use_lumped_rlc: bool
    use_kerr: bool
    use_mag_sources: bool

    # ---- output behaviour ----
    use_snapshot: bool
    use_monitor: bool
    use_flux_window: bool

    # ---- arrays / coeffs / configs ----
    fast_coeffs: Any = None
    cpml_params: Any = None
    cpml_axes: str = ""
    upml_coeffs: Any = None
    tfsf_cfg: Any = None
    tfsf_is_2d: bool = False
    bloch: Any = None  # per-axis Bloch phase for oblique-periodic TFSF (#404); None = real path
    debye_coeffs: Any = None
    lorentz_coeffs: Any = None
    aniso_eps: Any = None
    aniso_inv_eps: Any = None
    pec_edge_masks: Any = None
    pec_static_edge_masks: Any = None
    pec_occupancy: Any = None
    conformal_weights: Any = None
    kerr_chi3: Any = None
    ntff: Any = None
    pec_faces_frozen: Any = frozenset()
    pmc_faces_frozen: Any = frozenset()

    # ---- pre-extracted metadata ----
    src_meta: tuple = ()
    mag_src_meta: tuple = ()
    prb_meta: tuple = ()
    dft_meta: tuple = ()
    flux_meta: tuple = ()
    waveguide_meta: tuple = ()
    wire_sparam_meta: tuple = ()
    lumped_sparam_meta: tuple = ()
    rlc_meta: tuple = ()
    # issue #313 opt-in reference-plane channel (defaults keep every
    # existing caller byte-identical)
    use_wire_refplanes: bool = False
    wire_refplane_meta: tuple = ()
    # issue #677 node-thin surface-impedance sheet operator (defaults keep
    # every existing caller byte-identical)
    use_sheet_impedance: bool = False
    sheet_impedance: Any = None
    # issue #1179 design-box permittivity (defaults keep every existing
    # caller byte-identical)
    use_design_box: bool = False
    design_box: Any = None
    # issue #1183 design-box PEC occupancy (same default discipline)
    use_design_occupancy: bool = False
    design_occupancy: Any = None

    # ---- output extractors ----
    monitor_component: str = "ez"
    mon_idx: Any = None
    snapshot_extractor: Callable | None = None

    # ---- caller-imported helper callables ----
    apply_cpml_h: Callable | None = None
    apply_cpml_e: Callable | None = None
    apply_upml_h: Callable | None = None
    apply_upml_e: Callable | None = None
    apply_tfsf_h: Callable | None = None
    apply_tfsf_e: Callable | None = None
    update_tfsf_1d_h: Callable | None = None
    update_tfsf_1d_e: Callable | None = None
    update_tfsf_2d_h: Callable | None = None
    update_tfsf_2d_e: Callable | None = None
    init_ntff_data: Callable | None = None
    accumulate_ntff: Callable | None = None
    apply_kerr_ade: Callable | None = None
    update_rlc_element: Callable | None = None


def make_core_step(ctx: _StepContext):
    """Build the shared per-step Yee kernel from an explicit context.

    Returns ``core_step(carry, step_idx, src_vals, mag_src_vals)`` ->
    ``(new_carry, probe_out, extras)`` where ``extras`` is a dict carrying the
    optional per-step outputs each caller needs:
      * ``extras["snap_fields"]`` — snapshot field list (``use_snapshot``).
      * ``extras["monitor_val"]`` — monitored scalar (``use_monitor``).
    """
    materials = ctx.materials
    dt = ctx.dt
    dx = ctx.dx
    periodic = ctx.periodic
    grid = ctx.grid
    aniso_eps = ctx.aniso_eps
    aniso_inv_eps = ctx.aniso_inv_eps

    # #1043. ``apply_cpml_e`` builds its psi coefficient from a permittivity,
    # and it has to be the SAME one the Yee half of this timestep used, or the
    # two halves integrate different media and the combined update can
    # amplify (see that function's ``inv_eps_r_update`` docstring for the
    # derivation and the measured spectral radius). ``None`` on every path
    # that has no anisotropic array, which keeps those byte-identical.
    # The guard mirrors ``_update_e_with_optional_dispersion``'s own
    # ``debye is None and lorentz is None``: with a dispersion model active the
    # E update never consults the anisotropic arrays, so neither may this.
    #
    # #1210 made the PLAIN path per-component too: ``update_e`` builds its
    # coefficients from the mean of eps_r over each edge's four incident
    # cells, so ``materials.eps_r`` is no longer the permittivity the Yee half
    # used anywhere a material interface crosses the pad. The same argument
    # that threaded the subpixel arrays threads this one; where the pad is
    # homogeneous the mean IS ``materials.eps_r``, so those runs keep their
    # bytes.
    #
    # #1260 made the DISPERSIVE update per-component as well: its ε_∞ is the
    # same ``component_e_materials`` mean, so a dispersive run threads the same
    # array. (It used to fall back to the cell's ``materials.eps_r``.)
    _aniso_is_live = not (ctx.use_debye or ctx.use_lorentz)
    if _aniso_is_live and aniso_inv_eps is not None:
        cpml_inv_eps_r = aniso_inv_eps
    elif _aniso_is_live and aniso_eps is not None:
        cpml_inv_eps_r = tuple(1.0 / e for e in aniso_eps)
    else:
        _eps_edge, _ = component_e_materials(materials, periodic)
        cpml_inv_eps_r = tuple(1.0 / e for e in _eps_edge)

    # #677 surface-impedance sheet: Holland exponential-stepping A/B built
    # once from the FINAL run materials (background eps_r/sigma at the sheet
    # cells + the ctx's sigma_sheet), applied per step at tangential edges.
    if ctx.use_sheet_impedance:
        from rfx.materials.thin_conductor import (
            apply_sheet_impedance_e as _apply_sheet_e,
            sheet_update_coeffs as _sheet_update_coeffs,
        )
        _sheet_coeffs = _sheet_update_coeffs(
            ctx.sheet_impedance.sigma_sheet, materials, dt)

    def core_step(carry, step_idx, src_vals, mag_src_vals):
        st = carry["fdtd"]
        tfsf_h_state = None
        # #1163: a series RLC element is solved together with its edge field
        # and needs E^n there; nothing before the E update writes E, so the
        # carry's value IS E^n on every path (fast H+E kernel included).
        rlc_e_prev = (
            tuple(getattr(st, m.component)[m.i, m.j, m.k]
                  for m in ctx.rlc_meta)
            if ctx.use_lumped_rlc else ())

        if ctx.use_fast_he:
            # Fast path: combined H+E update with PEC baked into
            # pre-computed coefficients — eliminates separate apply_pec(),
            # coefficient recomputation, and reduces XLA scatter ops.
            st = update_he_fast(st, ctx.fast_coeffs)
        else:
            # H update
            if ctx.use_upml:
                st = ctx.apply_upml_h(st, ctx.upml_coeffs, periodic=periodic)
            else:
                st = update_h(st, materials, dt, dx, periodic=periodic,
                              stencil_order=ctx.stencil_order, bloch=ctx.bloch)
            if ctx.use_tfsf:
                st = ctx.apply_tfsf_h(st, ctx.tfsf_cfg, carry["tfsf"], dx, dt)
            if ctx.use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_h as _apply_wg_h
                for cfg_meta in ctx.waveguide_meta:
                    st = _apply_wg_h(st, cfg_meta, step_idx, dt, dx)
            if ctx.use_cpml:
                st, cpml_new = ctx.apply_cpml_h(
                    st, ctx.cpml_params, carry["cpml"], grid, ctx.cpml_axes,
                    materials=materials)
            # Stage 2 H damping — applied AFTER CPML-H so CPML cannot
            # un-zero H at Kottke-frozen PEC cells.  Threshold rather
            # than ``== 0.0`` so smooth-Kottke (eps_inside = 1e10)
            # cells (inv ≈ 1e-10 at f=1) are caught alongside exact-PEC
            # cells (inv = 0 from binary `pec_shapes`).  Smooth-Kottke
            # path uses ONLY the full-PEC mask (all three inv below
            # threshold); the pairwise per-component masks would trigger
            # spuriously at sigmoid-edge cells (where 2 of 3 components
            # are frozen due to Kottke anisotropy but the cell is
            # legitimately a partial-fill interface, not full PEC) —
            # killing wave propagation INTO the stub region.  Binary
            # Stage 2 path keeps the pairwise masks as before (boundary
            # cells of binary PEC need the corner-specific H zero).
            if ctx.use_aniso_inv:
                from rfx.boundaries.pec import apply_pec_h_mask
                _inv_xx, _inv_yy, _inv_zz = aniso_inv_eps
                _PEC_INV_THRESHOLD = 1e-9
                _xx0 = (_inv_xx < _PEC_INV_THRESHOLD)
                _yy0 = (_inv_yy < _PEC_INV_THRESHOLD)
                _zz0 = (_inv_zz < _PEC_INV_THRESHOLD)
                if ctx.aniso_inv_eps_smooth:
                    st = apply_pec_h_mask(
                        st,
                        pec_mask=_xx0 & _yy0 & _zz0,
                    )
                else:
                    st = apply_pec_h_mask(
                        st,
                        pec_mask=_xx0 & _yy0 & _zz0,
                        mask_hx=_yy0 & _zz0,
                        mask_hy=_xx0 & _zz0,
                        mask_hz=_xx0 & _yy0,
                    )
            if ctx.use_pmc_faces:
                from rfx.boundaries.pmc import apply_pmc_faces
                st = apply_pmc_faces(st, ctx.pmc_faces_frozen)
            if ctx.use_tfsf:
                if ctx.tfsf_is_2d:
                    tfsf_h_state = ctx.update_tfsf_2d_h(ctx.tfsf_cfg, carry["tfsf"], dx, dt)
                else:
                    tfsf_h_state = ctx.update_tfsf_1d_h(ctx.tfsf_cfg, carry["tfsf"], dx, dt)

            # Magnetic current (Schelkunoff M / J-magnetic) injection —
            # applied after H update so the Yee leapfrog ordering is
            # H^{n+1/2} += -dt/mu · M^{n+1/2}. The coefficient is
            # pre-baked into the waveform values at construction time.
            if ctx.use_mag_sources:
                for idx_m, (mi, mj, mk, mc) in enumerate(ctx.mag_src_meta):
                    h_field = getattr(st, mc)
                    h_field = h_field.at[mi, mj, mk].add(
                        mag_src_vals[idx_m].astype(h_field.dtype))
                    st = st._replace(**{mc: h_field})

            # Snapshot E^n before the linear E-update for the reactive Kerr
            # increment (#437): E^{n+1} = E^n + (E_lin - E^n)/(1 + chi3|E^n|^2/eps_r).
            e_prev_kerr = (st.ex, st.ey, st.ez) if ctx.use_kerr else None
            # Snapshot E^n for the #677 sheet operator (it REPLACES the
            # standard update at masked tangential edges with
            # A*E^n + B*curlH, so it needs the pre-update E).
            e_prev_sheet = (
                (st.ex, st.ey, st.ez) if ctx.use_sheet_impedance else None)
            # #1179: the design box REDOES the E update at its own cells from
            # the pre-update state.  The whole state, because it needs E^n and
            # the same H^{n+1/2} the update below consumes; H is not touched
            # between the two, and a NamedTuple alias costs nothing.
            st_prev_design = st if ctx.use_design_box else None

            if ctx.use_upml:
                if ctx.use_debye or ctx.use_lorentz:
                    raise ValueError("boundary='upml' does not yet support dispersion")
                st = ctx.apply_upml_e(st, ctx.upml_coeffs, periodic=periodic)
                debye_new = None
                lorentz_new = None
            else:
                st, debye_new, lorentz_new = _update_e_with_optional_dispersion(
                    st,
                    materials,
                    dt,
                    dx,
                    debye=(ctx.debye_coeffs, carry["debye"]) if ctx.use_debye else None,
                    lorentz=(ctx.lorentz_coeffs, carry["lorentz"]) if ctx.use_lorentz else None,
                    periodic=periodic,
                    aniso_eps=aniso_eps,
                    aniso_inv_eps=aniso_inv_eps,
                    stencil_order=ctx.stencil_order,
                    bloch=ctx.bloch,
                )

            # #1179 design box: redo the E update at the design cells with
            # coefficients built from the traced permittivity, leaving the
            # grid-wide ``materials`` constant.  Slot: immediately after the
            # E update, on the same H, before anything that reads or writes
            # E.  The fences in _build_step_setup guarantee that nothing
            # later in this step reads ``materials`` at a design cell.
            if ctx.use_design_box:
                st = update_e_box(
                    st, st_prev_design, ctx.design_box.bounds,
                    ctx.design_box.ca, ctx.design_box.cb, dx,
                    periodic=periodic,
                    stencil_order=ctx.stencil_order,
                    bloch=ctx.bloch,
                )

            # Reactive Kerr correction: scale the E-increment by eps_r/eps_eff (#437).
            if ctx.use_kerr:
                st = ctx.apply_kerr_ade(st, e_prev_kerr, ctx.kerr_chi3, materials.eps_r)

            if ctx.use_tfsf:
                st = ctx.apply_tfsf_e(st, ctx.tfsf_cfg, tfsf_h_state, dx, dt)
            if ctx.use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e
                for cfg_meta in ctx.waveguide_meta:
                    st = _apply_wg_e(st, cfg_meta, step_idx, dt, dx)
            if ctx.use_cpml:
                st, cpml_new = ctx.apply_cpml_e(
                    st, ctx.cpml_params, cpml_new, grid, ctx.cpml_axes,
                    materials=materials,
                    inv_eps_r_update=cpml_inv_eps_r)
            # Re-enforce Kottke-frozen E cells after CPML-E correction.
            # CPML adds a psi-driven correction that can thaw cells
            # where inv_eps==0; re-zero them here so the frozen
            # boundary condition is not violated.
            if ctx.use_aniso_inv:
                _inv_xx_r, _inv_yy_r, _inv_zz_r = aniso_inv_eps
                _PEC_INV_THRESHOLD = 1e-9
                st = st._replace(
                    ex=jnp.where(_inv_xx_r < _PEC_INV_THRESHOLD, 0.0, st.ex),
                    ey=jnp.where(_inv_yy_r < _PEC_INV_THRESHOLD, 0.0, st.ey),
                    ez=jnp.where(_inv_zz_r < _PEC_INV_THRESHOLD, 0.0, st.ez),
                )

            if ctx.use_pec_faces:
                st = apply_pec_faces(st, ctx.pec_faces_frozen)

            if ctx.use_conformal and not ctx.use_aniso_inv:
                # Stage 1 path. Stage 2 (use_aniso_inv) skips this —
                # the inv-eps tensor encodes the fully-PEC-cell zero
                # already, so this would be redundant double-zeroing.
                from rfx.geometry.conformal import apply_conformal_pec
                st = apply_conformal_pec(st, ctx.conformal_weights[0], ctx.conformal_weights[1], ctx.conformal_weights[2])
            if ctx.use_pec_edges:
                # #931 §1.7: the (Mx, My, Mz) realized once at setup.
                #
                # NOT an ``elif`` on the conformal branch. Dey-Mittra is a
                # subpixel UPDATE-COEFFICIENT model, not a second geometry
                # realization: ``apply_conformal_pec`` zeroes only edges
                # whose weight is exactly 0, and no edge of a one-cell PEC
                # slab is fully covered — both its faces sit ON the slab's
                # own boundary, so w = 1/2 there. While the waveguide
                # S-matrix lane folded interior PEC into sigma=1e10 the
                # conductor survived anyway; with that fold deleted (#931)
                # the ``elif`` dropped it outright — measured on the
                # conformal PEC-short battery, min|S11| 0.2296 against a
                # gate of 0.99, restored to 0.9942 by applying both.
                st = apply_pec_edges(st, ctx.pec_edge_masks)

            # #1183: the design occupancy REDOES the scaling at its window
            # from the field BEFORE the grid-wide one, for the same reason
            # the design box keeps the pre-update state -- the multiply is
            # not invertible where the static factor is 0.
            st_prev_occ = st if ctx.use_design_occupancy else None

            if ctx.use_pec_occupancy:
                st = apply_pec_occupancy(
                    st, ctx.pec_occupancy, ctx.periodic,
                    sheet_edge_masks=ctx.pec_static_edge_masks)

            # #1183 design occupancy: the traced 1 - M on its window, over
            # a field the static occupancy has not scaled. Outside the
            # window no design cell can reach, so the grid-wide factor
            # there is already the right one.
            if ctx.use_design_occupancy:
                st = apply_pec_occupancy_box(
                    st, st_prev_occ, ctx.design_occupancy.write,
                    ctx.design_occupancy.keep)

            # #677 node-thin surface-impedance sheet operator. Contract slot:
            # AFTER apply_pec_mask/apply_pec_occupancy (PEC wins on overlap —
            # PEC-owned edges are already excluded from the ctx masks at
            # build time, and running after keeps the ordering honest),
            # BEFORE the S-param DFT sampling and source injection below.
            # curlH comes from the SAME shared stencil helper update_e uses,
            # on the same H^{n+1/2} the E update consumed (H is unchanged
            # between the E update and this slot).
            if ctx.use_sheet_impedance:
                from rfx.core.yee import curl_h as _curl_h
                _scd = jnp.promote_types(st.ex.dtype, jnp.float32)
                _curls = _curl_h(
                    st.hx.astype(_scd), st.hy.astype(_scd),
                    st.hz.astype(_scd), dx, periodic,
                    ctx.stencil_order, ctx.bloch)
                st = _apply_sheet_e(
                    st, e_prev_sheet, _curls, ctx.sheet_impedance,
                    _sheet_coeffs)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        if ctx.use_lumped_rlc:
            new_rlc_states = []
            for rlc_st, meta, e_prev in zip(
                    carry["rlc_states"], ctx.rlc_meta, rlc_e_prev):
                st, rlc_st_new = ctx.update_rlc_element(
                    st, rlc_st, meta, e_prev)
                new_rlc_states.append(rlc_st_new)

        # Compute step time first; the lumped S-param DFT block below
        # needs `t` and accumulates BEFORE source injection per the
        # rfx/probes/probes.py update_sparam_probe docstring contract
        # ("sample after E-update/apply_pec but before apply_lumped_port
        # so V reflects only the cavity/load response, not the driving
        # waveform"; issue #72 — the JIT scan path violated it via PR #72
        # ordering, producing 5-10 dB train/eval |S11| disagreement on
        # near-matched antennas). The WIRE-port physical V/I/V_port block
        # moved to AFTER the soft-source loop below (issue #683, decided
        # by measurement 2026-08-29: only the post-injection sample is the
        # true field level E^{n+1} and satisfies the known-load circuit
        # law at an excited port — see
        # docs/design_notes/issue683_sampling_order_decision_protocol.md
        # section 9). The lumped ordering is deliberately unchanged: the
        # #683 measurement was made on wire ports, and flipping lumped on
        # a wire-port measurement would repeat the align-first-decide-
        # later mistake the ledger's #673/#672 entry warns about.
        t = step_idx.astype(jnp.float32) * dt

        # Wire-port DRIVE-REFERENCE DFT accumulation at the historical
        # PRE-injection slot (issue #683 x #764 decomposer recalibration,
        # docs/design_notes/issue683_decomposer_flip_predeclaration.md
        # section 3): the #308 receive-wave sign and Z0/n_cells
        # normalization were calibrated against the pre-injection drive
        # sample, so that sample is kept as its own channel `v_ref_dft` —
        # bit-identical to the historical `v_dft` — and feeds ONLY the
        # off-diagonal incident-wave denominator and the byte-frozen
        # legacy diagonal in decompose_wire_s_matrix. The physical
        # channels (v, i, v_port) are accumulated POST-injection below.
        if ctx.use_wire_sparams or ctx.use_lumped_sparams:
            from rfx.probes.probes import _ampere_loop
            # Both families advance their H-derived current by dt/2 to the
            # E time level; one import for both blocks.
            from rfx.core.dft_utils import half_step_current_phase as _half_i_phase
        if ctx.use_wire_sparams:
            new_wire_refs = []
            for accs, wp_meta in zip(carry["wire_sparam_accs"], ctx.wire_sparam_meta):
                v_ref_dft = accs[4]
                mi, mj, mk = wp_meta.mid_i, wp_meta.mid_j, wp_meta.mid_k
                v_ref = -getattr(st, wp_meta.component)[mi, mj, mk] * dx
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase = jnp.exp(-1j * 2.0 * jnp.pi * wp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_wire_refs.append((v_ref_dft + v_ref * phase, phase))

        # Lumped-port DRIVE-REFERENCE DFT accumulation at the historical
        # PRE-injection slot (issue #72): the #308 off-diagonal incident
        # wave is calibrated against this sample, so it is kept as its own
        # channel — bit-identical to the pre-decision `v_dft` — and feeds
        # ONLY the off-diagonal denominator in decompose_lumped_s_matrix.
        # The physical V/I are accumulated POST-injection below.  Mirrors
        # the wire-port block above (issue #683).
        if ctx.use_lumped_sparams:
            new_lumped_refs = []
            for accs, lp_meta in zip(carry["lumped_sparam_accs"], ctx.lumped_sparam_meta):
                v_ref_dft_l = accs[2]
                li, lj, lk = lp_meta.i, lp_meta.j, lp_meta.k
                v_ref_l = -getattr(st, lp_meta.component)[li, lj, lk] * dx
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase_l = jnp.exp(-1j * 2.0 * jnp.pi * lp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_lumped_refs.append((v_ref_dft_l + v_ref_l * phase_l, phase_l))

        # Reference-plane V/I DFT accumulation (issue #313 opt-in) — same
        # rect-DFT kernel as the port-cell channels.  This slot is before
        # the soft-source loop, but the planes sit >= 1 cell from every
        # source cell, so pre/post-injection sampling is identical here
        # and the #683 wire-port flip (which moved only the port-CELL
        # physical channels below the source loop) does not apply.
        if ctx.use_wire_refplanes:
            from rfx.probes.refplane import wire_refplane_step_vi
            new_refplane_accs = []
            for accs, rp_meta in zip(carry["wire_refplane_accs"],
                                     ctx.wire_refplane_meta):
                v_dft_r, im_dft_r, ip_dft_r = accs
                v_r, im_r, ip_r = wire_refplane_step_vi(st, rp_meta, dx)
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase_r = jnp.exp(-1j * 2.0 * jnp.pi * rp_meta.freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_refplane_accs.append((
                    v_dft_r + v_r * phase_r,
                    im_dft_r + im_r * phase_r,
                    ip_dft_r + ip_r * phase_r,
                ))

        # Soft sources — cast source value to field dtype to avoid
        # mixed-precision scatter warnings (float32 -> float16).
        for idx_s, (si, sj, sk, sc) in enumerate(ctx.src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s].astype(field.dtype))
            st = st._replace(**{sc: field})

        # Wire-port PHYSICAL V/I/V_port DFT accumulation AFTER soft-source
        # injection (issue #683, decided by measurement 2026-08-29;
        # protocol + results in docs/design_notes/
        # issue683_sampling_order_decision_protocol.md, recalibration in
        # issue683_decomposer_flip_predeclaration.md).  This block used to
        # sit BEFORE the source loop (the #72 wire slot).  The #683
        # known-load experiment refuted that sample at an EXCITED port:
        # pre-injection Re(V/I) does not track a known external load
        # (slope -0.62, intercept -81 Ohm vs the circuit law's +1/n_live
        # slope), and the pre-injection E is not any field time level of
        # the discrete update (Ampere-identity residual 3.25 vs 2.3e-7
        # post).  The post-injection sample here IS the true E^{n+1}: the
        # soft-source loop above is the last write to E in the step (the
        # TFSF call below only advances the auxiliary 1D/2D state), and
        # `t` stamping is unchanged (phase computed at the pre slot and
        # reused), so a PASSIVE port (no source at its own cells) reads
        # bit-identically to the old slot.  This also makes the uniform
        # lane agree with the NU lane (rfx/nonuniform.py), whose
        # post-injection slot the #683 run validated.
        if ctx.use_wire_sparams:
            new_wire_accs = []
            for accs, wp_meta, (v_ref_new, phase) in zip(
                    carry["wire_sparam_accs"], ctx.wire_sparam_meta,
                    new_wire_refs):
                v_dft, i_dft, vinc_dft, v_port_dft = accs[0], accs[1], accs[2], accs[3]
                mi, mj, mk = wp_meta.mid_i, wp_meta.mid_j, wp_meta.mid_k
                field_c = getattr(st, wp_meta.component)
                v = -field_c[mi, mj, mk] * dx
                # Whole-port gap voltage (issue #764): the discrete line
                # integral of E across the LIVE run,
                # V_port = sum_live(-E_c*dx).  Only this SUM is
                # KVL/Faraday-constrained on the staggered grid; a single
                # cell is not (a PEC short forces sum V_c = 0 while V_mid
                # stays finite).  ``live_cells`` is a static tuple, so the
                # unroll resolves at trace time; an empty tuple (pre-#764
                # spec constructor) degenerates to the single midpoint cell.
                _lc = wp_meta.live_cells or ((mi, mj, mk),)
                v_port = -sum(
                    field_c[ci, cj, ck] for (ci, cj, ck) in _lc) * dx
                # #692: the SHARED loop, not a fourth inline copy. This block
                # used to spell the six branches verbatim with a raw
                # `h[i-1]`, so a port at index 0 read H from the OPPOSITE
                # face of the domain — and this lane is the one described at
                # the top of this file as "an AD-compatible alternative to
                # the Python-loop extract_s_matrix path", i.e. it must agree
                # with `probes.port_current` cell for cell. Measured against
                # the helper on random H: the two spellings disagreed at 9 of
                # 21 sampled (index, component) cells, every one of them at
                # an index with a zero coordinate on a back-read axis.
                # `_ampere_loop` is jit-safe here: `component` is a static
                # str and `mi/mj/mk` are static Python ints (see
                # WireSParamSpec), so every branch resolves at trace time.
                # (I reads H only, so it is identical on both sides of the
                # source loop; the flip moves V by the same-step injection
                # increment at driven cells, which #683's gate G2 measured
                # as EXACTLY the pre/post lane difference.)
                i_val = _ampere_loop(
                    st, (mi, mj, mk), wp_meta.component, dx, periodic)
                # Yee half-step: I is H-derived (H^{n+1/2}) while V/V_port are
                # E-derived (E^{n+1}); advance the current sample by dt/2 so
                # both DFT channels share a reference time
                # (`dft_utils.half_step_current_phase`). The
                # E-derived channels keep the uncorrected `phase`.
                i_phase = phase * _half_i_phase(
                    wp_meta.freqs.astype(jnp.float64), dt).astype(jnp.complex64)
                new_wire_accs.append((
                    v_dft + v * phase,
                    i_dft + i_val * i_phase,
                    vinc_dft,
                    v_port_dft + v_port * phase,
                    v_ref_new,
                ))

        # Lumped-port PHYSICAL V/I DFT accumulation AFTER source injection.
        # Decided by the lumped known-load decision run
        # (scripts/diagnostics/lumped_port_known_load_line.py), which the
        # 2026-09-05 scope note said was the missing input: on a
        # parallel-plate line terminated in a known R, the pre-injection
        # slot read |S11| 0.714 / 1.248 / 4.757 against a closed form of
        # 0.333 / 0 / 0.333, and its terminal V/(Zc·I) was -0.167 / +0.110
        # / +0.659 where the load is 0.5 / 1.0 / 2.0.  The one-cell WIRE
        # port on the SAME cell — same sigma (setup_wire_port with
        # n_live=1), same injection (apply_wire_port with n_live=1) — read
        # 0.334 / 0.0004 / 0.333 and 0.500 / 0.999 / 1.988.  The physics
        # was never the difference; the extraction lane was.  `t` stamping
        # is unchanged (phase computed at the pre slot and reused), so a
        # PASSIVE port reads bit-identically to the old slot in V.
        if ctx.use_lumped_sparams:
            new_lumped_accs = []
            for accs, lp_meta, (v_ref_new_l, phase_l) in zip(
                    carry["lumped_sparam_accs"], ctx.lumped_sparam_meta,
                    new_lumped_refs):
                v_dft_l, i_dft_l = accs[0], accs[1]
                li, lj, lk = lp_meta.i, lp_meta.j, lp_meta.k
                v_l = -getattr(st, lp_meta.component)[li, lj, lk] * dx
                # #692: shared loop — see the wire-port block above.
                i_val_l = _ampere_loop(
                    st, (li, lj, lk), lp_meta.component, dx, periodic)
                # Yee half-step: I is H-derived (H^{n+1/2}), V is E-derived
                # (E^{n+1}).  Withheld on this lane until the slot above
                # was post-injection, because that is the correction's
                # premise (2026-09-05 scope note).
                i_phase_l = phase_l * _half_i_phase(
                    lp_meta.freqs.astype(jnp.float64), dt).astype(jnp.complex64)
                new_lumped_accs.append((
                    v_dft_l + v_l * phase_l,
                    i_dft_l + i_val_l * i_phase_l,
                    v_ref_new_l,
                ))

        if ctx.use_tfsf:
            if ctx.tfsf_is_2d:
                tfsf_new = ctx.update_tfsf_2d_e(ctx.tfsf_cfg, tfsf_h_state, dx, dt, t)
            else:
                tfsf_new = ctx.update_tfsf_1d_e(ctx.tfsf_cfg, tfsf_h_state, dx, dt, t)

        if ctx.use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                update_waveguide_port_probe,
            )

            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(carry["waveguide_port_accs"], ctx.waveguide_meta):
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
                # NOTE: this samples `st` AFTER source injection above.
                # The same docstring-contract concern as wire/lumped
                # applies here, but waveguide-port is out of scope for
                # this fix (issue #29 OPEN tracks waveguide-port issues).
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
                   for pi, pj, pk, pc in ctx.prb_meta]
        probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

        # NTFF accumulation
        if ctx.use_ntff:
            ntff_new = ctx.accumulate_ntff(
                carry["ntff"], st, ctx.ntff, dt, step_idx)

        if ctx.use_dft_planes:
            t_plane = st.step * dt
            new_dft_planes = []
            for acc, (component, axis, index, freqs, region) in zip(
                carry["dft_planes"], ctx.dft_meta
            ):
                field = getattr(st, component)
                if region is None:
                    lo1 = lo2 = 0
                    if axis == 0:
                        hi1, hi2 = field.shape[1], field.shape[2]
                    elif axis == 1:
                        hi1, hi2 = field.shape[0], field.shape[2]
                    else:
                        hi1, hi2 = field.shape[0], field.shape[1]
                else:
                    lo1, hi1, lo2, hi2 = region
                if axis == 0:
                    plane = field[index, lo1:hi1, lo2:hi2]
                elif axis == 1:
                    plane = field[lo1:hi1, index, lo2:hi2]
                else:
                    plane = field[lo1:hi1, lo2:hi2, index]
                phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t_plane)
                new_dft_planes.append(
                    acc + plane[None, :, :] * phase[:, None, None] * dt
                )

        # Flux monitor DFT accumulation (co-located E/H, finite-size region).
        if ctx.use_flux_monitors:
            t_flux = st.step * dt
            new_flux_accs = []
            if ctx.use_flux_window:
                from rfx.core.dft_utils import dft_window_weight as _dft_w
            for (e1_acc, e2_acc, h1_acc, h2_acc), fmeta in zip(
                carry["flux_monitors"], ctx.flux_meta
            ):
                if ctx.use_flux_window:
                    (ax, idx, fqs, comp_names, _lo1, _hi1, _lo2, _hi2,
                     _tot_steps, _win_name, _win_alpha) = fmeta
                else:
                    (ax, idx, fqs, comp_names, _lo1, _hi1, _lo2, _hi2) = fmeta
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
                # E is at time t_flux = step*dt; H is at t_flux - dt/2
                phase_e = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * t_f64)
                phase_h = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * (t_f64 - jnp.float64(dt * 0.5)))
                if ctx.use_flux_window:
                    # Streaming DFT window weight (rect=1.0 default; Tukey/Hann
                    # suppress late-time contributions from CPML reflections).
                    _w = _dft_w(st.step, _tot_steps, _win_name, _win_alpha).astype(jnp.float64)
                    kernel_e = (phase_e[:, None, None] * dt * _w).astype(jnp.complex128)
                    kernel_h = (phase_h[:, None, None] * dt * _w).astype(jnp.complex128)
                else:
                    kernel_e = (phase_e[:, None, None] * dt).astype(jnp.complex128)
                    kernel_h = (phase_h[:, None, None] * dt).astype(jnp.complex128)
                new_flux_accs.append((
                    e1_acc + e1.astype(jnp.float64)[None, :, :] * kernel_e,
                    e2_acc + e2.astype(jnp.float64)[None, :, :] * kernel_e,
                    h1_acc + h1.astype(jnp.float64)[None, :, :] * kernel_h,
                    h2_acc + h2.astype(jnp.float64)[None, :, :] * kernel_h,
                ))

        # ---- per-step extras (caller-specific outputs) ----
        extras: dict = {}
        if ctx.use_snapshot:
            extras["snap_fields"] = ctx.snapshot_extractor(st)
        if ctx.use_monitor:
            extras["monitor_val"] = getattr(st, ctx.monitor_component)[
                ctx.mon_idx[0], ctx.mon_idx[1], ctx.mon_idx[2]]

        # Rebuild carry
        new_carry: dict = {"fdtd": st}
        if ctx.use_cpml:
            new_carry["cpml"] = cpml_new
        if ctx.use_debye:
            new_carry["debye"] = debye_new
        if ctx.use_lorentz:
            new_carry["lorentz"] = lorentz_new
        if ctx.use_tfsf:
            new_carry["tfsf"] = tfsf_new
        if ctx.use_ntff:
            new_carry["ntff"] = ntff_new
        if ctx.use_dft_planes:
            new_carry["dft_planes"] = tuple(new_dft_planes)
        if ctx.use_flux_monitors:
            new_carry["flux_monitors"] = tuple(new_flux_accs)
        if ctx.use_waveguide_ports:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if ctx.use_wire_sparams:
            new_carry["wire_sparam_accs"] = tuple(new_wire_accs)
        if ctx.use_lumped_sparams:
            new_carry["lumped_sparam_accs"] = tuple(new_lumped_accs)
        if ctx.use_wire_refplanes:
            new_carry["wire_refplane_accs"] = tuple(new_refplane_accs)
        if ctx.use_lumped_rlc:
            new_carry["rlc_states"] = tuple(new_rlc_states)

        return new_carry, probe_out, extras

    return core_step


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
    aniso_inv_eps: tuple | None = None,
    aniso_inv_eps_smooth: bool = False,
    pec_mask: object | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    pec_edge_masks: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    wire_refplane_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
    mag_sources: list | None = None,
    stencil_order: int = 2,
    report_every: int | None = None,
    report_label: str = "",
    sheet_impedance: object | None = None,
    design_box: DesignBoxSpec | None = None,
    design_occupancy: DesignOccupancySpec | None = None,
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
        If provided, record a frame after every ``snapshot.interval``-th
        step (``n_steps // interval`` frames; see ``SnapshotSpec``), with
        their positions and times in ``SimResult.snapshot_axes``. Recording
        does not change any other output: the time series, DFT and port
        accumulators and the final state are bit-identical to the same run
        without a snapshot. ``interval=1`` records inside the step body, as
        it always has; a larger interval runs the steps in blocks of
        ``interval`` (``lax.scan`` over blocks of an inner ``lax.scan`` over
        steps, plus a scan over the remaining ``n_steps % interval``) and
        records after each block, so the step body is the one a run without
        a snapshot compiles. The remainder scan is a second compile of the
        step body; a remainder of ONE step is instead joined with the last
        step of the last block into a two-step scan (a one-step scan is
        inlined by XLA and changed the last bit), leaving the rest of that
        block as a scan of ``interval - 1`` steps. With ``report_every`` a
        chunk that does not start on a multiple of the interval adds a short
        head scan, and each distinct piece length compiles once
        (``rfx.snapshots.plan_snapshot_pieces`` lays the pieces out). With
        ``checkpoint_segments`` the interval must divide the segment length
        ``n_steps // checkpoint_segments``.
    checkpoint : bool
        If True, wrap the scan body with ``jax.checkpoint`` to
        trade compute for memory during reverse-mode AD.  Reduces
        backward-pass memory from O(n_steps) to O(1) per step.
    return_state : bool
        If False, do not expose the final FDTD state in the result.
        This shrinks the differentiable output surface for optimisation.
    report_every : int or None
        Issue #667. When set, emit one ``  [PROGRESS] ...`` line to stdout
        every *N* steps: steps done / total, wall elapsed, implied rate and
        ETA. ``None`` (default) is OFF and runs the unchanged single-scan
        code path, so existing callers are byte-identical by construction.

        When set, the same compiled scan is driven from the host in
        ``report_every``-step chunks with ``carry`` threaded through, so the
        result is a continuation and not a re-solve; the DFT accumulators
        and every port / flux / monitor state live in that carry.
        Bit-exactness against ``report_every=None`` is locked by
        ``tests/unit/runners/test_run_progress_reporting.py``.

        Costs, measured rather than hidden: each report inserts one device
        synchronisation (without it the host would dispatch every chunk at
        once and print a fabricated rate), all full chunks share one XLA
        executable while a ragged final chunk compiles once more, and the
        per-chunk outputs are joined once at the end with a bounded-arity
        grouped concatenate (a flat one takes an argument per chunk, which
        cost 232 s at 22,500 chunks against 0.99 s grouped).

        Forward-only: it reads the host wall clock, so it raises under
        ``jax.jit``/``grad``/``vmap`` and is rejected together with
        ``checkpoint_segments``. The trace check inspects the carry, the
        per-step ``xs`` and the material / geometry arrays — the routes
        ``forward()`` and ``optimize()`` actually trace through — rather
        than asking JAX whether a trace is active, which its current API
        does not expose. A tracer reaching the scan body by some other
        closure alone would print trace-time lines; the computed values
        stay byte-identical either way.
    report_label : str
        Short tag prefixed to each progress line, e.g. ``"MSL drive p1"``,
        so the per-drive solves of one ``compute_*_s_matrix`` call are
        distinguishable in a log. Ignored when ``report_every`` is None.
    design_box : DesignBoxSpec or None
        Issue #1179. One static box whose E update is redone from its own
        (usually traced) permittivity, leaving ``materials`` constant so the
        reverse-mode tape holds box-shaped arrays instead of grid-shaped
        ones. ``None`` (default) is the unchanged path.

        ``_resolve_design_box`` rejects, from what this function resolves:
        UPML, Debye/Lorentz, anisotropic eps, Kerr, ``stencil_order=4``, the
        Bloch path, a box reaching into the CPML absorber, and a box holding
        a source, magnetic source, lumped/wire S-param port or RLC cell or a
        surface-impedance sheet edge. It does NOT carry the checks that need
        the API object: the lane (non-uniform, distributed, ADI), the
        collision with ``eps_override`` / ``sigma_override`` /
        ``mu_r_override``, ``pec_occupancy_override``, a PASSIVE port (it
        leaves neither a source nor an accumulator here), and the promotion
        of ``materials.eps_r`` to the design dtype. Those are
        ``Simulation.forward``'s, in ``_resolve_design_box_override``; a
        caller reaching this function directly owns them.
    design_occupancy : DesignOccupancySpec or None
        Issue #1183. One static box whose PEC OCCUPANCY is the traced
        quantity, for the same reason and with the same effect on the tape:
        the ``1 - M`` scaling is redone on the box's window from the field
        before the grid-wide ``apply_pec_occupancy``, so what the backward
        pass keeps per step is window-shaped. ``None`` (default) is the
        unchanged path. The box's occupancy REPLACES ``pec_occupancy`` at
        the box cells; outside it, ``pec_occupancy`` is carried.

        ``_resolve_design_occupancy`` rejects a periodic axis, an empty box
        and a box off the grid. ``Simulation.forward`` owns the lane and the
        collision with ``design_eps_override``.

    Returns
    -------
    SimResult with final state, time series, and optional NTFF data.
    """
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []
    wire_port_sparams = wire_port_sparams or []
    lumped_port_sparams = lumped_port_sparams or []
    wire_refplane_sparams = wire_refplane_sparams or []
    lumped_rlc = lumped_rlc or []
    mag_sources = mag_sources or []

    # ---- shared setup (W6.2) ----
    _setup = _build_step_setup(
        grid=grid,
        materials=materials,
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
        debye=debye,
        lorentz=lorentz,
        tfsf=tfsf,
        sources=sources,
        probes=probes,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_ports=waveguide_ports,
        ntff=ntff,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        wire_port_sparams=wire_port_sparams,
        lumped_port_sparams=lumped_port_sparams,
        wire_refplane_sparams=wire_refplane_sparams,
        lumped_rlc=lumped_rlc,
        kerr_chi3=kerr_chi3,
        field_dtype=field_dtype,
        mag_sources=mag_sources,
        stencil_order=stencil_order,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
        design_occupancy=design_occupancy,
    )
    carry_init = _setup.carry_init
    dt = _setup.dt
    dx = _setup.dx
    periodic = _setup.periodic
    waveguide_meta = _setup.waveguide_meta
    wire_sparam_meta = _setup.wire_sparam_meta
    lumped_sparam_meta = _setup.lumped_sparam_meta
    wire_refplane_meta = _setup.wire_refplane_meta
    flux_meta = _setup.flux_meta_11   # 11-field: run() uses streaming DFT window
    use_snapshot = snapshot is not None
    use_flux_monitors = len(flux_monitors) > 0
    use_wire_sparams = len(wire_port_sparams) > 0
    use_lumped_sparams = len(lumped_port_sparams) > 0
    use_wire_refplanes = len(wire_refplane_sparams) > 0
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0

    # ---- run()-specific: fast-path pre-baked coefficients ----
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
    _ctx = _setup.ctx_kwargs
    _fast_eligible = (
        not _ctx["use_cpml"]
        and not _ctx["use_upml"]
        # the baked step applies no per-face masks after its H update, so a
        # magnetic wall would be neither electric nor magnetic there (#1164)
        and not _ctx["use_pmc_faces"]
        and not _ctx["use_tfsf"]
        and not _ctx["use_debye"]
        and not _ctx["use_lorentz"]
        and not _ctx["use_pec_edges"]
        and not _ctx["use_pec_occupancy"]
        and not _ctx["use_conformal"]
        and not _ctx["use_lumped_rlc"]
        and not _ctx["use_kerr"]
        and not _ctx["use_mag_sources"]
        # #677: the GPU baked fast path has an inline H+E update with no
        # sheet-operator slot, so a surface-impedance sheet is DELIBERATELY
        # unsupported here: an f0 sheet takes the standard path, which costs
        # performance only -- no accuracy loss, nothing silently wrong.
        # Per-plane baking is a parked capability, not a promised follow-up;
        # it waits on the thin-sheet plane-BC architecture decision (#701),
        # and the parking itself is backlog #787.
        and not _ctx["use_sheet_impedance"]
        # #1179: same reason as the sheet operator above -- the inline H+E
        # update has no slot for the design-box redo, and its coefficients
        # are baked from ``materials``, which is exactly the array a design
        # box stops carrying. A design box takes the standard path; that
        # costs GPU scatter kernels, nothing else.
        and not _ctx["use_design_box"]
        # #1183: and the same for the design occupancy. The fast path has
        # no occupancy slot at all -- it would drop the design variable and
        # return an all-zero gradient, which reads like a converged design.
        and not _ctx["use_design_occupancy"]
        and aniso_eps is None
        and periodic == (False, False, False)
    )
    # On GPU the baked-PEC path eliminates expensive scatter-update
    # kernel launches — always beneficial.  On CPU, XLA fuses scatter
    # ops efficiently so the extra coefficient arrays hurt at small-to-
    # medium grids; enable only when explicitly requested via GPU backend.
    # The GPU fast path has an INLINE 2nd-order stencil (update_he_fast),
    # so it must never be used for stencil_order=4 — that would silently
    # produce a 2nd-order result. Gate it on order==2.
    use_fast_he = _fast_eligible and _on_gpu and stencil_order == 2
    _fast_coeffs = (
        precompute_coeffs(materials, dt, dx, pec_faces=_setup.pec_faces,
                          periodic=periodic)
        if use_fast_he else None
    )

    # ---- run()-specific: snapshot setup ----
    # #1258: ``interval`` is honoured. interval == 1 keeps the pre-#1258 path
    # (the frame is an extra per-step output of the step body). interval > 1
    # records from the carry after each block of ``interval`` steps, so the
    # step body is the SAME one a run without a snapshot compiles and memory
    # scales with the frame count, not with n_steps.
    if use_snapshot:
        snap_interval = validate_snapshot_spec(snapshot)
        snap_components = tuple(snapshot.components)
        _take_snapshot = snapshot_extractor(snapshot)
    else:
        snap_interval = 1
        snap_components = ()
        _take_snapshot = None
    snap_in_body = use_snapshot and snap_interval == 1
    snap_by_block = use_snapshot and snap_interval > 1

    # ---- precompute source waveform matrix (n_steps, n_sources) ----
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # ---- magnetic source (H-field) waveform matrix ----
    if mag_sources:
        mag_src_waveforms = jnp.stack([s.waveform for s in mag_sources], axis=-1)
    else:
        mag_src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # ---- scan body (shared kernel; W6.1 kernel + W6.2 setup) ----
    _step_ctx = _StepContext(
        **_setup.ctx_kwargs,
        # run()-specific overrides
        use_fast_he=use_fast_he,
        use_snapshot=snap_in_body,
        use_monitor=False,
        use_flux_window=True,
        fast_coeffs=_fast_coeffs,
        flux_meta=flux_meta if use_flux_monitors else (),
        monitor_component="ez",
        mon_idx=None,
        snapshot_extractor=_take_snapshot if snap_in_body else None,
    )
    _core_step = make_core_step(_step_ctx)

    def step_fn(carry, xs):
        _step_idx, src_vals, mag_src_vals = xs
        new_carry, probe_out, extras = _core_step(
            carry, _step_idx, src_vals, mag_src_vals)
        if snap_in_body:
            output = (probe_out, extras["snap_fields"])
        else:
            output = (probe_out,)
        return new_carry, output

    def _make_recorder(body):
        """``record(carry, xs_seg, lo)``: scan ``xs_seg`` (global steps
        ``lo, lo+1, ...``) with ``body`` and take a frame after every step
        whose completed-step count is a multiple of ``snap_interval`` (#1258).

        The steps run as the pieces :func:`rfx.snapshots.plan_snapshot_pieces`
        lays out, in step order:

        * ``plain`` -- a scan of ``body``, with the frame read from the carry
          when the piece ends on a multiple;
        * ``blocks`` -- a scan over blocks of ``snap_interval`` steps whose
          body is an inner scan of ``body`` followed by the frame read from
          the carry;
        * ``rec`` -- a scan of ``body`` over two or three steps that also
          outputs each step's frame, of which the rows ending on a multiple
          are kept.

        A one-step plain piece never runs on its own (XLA inlines a loop that
        runs once, and the inlined step moved the last bit of Ex/Ey on a PEC
        box): the planner joins it with ONE step of a neighbour into a
        ``rec`` piece, so at most three frames exist before the unwanted ones
        are dropped, whatever the interval. Every piece calls the same
        ``body`` on the same ``xs`` rows in the same order, inside a loop of
        at least two steps, so the carry is the one a single scan produces;
        the probe rows are joined back in step order. Returns
        ``(carry, (probe_rows, [frames per component]))``, the output layout
        of the ``interval == 1`` path. A segment of one step is run as the
        one-step scan the unchunked path would also run.

        The block and recording bodies are built once per ``body`` so
        repeated calls (one per ``report_every`` chunk) reuse their traces
        and compiled executables.
        """
        m = snap_interval

        def block_body(c, block_xs):
            c, (p,) = jax.lax.scan(body, c, block_xs)
            return c, (p, _take_snapshot(c["fdtd"]))

        def rec_body(c, x):
            c, (p,) = body(c, x)
            return c, (p, _take_snapshot(c["fdtd"]))

        def record(carry, xs_seg, lo):
            n = int(jax.tree_util.tree_leaves(xs_seg)[0].shape[0])
            probe_parts, frame_parts = [], []
            for piece in plan_snapshot_pieces(n, lo, m):
                a, b = piece.start, piece.start + piece.length
                rows = jax.tree_util.tree_map(lambda x: x[a:b], xs_seg)
                if piece.kind == "plain":
                    carry, (p,) = jax.lax.scan(body, carry, rows)
                    probe_parts.append(p)
                    if piece.frame_rows:
                        frame_parts.append(
                            [f[None] for f in _take_snapshot(carry["fdtd"])])
                elif piece.kind == "blocks":
                    nb = piece.length // m
                    blocks = jax.tree_util.tree_map(
                        lambda x: x.reshape(nb, m, *x.shape[1:]), rows)
                    carry, (p, f) = jax.lax.scan(block_body, carry, blocks)
                    probe_parts.append(p.reshape(nb * m, *p.shape[2:]))
                    frame_parts.append(list(f))
                else:  # "rec"
                    carry, (p, f) = jax.lax.scan(rec_body, carry, rows)
                    probe_parts.append(p)
                    if piece.frame_rows:
                        idx = jnp.asarray(piece.frame_rows, dtype=jnp.int32)
                        frame_parts.append([ff[idx] for ff in f])

            if not probe_parts:
                # n == 0: the zero-length scan the other paths return.
                carry, (p,) = jax.lax.scan(body, carry, xs_seg)
                probe_parts.append(p)
            probes = (probe_parts[0] if len(probe_parts) == 1
                      else jnp.concatenate(probe_parts, axis=0))
            if frame_parts:
                frames = [
                    parts[0] if len(parts) == 1
                    else jnp.concatenate(parts, axis=0)
                    for parts in zip(*frame_parts)]
            else:
                frames = [
                    jnp.zeros((0,) + s.shape, s.dtype)
                    for s in jax.eval_shape(_take_snapshot, carry["fdtd"])]
            return carry, (probes, frames)

        return record

    # ---- run ----
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms, mag_src_waveforms)

    if checkpoint_segments is None:
        # Legacy path: optional per-step rematerialisation only. The scan
        # itself still keeps every step's carry, so peak memory grows
        # linearly with n_steps.
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        if report_every is None and snap_by_block:
            final_carry, outputs = _make_recorder(body)(carry_init, xs, 0)
        elif report_every is None:
            final_carry, outputs = jax.lax.scan(body, carry_init, xs)
        else:
            # Issue #667: same scan, driven from the host in chunks so a
            # multi-hour solve emits progress. The carry threads through
            # untouched (DFT accumulators, port/flux/monitor state, NTFF
            # compensation), so this is a continuation and not a re-solve.
            # The ``report_every is None`` branch above is byte-identical to
            # the pre-#667 code by construction.
            final_carry, outputs = scan_with_progress(
                body, carry_init, xs,
                n_steps=n_steps,
                report_every=report_every,
                label=report_label,
                # A tracer from forward()/optimize() arrives through the
                # material and geometry arrays, which the scan body captures
                # in its closure — carry_init and xs stay concrete under bare
                # grad/vmap, so checking only those would let the request
                # through and print trace-time lines.
                trace_probes=(materials, aniso_eps, aniso_inv_eps,
                              pec_mask, pec_edge_masks,
                              pec_occupancy, conformal_weights,
                              kerr_chi3, debye, lorentz, tfsf),
                # #1258: a chunk that does not start on a multiple of the
                # snapshot interval still records at the global multiples.
                chunk_scan=_make_recorder(body) if snap_by_block else None,
            )
    else:
        if report_every is not None:
            raise NotImplementedError(
                "report_every is not supported together with "
                "checkpoint_segments: segmented checkpointing is itself a "
                "scan-of-scans whose segment boundaries pin the AD "
                "rematerialisation points, and host-side progress chunking "
                "would have to cut the same axis. Progress reporting is a "
                "forward-only feature — drop checkpoint_segments for a "
                "monitored forward solve, or drop report_every for the "
                "memory-limited gradient solve."
            )
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
        if snap_by_block and s % snap_interval != 0:
            raise ValueError(
                f"snapshot interval={snap_interval} does not divide the "
                f"checkpoint segment length n_steps/checkpoint_segments="
                f"{n_steps}/{K}={s}: every segment must record the same "
                f"number of frames. Pick an interval that divides {s}, or a "
                f"checkpoint_segments whose segment length is a multiple of "
                f"{snap_interval}.")
        xs_segmented = jax.tree_util.tree_map(
            lambda a: a.reshape(K, s, *a.shape[1:]), xs)

        _record_segment = _make_recorder(step_fn) if snap_by_block else None

        def segment_body(carry, seg_xs):
            if snap_by_block:
                # Each segment starts on a multiple of s, hence of the
                # interval, so ``lo=0`` places its frames correctly.
                return _record_segment(carry, seg_xs, 0)
            new_carry, seg_outputs = jax.lax.scan(step_fn, carry, seg_xs)
            return new_carry, seg_outputs

        seg_body_ckpt = jax.checkpoint(
            segment_body, prevent_cse=False) if checkpoint else segment_body
        final_carry, seg_outputs = jax.lax.scan(
            seg_body_ckpt, carry_init, xs_segmented)
        # seg_outputs leaves: (K, per-segment rows, ...). Flatten back to
        # (K * rows, ...): n_steps for the probe rows (and for per-step
        # frames at interval 1), n_steps // interval for block frames.
        outputs = jax.tree_util.tree_map(
            lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:]),
            seg_outputs)

    if use_snapshot:
        time_series = outputs[0]
        # outputs[1] is a list of arrays, each (n_steps // interval, ...)
        snapshots = {comp: outputs[1][i]
                     for i, comp in enumerate(snap_components)}
        snap_axes = snapshot_axes(grid, snapshot, n_steps, dt=dt)
        for comp in snap_components:
            n_frames = int(snapshots[comp].shape[0])
            if n_frames != len(snap_axes[comp].steps):
                raise RuntimeError(
                    f"snapshot {comp!r} recorded {n_frames} frames, expected "
                    f"{len(snap_axes[comp].steps)} for n_steps={n_steps}, "
                    f"interval={snap_interval} (#1258)")
    else:
        time_series = outputs[0]
        snapshots = None
        snap_axes = None

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
                # Stamp the scan's authoritative dt so the post-scan rect-DFT
                # extractor uses the right Δt. The jit-safe in-scan probe
                # accumulator cannot run update_waveguide_port_probe's
                # ``float(dt)`` stamp, so a cfg built via init_waveguide_port
                # without dt= would otherwise keep dt=0 and silently zero every
                # extracted spectrum (S→0). The manual Python-loop path already
                # stamps dt via update_waveguide_port_probe; this makes the
                # compiled run() symmetric with it. The scan's own step, not
                # grid.dt: stencil_order=4 derates it.
                dt=float(dt),
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

    final_wire_refplanes = None
    if use_wire_refplanes:
        final_wire_refplanes = tuple(
            (rp_meta, accs)
            for rp_meta, accs in zip(wire_refplane_meta,
                                     final_carry["wire_refplane_accs"])
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
        wire_refplane_sparams=final_wire_refplanes,
        snapshot_axes=snap_axes,
        dt=dt,
    )


# ---------------------------------------------------------------------------
# Field-decay-based stopping criterion (Python loop + JIT step)
# ---------------------------------------------------------------------------

def _warn_static_remnant_cap_hit(state, materials, grid) -> None:
    """#388: on an ``until_decay`` CAP-HIT, warn if the interior remnant is electrostatic.

    A soft current source deposits net charge; the remnant's discrete-Poisson self-energy
    neither radiates nor is absorbed by CPML, so it floors the interior-energy criterion and
    ``until_decay`` silently cap-hits even though the radiating field has settled. This is a
    MEASURED post-run advisory (complements the waveform-predicted pre-run
    :meth:`_warn_until_decay_dc_floor`, which cannot see thin-source-cell-amplified floors).

    Detection uses PHYSICAL energy (½ε|E|² vs ½μ|H|²): a propagating/radiating field is
    equipartitioned (H-share ~0.5, and its instantaneous H-energy stays ~0.2 even at an E-max
    2f-slosh instant), while a static electrostatic remnant is E-only (H-share ~1e-10). The
    ``H-share < 1e-2`` gate separates them with enormous margin — no false-fire on a genuine
    still-ringing cap-hit.
    """
    import warnings as _w
    is_2d = getattr(grid, "is_2d", False)  # NonUniformGrid is always 3D (no is_2d attr)
    ix0, ix1 = grid.pad_x_lo, grid.nx - grid.pad_x_hi
    iy0, iy1 = grid.pad_y_lo, grid.ny - grid.pad_y_hi
    iz0, iz1 = (0, grid.nz) if is_2d else (grid.pad_z_lo, grid.nz - grid.pad_z_hi)
    sl = (slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1))
    eps_r = materials.eps_r[sl]
    mu_r = getattr(materials, "mu_r", None)
    mu_r = mu_r[sl] if (mu_r is not None and jnp.ndim(mu_r) > 0) else 1.0
    if is_2d:
        e2 = state.ez[sl] ** 2
        h2 = state.hx[sl] ** 2 + state.hy[sl] ** 2
    else:
        e2 = state.ex[sl] ** 2 + state.ey[sl] ** 2 + state.ez[sl] ** 2
        h2 = state.hx[sl] ** 2 + state.hy[sl] ** 2 + state.hz[sl] ** 2
    u_e = 0.5 * EPS_0 * float(jnp.sum(eps_r * e2))
    u_h = 0.5 * MU_0 * float(jnp.sum(mu_r * h2))
    tot = u_e + u_h
    if tot <= 0.0:
        return
    h_share = u_h / tot
    if h_share < 1e-2:
        _w.warn(
            f"run(until_decay=...) cap-hit at max_steps without the energy criterion firing, and "
            f"{100 * (1 - h_share):.1f}% of the final interior energy is electrostatic "
            f"(H-share {h_share:.1e}). A static soft-source charge remnant (discrete-Poisson "
            f"self-energy that neither radiates nor is absorbed by CPML) held the interior energy "
            f"above decay_by*peak_U while the radiating field settled, so the energy criterion "
            f"could not self-terminate. The purpose-built remedy is the radiated-flux stop "
            f"criterion (immune to this static floor — it terminates on outgoing-flux settling, "
            f"not interior energy): re-run with "
            f"radiated_flux_box=((x0,y0,z0),(x1,y1,z1)) enclosing the radiator. Otherwise reduce "
            f"the source's deposited DC (GaussianPulse cutoff / lower bandwidth), run a fixed "
            f"n_steps, or confirm settling from the probe envelope. (issue #388)",
            UserWarning,
        )


def run_until_decay(
    grid: Grid,
    materials: MaterialArrays,
    *,
    decay_by: float = 1e-3,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
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
    aniso_inv_eps: tuple | None = None,
    aniso_inv_eps_smooth: bool = False,
    pec_mask: object | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    pec_edge_masks: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
    mag_sources: list | None = None,
    checkpoint_segments: int | None = None,
    stencil_order: int = 2,
    report_every: int | None = None,
    report_label: str = "",
    sheet_impedance: object | None = None,
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
    decay_energy_consecutive : int
        On **absorbing** boundaries (``cpml`` / ``upml``) the stop uses the
        total interior-domain energy criterion (issue #169). It fires only
        after the interior energy ``U`` has stayed below ``decay_by * peak_U``
        on this many *consecutive* checks. Default ``2`` (MANDATORY minimum):
        the interior energy of a guided geometry is not null-free, dipping
        through transient inter-packet minima that recover; a single below-
        threshold check can false-fire on such a dip, so ``>= 2`` is required
        to absorb the chatter. Has no effect on closed/PEC boundaries (which
        use the instantaneous point-field fallback).
    radiated_flux_box : tuple or None
        Opt-in (#388) RADIATED-FLUX stop for absorbing boundaries: a Huygens box
        ``((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))`` in physical coordinates
        enclosing the structure (clear of the CPML). When set, the stop uses the
        outgoing Poynting flux through the box instead of the interior energy —
        a *radiation-settling* criterion that ignores non-radiating trapped
        energy (the static soft-source charge that FLOORS the energy criterion,
        and near-Nyquist grid buzz), because neither carries net radiated power.
        Appropriate for radiation / S-parameter measurements; the flux peak is
        tracked from the first check (the radiation peaks during the source
        drive). Default ``None`` keeps the interior-energy criterion, unchanged.
    flux_env_checks : int
        Number of recent checks whose ``|flux|`` max forms the envelope used by
        the radiated-flux stop, smoothing the 2f0 Poynting oscillation. Default 4.
    monitor_component : str
        Field component to monitor ("ez", "hy", etc.). Used only by the
        closed/PEC point-field fallback stop.
    monitor_position : tuple or None
        Physical position (x, y, z) to monitor. If None, use center of
        the domain.
    checkpoint_segments : int or None
        **Not supported** on the decay path.  ``run_until_decay`` uses a
        Python loop (not ``jax.lax.scan``), so scan-level gradient
        checkpointing does not apply.  Passing a non-None value raises
        ``NotImplementedError``.

    Notes
    -----
    **Differences from** :func:`run`:

    * ``checkpoint_segments`` is not supported (raises ``NotImplementedError``
      when not ``None``).
    * ``snapshot`` records a frame after every ``snapshot.interval``-th step
      of the loop, as :func:`run` does (#1258): ``actual_steps // interval``
      frames, positions and times in ``SimResult.snapshot_axes``. Reading
      the frame does not touch the stepped state.
    * ``checkpoint`` (``jax.checkpoint`` gradient tape) is accepted but
      **silently ignored** — gradient checkpointing has no effect on the
      Python-loop path.
    * Flux-monitor DFT accumulation uses a rectangular (no-window) weight
      instead of the streaming Hann window used by :func:`run`.  Flux values
      from the two paths are therefore not numerically identical even for
      the same step count.

    .. note::

       **The stop criterion depends on the boundary (issue #169, RESOLVED for
       absorbing boundaries).**

       * **Absorbing boundaries (``cpml`` / ``upml``)** — the stop is gated on
         the **total interior-domain energy** ``U = sum(E^2 + H^2)`` over the
         non-CPML interior slice, declared decayed once ``U < decay_by * peak_U``
         on ``decay_energy_consecutive`` consecutive checks. Because it is a
         whole-domain energy, it does not pass through the per-cell
         interference nulls that the old single-cell point-field stopper hit
         between slow-tail wave packets, so it **is** suitable for flux /
         S-parameter / transmission measurements on guided / low-loss
         geometries. (On the cv03-class eps=12 guide it now stops at the
         flux-converged transmission within ~0.13%, vs the prior ~7% under-run
         of the point-field stop.) ``decay_energy_consecutive >= 2`` is
         mandatory because the interior energy is *not* null-free — it dips
         through transient inter-packet minima that recover, so a single
         below-threshold check can false-fire.

       * **Closed / PEC boundaries** — domain energy does not decay in a
         lossless closed cavity, so the stop falls back to the historical
         *instantaneous* squared field at the single ``monitor_component`` cell
         (``val_sq < decay_by * peak_sq``). This fallback retains the original
         limitation: it is a valid decay witness only for lossy / radiating
         structures with a clean ring-down envelope, and is **not** suitable
         for flux / S-parameter / transmission gating on guided / low-loss
         closed geometries — for those use a fixed ``n_steps`` via
         :func:`run`.
    report_every : int or None
        Issue #667. When set, emit one ``  [PROGRESS] ...`` line every *N*
        steps plus a final line at the actual stop step. ``None`` (default)
        is OFF. This lane is already a Python loop, so the tick is a pure
        addition on Python ints — it never reads a field value and cannot
        perturb the result. The denominator is ``max_steps``, i.e. a CAP:
        the line marks it ``(cap)`` and the ETA is an upper bound, because a
        decay stop can fire at any check.
    report_label : str
        Short tag prefixed to each progress line. Ignored when
        ``report_every`` is None.

    Returns
    -------
    SimResult
    """
    if checkpoint_segments is not None:
        raise NotImplementedError(
            "checkpoint_segments is not supported by run_until_decay: "
            "this function uses a Python loop, not jax.lax.scan, so "
            "scan-level gradient checkpointing does not apply. "
            "Use run() with checkpoint_segments if you need scan-level "
            "checkpointing."
        )
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []
    wire_port_sparams = wire_port_sparams or []
    lumped_port_sparams = lumped_port_sparams or []
    lumped_rlc = lumped_rlc or []
    mag_sources = mag_sources or []

    # ---- shared setup (W6.2) ----
    _setup = _build_step_setup(
        grid=grid,
        materials=materials,
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
        debye=debye,
        lorentz=lorentz,
        tfsf=tfsf,
        sources=sources,
        probes=probes,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_ports=waveguide_ports,
        ntff=ntff,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        wire_port_sparams=wire_port_sparams,
        lumped_port_sparams=lumped_port_sparams,
        lumped_rlc=lumped_rlc,
        kerr_chi3=kerr_chi3,
        field_dtype=field_dtype,
        mag_sources=mag_sources,
        stencil_order=stencil_order,
        sheet_impedance=sheet_impedance,
    )
    carry = _setup.carry_init
    dx = _setup.dx
    waveguide_meta = _setup.waveguide_meta
    wire_sparam_meta = _setup.wire_sparam_meta
    lumped_sparam_meta = _setup.lumped_sparam_meta
    flux_meta_decay = _setup.flux_meta_8   # 8-field: no streaming DFT window
    use_flux_monitors = len(flux_monitors) > 0
    use_wire_sparams = len(wire_port_sparams) > 0
    use_lumped_sparams = len(lumped_port_sparams) > 0
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0

    # ---- monitor position ----
    if monitor_position is None:
        # Center of the physical domain
        cx = (grid.nx - 1) * dx / 2.0
        cy = (grid.ny - 1) * dx / 2.0
        cz = 0.0 if grid.is_2d else (grid.nz - 1) * dx / 2.0
        monitor_position = (cx, cy, cz)
    mon_idx = grid.position_to_index(monitor_position)

    # ---- JIT-compiled single step (shared kernel; W6.1 + W6.2 setup) ----
    # run_until_decay does NOT build the GPU fast-HE coeffs and uses the
    # historical rect (no-window) flux DFT: use_flux_window=False keeps the
    # 8-field flux_meta and skips the streaming window weight so decay-path
    # flux values stay bit-identical to the pre-refactor body. Unifying the
    # decay path onto windows is a deliberate follow-up.
    _step_ctx = _StepContext(
        **_setup.ctx_kwargs,
        # decay-path-specific overrides
        use_fast_he=False,
        use_snapshot=False,
        use_monitor=True,
        use_flux_window=False,
        fast_coeffs=None,
        flux_meta=flux_meta_decay if use_flux_monitors else (),
        monitor_component=monitor_component,
        mon_idx=mon_idx,
        snapshot_extractor=None,
    )
    _core_step = make_core_step(_step_ctx)

    @jax.jit
    def _single_step(carry_in, step_idx, src_vals, mag_src_vals):
        new_carry, probe_out, extras = _core_step(
            carry_in, step_idx, src_vals, mag_src_vals)
        return new_carry, probe_out, extras["monitor_val"]

    # #1258: frames are read from the carry after every interval-th step.
    if snapshot is not None:
        snap_interval = validate_snapshot_spec(snapshot)
        _take_snapshot = snapshot_extractor(snapshot)
        snap_frames: list = []

    # ---- precompute source waveforms up to max_steps ----
    if sources:
        src_waveforms = jnp.stack([s.waveform[:max_steps] if s.waveform.shape[0] >= max_steps
                                   else jnp.pad(s.waveform, (0, max_steps - s.waveform.shape[0]))
                                   for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((max_steps, 0), dtype=jnp.float32)

    if mag_sources:
        mag_src_waveforms = jnp.stack(
            [s.waveform[:max_steps] if s.waveform.shape[0] >= max_steps
             else jnp.pad(s.waveform, (0, max_steps - s.waveform.shape[0]))
             for s in mag_sources], axis=-1)
    else:
        mag_src_waveforms = jnp.zeros((max_steps, 0), dtype=jnp.float32)

    # ---- Python loop with decay check ----
    # Stop criterion depends on the boundary (issue #169):
    #   * absorbing (cpml/upml): TOTAL interior-domain energy decay. The energy
    #     leaves through the absorber, so U -> 0 and the criterion is a genuine
    #     convergence witness for flux / S-param / transmission. Requires
    #     decay_energy_consecutive >= 2 consecutive sub-threshold checks because
    #     the interior energy is NOT null-free (it dips through transient
    #     inter-packet minima that recover).
    #   * closed/PEC: domain energy never decays in a lossless cavity, so fall
    #     back to the historical instantaneous single-cell point-field stop.
    #     That branch is BYTE-IDENTICAL to the pre-#169 behavior.
    use_absorbing = boundary in ("cpml", "upml")

    # Non-CPML interior slice bounds (Python ints — never enter the traced
    # step; the reduction below is host-side, like the existing float()).
    _ix0, _ix1 = grid.pad_x_lo, grid.nx - grid.pad_x_hi
    _iy0, _iy1 = grid.pad_y_lo, grid.ny - grid.pad_y_hi
    if grid.is_2d:
        _iz0, _iz1 = 0, grid.nz
    else:
        _iz0, _iz1 = grid.pad_z_lo, grid.nz - grid.pad_z_hi

    def _interior_energy(state) -> float:
        """Total interior-domain field energy U = sum(E^2 + H^2) (host float)."""
        sx, sy, sz = (slice(_ix0, _ix1), slice(_iy0, _iy1), slice(_iz0, _iz1))
        if grid.is_2d:
            # 2d_tmz active fields: ez, hx, hy (ex/ey/hz are identically zero).
            u = (state.ez[sx, sy, sz] ** 2
                 + state.hx[sx, sy, sz] ** 2
                 + state.hy[sx, sy, sz] ** 2)
        else:
            u = (state.ex[sx, sy, sz] ** 2 + state.ey[sx, sy, sz] ** 2
                 + state.ez[sx, sy, sz] ** 2 + state.hx[sx, sy, sz] ** 2
                 + state.hy[sx, sy, sz] ** 2 + state.hz[sx, sy, sz] ** 2)
        return float(jnp.sum(u))

    # #388 opt-in RADIATED-FLUX stop: stop when the outgoing Poynting flux through a Huygens
    # box decays, instead of the interior energy. This is a *radiation-settling* criterion —
    # it ignores non-radiating trapped energy (the static soft-source charge that FLOORS the
    # energy criterion, and near-Nyquist grid buzz) because neither carries net radiated power.
    # It is the right stop for radiation / S-parameter measurements; it is opt-in (default
    # keeps the energy criterion, unchanged) because its semantics differ from energy-decay.
    use_flux_stop = radiated_flux_box is not None and use_absorbing
    if use_flux_stop:
        _flo = grid.position_to_index(radiated_flux_box[0])
        _fhi = grid.position_to_index(radiated_flux_box[1])
        _bl = (min(_flo[0], _fhi[0]), max(_flo[0], _fhi[0]),
               min(_flo[1], _fhi[1]), max(_flo[1], _fhi[1]),
               min(_flo[2], _fhi[2]), max(_flo[2], _fhi[2]))

    def _radiated_power(state) -> float:
        """Net outgoing Poynting flux P = ∮ (E×H)·n̂ dA over the box (co-located approx; a stop
        criterion needs only the DECAY, so the half-cell Yee stagger is neglected). The dA=dx²
        factor is a constant that cancels in P/peak_P."""
        ex, ey, ez = state.ex, state.ey, state.ez
        hx, hy, hz = state.hx, state.hy, state.hz
        il, ih, jl, jh, kl, kh = _bl
        jj, kk = slice(jl, jh), slice(kl, kh)
        ii = slice(il, ih)
        # +x/-x faces: S_x = Ey·Hz - Ez·Hy
        p = jnp.sum(ey[ih, jj, kk] * hz[ih, jj, kk] - ez[ih, jj, kk] * hy[ih, jj, kk])
        p -= jnp.sum(ey[il, jj, kk] * hz[il, jj, kk] - ez[il, jj, kk] * hy[il, jj, kk])
        # +y/-y faces: S_y = Ez·Hx - Ex·Hz
        p += jnp.sum(ez[ii, jh, kk] * hx[ii, jh, kk] - ex[ii, jh, kk] * hz[ii, jh, kk])
        p -= jnp.sum(ez[ii, jl, kk] * hx[ii, jl, kk] - ex[ii, jl, kk] * hz[ii, jl, kk])
        # +z/-z faces: S_z = Ex·Hy - Ey·Hx
        p += jnp.sum(ex[ii, jj, kh] * hy[ii, jj, kh] - ey[ii, jj, kh] * hx[ii, jj, kh])
        p -= jnp.sum(ex[ii, jj, kl] * hy[ii, jj, kl] - ey[ii, jj, kl] * hx[ii, jj, kl])
        return float(p)

    peak_sq = 0.0          # closed/PEC point-field running peak
    peak_U = 0.0           # absorbing interior-energy running peak (at checks)
    energy_below = 0       # consecutive sub-threshold energy checks
    peak_flux = 0.0        # #388 flux-stop: running peak of the |P| envelope
    flux_below = 0         # #388 flux-stop: consecutive sub-threshold checks
    flux_hist: list[float] = []   # recent |P| samples for the max-envelope
    decayed_fired = False  # #388: did the energy criterion fire (vs silently cap-hit)?
    all_probes = []
    actual_steps = 0

    # Issue #667 progress ticker. This lane is already a host loop, so the
    # tick is a pure addition: it reads Python ints (``actual_steps``,
    # ``report_every``) and never touches a traced value or a field value,
    # so it cannot perturb the result. ``max_steps`` is a CAP, not a known
    # length — the line says so and the ETA is an upper bound.
    _reporter = None
    if report_every is not None:
        _report_every = validate_report_every(report_every, n_steps=max_steps)
        # Probe the material/geometry arrays too, not just the carry: under
        # a bare grad/vmap the carry stays concrete and the tracer rides in
        # through the closure instead.
        check_not_traced(carry, materials, aniso_eps, aniso_inv_eps,
                         pec_mask, pec_edge_masks,
                         pec_occupancy, conformal_weights,
                         kerr_chi3, debye, lorentz, tfsf)
        _reporter = ProgressReporter(
            max_steps, label=report_label, total_is_cap=True)

    for step in range(max_steps):
        step_idx = jnp.array(step, dtype=jnp.int32)
        src_vals = src_waveforms[step]
        mag_src_vals = mag_src_waveforms[step]
        carry, probe_out, monitor_val = _single_step(carry, step_idx, src_vals, mag_src_vals)

        all_probes.append(probe_out)
        actual_steps = step + 1
        if snapshot is not None and actual_steps % snap_interval == 0:
            snap_frames.append(_take_snapshot(carry["fdtd"]))

        if _reporter is not None and actual_steps % _report_every == 0:
            # Block first: JAX dispatch is asynchronous, so an unsynchronised
            # tick would report the host's dispatch rate, not the solve rate.
            jax.block_until_ready(carry["fdtd"])
            _reporter.report(actual_steps)

        if use_absorbing and use_flux_stop:
            # #388 opt-in: RADIATED-FLUX stop. Compute the net outgoing Poynting flux at each
            # check; smooth the 2f0 oscillation with a max-envelope over the last
            # ``flux_env_checks`` checks; stop when the envelope decays below the threshold.
            # The flux PEAK is during the source drive, so track it from the FIRST check (unlike
            # the interior energy, whose peak is post-source); only the STOP check waits for
            # min_steps.
            if step % check_interval == 0:
                flux_hist.append(abs(_radiated_power(carry["fdtd"])))
                env = max(flux_hist[-flux_env_checks:])
                if env > peak_flux:
                    peak_flux = env
                if actual_steps >= min_steps and peak_flux > 0.0 and env < decay_by * peak_flux:
                    flux_below += 1
                    if flux_below >= decay_energy_consecutive:
                        decayed_fired = True
                        break
                else:
                    flux_below = 0
        elif use_absorbing:
            # Interior-energy criterion. The reduction is check-step-only (the
            # whole-domain sum is the expensive part), so we compute U ONLY on
            # an eligible check step — never every step.
            #
            # The PEAK is tracked from the FIRST check, like the flux branch
            # above, and only the STOP waits for min_steps. This block used to
            # do both behind ``actual_steps >= min_steps``, on the assumption
            # that the interior energy peaks after the source ends. That is
            # false for a domain that empties before min_steps: the reference
            # then IS the residual, no residual can fall another
            # ``decay_by`` below itself, and the run goes to max_steps
            # (#1078). Measured on the committed cv03-class guided fixture of
            # tests/unit/runners/test_decay_flux_convergence.py: true peak
            # 6.28e-10 at step 701, U = 6.13e-17 at step 2001 = 9.8e-8 of it,
            # while the post-min_steps maximum reads 6.13e-17 — a reference
            # 1.0e7 times too small.
            if decay_by > 0.0 and step % check_interval == 0:
                U = _interior_energy(carry["fdtd"])
                if U > peak_U:
                    peak_U = U
            # Forced-N escape preserved: decay_by=0.0 -> neither this branch
            # nor the peak branch above runs, so the loop is bounded by
            # min/max-steps alone; check_interval > max_steps -> the stop is
            # never evaluated.
            if decay_by > 0.0 and actual_steps >= min_steps and step % check_interval == 0:
                if U < decay_by * peak_U:
                    energy_below += 1
                    if energy_below >= decay_energy_consecutive:
                        decayed_fired = True
                        break
                else:
                    energy_below = 0
        else:
            # Closed/PEC fallback — BYTE-IDENTICAL to the pre-#169 point stop.
            # Decay check
            val_sq = float(monitor_val) ** 2
            if val_sq > peak_sq:
                peak_sq = val_sq

            if actual_steps >= min_steps and step % check_interval == 0 and peak_sq > 0.0:
                if val_sq < decay_by * peak_sq:
                    break

    if _reporter is not None and _reporter.last_reported != actual_steps:
        # The decay stop lands on an arbitrary step, so without this line the
        # log's last progress entry would understate the run by up to
        # report_every steps and never show where it actually stopped.
        jax.block_until_ready(carry["fdtd"])
        _reporter.report(actual_steps)

    # #388: measured static-remnant advisory on an absorbing-lane ENERGY cap-hit (the energy
    # criterion never fired). Complements the pre-run waveform-DC advisory. Not applicable to
    # the flux-stop lane (which rejects the static remnant by construction).
    if use_absorbing and not decayed_fired and not use_flux_stop:
        _warn_static_remnant_cap_hit(carry["fdtd"], materials, grid)

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
                # Stamp the scan dt (see run() above): the jit-safe in-scan
                # probe accumulator can't run update_waveguide_port_probe's
                # float(dt) stamp, so without this a cfg built without dt= keeps
                # dt=0 and the post-scan rect-DFT zeroes every spectrum.
                # The scan's own step (stencil_order=4 derates it).
                dt=float(_setup.dt),
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

    snapshots = snap_axes = None
    if snapshot is not None:
        if snap_frames:
            snapshots = {
                comp: jnp.stack([frame[i] for frame in snap_frames])
                for i, comp in enumerate(snapshot.components)}
        else:
            snapshots = {
                comp: jnp.zeros((0,) + s.shape, s.dtype)
                for comp, s in zip(
                    snapshot.components,
                    jax.eval_shape(_take_snapshot, carry["fdtd"]))}
        snap_axes = snapshot_axes(
            grid, snapshot, actual_steps, dt=_setup.dt)

    return SimResult(
        state=carry["fdtd"] if return_state else None,
        time_series=time_series,
        ntff_data=carry.get("ntff"),
        dft_planes=final_dft_planes,
        flux_monitors=final_flux_monitors,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        lumped_port_sparams=final_lumped_sparams,
        snapshots=snapshots,
        ntff_box=ntff,
        grid=grid,
        snapshot_axes=snap_axes,
        dt=_setup.dt,
    )
