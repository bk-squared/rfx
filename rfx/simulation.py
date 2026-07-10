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
    update_e, update_e_aniso, update_e_aniso_inv, update_h, EPS_0, _shift_bwd,
    precompute_coeffs, update_he_fast,
)
from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_occupancy


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
    wire_refplane_sparams : tuple | None
        Final reference-plane V/I DFT accumulators for the opt-in wire
        S-matrix plane path (issue #313).  Each entry is
        ``(WireRefPlaneSpec, (v_dft, i_minus_dft, i_plus_dft))``; two
        entries per opted port (plane slots 0 and 1).
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
    wire_refplane_sparams: tuple | None = None


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
    aniso_inv_eps: tuple | None = None,
    stencil_order: int = 2,
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
                        stencil_order=stencil_order), None, None

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
) -> "_SimSetup":
    """Build the shared setup artefacts used by both ``run`` and ``run_until_decay``.

    Returns a ``_SimSetup`` NamedTuple. Each driver extends ``ctx_kwargs``
    with its own driver-specific ``_StepContext`` fields before constructing
    the context.
    """
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
    _pec_faces_frozen = frozenset(_pec_faces) if use_pec_faces else frozenset()

    # ---- per-face PMC from grid.pmc_faces (T7 Phase 2 PR3) ----
    _pmc_faces = getattr(grid, "pmc_faces", None) or set()
    use_pmc_faces = bool(_pmc_faces)
    _pmc_faces_frozen = frozenset(_pmc_faces) if use_pmc_faces else frozenset()

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
    use_pec_mask = pec_mask is not None
    use_pec_occupancy = pec_occupancy is not None
    use_conformal = conformal_weights is not None
    # Stage 2: when aniso_inv_eps is set, the inverse-permittivity tensor
    # encodes both PEC behaviour and dielectric subpixel smoothing —
    # apply_conformal_pec is redundant and SKIPPED to avoid double-zeroing.
    use_aniso_inv = aniso_inv_eps is not None
    use_wire_sparams = len(wire_port_sparams) > 0
    use_lumped_sparams = len(lumped_port_sparams) > 0
    wire_refplane_sparams = wire_refplane_sparams or []
    use_wire_refplanes = len(wire_refplane_sparams) > 0
    use_lumped_rlc = len(lumped_rlc) > 0
    use_kerr = kerr_chi3 is not None
    use_mag_sources = len(mag_sources) > 0

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
    init_ntff_data_fn = accumulate_ntff_fn = None
    apply_kerr_ade = None
    update_rlc_element = None
    debye_coeffs = lorentz_coeffs = None

    # ---- initialise states ----
    _field_dtype = field_dtype if field_dtype is not None else jnp.float32
    fdtd = init_state(grid.shape, field_dtype=_field_dtype)
    carry_init: dict = {"fdtd": fdtd}

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml
        from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h
        cpml_params, cpml_state = init_cpml(grid)
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
            from rfx.sources.tfsf_2d import update_tfsf_2d_h, update_tfsf_2d_e

    if use_ntff:
        from rfx.farfield import init_ntff_data as _init_ntff_data
        from rfx.farfield import accumulate_ntff as _accumulate_ntff
        init_ntff_data_fn = _init_ntff_data
        accumulate_ntff_fn = _accumulate_ntff
        carry_init["ntff"] = _init_ntff_data(ntff)

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
        carry_init["lumped_sparam_accs"] = tuple(
            (
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # v_dft
                jnp.zeros(len(lp.freqs), dtype=_sparam_acc_dtype),  # i_dft
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
        (probe.component, probe.axis, probe.index, probe.freqs)
        for probe in dft_planes
    )
    waveguide_meta = tuple(waveguide_ports)

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
        use_pec_mask=use_pec_mask,
        use_pec_occupancy=use_pec_occupancy,
        use_conformal=use_conformal,
        use_wire_sparams=use_wire_sparams,
        use_lumped_sparams=use_lumped_sparams,
        use_wire_refplanes=use_wire_refplanes,
        use_lumped_rlc=use_lumped_rlc,
        use_kerr=use_kerr,
        use_mag_sources=use_mag_sources,
        cpml_params=cpml_params,
        cpml_axes=cpml_axes,
        upml_coeffs=upml_coeffs,
        tfsf_cfg=tfsf_cfg,
        tfsf_is_2d=_tfsf_is_2d,
        debye_coeffs=debye_coeffs,
        lorentz_coeffs=lorentz_coeffs,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        pec_mask=pec_mask,
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
    use_pec_mask: bool
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
    debye_coeffs: Any = None
    lorentz_coeffs: Any = None
    aniso_eps: Any = None
    aniso_inv_eps: Any = None
    pec_mask: Any = None
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
    pec_axes = ctx.pec_axes
    aniso_eps = ctx.aniso_eps
    aniso_inv_eps = ctx.aniso_inv_eps

    def core_step(carry, step_idx, src_vals, mag_src_vals):
        st = carry["fdtd"]
        tfsf_h_state = None

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
                              stencil_order=ctx.stencil_order)
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
                )

            # Kerr nonlinear ADE correction (after linear E-update)
            if ctx.use_kerr:
                st = ctx.apply_kerr_ade(st, ctx.kerr_chi3, dt)

            if ctx.use_tfsf:
                st = ctx.apply_tfsf_e(st, ctx.tfsf_cfg, tfsf_h_state, dx, dt)
            if ctx.use_waveguide_ports:
                from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e
                for cfg_meta in ctx.waveguide_meta:
                    st = _apply_wg_e(st, cfg_meta, step_idx, dt, dx)
            if ctx.use_cpml:
                st, cpml_new = ctx.apply_cpml_e(
                    st, ctx.cpml_params, cpml_new, grid, ctx.cpml_axes,
                    materials=materials)
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

            if pec_axes:
                st = apply_pec(st, axes=pec_axes)
            if ctx.use_pec_faces:
                st = apply_pec_faces(st, ctx.pec_faces_frozen)

            if ctx.use_conformal and not ctx.use_aniso_inv:
                # Stage 1 path. Stage 2 (use_aniso_inv) skips this —
                # the inv-eps tensor encodes the fully-PEC-cell zero
                # already, so this would be redundant double-zeroing.
                from rfx.geometry.conformal import apply_conformal_pec
                st = apply_conformal_pec(st, ctx.conformal_weights[0], ctx.conformal_weights[1], ctx.conformal_weights[2])
            elif ctx.use_pec_mask:
                from rfx.boundaries.pec import apply_pec_mask
                st = apply_pec_mask(st, ctx.pec_mask)

            if ctx.use_pec_occupancy:
                st = apply_pec_occupancy(st, ctx.pec_occupancy)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        if ctx.use_lumped_rlc:
            new_rlc_states = []
            for rlc_st, meta in zip(carry["rlc_states"], ctx.rlc_meta):
                st, rlc_st_new = ctx.update_rlc_element(st, rlc_st, meta)
                new_rlc_states.append(rlc_st_new)

        # Compute step time first; wire/lumped S-param DFT blocks below
        # need `t` and must accumulate BEFORE source injection per the
        # rfx/probes/probes.py update_sparam_probe docstring contract
        # ("sample after E-update/apply_pec but before apply_lumped_port
        # so V reflects only the cavity/load response, not the driving
        # waveform"). The JIT scan path violated it via PR #72 ordering,
        # producing 5–10 dB train/eval |S11| disagreement on near-matched
        # antennas where source-injection contamination is large relative
        # to V (issue #72).
        t = step_idx.astype(jnp.float32) * dt

        # Wire port S-param DFT accumulation BEFORE source injection so
        # that sampled V/I reflects only the load/cavity response.
        if ctx.use_wire_sparams:
            new_wire_accs = []
            for accs, wp_meta in zip(carry["wire_sparam_accs"], ctx.wire_sparam_meta):
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
        if ctx.use_lumped_sparams:
            new_lumped_accs = []
            for accs, lp_meta in zip(carry["lumped_sparam_accs"], ctx.lumped_sparam_meta):
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

        # Reference-plane V/I DFT accumulation (issue #313 opt-in) — same
        # pre-source-injection sample point and rect-DFT kernel as the
        # port-cell channels above.  The planes sit >= 1 cell from every
        # source cell, so pre/post-injection sampling is identical here.
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
            for acc, (component, axis, index, freqs) in zip(carry["dft_planes"], ctx.dft_meta):
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
        and not _ctx["use_tfsf"]
        and not _ctx["use_debye"]
        and not _ctx["use_lorentz"]
        and not _ctx["use_pec_mask"]
        and not _ctx["use_pec_occupancy"]
        and not _ctx["use_conformal"]
        and not _ctx["use_lumped_rlc"]
        and not _ctx["use_kerr"]
        and not _ctx["use_mag_sources"]
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
        precompute_coeffs(materials, dt, dx, pec_axes=_setup.pec_axes)
        if use_fast_he else None
    )

    # ---- run()-specific: snapshot setup ----
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
        use_snapshot=use_snapshot,
        use_monitor=False,
        use_flux_window=True,
        fast_coeffs=_fast_coeffs,
        flux_meta=flux_meta if use_flux_monitors else (),
        monitor_component="ez",
        mon_idx=None,
        snapshot_extractor=_take_snapshot if use_snapshot else None,
    )
    _core_step = make_core_step(_step_ctx)

    def step_fn(carry, xs):
        _step_idx, src_vals, mag_src_vals = xs
        new_carry, probe_out, extras = _core_step(
            carry, _step_idx, src_vals, mag_src_vals)
        if use_snapshot:
            output = (probe_out, extras["snap_fields"])
        else:
            output = (probe_out,)
        return new_carry, output

    # ---- run ----
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms, mag_src_waveforms)

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
                # Stamp the scan's authoritative dt so the post-scan rect-DFT
                # extractor uses the right Δt. The jit-safe in-scan probe
                # accumulator cannot run update_waveguide_port_probe's
                # ``float(dt)`` stamp, so a cfg built via init_waveguide_port
                # without dt= would otherwise keep dt=0 and silently zero every
                # extracted spectrum (S→0). The manual Python-loop path already
                # stamps dt via update_waveguide_port_probe; this makes the
                # compiled run() symmetric with it.
                dt=float(grid.dt),
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
    decay_energy_consecutive: int = 2,
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
    * ``snapshot`` is accepted for API symmetry but **silently ignored** —
      the Python loop does not accumulate per-step field snapshots.
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
         :func:`run` (see ``examples/crossval/03_straight_waveguide_flux.py``).

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
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        wire_port_sparams=wire_port_sparams,
        lumped_port_sparams=lumped_port_sparams,
        lumped_rlc=lumped_rlc,
        kerr_chi3=kerr_chi3,
        field_dtype=field_dtype,
        mag_sources=mag_sources,
        stencil_order=stencil_order,
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

    peak_sq = 0.0          # closed/PEC point-field running peak
    peak_U = 0.0           # absorbing interior-energy running peak (at checks)
    energy_below = 0       # consecutive sub-threshold energy checks
    all_probes = []
    actual_steps = 0

    for step in range(max_steps):
        step_idx = jnp.array(step, dtype=jnp.int32)
        src_vals = src_waveforms[step]
        mag_src_vals = mag_src_waveforms[step]
        carry, probe_out, monitor_val = _single_step(carry, step_idx, src_vals, mag_src_vals)

        all_probes.append(probe_out)
        actual_steps = step + 1

        if use_absorbing:
            # Interior-energy criterion. The reduction is check-step-only (the
            # whole-domain sum is the expensive part), so we compute U ONLY on
            # an eligible check step — never every step.
            if actual_steps >= min_steps and step % check_interval == 0:
                U = _interior_energy(carry["fdtd"])
                if U > peak_U:
                    peak_U = U
                # Forced-N escape preserved: decay_by=0.0 -> U < 0 is never
                # true (U >= 0) -> never fires; check_interval > max_steps ->
                # this branch is never entered. min/max-steps bound the loop.
                if U < decay_by * peak_U:
                    energy_below += 1
                    if energy_below >= decay_energy_consecutive:
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
                dt=float(grid.dt),
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
