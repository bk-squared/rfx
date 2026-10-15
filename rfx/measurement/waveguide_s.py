"""Waveguide travelling waves and S-parameter extraction."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from rfx.boundaries.axes import resolve_cpml_axes
from rfx.model.waveguide_modes import (
    C0_LOCAL,
    WaveguidePortConfig,
    _compute_beta,
    _compute_mode_impedance,
)
from rfx.measurement.settling import (
    settling_db_from_port_records,
)


def _co_located_current_spectrum(cfg: WaveguidePortConfig, current_dft: jnp.ndarray) -> jnp.ndarray:
    from rfx.measurement.modal import legacy_current_spectrum
    return legacy_current_spectrum(cfg, current_dft)


def _extract_global_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return global (+x/-x) modal waves from colocated modal V/I spectra."""
    z_mode = _compute_mode_impedance(
        cfg.freqs,
        cfg.f_cutoff,
        cfg.mode_type,
        dt=cfg.dt,
        dx=cfg.dx,
    )
    forward = 0.5 * (voltage_dft + z_mode * current_dft)
    backward = 0.5 * (voltage_dft - z_mode * current_dft)
    return forward, backward


def _rect_dft(time_series: jnp.ndarray, freqs: jnp.ndarray, dt: float,
              n_valid: jnp.ndarray | int, kind: str = 'E') -> jnp.ndarray:
    from rfx.measurement.modal import rect_dft
    return rect_dft(time_series, freqs, dt, n_valid, kind)


def _extract_global_waves_from_time_series(
    cfg: WaveguidePortConfig,
    voltage_t: jnp.ndarray,
    current_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Post-scan: rect-DFT the time series, then run the standard wave decomp.

    Mirrors ``_extract_global_waves`` but takes raw time-series inputs
    and computes the DFT in post-processing — eliminating the in-scan
    windowed/gated accumulator path that integrates partially-decayed
    standing waves and produces ``|S11| → ∞`` as the gate widens.
    """
    n_valid = cfg.n_steps_recorded
    voltage_dft = _rect_dft(voltage_t, cfg.freqs, cfg.dt, n_valid)
    current_dft = _rect_dft(current_t, cfg.freqs, cfg.dt, n_valid, 'H')
    return _extract_global_waves(cfg, voltage_dft, current_dft)


def _extract_port_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return port-local (incident, outgoing) waves.

    For a positive-direction port (`+x`, `+y`, `+z`), incident is the local
    forward wave and outgoing is the local backward wave. For a negative
    direction port, the mapping is reversed.
    """
    forward, backward = _extract_global_waves(cfg, voltage_dft, current_dft)
    if cfg.direction.startswith("+"):
        return forward, backward
    return backward, forward


def _extract_port_waves_from_time_series(
    cfg: WaveguidePortConfig,
    voltage_t: jnp.ndarray,
    current_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time-series variant of ``_extract_port_waves`` (post-scan rect DFT)."""
    forward, backward = _extract_global_waves_from_time_series(
        cfg, voltage_t, current_t
    )
    if cfg.direction.startswith("+"):
        return forward, backward
    return backward, forward


def waveguide_plane_positions(cfg: WaveguidePortConfig) -> dict[str, float]:
    """Physical source/reference/probe positions along the port normal axis.

    When `cfg` comes from `rfx.api.Simulation`, these are domain-relative
    physical coordinates along the active port-normal axis. For lower-level
    manual setups without `x_position`,
    they fall back to the padded-grid coordinate origin.
    """
    return {
        "source": cfg.source_x_m,
        "reference": cfg.reference_x_m,
        "probe": cfg.probe_x_m,
    }


def _mode_indices_from_config(cfg: WaveguidePortConfig) -> tuple[int, int]:
    """Best-effort extraction of (m, n) mode indices from a config.

    Uses the cutoff frequency and aperture dimensions to identify the mode.
    """
    if getattr(cfg, "mode_indices", (0, 0)) != (0, 0):
        return cfg.mode_indices
    a, b = cfg.a, cfg.b
    fc = cfg.f_cutoff
    kc = 2 * np.pi * fc / C0_LOCAL
    kc_sq = kc ** 2

    best_err = float('inf')
    best_mn = (0, 0)
    for m in range(0, 8):
        for n in range(0, 8):
            if m == 0 and n == 0:
                continue
            kc_sq_an = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
            err = abs(kc_sq - kc_sq_an) / max(kc_sq_an, 1e-30)
            if err < best_err:
                best_err = err
                best_mn = (m, n)
    if best_err < 0.35:
        return best_mn
    return (0, 0)


def _shift_modal_waves(
    forward: jnp.ndarray,
    backward: jnp.ndarray,
    beta: jnp.ndarray,
    shift_m: float,
    step_sign: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shift modal waves to a new reference plane.

    `shift_m` is in GLOBAL coordinates along the port normal axis. For a
    "+x" port (``step_sign=+1``) `shift_m > 0` moves the reporting plane
    downstream (local-forward). For a "-x" port (``step_sign=-1``) the
    local-forward direction is -x in global coords, so the incident-wave
    phase flips: we convert to local via ``local_shift = shift_m *
    step_sign`` and apply the same formula.

    Before 2026-04-22 the function ignored ``step_sign`` and applied
    the "+x" formula unconditionally — this silently gave the wrong sign
    on any negative-direction port, which is why user-specified
    ``reference_plane`` shifts produced only half-corrected phases for
    "-x"/"-y"/"-z" ports and the crossval/11 phase offset never closed.
    """
    if shift_m == 0.0:
        return forward, backward
    local_shift = shift_m * step_sign
    shift = jnp.asarray(local_shift, dtype=beta.dtype)
    forward_shifted = forward * jnp.exp(-1j * beta * shift)
    backward_shifted = backward * jnp.exp(+1j * beta * shift)
    return forward_shifted, backward_shifted


def extract_waveguide_sparams(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
    probe_shift: float = 0.0,
    v_ref_dft: jnp.ndarray | None = None,
    i_ref_dft: jnp.ndarray | None = None,
    v_probe_dft: jnp.ndarray | None = None,
    i_probe_dft: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (S11, S21) with optional reference-plane shifts.

    The Phase 2 cleanup (2026-04-25) removed the in-scan DFT accumulators
    (``cfg.v_ref_dft`` etc.) from ``WaveguidePortConfig``. Callers that
    used to read those fields off the config now must either:

    1. Pass V/I DFT spectra explicitly via the ``*_dft`` keyword args
       (used by direct unit-test mock paths and by callers that have
       precomputed the spectra outside the FDTD scan); OR
    2. Use ``extract_waveguide_port_waves`` /
       ``extract_waveguide_s_matrix`` which run the canonical post-scan
       rectangular full-record DFT on ``cfg.v_ref_t`` / ``cfg.i_ref_t``.

    When the explicit DFTs are not provided this function falls back to
    the post-scan rect-DFT path on the recorded time series.

    Parameters
    ----------
    ref_shift : float
        Metres to shift the reference-plane reporting location relative to the
        stored reference probe. Positive is downstream (+x), negative upstream.
    probe_shift : float
        Metres to shift the probe-plane reporting location relative to the
        stored probe plane. Positive is downstream (+x), negative upstream.
    v_ref_dft, i_ref_dft, v_probe_dft, i_probe_dft : array or None
        Optional precomputed V/I spectra at the reference and probe
        planes. When all are given the wave decomposition runs against
        them directly; otherwise the recorded time-series on ``cfg`` is
        used.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx)
    step_sign = 1 if cfg.direction.startswith("+") else -1
    explicit = (
        v_ref_dft is not None and i_ref_dft is not None
        and v_probe_dft is not None and i_probe_dft is not None
    )
    if explicit:
        a_ref, b_ref = _extract_port_waves(cfg, v_ref_dft, i_ref_dft)
        a_probe, b_probe = _extract_port_waves(cfg, v_probe_dft, i_probe_dft)
    else:
        a_ref, b_ref = _extract_port_waves_from_time_series(
            cfg, cfg.v_ref_t, cfg.i_ref_t,
        )
        a_probe, b_probe = _extract_port_waves_from_time_series(
            cfg, cfg.v_probe_t, cfg.i_probe_t,
        )
    a_ref, b_ref = _shift_modal_waves(a_ref, b_ref, beta, ref_shift, step_sign)
    a_probe, b_probe = _shift_modal_waves(a_probe, b_probe, beta, probe_shift, step_sign)
    safe_ref = jnp.where(jnp.abs(a_ref) > 0, a_ref, jnp.ones_like(a_ref))
    s11 = b_ref / safe_ref
    s21 = a_probe / safe_ref
    return s11, s21


def extract_waveguide_s21(cfg: WaveguidePortConfig,
                          dt: float = 0.0,
                          *,
                          ref_shift: float = 0.0,
                          probe_shift: float = 0.0) -> jnp.ndarray:
    """Extract S21 from forward-wave modal amplitudes with optional de-embedding."""
    _, s21 = extract_waveguide_sparams(
        cfg,
        ref_shift=ref_shift,
        probe_shift=probe_shift,
    )
    return s21


def extract_waveguide_s11(cfg: WaveguidePortConfig,
                          *,
                          ref_shift: float = 0.0) -> jnp.ndarray:
    """Extract S11 from backward/forward modal amplitudes with optional de-embedding."""
    s11, _ = extract_waveguide_sparams(cfg, ref_shift=ref_shift)
    return s11


def extract_waveguide_port_waves(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return port-local (incident, outgoing) waves at a shifted reference plane.

    The spectra are recomputed POST-SCAN from the per-step modal V/I
    time series via a rectangular full-record DFT (matching OpenEMS's
    ``utilities.DFT_time2freq``). The legacy in-scan windowed/gated
    DFT accumulators were removed in the Phase 2 cleanup
    (2026-04-25): they integrated partially-decayed standing waves
    and produced ``|S11|`` that grew without bound as the gate widened.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx)
    step_sign = 1 if cfg.direction.startswith("+") else -1
    a_ref, b_ref = _extract_port_waves_from_time_series(
        cfg, cfg.v_ref_t, cfg.i_ref_t,
    )
    return _shift_modal_waves(a_ref, b_ref, beta, ref_shift, step_sign)


def _s_matrix_dtypes(field_dtype, port_cfgs):
    """Promote field, record, frequency, and JAX default float precision."""
    # Preserve the frequency-based records from init_waveguide_port, widening
    # for field storage and the JAX default so extraction is never narrower
    # than main, including explicit float32 frequencies with x64 enabled.
    record_dtype = jnp.result_type(
        jnp.float32 if field_dtype is None else field_dtype,
        jnp.float32,
        jnp.result_type(float),
        *(cfg.v_probe_t.dtype for cfg in port_cfgs),
        *(cfg.freqs.dtype for cfg in port_cfgs),
    )
    return record_dtype, jnp.result_type(record_dtype, 1j)


def _s_matrix_port_waves(cfg, *, ref_shift):
    """Store modal amplitudes with the JAX default float precision floor."""
    waves = extract_waveguide_port_waves(cfg, ref_shift=ref_shift)
    complex_dtype = jnp.result_type(
        cfg.v_ref_t.dtype, cfg.freqs.dtype, jnp.result_type(float), 1j,
    )
    return tuple(wave.astype(complex_dtype) for wave in waves)


def extract_waveguide_s_matrix(
    grid,
    materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str | None = None,
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, float] | None = None,
    aniso_eps: tuple | None = None,
    conformal_weights: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    checkpoint_segments: int | None = None,
    return_settling: bool = False,
    settling_details: list | None = None,
    sheet_impedance: object | None = None,
    pec_edge_masks: tuple | None = None,
    field_dtype=None,
) -> "jnp.ndarray | tuple[jnp.ndarray, np.ndarray]":
    """Assemble an x-directed waveguide S-matrix via one-driven-port-at-a-time runs.

    ``field_dtype`` is forwarded to every core run; ``None`` uses float32.
    Modal records preserve their precision and widen for field storage.
    Spectra and S promote the record and frequency dtypes.

    PEC realization (#931 §1.7): interior ``Box(material='pec')`` walls
    reach this lane as the realized edge masks ``pec_edge_masks``
    (``rfx.boundaries.pec.realized_pec_edge_masks``), applied per step by
    the shared ``apply_pec_edges``.  Until #931 the caller folded the PEC
    CELL mask into ``sigma = 1e10`` instead — a fourth realization of the
    same geometry, and one that damped only the components indexed by the
    occupied cell, so a one-cell-thick wall or iris got its lower face and
    never its far one.  An empty-guide REFERENCE run gets no interior edge
    masks; per-port reference geometry supplies its own.


    ``return_settling=True`` (issue #538) additionally returns the
    per-driven-run pole-tail amplitude witness: ``(S, settling_db)`` with
    ``settling_db`` shape ``(n_ports,)`` from
    :func:`settling_db_from_port_records` (NaN per run under tracing).
    Default ``False`` keeps the legacy single-array return for existing
    callers, and the S values are identical either way — the witness is
    post-processing of records the scan already produces.

    checkpoint_segments : int or None
        When set, enables segmented gradient checkpointing on each per-port
        FDTD run (forwarded to ``rfx.simulation.run`` as
        ``checkpoint=True, checkpoint_segments=K``).  See
        :meth:`Simulation.compute_waveguide_s_matrix` for full semantics.
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_matrix requires at least two waveguide ports"
        )
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in port_cfgs)
    if len(ref_shifts) != len(port_cfgs):
        raise ValueError("ref_shifts must match the number of waveguide ports when provided")

    from rfx.simulation import run as run_simulation

    template_cfgs = tuple(port_cfgs)
    record_dtype, complex_dtype = _s_matrix_dtypes(field_dtype, template_cfgs)
    n_ports = len(template_cfgs)
    # G-AD-WIRE: use a list-of-columns accumulator so the final jnp.stack
    # keeps all b_recv/safe_a values on the JAX tape.  The old np.zeros +
    # np.array item-assign pattern concretised every JAX array and broke
    # jax.grad.  Shape: s_cols[drive_idx] = list of (n_freqs,) jnp arrays
    # in recv order; stacked to (n_ports, n_freqs) then assembled to
    # (n_ports, n_ports, n_freqs) via jnp.stack(axis=1).
    s_cols: list[jnp.ndarray] = []
    settling_runs: list[float] = []

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t, dtype=record_dtype)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    for drive_idx in range(n_ports):
        driven_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        _wg_checkpoint = checkpoint_segments is not None
        result = run_simulation(
            grid,
            materials,
            n_steps,
            boundary=boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            waveguide_ports=driven_cfgs,
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=pec_edge_masks,
            checkpoint=_wg_checkpoint,
            checkpoint_segments=checkpoint_segments,
            sheet_impedance=sheet_impedance,
            field_dtype=field_dtype,
        )
        final_cfgs = result.waveguide_ports or ()
        if len(final_cfgs) != n_ports:
            raise RuntimeError("waveguide S-matrix extraction expected one final config per port")
        if return_settling:
            value, detail = settling_db_from_port_records(
                final_cfgs, freq_max=getattr(grid, "freq_max", None),
                return_detail=True)
            settling_runs.append(value)
            if settling_details is not None:
                settling_details.append(detail)

        # NB: run() stamps grid.dt onto every returned waveguide cfg, so the
        # post-scan rect-DFT extractor below uses the correct Δt even when the
        # cfgs were built via init_waveguide_port without dt=.

        a_drive, b_drive = _s_matrix_port_waves(
            final_cfgs[drive_idx],
            ref_shift=ref_shifts[drive_idx],
        )
        safe_a = jnp.where(jnp.abs(a_drive) > 0, a_drive, jnp.ones_like(a_drive))
        col_slices: list[jnp.ndarray] = []
        for recv_idx, cfg in enumerate(final_cfgs):
            _a_recv, b_recv = _s_matrix_port_waves(
                cfg,
                ref_shift=ref_shifts[recv_idx],
            )
            # G-AD-WIRE: keep on JAX tape — no np.array() concretisation.
            col_slices.append(b_recv / safe_a)
        # Stack recv slices into column (n_ports, n_freqs).
        s_cols.append(jnp.stack(col_slices, axis=0))

    # Assemble full S-matrix: stack drive columns along axis=1 →
    # (n_ports, n_ports, n_freqs).
    s_matrix = jnp.stack(s_cols, axis=1).astype(complex_dtype)
    if return_settling:
        return s_matrix, np.asarray(settling_runs, dtype=float)
    return s_matrix


def extract_waveguide_s_matrix_flux(
    grid,
    materials,
    ref_materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str | None = None,
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_debye: tuple | None = None,
    ref_lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, ...] | None = None,
    aniso_eps: tuple | None = None,
    conformal_weights: tuple | None = None,
    ref_aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    ref_aniso_inv_eps: tuple | None = None,
    ref_materials_per_port: "list | None" = None,
    sheet_impedance: object | None = None,
    pec_edge_masks: tuple | None = None,
    ref_pec_edge_masks_per_port: "list | None" = None,
    checkpoint_segments: int | None = None,
    return_settling: bool = False,
    settling_details: list | None = None,
    field_dtype=None,
) -> "jnp.ndarray | tuple[jnp.ndarray, np.ndarray]":
    """Hybrid power-flux magnitude + modal phase waveguide S-matrix.

    ``field_dtype`` is forwarded to every core run; ``None`` uses float32.
    Modal records preserve their precision and widen for field storage.
    Spectra and S promote the record and frequency dtypes.

    PEC realization (#931 §1.7): interior ``Box(material='pec')`` walls
    reach this lane as the realized edge masks ``pec_edge_masks``
    (``rfx.boundaries.pec.realized_pec_edge_masks``), applied per step by
    the shared ``apply_pec_edges``.  Until #931 the caller folded the PEC
    CELL mask into ``sigma = 1e10`` instead — a fourth realization of the
    same geometry, and one that damped only the components indexed by the
    occupied cell, so a one-cell-thick wall or iris got its lower face and
    never its far one.  An empty-guide REFERENCE run gets no interior edge
    masks; per-port reference geometry supplies its own.


    ``return_settling=True`` (issue #538): returns ``(S, settling_db)``;
    the witness per drive is the WORST of the device and reference runs'
    port records (the reference run produces the ``P_inc``/``a_inc``
    normalization, so its truncation corrupts the same S values). Scope:
    modal port records only — non-modal power crossing the flux plane
    enters the |S| magnitudes but is orthogonal-projected out of the
    witness. NaN per run under tracing. Default ``False`` keeps the
    legacy single-array return; S is identical either way.

    ``ref_materials_per_port`` (optional): a per-driven-port list of reference
    materials. These must be PEC-FOLDED (walls as sigma≈1e10) exactly like the
    device ``materials`` the caller passes in — NOT raw ``_build_materials``
    output, where interior ``Box(material='pec')`` walls are NOT yet in
    eps/sigma (they are folded from a pec-mask by ``compute_waveguide_s_matrix``
    before this call). For a STRAIGHT guide the device PEC walls also bound the
    vacuum reference (walls come from ``pec_axes``), so a single vacuum
    ``ref_materials`` yields the correct guided incident power P_inc. For a
    JUNCTION (T/branch, walls in ``materials`` with ``cpml_axes`` open), a vacuum
    reference radiates into free space and P_inc is mis-powered (all |S| inflate
    hard, non-passive). Passing ``ref_materials_per_port[i]`` = a matched STRAIGHT
    guide for driven arm ``i`` (its walls extended through, no junction) moves
    P_inc toward the true single-mode guided incident power.

    NOT passive-by-construction (verified 2026-07-05, compact 3-port T-junction):
    the per-port straight-guide reference is NECESSARY but NOT SUFFICIENT — it
    fixes |S11| (1.86→0.49, physical) and roughly halves the blow-up (max|S|
    9.8→3.9), yet does NOT achieve passivity on a compact junction (residual
    max|S|≈3.9, per-drive energy sum 2-6×). Additional multi-port de-embedding
    (probe-plane placement, higher-order-mode filtering, larger port-to-junction
    separation) is still required. See the skipped ``test_api.py`` T-junction
    reciprocity test.

    For each driven port ``i``:

      1. Reference run (empty guide, port ``i`` driven) → incident
         Poynting flux ``P_inc`` at the driven port probe plane.
      2. Device run (port ``i`` driven) → Poynting flux at every port
         probe plane.

    Magnitudes are assembled from power ratios::

        |S_ii|² = |F_ref[i] − F_dev[i]| / |F_ref[i]|   (reflection)
        |S_ji|² = |F_dev[j]|            / |F_ref[i]|   (transmission, j≠i)

    Phase is taken from the modal V/I decomposition of the device run.
    This resolves both the Z_TE impedance-mismatch error that corrupts
    ``normalize=False`` S11 (~3 % at dx/λ=0.07) and the round-trip
    dispersion error in the ``normalize=True`` diagonal formula.

    Parameters mirror ``extract_waveguide_s_params_normalized``; see that
    function for argument documentation.
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_matrix_flux requires at least two waveguide ports"
        )
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in port_cfgs)
    if len(ref_shifts) != len(port_cfgs):
        raise ValueError("ref_shifts must match the number of waveguide ports when provided")

    from rfx.simulation import run as run_simulation
    from rfx.probes.probes import flux_spectrum
    from rfx.measurement import grid_flux_monitor

    _AXIS_IDX = {"x": 0, "y": 1, "z": 2}

    template_cfgs = tuple(port_cfgs)
    record_dtype, complex_dtype = _s_matrix_dtypes(field_dtype, template_cfgs)
    n_ports = len(template_cfgs)
    freqs = template_cfgs[0].freqs
    # jnp-native assembly (issue #148): columns are collected per drive
    # port and stacked at the end — no in-place np mutation, so traced
    # values flow through to the returned S-matrix.
    s_columns: list = []
    settling_runs: list[float] = []

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t, dtype=record_dtype)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    def _make_flux_monitors():
        from rfx.boundaries.pec import resolve_wall_faces
        monitors = [
            grid_flux_monitor(
                grid, axis=_AXIS_IDX[cfg.normal_axis],
                index=cfg.probe_x,
                freqs=freqs,
                dft_total_steps=n_steps,
                periodic=periodic,
                lo1=cfg.u_lo, hi1=cfg.u_hi,
                lo2=cfg.v_lo, hi2=cfg.v_hi,
                pmc_faces=resolve_wall_faces(grid, periodic or (False, False, False))[1],
            )
            for cfg in template_cfgs
        ]
        return [mon._replace(
            e1_dft=mon.e1_dft.astype(complex_dtype),
            e2_dft=mon.e2_dft.astype(complex_dtype),
            h1_dft=mon.h1_dft.astype(complex_dtype),
            h2_dft=mon.h2_dft.astype(complex_dtype),
        ) for mon in monitors]

    common_run_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
    )

    for drive_idx in range(n_ports):
        # --- Reference run (empty guide, port drive_idx driven) ---
        ref_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        # Per-port matched-straight-guide reference for junctions (else vacuum).
        ref_mat_drive = (
            ref_materials_per_port[drive_idx]
            if ref_materials_per_port is not None else ref_materials
        )
        _flux_checkpoint = checkpoint_segments is not None
        # #677: the vacuum/straight-guide REFERENCE deliberately runs with
        # NO sheet ctx (explicit strip — the sheet no longer rides
        # materials.sigma, so vacuum materials alone do not strip it).
        _ref_edges = (
            None if ref_pec_edge_masks_per_port is None
            else ref_pec_edge_masks_per_port[drive_idx])
        ref_result = run_simulation(
            grid, ref_mat_drive, n_steps,
            debye=ref_debye, lorentz=ref_lorentz,
            waveguide_ports=ref_cfgs,
            flux_monitors=_make_flux_monitors(),
            aniso_eps=ref_aniso_eps,
            aniso_inv_eps=ref_aniso_inv_eps,
            pec_edge_masks=_ref_edges,
            checkpoint=_flux_checkpoint,
            checkpoint_segments=checkpoint_segments,
            field_dtype=field_dtype,
            **common_run_kw,
        )
        ref_final_cfgs = ref_result.waveguide_ports or ()
        ref_final_mons = ref_result.flux_monitors or ()
        if len(ref_final_cfgs) != n_ports:
            raise RuntimeError("waveguide S-matrix extraction expected one final config per port")

        # Incident power: signed flux at driven port in the reference guide.
        # |F_ref| = P_inc regardless of port direction (+x vs −x).
        # jnp-native (issue #148): no np.array() concretization — keeps
        # the whole flux extraction on the AD tape for
        # eps_override-traced gradients.
        F_ref_drive = flux_spectrum(ref_final_mons[drive_idx])  # (n_freqs,)
        P_inc = jnp.abs(F_ref_drive)
        safe_P_inc = jnp.where(P_inc > 1e-60, P_inc, jnp.ones_like(P_inc))

        # Modal incident wave (reference run) — used to determine phase reference
        a_inc_ref, _ = _s_matrix_port_waves(
            ref_final_cfgs[drive_idx], ref_shift=ref_shifts[drive_idx]
        )

        # --- Device run (port drive_idx driven) ---
        dev_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        dev_result = run_simulation(
            grid, materials, n_steps,
            debye=debye, lorentz=lorentz,
            waveguide_ports=dev_cfgs,
            flux_monitors=_make_flux_monitors(),
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=pec_edge_masks,
            checkpoint=_flux_checkpoint,
            checkpoint_segments=checkpoint_segments,
            sheet_impedance=sheet_impedance,
            field_dtype=field_dtype,
            **common_run_kw,
        )
        dev_final_cfgs = dev_result.waveguide_ports or ()
        dev_final_mons = dev_result.flux_monitors or ()
        if len(dev_final_cfgs) != n_ports:
            raise RuntimeError("waveguide S-matrix extraction expected one final config per port")
        if return_settling:
            # worst over BOTH runs of the pair: the reference run's records
            # feed P_inc/a_inc, so its truncation corrupts the same S values
            # (review finding — max is the conservative composition; NaN
            # from a traced run propagates instead of being max()-eaten)
            _sd_dev, _detail_dev = settling_db_from_port_records(
                dev_final_cfgs, freq_max=getattr(grid, "freq_max", None), return_detail=True)
            _sd_ref, _detail_ref = settling_db_from_port_records(
                ref_final_cfgs, freq_max=getattr(grid, "freq_max", None), return_detail=True)
            settling_runs.append(
                float("nan") if (np.isnan(_sd_dev) or np.isnan(_sd_ref))
                else max(_sd_dev, _sd_ref))
            if settling_details is not None:
                from rfx.sparams._tail_witness import combine_witness_details
                settling_details.append(combine_witness_details(_detail_dev, _detail_ref))

        F_dev_drive = flux_spectrum(dev_final_mons[drive_idx])

        col_entries = []
        for recv_idx, dev_cfg in enumerate(dev_final_cfgs):
            # Phase from modal V/I decomposition of device run
            _, b_recv_dev = _s_matrix_port_waves(
                dev_cfg, ref_shift=ref_shifts[recv_idx]
            )
            safe_a = jnp.where(jnp.abs(a_inc_ref) > 1e-60, a_inc_ref, jnp.ones_like(a_inc_ref))
            # AD-safe angle (double-where): angle() has an undefined
            # gradient at 0; angle(1)=0 matches np.angle(0)=0 so the
            # primal is unchanged.
            ratio = b_recv_dev / safe_a
            ratio_ok = jnp.abs(ratio) > 0.0
            phase = jnp.angle(jnp.where(ratio_ok, ratio, jnp.ones_like(ratio)))

            if recv_idx == drive_idx:
                # Reflection: ref_flux − dev_flux = power that turned around
                P_num = jnp.abs(F_ref_drive - F_dev_drive)
            else:
                # Transmission: net flux through receiving port's probe plane
                P_num = jnp.abs(flux_spectrum(dev_final_mons[recv_idx]))
            # AD-safe sqrt (double-where): a perfect match/null makes the
            # power ratio exactly 0, where d(sqrt)/dx = inf would leak
            # 0*inf=nan through the backward pass; primal stays exactly
            # sqrt(x) for x>0 and exactly 0 at x=0.
            p_ratio = P_num / safe_P_inc
            p_ok = p_ratio > 0.0
            mag = jnp.where(
                p_ok, jnp.sqrt(jnp.where(p_ok, p_ratio, jnp.ones_like(p_ratio))), 0.0
            )

            # Modal records and flux accumulators use the precision selected
            # for this run, including the reference's incident spectrum.
            col_entries.append(mag * jnp.exp(1j * phase))

        # (n_ports, n_freqs) column for this drive port
        s_columns.append(jnp.stack(col_entries, axis=0))

    # Stack drive columns on axis 1 -> (recv, drive, n_freqs)
    s_matrix = jnp.stack(s_columns, axis=1).astype(complex_dtype)
    if return_settling:
        return s_matrix, np.asarray(settling_runs, dtype=float)
    return s_matrix


def extract_waveguide_s_params_normalized(
    grid,
    materials,
    ref_materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str | None = None,
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_debye: tuple | None = None,
    ref_lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, ...] | None = None,
    aniso_eps: tuple | None = None,
    conformal_weights: tuple | None = None,
    ref_aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    ref_aniso_inv_eps: tuple | None = None,
    checkpoint_segments: int | None = None,
    return_settling: bool = False,
    settling_details: list | None = None,
    sheet_impedance: object | None = None,
    pec_edge_masks: tuple | None = None,
    field_dtype=None,
) -> "jnp.ndarray | tuple[jnp.ndarray, np.ndarray]":
    """Two-run normalized waveguide S-matrix.

    ``field_dtype`` is forwarded to every core run; ``None`` uses float32.
    Modal records preserve their precision and widen for field storage.
    Spectra and S promote the record and frequency dtypes.

    PEC realization (#931 §1.7): interior ``Box(material='pec')`` walls
    reach this lane as the realized edge masks ``pec_edge_masks``
    (``rfx.boundaries.pec.realized_pec_edge_masks``), applied per step by
    the shared ``apply_pec_edges``.  Until #931 the caller folded the PEC
    CELL mask into ``sigma = 1e10`` instead — a fourth realization of the
    same geometry, and one that damped only the components indexed by the
    occupied cell, so a one-cell-thick wall or iris got its lower face and
    never its far one.  An empty-guide REFERENCE run gets no interior edge
    masks; per-port reference geometry supplies its own.


    Cancels Yee-grid numerical dispersion for **transmission** (off-diagonal)
    terms by normalizing device outgoing waves against reference-run waves
    measured at the same port location.

    For each driven port j:
      1. Run reference (empty guide) with port j driven.
      2. Run device with port j driven.
      3. Off-diagonal (i != j):
           S_ij = b_out_device[i] / b_out_reference[i]
         The reference b_out[i] accumulates the same one-way dispersion as
         the device wave, so the ratio cancels the bias exactly.
      4. Diagonal (i == j):
           S_jj = (b_out_device[j] - b_out_reference[j]) / a_inc_reference[j]
         The subtraction removes the empty-guide CPML/discretization
         contribution at the driven port so S_jj measures only the
         device-induced reflection.

    **Known limitation — S11 of strong reflectors**: the diagonal formula
    above does NOT cancel dispersion for reflection. The device wave is a
    round-trip (port → obstacle → port) while the reference wave is
    one-way (port → CPML). The dispersion accumulated over these different
    path lengths cannot cancel, and the source-plane impedance mismatch
    (Z_TE_exact vs Z_TE_num, ~3 % at dx/λ=0.07) shows up as coherent
    standing-wave error in b_ref[j] — producing ±10–20 % swings in |S11|.
    For strong reflectors use ``normalize=False`` instead; Yee dispersion
    contributes < 2 % to |S11| magnitude at dx/λ < 0.05.

    A power-flux (Poynting-vector) extraction analogous to Meep's
    ``add_flux`` would eliminate this limitation for both S11 and S21.
    Tracked at https://github.com/bk-squared/rfx/issues.

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    materials : MaterialArrays
        Device material arrays.
    ref_materials : MaterialArrays
        Reference (empty waveguide) material arrays.
    port_cfgs : list[WaveguidePortConfig]
        Port configurations (same for both runs).
    n_steps : int
        Number of timesteps for each run.
    boundary, cpml_axes, pec_axes, periodic : str / tuple
        Boundary conditions (same for both runs).
    debye, lorentz : tuple or None
        Dispersion specs for the device run.
    ref_debye, ref_lorentz : tuple or None
        Dispersion specs for the reference run.
    ref_shifts : list[float] or None
        Optional reference-plane shifts per port.

    Returns
    -------
    jnp.ndarray
        Normalized S-matrix of shape (n_ports, n_ports, n_freqs), complex.
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_params_normalized requires at least two waveguide ports"
        )
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in port_cfgs)
    if len(ref_shifts) != len(port_cfgs):
        raise ValueError("ref_shifts must match the number of waveguide ports when provided")

    from rfx.simulation import run as run_simulation

    template_cfgs = tuple(port_cfgs)
    record_dtype, complex_dtype = _s_matrix_dtypes(field_dtype, template_cfgs)
    n_ports = len(template_cfgs)
    n_freqs = len(template_cfgs[0].freqs)
    s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=complex_dtype)

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t, dtype=record_dtype)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    common_run_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
    )
    # ``aniso_eps`` is per-component smoothed permittivity for the device
    # geometry. The reference run is vacuum and has no ε interfaces, so
    # it always passes aniso_eps=None — the two-run cancellation only
    # benefits if the smoothed ε is applied to the device run.

    settling_runs: list[float] = []
    for drive_idx in range(n_ports):
        # --- Reference run: extract waves at all ports ---
        ref_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        _norm_checkpoint = checkpoint_segments is not None
        # #677: the vacuum REFERENCE deliberately runs with NO sheet ctx
        # (explicit strip — vacuum materials alone do not strip a sheet
        # that no longer rides materials.sigma).
        ref_result = run_simulation(
            grid, ref_materials, n_steps,
            debye=ref_debye, lorentz=ref_lorentz,
            waveguide_ports=ref_cfgs,
            aniso_inv_eps=ref_aniso_inv_eps,
            checkpoint=_norm_checkpoint,
            checkpoint_segments=checkpoint_segments,
            field_dtype=field_dtype,
            **common_run_kw,
        )
        ref_final_cfgs = ref_result.waveguide_ports or ()
        if len(ref_final_cfgs) != n_ports:
            raise RuntimeError(
                "waveguide S-matrix extraction expected one final config per port"
            )

        # Incident wave at the driven port (for diagonal normalization)
        a_inc_ref, _ = _s_matrix_port_waves(
            ref_final_cfgs[drive_idx],
            ref_shift=ref_shifts[drive_idx],
        )
        a_inc_ref_np = np.array(a_inc_ref)
        safe_a_inc = np.where(
            np.abs(a_inc_ref_np) > 1e-30,
            a_inc_ref_np,
            np.ones_like(a_inc_ref_np),
        )

        # Outgoing waves at all ports (for off-diagonal normalization)
        b_out_ref = []
        for recv_idx in range(n_ports):
            _, b_ref_i = _s_matrix_port_waves(
                ref_final_cfgs[recv_idx],
                ref_shift=ref_shifts[recv_idx],
                )
            b_out_ref.append(np.array(b_ref_i))

        # --- Device run: extract outgoing waves at every port ---
        dev_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        dev_result = run_simulation(
            grid, materials, n_steps,
            debye=debye, lorentz=lorentz,
            waveguide_ports=dev_cfgs, aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=pec_edge_masks,
            checkpoint=_norm_checkpoint,
            checkpoint_segments=checkpoint_segments,
            sheet_impedance=sheet_impedance,
            field_dtype=field_dtype,
            **common_run_kw,
        )
        dev_final_cfgs = dev_result.waveguide_ports or ()
        if len(dev_final_cfgs) != n_ports:
            raise RuntimeError(
                "waveguide S-matrix extraction expected one final config per port"
            )
        if return_settling:
            # worst over BOTH runs of the pair (same reasoning as the flux
            # variant: the reference run produces the normalization records)
            _sd_dev, _detail_dev = settling_db_from_port_records(
                dev_final_cfgs, freq_max=getattr(grid, "freq_max", None), return_detail=True)
            _sd_ref, _detail_ref = settling_db_from_port_records(
                ref_final_cfgs, freq_max=getattr(grid, "freq_max", None), return_detail=True)
            settling_runs.append(
                float("nan") if (np.isnan(_sd_dev) or np.isnan(_sd_ref))
                else max(_sd_dev, _sd_ref))
            if settling_details is not None:
                from rfx.sparams._tail_witness import combine_witness_details
                settling_details.append(combine_witness_details(_detail_dev, _detail_ref))

        for recv_idx, cfg in enumerate(dev_final_cfgs):
            _, b_recv_dev = _s_matrix_port_waves(
                cfg,
                ref_shift=ref_shifts[recv_idx],
                )
            b_recv_dev_np = np.array(b_recv_dev)

            if recv_idx == drive_idx:
                # Diagonal: subtract the reference-run reflection so the
                # result captures only device-induced reflection.
                s_matrix[recv_idx, drive_idx, :] = (
                    b_recv_dev_np - b_out_ref[recv_idx]
                ) / safe_a_inc
            else:
                # Off-diagonal: normalize by reference outgoing at same port
                safe_b_ref = np.where(
                    np.abs(b_out_ref[recv_idx]) > 1e-30,
                    b_out_ref[recv_idx],
                    np.ones_like(b_out_ref[recv_idx]),
                )
                s_matrix[recv_idx, drive_idx, :] = b_recv_dev_np / safe_b_ref

    if return_settling:
        import numpy as _np
        return jnp.asarray(s_matrix), _np.asarray(settling_runs, dtype=float)
    return jnp.asarray(s_matrix)


def extract_multimode_s_matrix(
    grid,
    materials,
    port_mode_cfgs: list[list[WaveguidePortConfig]],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str | None = None,
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_shifts: list[float] | None = None,
    aniso_eps: tuple | None = None,
    conformal_weights: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    pec_edge_masks: tuple | None = None,
    field_dtype=None,
    settling_details: list | None = None,
) -> tuple[jnp.ndarray, list[tuple[int, int, str, tuple[int, int]]]]:
    """Assemble a multi-mode waveguide S-matrix.

    ``field_dtype`` is forwarded to every core run; ``None`` uses float32.
    Modal records preserve their precision and widen for field storage.
    Spectra and S promote the record and frequency dtypes.

    Each physical port may have multiple modes.  The S-matrix indices
    enumerate (port_index, mode_index) pairs.

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    materials : MaterialArrays
        Material arrays.
    port_mode_cfgs : list of list of WaveguidePortConfig
        ``port_mode_cfgs[p]`` is a list of configs for physical port ``p``,
        one per mode.
    n_steps : int
        Number of timesteps per run.
    ref_shifts : list[float] or None
        Reference-plane shifts per physical port.  Applied to all modes
        of each port.

    Returns
    -------
    s_matrix : jnp.ndarray, shape (N, N, n_freqs)
        S-matrix where N = sum of modes across all ports.
    mode_map : list of (port_idx, mode_idx, mode_type, (m, n))
        Ordering of rows/columns in the S-matrix.
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    from rfx.simulation import run as run_simulation

    # Flatten to a linear list, keeping track of (port_idx, mode_within_port)
    flat_cfgs: list[WaveguidePortConfig] = []
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []

    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(mode_cfgs):
            flat_cfgs.append(cfg)
            # Extract mode info from the config
            m_idx = _mode_indices_from_config(cfg)
            mode_map.append((port_idx, mode_within, cfg.mode_type, m_idx))

    record_dtype, complex_dtype = _s_matrix_dtypes(field_dtype, flat_cfgs)
    n_total = len(flat_cfgs)
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=complex_dtype)

    n_ports = len(port_mode_cfgs)
    if ref_shifts is None:
        ref_shifts = [0.0] * n_ports
    if len(ref_shifts) != n_ports:
        raise ValueError("ref_shifts must have one entry per physical port")

    # Expand ref_shifts to flat indexing
    flat_ref_shifts = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for _ in mode_cfgs:
            flat_ref_shifts.append(ref_shifts[port_idx])

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t, dtype=record_dtype)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    # Drive each mode one at a time
    for drive_flat_idx in range(n_total):
        driven_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        # Set the driven mode amplitude to the original first-mode amplitude
        orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0
        driven_cfgs[drive_flat_idx] = driven_cfgs[drive_flat_idx]._replace(
            src_amp=orig_amp,
        )

        result = run_simulation(
            grid, materials, n_steps,
            boundary=boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            waveguide_ports=driven_cfgs,
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=pec_edge_masks,
            field_dtype=field_dtype,
        )
        final_cfgs = result.waveguide_ports or ()
        if settling_details is not None:
            _, detail = settling_db_from_port_records(
                final_cfgs, freq_max=getattr(grid, 'freq_max', None), return_detail=True)
            settling_details.append(detail)
        if len(final_cfgs) != n_total:
            raise RuntimeError(
                f"Expected {n_total} final waveguide configs, got {len(final_cfgs)}"
            )

        a_drive, _ = _s_matrix_port_waves(
            final_cfgs[drive_flat_idx],
            ref_shift=flat_ref_shifts[drive_flat_idx],
        )
        safe_a = jnp.where(jnp.abs(a_drive) > 0, a_drive, jnp.ones_like(a_drive))

        for recv_idx, cfg in enumerate(final_cfgs):
            _, b_recv = _s_matrix_port_waves(
                cfg,
                ref_shift=flat_ref_shifts[recv_idx],
            )
            s_matrix[recv_idx, drive_flat_idx, :] = np.array(b_recv / safe_a)

    return jnp.asarray(s_matrix), mode_map


def _modal_net_power(cfg: WaveguidePortConfig) -> np.ndarray:
    """Per-mode net real power through the reference plane.

    P_m(f) = 0.5 · Re(V_m(f) · conj(I_m(f)))

    where V_m, I_m are the recorded modal voltage/current at the reference
    plane (rect full-record DFT). Because V_m and I_m are independent
    field measurements (E-line integral and H-loop), their product is the
    physical modal power flux — no modal-impedance (Z_TE) assumption,
    which is exactly the discretization bias that corrupts the
    wave-decomposition ``normalize=False`` reflection and the two-run
    ``normalize=True`` cross-mode normalization. Sign follows the port's
    recorded V/I convention; callers use magnitudes / signed differences.
    """
    v_dft = np.array(_rect_dft(cfg.v_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded))
    i_dft = np.array(_rect_dft(cfg.i_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded, 'H'))
    # The shared kernel already stamps current at the physical H half-step.
    return 0.5 * np.real(v_dft * np.conj(i_dft))


def extract_multimode_s_matrix_flux(
    grid,
    materials,
    ref_materials,
    port_mode_cfgs: list[list[WaveguidePortConfig]],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str | None = None,
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_debye: tuple | None = None,
    ref_lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, ...] | None = None,
    aniso_eps: tuple | None = None,
    conformal_weights: tuple | None = None,
    ref_aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    ref_aniso_inv_eps: tuple | None = None,
    pec_edge_masks: tuple | None = None,
    field_dtype=None,
    settling_details: list | None = None,
) -> tuple[jnp.ndarray, list[tuple[int, int, str, tuple[int, int]]]]:
    """Power-flux multi-mode waveguide S-matrix.

    ``field_dtype`` is forwarded to every core run; ``None`` uses float32.
    Modal records preserve their precision and widen for field storage.
    Spectra and S promote the record and frequency dtypes.

    Multi-mode analogue of :func:`extract_waveguide_s_matrix_flux`. For
    each driven modal channel ``(p, m)``:

      1. Reference run (empty guide, channel ``(p,m)`` driven) → incident
         modal net power ``P_inc`` at the driven channel and the per-channel
         net modal power ``F_ref``.
      2. Device run (channel ``(p,m)`` driven) → per-channel net modal
         power ``F_dev``.

    Magnitudes use per-mode power ratios::

        |S_{qn,pm}|² = |F_ref[qn] − F_dev[qn]| / |P_inc|   (q==p, n==m: reflection)
        |S_{qn,pm}|² = |F_dev[qn]|             / |P_inc|   (otherwise: transmission / mode conversion)

    All ratios are referenced to the (always-nonzero) incident modal power,
    so cross-mode channels do not hit the 0/0 blow-up of the two-run
    normalized scheme (the single-mode
    :func:`extract_waveguide_s_params_normalized` ``b_dev/b_ref`` with
    ``b_ref ≈ 0`` for an uncoupled empty-guide channel — its multi-mode
    generalization is unsupported for exactly this reason). Phase is taken
    from the modal V/I wave decomposition of the device run.

    Per-mode net power ``F`` comes from :func:`_modal_net_power`
    (``0.5·Re(V·I*)``), avoiding the modal-impedance discretization bias.

    Parameters mirror :func:`extract_waveguide_s_params_normalized`.

    Returns
    -------
    s_matrix : jnp.ndarray, shape (N, N, n_freqs)
    mode_map : list of (port_idx, mode_idx, mode_type, (m, n))
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    from rfx.simulation import run as run_simulation

    flat_cfgs: list[WaveguidePortConfig] = []
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(mode_cfgs):
            flat_cfgs.append(cfg)
            mode_map.append((
                port_idx,
                mode_within,
                cfg.mode_type,
                _mode_indices_from_config(cfg),
            ))

    if len(flat_cfgs) < 2:
        raise ValueError(
            "extract_multimode_s_matrix_flux requires at least two modal channels"
        )

    record_dtype, complex_dtype = _s_matrix_dtypes(field_dtype, flat_cfgs)
    n_total = len(flat_cfgs)
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=complex_dtype)

    n_ports = len(port_mode_cfgs)
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in range(n_ports))
    if len(ref_shifts) != n_ports:
        raise ValueError("ref_shifts must have one entry per physical port")
    flat_ref_shifts: list[float] = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        flat_ref_shifts.extend([float(ref_shifts[port_idx])] * len(mode_cfgs))

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t, dtype=record_dtype)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    common_run_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
    )
    orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0

    for drive_flat_idx in range(n_total):
        # --- Reference run (empty guide, channel drive_flat_idx driven) ---
        ref_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        ref_cfgs[drive_flat_idx] = ref_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        ref_result = run_simulation(
            grid, ref_materials, n_steps,
            debye=ref_debye, lorentz=ref_lorentz,
            waveguide_ports=ref_cfgs,
            aniso_eps=ref_aniso_eps,
            aniso_inv_eps=ref_aniso_inv_eps,
            field_dtype=field_dtype,
            **common_run_kw,
        )
        ref_final_cfgs = ref_result.waveguide_ports or ()
        if len(ref_final_cfgs) != n_total:
            raise RuntimeError("multimode flux extraction expected one final config per channel")

        F_ref = [_modal_net_power(cfg) for cfg in ref_final_cfgs]
        P_inc = np.abs(F_ref[drive_flat_idx])
        safe_P_inc = np.where(P_inc > 1e-60, P_inc, np.ones_like(P_inc))
        a_inc_ref, _ = _s_matrix_port_waves(
            ref_final_cfgs[drive_flat_idx],
            ref_shift=flat_ref_shifts[drive_flat_idx],
        )
        a_inc_ref = np.array(a_inc_ref)
        safe_a_inc = np.where(np.abs(a_inc_ref) > 1e-60, a_inc_ref, np.ones_like(a_inc_ref))

        # --- Device run (channel drive_flat_idx driven) ---
        dev_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        dev_cfgs[drive_flat_idx] = dev_cfgs[drive_flat_idx]._replace(src_amp=orig_amp)
        dev_result = run_simulation(
            grid, materials, n_steps,
            debye=debye, lorentz=lorentz,
            waveguide_ports=dev_cfgs,
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=pec_edge_masks,
            field_dtype=field_dtype,
            **common_run_kw,
        )
        dev_final_cfgs = dev_result.waveguide_ports or ()
        if settling_details is not None:
            from rfx.sparams._tail_witness import combine_witness_details
            _, ref_detail = settling_db_from_port_records(
                ref_final_cfgs, freq_max=getattr(grid, 'freq_max', None), return_detail=True)
            _, dev_detail = settling_db_from_port_records(
                dev_final_cfgs, freq_max=getattr(grid, 'freq_max', None), return_detail=True)
            settling_details.append(combine_witness_details(dev_detail, ref_detail))
        if len(dev_final_cfgs) != n_total:
            raise RuntimeError("multimode flux extraction expected one final config per channel")

        F_dev = [_modal_net_power(cfg) for cfg in dev_final_cfgs]

        for recv_idx, dev_cfg in enumerate(dev_final_cfgs):
            _, b_recv_dev = _s_matrix_port_waves(
                dev_cfg, ref_shift=flat_ref_shifts[recv_idx]
            )
            phase = np.angle(np.array(b_recv_dev) / safe_a_inc)

            if recv_idx == drive_flat_idx:
                P_refl = np.abs(F_ref[recv_idx] - F_dev[recv_idx])
                mag = np.sqrt(P_refl / safe_P_inc)
            else:
                P_out = np.abs(F_dev[recv_idx])
                mag = np.sqrt(P_out / safe_P_inc)

            s_matrix[recv_idx, drive_flat_idx, :] = mag * np.exp(1j * phase)

    return jnp.asarray(s_matrix), mode_map
