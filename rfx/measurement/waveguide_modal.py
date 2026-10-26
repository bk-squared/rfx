"""Waveguide modal projections and overlap records."""

from __future__ import annotations

from typing import NamedTuple
import jax.numpy as jnp
from rfx.model.waveguide_modes import (
    WaveguidePortConfig,
    _aperture_dA,
    _compute_beta,
    _compute_mode_impedance,
)
from rfx.measurement.waveguide_s import (
    _shift_modal_waves,
)
from rfx.stepping.waveguide_source import (
    _plane_indexer,
)


def _plane_field(field, cfg: WaveguidePortConfig, plane_index: int):
    """Extract a tangential E/H field slice on the port plane."""
    return field[_plane_indexer(cfg, plane_index)]


def _plane_h_field(field, cfg: WaveguidePortConfig, plane_index: int):
    """Extract a tangential H field slice averaged to the E plane along the normal."""
    prev_index = plane_index - 1
    return 0.5 * (
        _plane_field(field, cfg, plane_index)
        + _plane_field(field, cfg, prev_index)
    )


def _plane_h_field_at_dual(
    field,
    cfg: WaveguidePortConfig,
    plane_index: int,
    h_offset: tuple[float, float],
):
    """Extract H on the normal-averaged plane and optional transverse dual mesh."""
    base = _plane_h_field(field, cfg, plane_index)
    if h_offset == (0.0, 0.0):
        return base
    out = base
    for axis, offset in enumerate(h_offset):
        if offset == 0.0:
            continue
        if offset != 0.5:
            raise ValueError(
                "h_offset currently supports only 0.0 or 0.5 per axis, "
                f"got {h_offset!r}"
            )
        # Symmetric centered co-location average (matches the stored-profile
        # _shift_profile_to_dual stencil in lockstep). Edge-clamped, NOT
        # wrapped: simulated tangential H is nonzero at PEC walls, so a
        # wraparound would leak. See _shift_profile_to_dual docstring.
        n = out.shape[axis]
        plus = jnp.concatenate(
            [jnp.take(out, jnp.array([0]), axis=axis),
             jnp.take(out, jnp.arange(n - 1), axis=axis)],
            axis=axis,
        )
        minus = jnp.concatenate(
            [jnp.take(out, jnp.arange(1, n), axis=axis),
             jnp.take(out, jnp.array([n - 1]), axis=axis)],
            axis=axis,
        )
        out = 0.25 * (plus + 2.0 * out + minus)
    return out


def modal_voltage(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal voltage: V = integral E_t . e_mode dA."""
    e_u_sim = _plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
    e_v_sim = _plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
    dA = _aperture_dA(cfg)
    return jnp.sum((e_u_sim * cfg.ey_profile + e_v_sim * cfg.ez_profile) * dA)


def modal_current(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal current: I = integral H_t . h_mode dA.

    H is averaged between x_idx-1 and x_idx to co-locate with E
    on the Yee grid (H sits at x+1/2, E sits at x).
    """
    h_u_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset,
    )
    h_v_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset,
    )
    dA = _aperture_dA(cfg)
    return jnp.sum((h_u_sim * cfg.hy_profile + h_v_sim * cfg.hz_profile) * dA)


def mode_self_overlap(cfg: WaveguidePortConfig, dx: float) -> float:
    """Mode self-overlap: C = ∫∫ (e_mode × h*_mode) · n̂ dA.

    For x-normal: (e_mode × h_mode) · x̂ = ey*hz - ez*hy.
    Returns a positive real scalar for a properly defined mode.
    """
    cross = (cfg.ey_profile * cfg.hz_profile
             - cfg.ez_profile * cfg.hy_profile)
    dA = _aperture_dA(cfg)
    return float(jnp.sum(cross * dA))


def overlap_modal_amplitude(
    state, cfg: WaveguidePortConfig, x_idx: int, dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward/backward modal amplitudes via overlap integral.

    Uses the Lorentz reciprocity overlap:
      P1 = ∫∫ (E_sim × H*_mode) · n̂ dA
      P2 = ∫∫ (E*_mode × H_sim) · n̂ dA
      a_forward  = (P1 + P2) / (2 * C_mode)
      a_backward = (P1 - P2) / (2 * C_mode)

    For x-normal port with n̂ = x̂:
      (E × H_mode) · x̂ = Ey_sim * Hz_mode - Ez_sim * Hy_mode
      (E_mode × H) · x̂ = Ey_mode * Hz_sim - Ez_mode * Hy_sim

    Returns (a_forward, a_backward) as scalars.
    """
    e_u_sim = _plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
    e_v_sim = _plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
    h_u_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset,
    )
    h_v_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset,
    )

    dA = _aperture_dA(cfg)
    # P1 = ∫ (E_sim × H_mode) · n̂ dA
    p1 = jnp.sum((e_u_sim * cfg.hz_profile - e_v_sim * cfg.hy_profile) * dA)
    # P2 = ∫ (E_mode × H_sim) · n̂ dA
    p2 = jnp.sum((cfg.ey_profile * h_v_sim - cfg.ez_profile * h_u_sim) * dA)

    c_mode = mode_self_overlap(cfg, dx)
    safe_c = max(abs(c_mode), 1e-30)

    a_fwd = (p1 + p2) / (2.0 * safe_c)
    a_bwd = (p1 - p2) / (2.0 * safe_c)
    return a_fwd, a_bwd


def update_waveguide_port_probe(cfg: WaveguidePortConfig, state,
                                dt: float, dx: float) -> WaveguidePortConfig:
    from rfx.measurement.modal import record_waveguide
    return record_waveguide(cfg, state, dt, dx)


class OverlapDFTAccumulators(NamedTuple):
    """DFT accumulators for overlap-based modal extraction.

    Stores the two raw cross-product DFTs (P1 = modal voltage, P2 = modal
    current) at the reference and probe planes. ``update_overlap_dft``
    accumulates a rect-window streaming DFT every step (Phase 2 cleanup
    removed the windowed/gated path that previously lived on
    ``WaveguidePortConfig``).
    """
    p1_ref_dft: jnp.ndarray     # (n_freqs,) complex — ∫(E_sim × h_mode)·n̂ dA at ref
    p2_ref_dft: jnp.ndarray     # (n_freqs,) complex — ∫(e_mode × H_sim)·n̂ dA at ref
    p1_probe_dft: jnp.ndarray   # (n_freqs,) complex — ∫(E_sim × h_mode)·n̂ dA at probe
    p2_probe_dft: jnp.ndarray   # (n_freqs,) complex — ∫(e_mode × H_sim)·n̂ dA at probe


def init_overlap_dft(freqs: jnp.ndarray) -> OverlapDFTAccumulators:
    """Initialize zero-valued overlap DFT accumulators."""
    nf = len(freqs)
    # Canonical complex dtype — see init_waveguide_port for rationale.
    zeros_c = jnp.zeros(nf, dtype=jnp.array(0j).dtype)
    return OverlapDFTAccumulators(
        p1_ref_dft=zeros_c,
        p2_ref_dft=zeros_c,
        p1_probe_dft=zeros_c,
        p2_probe_dft=zeros_c,
    )


def _overlap_cross_products(
    state, cfg: WaveguidePortConfig, x_idx: int, dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the two raw overlap cross-product integrals at a plane.

    P1 = ∫(E_sim × h_mode) · n̂ dA  (= modal voltage V)
    P2 = ∫(e_mode × H_sim) · n̂ dA  (= modal current I)

    Returns (P1, P2) as scalars.
    """
    dA = _aperture_dA(cfg)
    e_u_sim = _plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
    e_v_sim = _plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
    h_u_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset,
    )
    h_v_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset,
    )

    # Preserve the production V/I-equivalent DFT path even when the stored H
    # template is sampled on a transverse dual mesh (`h_offset != 0`).  In the
    # unshifted case these are exactly the Lorentz cross-products because
    # h_mode = n̂×e_mode; with dual-mesh H they remain the modal voltage/current
    # quantities consumed by `_overlap_to_waves`.
    p1 = jnp.sum((e_u_sim * cfg.ey_profile + e_v_sim * cfg.ez_profile) * dA)
    p2 = jnp.sum((h_u_sim * cfg.hy_profile + h_v_sim * cfg.hz_profile) * dA)

    return p1, p2


def update_overlap_dft(
    acc: OverlapDFTAccumulators,
    cfg: WaveguidePortConfig,
    state,
    dt: float,
    dx: float,
) -> OverlapDFTAccumulators:
    """Accumulate overlap cross-product DFTs at reference and probe planes.

    Computes P1 and P2 at each timestep and accumulates into running DFTs.
    At extraction time these are combined with frequency-dependent mode
    impedance to yield forward/backward modal amplitudes.
    """
    p1_ref, p2_ref = _overlap_cross_products(state, cfg, cfg.ref_x, dx)
    p1_probe, p2_probe = _overlap_cross_products(state, cfg, cfg.probe_x, dx)

    # Rect-window streaming DFT (constant unit weight). The Phase 2
    # cleanup (2026-04-25) deleted the windowed/gated path on the V/I
    # extractor; the overlap path historically shared the same gating
    # knobs but those fields were also removed from
    # ``WaveguidePortConfig``. The S-param result is unchanged for runs
    # that previously used ``dft_window='rect'`` and no early gate.
    from rfx.measurement.dft import phase as dft_phase
    phase = dft_phase(state.step-1, cfg.freqs, dt, 'E')
    h_phase = dft_phase(state.step-1, cfg.freqs, dt, 'H')

    return OverlapDFTAccumulators(
        p1_ref_dft=acc.p1_ref_dft + p1_ref * phase,
        p2_ref_dft=acc.p2_ref_dft + p2_ref * h_phase,
        p1_probe_dft=acc.p1_probe_dft + p1_probe * phase,
        p2_probe_dft=acc.p2_probe_dft + p2_probe * h_phase,
    )


def _overlap_to_waves(
    p1_dft: jnp.ndarray,
    p2_dft: jnp.ndarray,
    z_mode: jnp.ndarray,
    c_stored: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert overlap cross-product DFTs to forward/backward modal amplitudes.

    a_fwd(f) = 0.5 * (P1(f) + Z_mode(f) * P2(f)) / C_stored
    a_bwd(f) = 0.5 * (P1(f) - Z_mode(f) * P2(f)) / C_stored

    The Z_mode scaling compensates for the stored h_mode profiles being
    un-normalized by impedance (h_stored = Z_mode * H_mode_physical).
    """
    safe_c = max(abs(c_stored), 1e-30)
    a_fwd = 0.5 * (p1_dft + z_mode * p2_dft) / safe_c
    a_bwd = 0.5 * (p1_dft - z_mode * p2_dft) / safe_c
    return a_fwd, a_bwd


def extract_waveguide_sparams_overlap(
    acc: OverlapDFTAccumulators,
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
    probe_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (S11, S21) from overlap-integral DFT accumulators.

    Combines the stored cross-product DFTs with frequency-dependent mode
    impedance and mode self-overlap normalization to produce S-parameters.

    Parameters
    ----------
    acc : OverlapDFTAccumulators
        Accumulated overlap DFTs from the simulation.
    cfg : WaveguidePortConfig
        Port configuration (for frequency, cutoff, and mode profile info).
    ref_shift : float
        Metres to shift the reference-plane reporting location.
    probe_shift : float
        Metres to shift the probe-plane reporting location.

    Returns
    -------
    (s11, s21) : tuple of jnp.ndarray
        S-parameters at the (optionally shifted) planes.
    """
    from rfx.measurement.waveguide_s import require_probe_medium
    require_probe_medium(cfg)
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx, medium=cfg.medium)
    z_mode = _compute_mode_impedance(
        cfg.freqs,
        cfg.f_cutoff,
        cfg.mode_type,
        medium=cfg.medium,
        dt=cfg.dt,
        dx=cfg.dx,
    )
    c_stored = mode_self_overlap(cfg, cfg.dx)
    p2_ref = acc.p2_ref_dft
    p2_probe = acc.p2_probe_dft

    # Convert cross-product DFTs to global forward/backward waves
    fwd_ref, bwd_ref = _overlap_to_waves(
        acc.p1_ref_dft, p2_ref, z_mode, c_stored
    )
    fwd_probe, bwd_probe = _overlap_to_waves(
        acc.p1_probe_dft, p2_probe, z_mode, c_stored
    )

    # Port-local wave mapping
    if cfg.direction.startswith("+"):
        a_inc_ref, a_out_ref = fwd_ref, bwd_ref
        a_inc_probe = fwd_probe
    else:
        a_inc_ref, a_out_ref = bwd_ref, fwd_ref
        a_inc_probe = bwd_probe

    # Apply reference-plane shifts (global-frame shift; sign handled in _shift_modal_waves)
    step_sign = 1 if cfg.direction.startswith("+") else -1
    a_inc_ref, a_out_ref = _shift_modal_waves(a_inc_ref, a_out_ref, beta, ref_shift, step_sign)
    a_inc_probe, _ = _shift_modal_waves(
        a_inc_probe,
        jnp.zeros_like(a_inc_probe),
        beta,
        probe_shift,
        step_sign,
    )

    safe_ref = jnp.where(
        jnp.abs(a_inc_ref) > 0, a_inc_ref, jnp.ones_like(a_inc_ref)
    )
    s11 = a_out_ref / safe_ref
    s21 = a_inc_probe / safe_ref

    return s11, s21

