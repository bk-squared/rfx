"""Modal record adapters; projection remains in the waveguide source module."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfx.sources.waveguide_port import WaveguidePortConfig

import jax.numpy as jnp

from .dft import phase, transform


def record_waveguide(cfg: WaveguidePortConfig, state,
                     dt: float, dx: float) -> WaveguidePortConfig:
    """Record per-step modal V and I at ref and probe planes.

    The S-parameter spectra are computed POST-SCAN by
    ``_extract_global_waves_from_time_series`` via a rectangular
    full-record DFT on the recorded arrays. No in-scan windowed DFT
    accumulators are kept (Phase 2 cleanup, 2026-04-25). The previous
    in-scan windowed/gated path silently integrated partially-decayed
    standing waves and produced ``|S11| → ∞`` as the gate widened.
    """
    from rfx.sources.waveguide_port import modal_voltage, modal_current
    t = state.step * dt

    v_ref = modal_voltage(state, cfg, cfg.ref_x, dx)
    v_probe = modal_voltage(state, cfg, cfg.probe_x, dx)
    i_ref = modal_current(state, cfg, cfg.ref_x, dx)
    i_probe = modal_current(state, cfg, cfg.probe_x, dx)

    arg = (t - cfg.src_t0) / cfg.src_tau
    v_inc = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    # Preserve the historical sample set: scan step N-1 remains excluded.
    # Record the retained samples at slot n with the physical stamp.
    n_t = cfg.v_probe_t.shape[0]
    step = jnp.where(state.step < n_t, jnp.asarray(state.step, dtype=jnp.int32)-1, n_t)
    return cfg._replace(
        dt=float(dt),
        v_probe_t=cfg.v_probe_t.at[step].set(
            jnp.asarray(v_probe, cfg.v_probe_t.dtype), mode='drop'),
        v_ref_t=cfg.v_ref_t.at[step].set(
            jnp.asarray(v_ref, cfg.v_ref_t.dtype), mode='drop'),
        i_probe_t=cfg.i_probe_t.at[step].set(
            jnp.asarray(i_probe, cfg.i_probe_t.dtype), mode='drop'),
        i_ref_t=cfg.i_ref_t.at[step].set(
            jnp.asarray(i_ref, cfg.i_ref_t.dtype), mode='drop'),
        v_inc_t=cfg.v_inc_t.at[step].set(
            jnp.asarray(v_inc, cfg.v_inc_t.dtype), mode='drop'),
        n_steps_recorded=jnp.maximum(cfg.n_steps_recorded,
                                     jnp.minimum(step + 1, n_t)),
    )



def rect_dft(time_series, freqs, dt, n_valid, kind='E'):
    """Waveguide single-sided spectrum of physical slot-n records."""
    return 2 * transform(time_series, freqs, dt, kind, n_valid=n_valid)


def legacy_current_spectrum(cfg, current_dft):
    """Compatibility for explicitly supplied, E-stamped legacy H spectra."""
    if cfg.dt <= 0:
        return current_dft
    return current_dft * (phase(0, cfg.freqs, cfg.dt, 'H', dtype=current_dft.dtype)
                          / phase(0, cfg.freqs, cfg.dt, 'E', dtype=current_dft.dtype))
