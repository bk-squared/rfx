"""Modal record adapters; projection remains in the waveguide source module."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfx.sources.waveguide_port import WaveguidePortConfig

import jax.numpy as jnp

from .dft import transform


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
    # A call before the first step (state.step == 0) has no slot: index n_t is
    # dropped, where -1 would have written the last slot.
    step = jnp.where((state.step >= 1) & (state.step < n_t),
                     jnp.asarray(state.step, dtype=jnp.int32)-1, n_t)
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
        # The count of samples written: the dropped last step is not one.
        n_steps_recorded=jnp.maximum(cfg.n_steps_recorded, jnp.minimum(
            jnp.asarray(state.step, dtype=jnp.int32), n_t - 1)),
    )


def recorded(cfg, name):
    """A port's time record cut to the samples written (no unwritten tail).

    A full scan leaves the last slot unwritten; a reader of the raw record
    (the settling witness, the source-end search) must not take that zero
    for a sample, or a source still on at the end reads as having ended.
    """
    from rfx.core.jax_utils import is_tracer
    record, count = getattr(cfg, name), getattr(cfg, 'n_steps_recorded', None)
    if count is None or is_tracer(count) or is_tracer(record):
        return record
    count = int(count)
    return record[:count] if 0 < count < len(record) else record



def rect_dft(time_series, freqs, dt, n_valid, kind='E'):
    """Waveguide single-sided spectrum of physical slot-n records."""
    return 2 * transform(time_series, freqs, dt, kind, n_valid=n_valid)


def legacy_current_spectrum(cfg, current_dft):
    """An E-stamped current spectrum turned to the H stamp.

    No production caller since S2 M2 (records are stamped by the kernel).
    Kept behind ``waveguide_port._co_located_current_spectrum`` because
    tests/unit/sparams/test_rf_audit_fixes.py and the diagnostics scripts
    convert spectra they formed at E's time with it.
    """
    from rfx.core.dft_utils import half_step_current_phase
    return current_dft * half_step_current_phase(cfg.freqs, cfg.dt).astype(current_dft.dtype)
