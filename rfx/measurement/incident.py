"""Measured incident spectra from TF/SF auxiliary lines."""

from __future__ import annotations

from rfx.sources.tfsf import TFSFConfig
from rfx.sources.tfsf import TFSFState
from rfx.sources.tfsf import update_tfsf_1d_e
from rfx.sources.tfsf import update_tfsf_1d_h
from rfx.sources.tfsf_oblique_open import MethodBConfig
from rfx.sources.tfsf_oblique_open import MethodBState
import jax.numpy as jnp
import numpy as np


def measure_normal_incident_spectrum(
    cfg: TFSFConfig,
    state: TFSFState,
    n_steps: int,
    freqs,
    dt: float,
    *,
    reference_index: int | None = None,
) -> np.ndarray:
    """Spectrum of the incident E the normal-incidence plane wave actually carries.

    The auxiliary line ADDS the source waveform to one E node
    (``update_tfsf_1d_e``: ``e1d.at[src_idx].add(src_val)``). That is a soft
    source: it launches half of the added field each way, scaled by the Yee
    update, so the wave that crosses the TF/SF plane is not the waveform. From
    the 1-D update equations, per frequency,

        E_launched / waveform = 1 / (2 S cos(k~ dx / 2)),  S = c dt / dx,

    with k~ the grid wavenumber (sin(w dt / 2) = S sin(k~ dx / 2)). At the
    uniform-grid default S = 0.99/sqrt(3) = 0.5716 that is 0.875 (-1.16 dB)
    before the cos term, -1.14 dB at 40 cells per wavelength. sigma goes as
    |E_scat / E_inc|^2 and E_scat follows the launched wave, so an RCS
    normalized by the waveform itself is low by that same number of dB,
    1.14 dB at 40 cells per wavelength (issue #820).

    This replays the auxiliary line standalone from a fresh zero state (the
    caller's ``state`` is only a shape/dtype template and is never mutated) for
    ``n_steps``, with the same time argument ``run()`` passes it, and DFTs
    ``e1d`` at ``cfg.i0`` -- the node mapped onto the 3-D TF/SF plane
    ``x_lo`` -- as ``sum_n e(n) exp(-j w n dt) dt``, the convention of
    :func:`rfx.sources.tfsf_oblique_open.measure_incident_spectrum`. The DFT
    window is the 3-D run's ``n_steps``, so truncation matches the far-field
    DFT it normalizes. The auxiliary line and the 3-D grid have the same
    dispersion along x at normal incidence, so the magnitude is the same
    anywhere in the total-field region; the phase reference is the auxiliary
    line's own, and only ``abs()`` of the result is meant for normalization.

    For complex normalization, pass ``reference_index``: the padded 3-D x
    index of an E-node plane inside the total-field slab. The auxiliary
    sample then follows that plane and its E field is stamped at
    ``(n + 1) * dt``, matching ``accumulate_ntff`` and the actual 3-D E
    update. This preserves the incident phase, including source delay and
    grid dispersion; it does not replace a record-length/mesh witness.
    Omitting this argument retains the historical magnitude-only result.

    Normal-incidence ``TFSFConfig`` only; the oblique open-domain path has
    its own replay (``tfsf_oblique_open.measure_incident_spectrum``).
    """
    from jax import lax

    if getattr(cfg, "closed_box", False):
        from rfx.sources.tfsf_discrete_plane_wave import normal_incident_spectrum
        return normal_incident_spectrum(cfg, state, n_steps, freqs, dt, reference_index)
    if not isinstance(cfg, TFSFConfig):
        raise TypeError(
            "measure_normal_incident_spectrum takes the normal-incidence "
            f"TFSFConfig; got {type(cfg).__name__}. The open-domain oblique "
            "path uses rfx.sources.tfsf_oblique_open.measure_incident_spectrum."
        )
    zero = state._replace(
        e1d=jnp.zeros_like(state.e1d),
        h1d=jnp.zeros_like(state.h1d),
        psi_e_lo=jnp.zeros_like(state.psi_e_lo),
        psi_e_hi=jnp.zeros_like(state.psi_e_hi),
        psi_h_lo=jnp.zeros_like(state.psi_h_lo),
        psi_h_hi=jnp.zeros_like(state.psi_h_hi),
        step=jnp.array(0, dtype=jnp.int32),
    )
    i_ref = int(cfg.i0)
    if reference_index is not None:
        if (not isinstance(reference_index, (int, np.integer))
                or isinstance(reference_index, (bool, np.bool_))
                or not cfg.x_lo <= reference_index <= cfg.x_hi):
            raise ValueError("reference_index must be an integer E-node index inside the TFSF slab")
        i_ref += int(reference_index) - int(cfg.x_lo)

    def _step(st, step_idx):
        # Same order and time argument as the run() scan body.
        st = update_tfsf_1d_h(cfg, st, cfg.dx_1d, dt)
        t = step_idx.astype(jnp.float32) * dt
        st = update_tfsf_1d_e(cfg, st, cfg.dx_1d, dt, t)
        return st, st.e1d[i_ref]

    _, rec = lax.scan(_step, zero, jnp.arange(n_steps, dtype=jnp.int32))
    rec = np.asarray(rec, dtype=np.float64)
    times = (np.arange(n_steps) + (reference_index is not None)) * float(dt)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    out = np.empty(len(freqs), dtype=np.complex128)
    for i, f in enumerate(freqs):
        out[i] = np.sum(rec * np.exp(-1j * 2.0 * np.pi * f * times)) * float(dt)
    return out


def measure_incident_spectrum(
    cfg: MethodBConfig,
    state: MethodBState,
    n_steps: int,
    freqs,
    dx: float,
) -> np.ndarray:
    """DFT of the ACTUAL 1D-aux incident field at the TFSF-box-centre projection.

    Replays the Method-B 1D auxiliary FDTD standalone (a fresh zero state — the
    caller's ``state`` is only a shape template and is never mutated) for
    ``n_steps`` and DFTs the recorded ``e1d`` at the aux index onto which the
    box centre projects.  This is the incident spectral amplitude the 4-edge
    box actually injects, so using it as the RCS normalization captures what an
    analytic waveform DFT cannot. Historical measurements below predate #1812
    and its declared-pulse timing (issue #471 F7 audit, gate grid
    f0=5 GHz, dx=2 mm):

    * the then-hardcoded ``t0=40·dt, tau=12·dt`` modulated Gaussian instead of the
      normal path's assumed differentiated Gaussian (+8.83 dB at f0 — the
      review's number, reproduced exactly), AND
    * the source->aux launch/absorption response the review neglected
      (-2.25 dB at theta=20 deg), which is THETA-DEPENDENT (-4.32 dB at 40 deg)
      because ``d_aux`` is dispersion-matched per angle, so the pulse's spatial
      extent vs the lo-CPML clearance changes with the angle.

    Net assumed-vs-actual mismatch: +6.58 dB (20 deg) / +4.51 dB (40 deg) —
    the pre-fix absolute-sigma inflation.  Cost: ``n_steps`` scans of the
    ~O(500)-cell 1D aux (well under a second; runs once per compute_rcs call).

    The DFT window equals the 3D run window (same ``n_steps``), so truncation
    semantics match the NTFF far-field DFT it normalizes.
    """
    from jax import lax

    zero = state._replace(
        e1d=jnp.zeros_like(state.e1d),
        h1d=jnp.zeros_like(state.h1d),
        psi_e_lo=jnp.zeros_like(state.psi_e_lo),
        psi_e_hi=jnp.zeros_like(state.psi_e_hi),
        psi_h_lo=jnp.zeros_like(state.psi_h_lo),
        psi_h_hi=jnp.zeros_like(state.psi_h_hi),
        step=jnp.array(0, dtype=jnp.int32),
    )
    # Box-centre projection onto the aux line (same mapping as the gather
    # tables: idx_E = (x*cos + y*sin)*dx/d_aux + src_idx). The plane-wave
    # magnitude is position-independent, so the exact interior index is not
    # critical; the centre is the semantically-right target location.
    cx = 0.5 * (cfg.x_lo + cfg.x_hi)
    cy = 0.5 * (cfg.y_lo + cfg.y_hi)
    ref = int(round(((cx * cfg.cos_theta + cy * cfg.sin_theta) * dx)
                    / cfg.d_aux + cfg.src_idx))

    ts = jnp.arange(n_steps, dtype=jnp.float32) * cfg.dt

    def _step(st, t):
        st = update_tfsf_1d_h(cfg, st, cfg.dx_1d, cfg.dt)
        st = update_tfsf_1d_e(cfg, st, cfg.dx_1d, cfg.dt, t)
        return st, st.e1d[ref]

    _, rec = lax.scan(_step, zero, ts)
    rec = np.asarray(rec, dtype=np.float64)
    times = np.arange(n_steps) * cfg.dt
    freqs = np.asarray(freqs, dtype=np.float64)
    out = np.empty(len(freqs), dtype=np.complex128)
    for i, f in enumerate(freqs):
        out[i] = np.sum(rec * np.exp(-1j * 2.0 * np.pi * f * times)) * cfg.dt
    return out
