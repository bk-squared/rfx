"""JAX-traceable post-forward 3-probe wave decomposition for MSL S-params.

The validated extractor in :mod:`rfx.sources.msl_port`
(:func:`extract_msl_s_params`, :func:`msl_forward_amplitude`) lives in
:func:`Simulation.compute_msl_s_matrix` — an *imperative* orchestrator
that uses ``np.asarray`` and runs FDTD per port, blocking ``jax.grad``.
This module re-implements the same 3-probe quadratic recurrence in
``jax.numpy`` so it can run *post-* :meth:`Simulation.forward` on the
returned ``ForwardResult.time_series`` (a JAX-traceable jnp.ndarray).

Two pieces:

  * :func:`register_msl_wave_probes` — adds 8 point probes per 2-port
    line (4 per port: Ez at probe 1/2/3 for V-proxy, Hy at probe 1
    for I-proxy) via :meth:`Simulation.add_probe` and returns the
    integer column indices into ``time_series``.

  * :func:`extract_msl_s_params_jax` — windowed single-bin DFT at
    the target frequencies, then the 3-probe quadratic recurrence
    `q² − ((V₁+V₃)/V₂)·q + 1 = 0` to solve for the per-Δ phasor
    ``q``, the forward amplitude ``α`` (at the upstream-most probe
    of each port), and ``S11 = γ/α``, ``S21 = α_passive/α_driven``.

The scalar Ez / Hy proxies (point probes at substrate-mid and trace-
top) match the line-integral V/I to a frequency-independent
normalization that cancels in every quantity returned here (S11 is a
ratio of α and γ; S21 is a ratio of α's; the 3-probe recurrence's
coefficient ``(V₁+V₃)/V₂`` is itself a ratio). The absolute ``Z0``
recovery is *not* normalization-cancelling and is therefore intentionally
omitted from this helper — use :meth:`compute_msl_s_matrix` for Z0.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MSLWaveProbeSet:
    """Indices into ``ForwardResult.time_series`` for a single port's
    4-probe set (3 Ez + 1 Hy)."""
    ez1_col: int
    ez2_col: int
    ez3_col: int
    hy1_col: int
    delta: float                # spacing between probes 1 and 2 (= 2 ↔ 3)


def register_msl_wave_probes(
    sim,
    *,
    feed_x: float,
    direction: str,                 # "+x" or "-x"
    y_centre: float,
    z_ez: float,                    # physical z of Ez probes (substrate mid)
    z_hy: float,                    # physical z of Hy probes (trace top)
    n_offset_cells: int = 5,
    n_spacing_cells: int = 3,
) -> MSLWaveProbeSet:
    """Register 4 point probes per port for the 3-probe wave decomposition.

    Probe placement matches :func:`rfx.sources.msl_port.msl_probe_x_coords`:
    starting ``n_offset_cells`` downstream of ``feed_x``, with the next
    two probes at ``n_spacing_cells`` further along the propagation
    direction.

    Returns
    -------
    MSLWaveProbeSet
        Carries the integer column indices into ``ForwardResult.time_series``
        for ``ez1``, ``ez2``, ``ez3`` and ``hy1``, plus the physical
        spacing ``delta`` between adjacent probes.
    """
    assert direction in ("+x", "-x")
    sign = 1 if direction == "+x" else -1
    dx = float(sim._dx)

    x1 = feed_x + sign * n_offset_cells * dx
    x2 = x1 + sign * n_spacing_cells * dx
    x3 = x2 + sign * n_spacing_cells * dx
    delta = float(abs(x2 - x1))

    # Snapshot existing probe count → returned columns are appended.
    base = len(sim._probes)
    sim.add_probe(position=(x1, y_centre, z_ez), component="ez")
    sim.add_probe(position=(x2, y_centre, z_ez), component="ez")
    sim.add_probe(position=(x3, y_centre, z_ez), component="ez")
    sim.add_probe(position=(x1, y_centre, z_hy), component="hy")
    return MSLWaveProbeSet(
        ez1_col=base, ez2_col=base + 1, ez3_col=base + 2, hy1_col=base + 3,
        delta=delta,
    )


def _windowed_dft(ts_col: jnp.ndarray, dt: float,
                  freqs: jnp.ndarray) -> jnp.ndarray:
    """Hann-windowed single-bin DFT of one time-series column at ``freqs``.

    JAX-friendly (avoids ``jnp.fft.rfft`` overhead when only N freqs are
    needed; ``ts_col`` is float32, output is complex64).
    """
    n = ts_col.shape[0]
    # 0..n-1 hann window
    win = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * jnp.arange(n) / max(n - 1, 1))
    sig = ts_col * win
    t = jnp.arange(n, dtype=jnp.float32) * dt
    # X(f) = sum_n sig[n] * exp(-j 2π f t[n])
    phase = -2j * jnp.pi * jnp.asarray(freqs, dtype=jnp.float32)[:, None] * t[None, :]
    return jnp.sum(sig[None, :].astype(jnp.complex64) * jnp.exp(phase), axis=1)


def _solve_3probe_jax(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray,
                      i1: jnp.ndarray | None, eps: float = 1e-30
                      ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX port of :func:`rfx.sources.msl_port._solve_3probe`.

    Returns ``(alpha, gamma, q)`` — forward and backward amplitudes at
    probe 1, and the per-Δ phasor.  Same root-selection logic as the
    numpy version: pick the q whose phase is closer to ``v2/v1``.
    """
    coeff = (v1 + v3) / (v2 + eps)
    disc = coeff ** 2 - 4.0
    sqrt_disc = jnp.sqrt(disc.astype(jnp.complex64))
    q_plus = (coeff + sqrt_disc) / 2.0
    q_minus = (coeff - sqrt_disc) / 2.0
    ratio = v2 / (v1 + eps)
    err_plus = jnp.abs(q_plus - ratio)
    err_minus = jnp.abs(q_minus - ratio)
    use_plus = err_plus < err_minus
    q = jnp.where(use_plus, q_plus, q_minus)
    denom = (q * q - 1.0) + eps
    alpha = (q * v2 - v1) / denom
    gamma = q * (v1 * q - v2) / denom
    return alpha, gamma, q


def extract_msl_s_params_jax(
    time_series: jnp.ndarray,
    driven: MSLWaveProbeSet,
    passive: MSLWaveProbeSet,
    *,
    dt: float,
    freqs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-traceable S11 + S21 from the 3-probe wave decomposition.

    Both S-parameters are dimensionless ratios — independent of the
    Ez↔V and Hy↔I proxy normalisations — so this helper does not
    require the substrate height / trace width metadata that the
    imperative :func:`extract_msl_s_params` uses for the absolute Z0.

    Parameters
    ----------
    time_series : (n_steps, n_probes) jnp.ndarray
        From ``Simulation.forward(...).time_series``.
    driven, passive : MSLWaveProbeSet
        From :func:`register_msl_wave_probes`.  ``driven`` is the port
        carrying the source; ``passive`` is the matched-terminated port.
    dt : float
        Yee timestep (``grid.dt``).
    freqs : (n_freqs,) jnp.ndarray
        Target frequencies for S-parameter evaluation.

    Returns
    -------
    s11, s21 : (n_freqs,) jnp.ndarray, complex64
        ``s11`` at the driven port; ``s21`` from driven → passive.
    """
    v1d = _windowed_dft(time_series[:, driven.ez1_col], dt, freqs)
    v2d = _windowed_dft(time_series[:, driven.ez2_col], dt, freqs)
    v3d = _windowed_dft(time_series[:, driven.ez3_col], dt, freqs)
    v1p = _windowed_dft(time_series[:, passive.ez1_col], dt, freqs)
    v2p = _windowed_dft(time_series[:, passive.ez2_col], dt, freqs)
    v3p = _windowed_dft(time_series[:, passive.ez3_col], dt, freqs)

    alpha_d, gamma_d, _ = _solve_3probe_jax(v1d, v2d, v3d, None)
    alpha_p, _, _ = _solve_3probe_jax(v1p, v2p, v3p, None)

    eps = 1e-30
    s11 = gamma_d / (alpha_d + eps)
    s21 = alpha_p / (alpha_d + eps)
    return s11, s21
