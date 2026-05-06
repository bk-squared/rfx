"""JAX-traceable post-forward 3-probe wave decomposition for MSL S-params.

The validated extractor in :mod:`rfx.sources.msl_port`
(:func:`extract_msl_s_params`, :func:`msl_forward_amplitude`) lives in
:func:`Simulation.compute_msl_s_matrix` — an *imperative* orchestrator
that uses ``np.asarray`` and runs FDTD per port, blocking ``jax.grad``.
This module re-implements the same 3-probe quadratic recurrence in
``jax.numpy`` so it can run *post-* :meth:`Simulation.forward` on the
returned ``ForwardResult.time_series`` / ``ForwardResult.dft_planes``
(JAX-traceable arrays / dict of JAX-traceable accumulators).

Four pieces, paired into a scalar (point-probe) lane and a plane
(plane-probe) lane.  The plane lane is the higher-fidelity follow-up
landed 2026-05-07 (gap #2/#4 in
``docs/agent-memory/rfx-known-issues.md``).

Scalar lane (legacy, single-Ez / single-Hy point probes):

  * :func:`register_msl_wave_probes` — adds 4 point probes per port
    (Ez at probe 1/2/3 for V-proxy, Hy at probe 1 for I-proxy).
  * :func:`extract_msl_s_params_jax` — Hann-windowed single-bin DFT
    over the time-series columns, then the 3-probe quadratic
    recurrence ``q² − ((V₁+V₃)/V₂)·q + 1 = 0``.

  Bias note (the reason the plane lane exists): the scalar Ez probe
  at substrate-mid is a *single-cell sample* of the quasi-TEM mode,
  which under-resolves the modal voltage integral compared to the
  plane-line integral used by ``compute_msl_s_matrix``.  The Y2 demo
  (``examples/inverse_design/msl_stub_notch_tuning.py``) measures a
  ~15-20 % notch-frequency bias at dx = h_sub/2 (2 substrate cells)
  on the scalar lane that disappears on the plane lane.

Plane lane (recommended for engineering accuracy):

  * :func:`register_msl_plane_probes` — adds 4 plane DFT probes per
    port (Ez planes at x=x₁/x₂/x₃ for V-proxy, Hy plane at x=x₁ for
    I-proxy) via :meth:`Simulation.add_dft_plane_probe` and returns
    the static integration metadata (j_centre, k_lo/k_hi, k_h,
    j_lo_ext/j_hi_ext, dy_arr / dz_arr slices) needed to mirror
    ``compute_msl_s_matrix``'s plane integrals.

  * :func:`extract_msl_s_params_jax_plane` — pulls the plane DFT
    accumulators from ``ForwardResult.dft_planes`` (populated since
    2026-05-07; see `feat(forward): expose dft_planes accumulators`),
    line-integrates Ez over the substrate column at the trace
    centerline for V, area-integrates Hy over a fringing-y-extended
    slab below the trace surface for I, and feeds the result into
    the same 3-probe recurrence as the scalar lane.

Both lanes return dimensionless ``S11`` and ``S21``.  Absolute ``Z0``
is intentionally omitted (use :meth:`compute_msl_s_matrix` for Z0 —
its sign convention requires the I integral; the scalar lane omits I
entirely, the plane lane carries it for the future Z0 add-on).
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


# ---------------------------------------------------------------------------
# Plane lane — plane-integrated V/I, mirrors compute_msl_s_matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MSLPlaneProbeSet:
    """Plane DFT probe names + static integration metadata for one MSL port.

    The integration indices and per-axis cell-size slices are computed
    once at registration time from the grid + ``MSLPort`` cross-section
    and held as Python / JAX-traceable static metadata.  At extraction
    time, V is the line-integral of Ez over the substrate column at the
    trace centerline; I is the area-integral of Hy over a fringing-y-
    extended slab placed in the substrate just below the trace surface.
    """
    ez1_name: str
    ez2_name: str
    ez3_name: str
    hy_name: str
    j_centre: int
    k_lo: int
    k_hi: int           # inclusive substrate column for Ez integration
    k_h: int            # z index for Hy integration (k_top - 1)
    j_lo: int
    j_hi: int           # inclusive y-extended slab for Hy integration
    dz_slice: jnp.ndarray   # (k_hi - k_lo + 1,) dz weights for V
    dy_slice: jnp.ndarray   # (j_hi - j_lo + 1,) dy weights for I
    direction_sign: float   # -1 for "+x" (flips raw Hy integral), +1 for "-x"
    delta: float            # adjacent-probe spacing (for diagnostics)


def register_msl_plane_probes(
    sim,
    *,
    port_index: int,
    freqs: jnp.ndarray,
    name_prefix: str | None = None,
) -> MSLPlaneProbeSet:
    """Register 4 plane DFT probes for a registered MSL port + return metadata.

    Mirrors the imperative path inside ``compute_msl_s_matrix``
    (``rfx/api.py:2955-3020``):

      * V at probe k=1,2,3: plane DFT of Ez on a yz plane at
        x = msl_probe_x_coords()[k-1].  Integrated as
        ``V_f = Σ_{k=k_lo..k_hi} ez_plane[:, j_centre, k] * dz_arr[k]``.
      * I at probe 1: plane DFT of Hy on the same yz plane.  Integrated
        as ``I_f = sign · Σ_{j=j_lo..j_hi} hy_plane[:, j, k_h] * dy_arr[j]``
        where ``k_h = k_top - 1`` is one cell below the trace surface
        and the y-window extends 2·h_sub past the trace edges to
        capture fringing return current.

    Parameters
    ----------
    sim : Simulation
        Must already have ``add_msl_port`` called for the port we are
        instrumenting; we read its descriptor from ``sim._msl_ports``.
    port_index : int
        Index into ``sim._msl_ports``.
    freqs : (n_freqs,) jnp.ndarray
        Target frequencies — same convention as ``add_dft_plane_probe``.
    name_prefix : str, optional
        Prefix for the four registered DFT-plane names.  Default
        ``f"msl_p{port_index}"``.

    Returns
    -------
    MSLPlaneProbeSet
    """
    import numpy as np
    from rfx.sources.msl_port import (
        MSLPort, _msl_yz_cells, msl_probe_x_coords,
    )

    if name_prefix is None:
        name_prefix = f"msl_p{port_index}"

    pe = sim._msl_ports[port_index]
    grid = sim._build_grid()
    x_feed, y_centre, z_lo = pe.position
    mp = MSLPort(
        feed_x=float(x_feed),
        y_lo=float(y_centre - pe.width / 2),
        y_hi=float(y_centre + pe.width / 2),
        z_lo=float(z_lo),
        z_hi=float(z_lo + pe.height),
        direction=pe.direction,
        impedance=pe.impedance,
        excitation=pe.waveform,
    )

    pxs = msl_probe_x_coords(
        grid, mp,
        n_offset_cells=pe.n_probe_offset
            if pe.n_probe_offset is not None else 5,
        n_spacing_cells=pe.n_probe_spacing
            if pe.n_probe_spacing is not None else 3,
    )

    # Cross-section index metadata, identical to compute_msl_s_matrix.
    cells = _msl_yz_cells(grid, mp)
    j_set = sorted({c[1] for c in cells})
    k_set = sorted({c[2] for c in cells})
    j_lo_inner, j_hi_inner = j_set[0], j_set[-1]
    k_lo, k_hi = k_set[0], k_set[-1]
    j_centre = (j_lo_inner + j_hi_inner) // 2
    k_top = k_hi
    k_h = max(k_lo, k_top - 1)

    # Per-axis cell-size arrays (uniform OR non-uniform mesh).
    def _profile(axis: str, n: int) -> np.ndarray:
        attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
        prof = getattr(grid, attr, None)
        if prof is not None:
            return np.asarray(prof, dtype=float)
        return np.full(n, float(grid.dx), dtype=float)

    dy_arr = _profile("y", grid.ny)
    dz_arr = _profile("z", grid.nz)

    # y-extended slab for I — 2·h_sub fringing margin on each side.
    height = mp.z_hi - mp.z_lo
    dy_local = float(dy_arr[j_centre])
    n_y_margin = max(2, int(round(2 * height / dy_local)))
    j_lo_ext = max(0, j_lo_inner - n_y_margin)
    j_hi_ext = min(int(grid.ny) - 1, j_hi_inner + n_y_margin)

    direction_sign = -1.0 if mp.direction == "+x" else 1.0
    delta = float(abs(pxs[1] - pxs[0]))

    # Register the 4 plane DFT probes.  The accumulators are filled
    # inside the JIT scan body and surfaced through
    # ``ForwardResult.dft_planes[name]``.
    ez_names = [f"{name_prefix}_ez{q+1}" for q in range(3)]
    hy_name = f"{name_prefix}_hy"
    for q in range(3):
        sim.add_dft_plane_probe(
            axis="x", coordinate=float(pxs[q]),
            component="ez", freqs=freqs, name=ez_names[q],
        )
    sim.add_dft_plane_probe(
        axis="x", coordinate=float(pxs[0]),
        component="hy", freqs=freqs, name=hy_name,
    )

    return MSLPlaneProbeSet(
        ez1_name=ez_names[0], ez2_name=ez_names[1], ez3_name=ez_names[2],
        hy_name=hy_name,
        j_centre=int(j_centre),
        k_lo=int(k_lo), k_hi=int(k_hi),
        k_h=int(k_h),
        j_lo=int(j_lo_ext), j_hi=int(j_hi_ext),
        dz_slice=jnp.asarray(dz_arr[k_lo:k_hi + 1], dtype=jnp.float32),
        dy_slice=jnp.asarray(dy_arr[j_lo_ext:j_hi_ext + 1], dtype=jnp.float32),
        direction_sign=direction_sign,
        delta=delta,
    )


def _v_from_plane(fr, plane_name: str, p: MSLPlaneProbeSet) -> jnp.ndarray:
    """V_f = ∫ Ez dz along substrate column at j=j_centre on this plane."""
    plane = fr.dft_planes[plane_name].accumulator     # (n_freqs, ny, nz)
    column = jax.lax.dynamic_slice_in_dim(
        plane[:, p.j_centre, :], p.k_lo, p.k_hi - p.k_lo + 1, axis=-1,
    )
    return jnp.sum(column * p.dz_slice[None, :], axis=-1)


def _i_from_plane(fr, plane_name: str, p: MSLPlaneProbeSet) -> jnp.ndarray:
    """I_f = sign · ∫ Hy dy on the trace-bottom slab at k=k_h."""
    plane = fr.dft_planes[plane_name].accumulator     # (n_freqs, ny, nz)
    slab = jax.lax.dynamic_slice_in_dim(
        plane[:, :, p.k_h], p.j_lo, p.j_hi - p.j_lo + 1, axis=-1,
    )
    raw = jnp.sum(slab * p.dy_slice[None, :], axis=-1)
    return p.direction_sign * raw


def extract_msl_s_params_jax_plane(
    fr,
    driven: MSLPlaneProbeSet,
    passive: MSLPlaneProbeSet,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-traceable plane-integrated 2-port S11/S21 from MSL plane probes.

    Mirrors ``compute_msl_s_matrix``'s plane V/I integrals (line-Ez for
    V, area-Hy for I) and runs the same 3-probe quadratic recurrence
    used by the scalar lane :func:`extract_msl_s_params_jax`.  The plane
    integrals match the validated imperative path bit-for-bit at the
    integration level, so the residual 15-20 % notch-frequency bias on
    the scalar lane disappears.

    Parameters
    ----------
    fr : ForwardResult
        From ``Simulation.forward(...)``.  Must carry
        ``dft_planes`` (populated since 2026-05-07 via the
        ``_forward_from_materials`` plumbing — gap #4 fix).
    driven, passive : MSLPlaneProbeSet
        From :func:`register_msl_plane_probes`.

    Returns
    -------
    s11, s21 : (n_freqs,) jnp.complex64
    """
    if fr.dft_planes is None:
        raise ValueError(
            "ForwardResult.dft_planes is None.  Did you register plane "
            "probes via register_msl_plane_probes BEFORE calling forward()?"
        )

    v1d = _v_from_plane(fr, driven.ez1_name, driven)
    v2d = _v_from_plane(fr, driven.ez2_name, driven)
    v3d = _v_from_plane(fr, driven.ez3_name, driven)
    v1p = _v_from_plane(fr, passive.ez1_name, passive)
    v2p = _v_from_plane(fr, passive.ez2_name, passive)
    v3p = _v_from_plane(fr, passive.ez3_name, passive)

    alpha_d, gamma_d, _ = _solve_3probe_jax(v1d, v2d, v3d, None)
    alpha_p, _, _ = _solve_3probe_jax(v1p, v2p, v3p, None)

    eps = 1e-30
    s11 = gamma_d / (alpha_d + eps)
    s21 = alpha_p / (alpha_d + eps)
    return s11, s21
