"""JAX-traceable post-forward N-probe wave decomposition for MSL S-params.

The canonical extractor is :func:`extract_msl_nprobe` — an N-probe
least-squares wave decomposition (issue #80 Fix C) that removes the
3-probe ``q→1`` singularity and is fully JAX-traceable.

Plane-probe registration helpers:

  * :func:`register_msl_plane_probes` — adds 4 plane DFT probes per
    port (Ez planes at x=x₁/x₂/x₃ for V-proxy, Hy plane at x=x₁ for
    I-proxy) via :meth:`Simulation.add_dft_plane_probe` and returns
    the static integration metadata (j_centre, k_lo/k_hi, k_h,
    j_lo_ext/j_hi_ext, dy_arr / dz_arr slices) needed to mirror
    ``compute_msl_s_matrix``'s plane integrals.

Point-probe registration helper:

  * :func:`register_msl_wave_probes` — adds 4 point probes per port
    (Ez at probe 1/2/3 for V-proxy, Hy at probe 1 for I-proxy).

Both return probe sets consumed by :func:`extract_msl_nprobe`.
:func:`extract_msl_nprobe` returns dimensionless ``S11`` and ``S21``.
Absolute ``Z0`` is intentionally omitted (use
:meth:`compute_msl_s_matrix` for Z0).
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


_Q_EPS = 1e-30


@jax.custom_jvp
def _solve_q(v1: jnp.ndarray, v2: jnp.ndarray, v3: jnp.ndarray) -> jnp.ndarray:
    """Solve q² − c·q + 1 = 0 for the physical root with branch-cut-safe AD.

    The two roots are reciprocals (q₊·q₋ = 1).  Primal forward picks
    the root whose phase matches the observed step ratio v2/v1, then
    `jnp.where` returns it.  Reverse-mode AD through `jnp.where` and
    `jnp.sqrt(complex)` is correct on each branch *interior* but
    silently drops the implicit derivative when adjacent samples
    straddle the √disc branch cut at |q|→1.  This `custom_jvp`
    computes ∂q/∂c via the implicit-function rule:

        2q·dq − q·dc − c·dq = 0   ⇒   dq/dc = q² / (q² − 1)

    This is continuous on both branches, blows up only at the genuine
    degeneracy q² = 1 (handled by the `+eps` regularizer).  The √disc
    discontinuity is now contained inside the `custom_jvp` boundary
    where AD never traverses it.
    """
    coeff = (v1 + v3) / (v2 + _Q_EPS)
    disc = coeff ** 2 - 4.0
    sqrt_disc = jnp.sqrt(disc.astype(jnp.complex64))
    q_plus = (coeff + sqrt_disc) / 2.0
    q_minus = (coeff - sqrt_disc) / 2.0
    ratio = v2 / (v1 + _Q_EPS)
    use_plus = jnp.abs(q_plus - ratio) < jnp.abs(q_minus - ratio)
    return jnp.where(use_plus, q_plus, q_minus)


@_solve_q.defjvp
def _solve_q_jvp(primals, tangents):
    v1, v2, v3 = primals
    dv1, dv2, dv3 = tangents
    q = _solve_q(v1, v2, v3)
    # c = (v1 + v3) / (v2 + eps);  dc applied via product/quotient rule
    inv_v2 = 1.0 / (v2 + _Q_EPS)
    dc = (dv1 + dv3) * inv_v2 - (v1 + v3) * dv2 * inv_v2 ** 2
    dq_dc = q * q / (q * q - 1.0 + _Q_EPS)
    return q, dq_dc * dc


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



# ---------------------------------------------------------------------------
# Fix C — N-probe least-squares wave decomposition (issue #80)
# ---------------------------------------------------------------------------
#
# The 3-probe quadratic ``q + 1/q = (V1 + V3)/V2`` becomes singular as
# ``β·Δ → 0`` (the q→1 degeneracy): the discriminant ``coeff² − 4 → 0`` so
# root selection is noise-dominated and the extracted ``|q|`` drifts above
# 1 (non-physical for a passive line) — see issue #80.
#
# The N-probe extractor below removes that singularity entirely.  Per
# frequency it fits
#
#     V_n = α · exp(−jβ·x_n) + γ · exp(+jβ·x_n)
#
# over all N ≥ 3 voltage probes at known positions ``x_n``.  Two stages:
#
#   (a) Estimate β ROBUSTLY from the whole probe array.  β enters the
#       model non-linearly, so we scan a small window of trial β values
#       centred on the analytic Hammerstad-Jensen guess
#       ``β₀ = ω·√ε_eff/c`` and, for each trial, solve the *linear*
#       (α, γ) least-squares problem and score it by the residual.  The
#       residual is a smooth function of β, so a quadratic-refinement
#       step around the best grid node gives a sub-grid β.  Anchoring
#       the scan on the analytic guess is what makes this robust — a
#       single probe pair (the 3-probe approach) cannot do it.
#
#   (b) With β fixed, ``V_n = α·e^{−jβx_n} + γ·e^{+jβx_n}`` is LINEAR in
#       ``(α, γ)``.  The over-determined N×2 system is solved by
#       ``jnp.linalg.lstsq`` (SVD-based, JVP rule built into JAX).  No
#       quadratic, no branch cut, no q→1 singularity.
#
# Differentiability: ``jnp.linalg.lstsq`` / ``jnp.linalg.svd`` carry JVP
# rules in JAX, so ``jax.grad`` flows natively through the whole
# extractor — no hand-written ``custom_jvp`` is needed.  The β scan uses
# a fixed Python-int grid size (static) and ``jax.lax`` reductions, so
# it is JIT- and grad-safe.


def _lstsq_alpha_gamma(
    v: jnp.ndarray, x: jnp.ndarray, beta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve ``V_n = α e^{−jβx_n} + γ e^{+jβx_n}`` for one β by SVD lstsq.

    Parameters
    ----------
    v : (N,) complex
        Voltage phasors at the N probe positions.
    x : (N,) real
        Probe positions (metres), measured along the propagation axis.
    beta : scalar complex
        Trial propagation constant (rad/m).  Complex β allows a small
        imaginary (loss) part.

    Returns
    -------
    alpha, gamma : scalar complex
        Forward / backward wave amplitudes at ``x = 0``.
    residual : scalar real
        L2 norm of ``V − (α e^{−jβx} + γ e^{+jβx})`` — the fit quality.
    """
    x_c = x.astype(jnp.complex64)
    col_fwd = jnp.exp(-1j * beta * x_c)
    col_bwd = jnp.exp(+1j * beta * x_c)
    a_mat = jnp.stack([col_fwd, col_bwd], axis=-1)        # (N, 2)
    # SVD-based least squares; rcond=None uses the JAX default cutoff.
    sol, _, _, _ = jnp.linalg.lstsq(a_mat, v, rcond=None)
    alpha = sol[0]
    gamma = sol[1]
    pred = a_mat @ sol
    residual = jnp.sqrt(jnp.sum(jnp.abs(v - pred) ** 2))
    return alpha, gamma, residual


# Module-level β-scan grid resolution.  Static (Python int) so the scan
# stays JIT-shape-stable and grad-safe.
_BETA_SCAN_NODES = 41
# Scan half-width as a fraction of the analytic β₀ guess.  ±35 % brackets
# the Hammerstad-Jensen ε_eff uncertainty (~0.5 %) plus FDTD numerical
# dispersion and any moderate substrate-εr mismatch with wide margin.
_BETA_SCAN_FRAC = 0.35


def _estimate_beta(
    v: jnp.ndarray, x: jnp.ndarray, beta0: jnp.ndarray
) -> jnp.ndarray:
    """Robust β estimate: residual scan around the analytic guess + refine.

    Stage (a) of the N-probe extractor.  ``beta0`` is the analytic
    Hammerstad-Jensen guess ``ω·√ε_eff/c``; the scan brackets it by
    ``±_BETA_SCAN_FRAC`` and the minimum-residual node is refined by a
    3-point parabolic interpolation.  Fully JAX-traceable: the grid size
    is a static Python int and all reductions use ``jnp``.
    """
    beta0 = jnp.real(beta0)
    lo = beta0 * (1.0 - _BETA_SCAN_FRAC)
    hi = beta0 * (1.0 + _BETA_SCAN_FRAC)
    grid = jnp.linspace(lo, hi, _BETA_SCAN_NODES)

    def _resid(b):
        _, _, r = _lstsq_alpha_gamma(v, x, b.astype(jnp.complex64))
        return r

    resids = jax.vmap(_resid)(grid)
    k = jnp.argmin(resids)
    # Clamp so the 3-point parabolic stencil stays in range.
    k = jnp.clip(k, 1, _BETA_SCAN_NODES - 2)
    b_lo, b_mid, b_hi = grid[k - 1], grid[k], grid[k + 1]
    r_lo, r_mid, r_hi = resids[k - 1], resids[k], resids[k + 1]
    # Parabolic vertex offset (in grid-step units), clamped to [-1, 1].
    denom = (r_lo - 2.0 * r_mid + r_hi)
    num = 0.5 * (r_lo - r_hi)
    frac = jnp.where(jnp.abs(denom) > 1e-20, num / denom, 0.0)
    frac = jnp.clip(frac, -1.0, 1.0)
    step = b_mid - b_lo
    return (b_mid + frac * step).astype(jnp.complex64)


def extract_msl_nprobe(
    v: jnp.ndarray,
    x: jnp.ndarray,
    i1: jnp.ndarray,
    beta0: jnp.ndarray,
    *,
    z0_hj: float | jnp.ndarray | None = None,
) -> dict:
    """N-probe least-squares MSL wave decomposition (issue #80 Fix C).

    Robust, JAX-differentiable replacement for the 3-probe quadratic
    extractor.  Per frequency it fits ``V_n = α e^{−jβx_n} + γ e^{+jβx_n}``
    over all ``N ≥ 3`` probes, eliminating the q→1 singularity.

    Parameters
    ----------
    v : (n_freqs, N) complex
        Voltage phasors at the N probe planes (probe index is the last
        axis).  ``x[n]`` is the position of column ``n``.
    x : (N,) real
        Probe positions in metres along the propagation axis.  May be a
        signed coordinate; only differences enter the model.
    i1 : (n_freqs,) complex
        Line current at probe 0 (used for the absolute Z0 recovery).
    beta0 : (n_freqs,) real or complex
        Analytic Hammerstad-Jensen propagation-constant guess per
        frequency, ``ω·√ε_eff/c``.  Anchors the β scan (stage a).
    z0_hj : float or (n_freqs,), optional
        Analytic Hammerstad-Jensen Z0.  When provided it is returned in
        the result dict for the honesty-guard sanity check; the extractor
        itself does not depend on it.

    Returns
    -------
    dict with keys
        ``s11``   : (n_freqs,) complex — γ/α at the probe-0 reference plane.
        ``z0``    : (n_freqs,) complex — (α − γ)/I1.
        ``alpha`` : (n_freqs,) complex — forward wave amplitude at x=0.
        ``gamma`` : (n_freqs,) complex — backward wave amplitude at x=0.
        ``beta``  : (n_freqs,) complex — fitted propagation constant.
        ``q``     : (n_freqs,) complex — ``exp(-jβΔ)`` for the honesty
                     guard, Δ = x[1] − x[0].  ``|q| < 1`` when healthy.
        ``residual`` : (n_freqs,) real — per-frequency L2 fit residual.
        ``z0_hj`` : the passed-through analytic Z0 (or ``None``).

    Notes
    -----
    The model is anchored at ``x = 0``.  ``x`` is shifted internally so
    its first entry sits at the origin; ``α``/``γ``/``S11`` are therefore
    referenced to probe 0 — identical to the 3-probe extractor's
    convention.
    """
    eps = 1e-30
    v = jnp.asarray(v, dtype=jnp.complex64)
    if v.ndim == 1:
        v = v[None, :]
    x = jnp.asarray(x, dtype=jnp.float32)
    # Reference the model at probe 0 so α/γ are probe-0 amplitudes.
    x = x - x[0]
    i1 = jnp.asarray(i1, dtype=jnp.complex64)
    beta0 = jnp.asarray(beta0)
    if beta0.ndim == 0:
        beta0 = jnp.broadcast_to(beta0, (v.shape[0],))

    def _per_freq(v_row, b0):
        beta = _estimate_beta(v_row, x, b0)
        alpha, gamma, residual = _lstsq_alpha_gamma(v_row, x, beta)
        return alpha, gamma, beta, residual

    alpha, gamma, beta, residual = jax.vmap(_per_freq)(v, beta0)

    z0 = (alpha - gamma) / (i1 + eps)
    s11 = gamma / (alpha + eps)
    delta = x[1] - x[0]
    q = jnp.exp(-1j * beta * delta.astype(jnp.complex64))

    return dict(
        s11=s11,
        z0=z0,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        q=q,
        residual=residual,
        z0_hj=z0_hj,
    )
