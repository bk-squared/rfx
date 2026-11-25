"""JAX-traceable post-forward N-probe wave decomposition for MSL S-params.

The canonical extractor is :func:`extract_msl_nprobe` — an N-probe
least-squares wave decomposition that removes the
3-probe ``q→1`` singularity and is fully JAX-traceable.

Plane-probe registration helpers:

  * :func:`register_msl_plane_probes` — adds 7 plane DFT probes per
    port (Ez planes at x=x₁/x₂/x₃ for V, bracketing Hy + Hz planes for
    the closed-loop I at x=x₁) via :meth:`Simulation.add_dft_plane_probe` and
    returns the static integration metadata (trace-PEC-search-derived
    j/k spans, per-cell dz/dy profiles, the leapfrog half-step phase)
    that :func:`_v_from_plane` / :func:`_i_from_plane` feed straight
    into the PRODUCTION primitives (:func:`rfx.api._sparams.
    msl_modal_voltage`, :func:`rfx.sources.msl_port.msl_loop_current`)
    -- transverse integrals and H-to-E-plane interpolation are shared with production.

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

from rfx.measurement.msl_waves import _lstsq_alpha_gamma
from rfx.measurement.phasors import exp_j


@dataclass(frozen=True)
class MSLWaveProbeSet:
    """Indices into ``ForwardResult.time_series`` for a single port's
    4-probe set (3 Ez + 1 Hy).

    ``delta`` is the physical spacing between probes 1 and 2, which equals
    the spacing between probes 2 and 3.
    """
    ez1_col: int
    ez2_col: int
    ez3_col: int
    hy1_col: int
    delta: float


def register_msl_wave_probes(
    sim,
    *,
    feed_x: float,
    direction: str,
    y_centre: float,
    z_ez: float,
    z_hy: float,
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

    ``z_ez`` is the physical z of the Ez probes at the substrate middle and
    ``z_hy`` is the physical z of the Hy probe at the trace top.
    """
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    sign = 1 if direction == "+x" else -1
    grid = sim._build_realized_grid()
    dx = float(grid.cells(0)[0])

    x1 = feed_x + sign * n_offset_cells * dx
    x2 = x1 + sign * n_spacing_cells * dx
    x3 = x2 + sign * n_spacing_cells * dx
    delta = float(abs(x2 - x1))

    base = len(sim._probes)
    sim.add_probe(position=(x1, y_centre, z_ez), component="ez")
    sim.add_probe(position=(x2, y_centre, z_ez), component="ez")
    sim.add_probe(position=(x3, y_centre, z_ez), component="ez")
    sim.add_probe(position=(x1, y_centre, z_hy), component="hy")
    return MSLWaveProbeSet(
        ez1_col=base, ez2_col=base + 1, ez3_col=base + 2, hy1_col=base + 3,
        delta=delta,
    )


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
    inv_v2 = 1.0 / (v2 + _Q_EPS)
    dc = (dv1 + dv3) * inv_v2 - (v1 + v3) * dv2 * inv_v2 ** 2
    dq_dc = q * q / (q * q - 1.0 + _Q_EPS)
    return q, dq_dc * dc



@dataclass(frozen=True)
class MSLPlaneProbeSet:
    """Plane DFT probe names + static integration metadata for one MSL port.

    Every index/array here is computed once at registration time by the
    SAME calls ``compute_msl_s_matrix`` makes (:func:`rfx.sources.
    msl_port.msl_cross_section_span` for the geometry, the PEC-mask walk
    for the rasterized trace node) so :func:`_v_from_plane` /
    :func:`_i_from_plane` can hand them straight to the production
    primitives instead of re-deriving V/I locally.

    ``k_hi`` is an exclusive bound, while ``k_lo`` and the spans ``j_lo`` to
    ``j_hi`` and ``k_trace_lo`` to ``k_trace_hi`` are inclusive.
    """
    ez1_name: str
    ez2_name: str
    ez3_name: str
    hy_name: str
    hz_name: str
    j_centre: int
    k_lo: int
    k_hi: int
    j_lo: int
    j_hi: int
    k_trace_lo: int
    k_trace_hi: int
    dz_arr: jnp.ndarray
    dy_arr: jnp.ndarray
    direction: str
    delta: float
    hy_left_name: str | None = None
    hz_left_name: str | None = None
    h_weights: tuple[float, float] | None = None


def register_msl_plane_probes(
    sim,
    *,
    port_index: int,
    freqs: jnp.ndarray,
    name_prefix: str | None = None,
) -> MSLPlaneProbeSet:
    """Register 7 plane DFT probes for a registered MSL port + return metadata.

    Computes the SAME geometry ``compute_msl_s_matrix`` computes
    (``rfx/api/_sparams.py:2953-3072``) so :func:`_v_from_plane` /
    :func:`_i_from_plane` can hand it straight to the production
    primitives:

      * V at probe k=1,2,3: plane DFT of Ez on a yz plane at
        x = msl_probe_x_coords()[k-1], integrated ground (``k_lo``) to
        the rasterized trace's bottom node (``k_hi``, EXCLUSIVE) by
        :func:`rfx.api._sparams.msl_modal_voltage`.  ``k_hi`` comes from
        walking the PEC mask up from the substrate top, exactly as
        ``compute_msl_s_matrix``'s ``trace_k_per_port`` does — NOT the
        ``round(h_sub/dx)`` proxy, which is one substrate edge short for
        ``frac(h_sub/dx) in (0, 0.5)`` (
        see ``msl_modal_voltage``'s docstring).
      * I at probe 1: the closed Ampere loop ``∮H·dl`` around the trace
        conductor from Hy + Hz on the two bracketing Yee half-planes,
        interpolated to the voltage plane before
        :func:`rfx.sources.msl_port.msl_loop_current`.

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
        Prefix for the seven registered DFT-plane names. Default
        ``f"msl_p{port_index}"``.

    Returns
    -------
    MSLPlaneProbeSet

    The H-plane registration coordinates name array indices, and the
    stencil weights account for H's physical half-cell offset.
    """
    import numpy as np
    from rfx.sources.msl_port import (
        msl_cross_section_span, msl_h_plane_stencil, msl_port_from_entry,
        msl_probe_x_coords,
    )

    if name_prefix is None:
        name_prefix = f"msl_p{port_index}"

    pe = sim._msl_ports[port_index]

    if pe.direction not in ("+x", "-x"):
        raise NotImplementedError(
            f"register_msl_plane_probes supports direction '+x'/'-x' only, "
            f"got {pe.direction!r} for port {port_index}. This is the "
            f"retained 3-probe DIAGNOSTIC primitive, not the production "
            f"S-matrix path; use Simulation.compute_msl_s_matrix(), which "
            f"handles every supported direction."
        )

    _is_nonuniform = (
        getattr(sim, "_dx_profile", None) is not None
        or getattr(sim, "_dy_profile", None) is not None
        or getattr(sim, "_dz_profile", None) is not None
    )
    if _is_nonuniform:
        raise NotImplementedError(
            "register_msl_plane_probes does not support a non-uniform mesh "
            "(sim._dx_profile / _dy_profile / _dz_profile set). "
            "sim._build_grid() always returns the uniform Grid regardless "
            "of these profiles, so this diagnostic path cannot build the "
            "NonUniformGrid compute_msl_s_matrix's non-uniform branch uses "
            "for its trace-PEC search. Use Simulation.compute_msl_s_matrix(), "
            "which builds that grid via build_nonuniform_grid(...)."
        )

    grid = sim._build_grid()
    mp = msl_port_from_entry(pe)

    pxs = msl_probe_x_coords(
        grid, mp,
        n_offset_cells=pe.n_probe_offset
            if pe.n_probe_offset is not None else 5,
        n_spacing_cells=pe.n_probe_spacing
            if pe.n_probe_spacing is not None else 3,
    )
    h_stencil = msl_h_plane_stencil(grid, mp, pxs[0])

    span = msl_cross_section_span(grid, mp, require_contiguous_width=True)
    j_centre = span["w_centre"]
    j_lo, j_hi = span["w_lo"], span["w_hi"]
    k_lo = span["n_lo"]

    dy_arr = np.asarray(grid.cells('y'), dtype=float)
    dz_arr = np.asarray(grid.cells('z'), dtype=float)

    k_trace_lo, k_trace_hi = _realized_trace_planes(sim, grid, span, j_centre)
    if k_trace_lo is None:
        raise RuntimeError(
            "register_msl_plane_probes: no realized PEC trace conductor "
            "found above the substrate top for MSL port "
            f"{port_index} ({pe.name!r}); the closed Ampere-loop current "
            "needs the trace. Declare the microstrip trace as a "
            "Box(material='pec') (a volume) or as a zero-thickness Box / "
            "add_thin_conductor (a sheet)."
        )

    delta = float(abs(pxs[1] - pxs[0]))

    ez_names = [f"{name_prefix}_ez{q+1}" for q in range(3)]
    hy_name = f"{name_prefix}_hy"
    hz_name = f"{name_prefix}_hz"
    hy_left_name = f"{hy_name}_left"
    hz_left_name = f"{hz_name}_left"
    for q in range(3):
        sim.add_dft_plane_probe(
            axis="x", coordinate=float(pxs[q]),
            component="ez", freqs=freqs, name=ez_names[q],
        )
    for coordinate, names in (
        (h_stencil["registration_coordinates"][1], (hy_name, hz_name)),
        (h_stencil["registration_coordinates"][0], (hy_left_name, hz_left_name)),
    ):
        for component, name in zip(("hy", "hz"), names):
            sim.add_dft_plane_probe(
                axis=h_stencil["axis"], coordinate=coordinate,
                component=component, freqs=freqs, name=name,
            )

    return MSLPlaneProbeSet(
        ez1_name=ez_names[0], ez2_name=ez_names[1], ez3_name=ez_names[2],
        hy_name=hy_name, hz_name=hz_name,
        j_centre=int(j_centre),
        k_lo=int(k_lo), k_hi=k_trace_lo,
        j_lo=int(j_lo), j_hi=int(j_hi),
        k_trace_lo=k_trace_lo, k_trace_hi=k_trace_hi,
        dz_arr=jnp.asarray(dz_arr, dtype=jnp.float32),
        dy_arr=jnp.asarray(dy_arr, dtype=jnp.float32),
        direction=mp.direction,
        delta=delta,
        hy_left_name=hy_left_name,
        hz_left_name=hz_left_name,
        h_weights=h_stencil["weights"],
    )


def realized_trace_planes_on_column(pec_edge_masks, normal_idx, ij, k_from,
                                    periodic=(False, False, False)):
    """Lowest and highest realized PEC wall plane on one column, at or
    above ``k_from`` along ``normal_idx``.

    Replaces the ``pec_mask`` CELL scan every MSL trace detector used.  A
    volume trace answers with its LOWER wall plane and its far face; a
    sheet trace answers with its single plane.  ``(None, None)`` when the
    column carries no realized PEC above ``k_from``.  ``periodic`` is the
    run's periodic-axis flags, forwarded so a column on the seam node of a periodic
    in-plane axis finds its backward incident edge.
    """
    from rfx.boundaries.pec import realized_wall_planes
    from rfx.core.jax_utils import is_tracer
    if pec_edge_masks is None or any(is_tracer(mask) for mask in pec_edge_masks):
        return None, None
    planes = [k for k in realized_wall_planes(
        pec_edge_masks, normal_idx, ij=ij, periodic=periodic)
        if k >= int(k_from)]
    if not planes:
        return None, None
    return int(min(planes)), int(max(planes))


def _realized_trace_planes(sim, grid, span, j_centre):
    """``realized_trace_planes_on_column`` for the uniform diagnostic path."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    _pec_sheets: list = []
    _pec_wires: list = []
    _assembled = sim._assemble_materials(
        grid, pec_sheets=_pec_sheets, pec_wires=_pec_wires)
    _pec_mask = _assembled[3]
    if _pec_mask is None and not _pec_sheets and not _pec_wires:
        return None, None
    _periodic = sim._periodic_flags()
    with jax.ensure_compile_time_eval():
        masks = realized_pec_edge_masks(
            _pec_mask, sheets=tuple(_pec_sheets), wires=tuple(_pec_wires),
            periodic=_periodic)
    ij = tuple(
        span["i_feed"] if c == span["prop_idx"] else int(j_centre)
        for c in range(3) if c != span["normal_idx"]
    )
    return realized_trace_planes_on_column(
        masks, span["normal_idx"], ij, span["n_hi"], periodic=_periodic)


def _v_from_plane(fr, plane_name: str, p: MSLPlaneProbeSet) -> jnp.ndarray:
    """V_f = production ``msl_modal_voltage`` on this plane's Ez accumulator.

    It calls
    :func:`rfx.api._sparams.msl_modal_voltage` directly with the SAME
    ``j_centre`` / ``k_lo`` / ``k_hi`` (EXCLUSIVE, the rasterized trace's
    bottom node) / ``dz_arr`` that ``compute_msl_s_matrix`` computes, so
    it cannot re-drift from production.
    """
    from rfx.api._sparams import msl_modal_voltage
    plane = fr.dft_planes[plane_name].accumulator
    return msl_modal_voltage(
        plane, j_centre=p.j_centre, k_lo=p.k_lo, k_hi=p.k_hi,
        dz_arr=p.dz_arr, dtype=plane.dtype,
    )


def _i_from_plane(fr, plane_name: str, p: MSLPlaneProbeSet) -> jnp.ndarray:
    """I_f = production ``msl_loop_current`` closed Ampere loop.

    ``plane_name`` names the retained right Hy plane; its left neighbour
    and both Hz planes are registered by :func:`register_msl_plane_probes`.
    Interpolate each physically stamped H pair to the voltage E-node. Missing bracketing metadata or data is an error.
    A single Hy slab undercounts by ~1.5x compared with the closed loop. The shared DFT owns the half-step phase; this
    projector
    calls :func:`rfx.sources.msl_port.msl_loop_current` directly with the
    SAME trace span ``compute_msl_s_matrix`` uses, so the sign
    convention comes from ``msl_loop_current``/``msl_axis_roles`` alone —
    no extra direction multiply here (that would double-apply the sign).
    """
    from rfx.sources.msl_port import msl_collocate_h_planes, msl_loop_current

    if p.hy_left_name is None or p.hz_left_name is None or p.h_weights is None:
        raise ValueError(
            "MSL current requires bracketing H-plane metadata; "
            "re-register the plane probes and acquire both H samples")
    planes = getattr(fr, "dft_planes", None) or {}
    required = (p.hy_left_name, plane_name, p.hz_left_name, p.hz_name)
    missing = [name for name in required if name not in planes]
    if missing:
        raise ValueError(f"MSL current requires bracketing H-plane data; missing {missing}")
    hy_plane = msl_collocate_h_planes(
        planes[p.hy_left_name].accumulator, planes[plane_name].accumulator, p.h_weights)
    hz_plane = msl_collocate_h_planes(
        planes[p.hz_left_name].accumulator, planes[p.hz_name].accumulator, p.h_weights)
    return msl_loop_current(
        hy_plane, hz_plane,
        j_lo=p.j_lo, j_hi=p.j_hi,
        k_trace_lo=p.k_trace_lo, k_trace_hi=p.k_trace_hi,
        dy_arr=p.dy_arr, dz_arr=p.dz_arr,
        direction=p.direction,
    )





_BETA_SCAN_NODES = 41
_BETA_SCAN_FRAC = 0.35


def _estimate_beta(
    v: jnp.ndarray, x: jnp.ndarray, beta0: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Robust β estimate: residual scan around the analytic guess + refine.

    Stage (a) of the N-probe extractor.  ``beta0`` is the analytic
    Hammerstad-Jensen guess ``ω·√ε_eff/c``; the scan brackets it by
    ``±_BETA_SCAN_FRAC`` and the minimum-residual node is refined by a
    3-point parabola through the squared residual.  Fully JAX-traceable: the grid size
    is a static Python int and all reductions use ``jnp``.

    Returns
    -------
    beta : scalar complex
        The refined β estimate.
    railed : scalar bool
        True when the scan FAILED TO BRACKET the optimum — the raw
        argmin sits at either edge node of the ±``_BETA_SCAN_FRAC``
        window, or the refined β lands within half a grid step of a
        window limit.  A railed β (and the Z0 derived from it) is the
        window limit, not a measurement: the true optimum lies at or
        beyond the scan edge (a substrate-εr / HJ-ε_eff
        mismatch large enough to push the true β outside ±35% used to
        be returned silently pinned at ``0.974·rail``).  Pure edge
        conditions — no tunable threshold.

    On the normalized probe vector, a parabola second difference with
    ``|denom|`` at most 1e-20 means the squared-residual curve is flat to noise.
    """
    beta0 = jnp.real(beta0)
    lo = beta0 * (1.0 - _BETA_SCAN_FRAC)
    hi = beta0 * (1.0 + _BETA_SCAN_FRAC)
    grid = jnp.linspace(lo, hi, _BETA_SCAN_NODES)

    v_scale = jnp.max(jnp.abs(v))
    v_n = v / jnp.where(v_scale > 0.0, v_scale, 1.0)

    def _resid(b):
        _, _, r = _lstsq_alpha_gamma(v_n, x, b.astype(jnp.complex64))
        return r

    resids = jax.vmap(_resid)(grid)
    k_raw = jnp.argmin(resids)
    k = jnp.clip(k_raw, 1, _BETA_SCAN_NODES - 2)
    b_lo, b_mid, _ = grid[k - 1], grid[k], grid[k + 1]
    r_lo, r_mid, r_hi = (resids[k - 1] ** 2, resids[k] ** 2,
                         resids[k + 1] ** 2)
    denom = (r_lo - 2.0 * r_mid + r_hi)
    num = 0.5 * (r_lo - r_hi)
    ok = jnp.abs(denom) > 1e-20
    safe_denom = jnp.where(ok, denom, 1.0)
    frac = jnp.where(ok, num / safe_denom, 0.0)
    frac = jnp.clip(frac, -1.0, 1.0)
    step = b_mid - b_lo
    beta = (b_mid + frac * step).astype(jnp.complex64)
    half = 0.5 * (grid[1] - grid[0])
    beta_r = jnp.real(beta)
    railed = (
        (k_raw <= 0)
        | (k_raw >= _BETA_SCAN_NODES - 1)
        | (beta_r <= grid[0] + half)
        | (beta_r >= grid[-1] - half)
    )
    return beta, railed


def _msl_fit_conditioning(x, beta, alpha, gamma):
    """Return phase span and per-probe fit conditioning, detached from AD.

    Centering changes only the wave amplitudes. Subtracting a multiple of
    sin(beta*y) from the beta column leaves its orthogonal residual unchanged:
    y*cos(beta*y) becomes y*(cos(beta*y) - sinc(beta*y/pi)). This exact
    identity avoids cancellation for the odd term only. The even term
    cos(beta*y) - sinc(beta*y/pi) ≈ -(beta*y)**2/3 still subtracts two
    numbers near 1 in float32; it dominates sigma_perp only near a voltage
    null at the centre (a + g ≈ 0). No small-phase series is used.
    QR uses the equivalent
    well-scaled wave basis {cos(beta*y), y*sinc(beta*y/pi)}.
    """
    x, beta, alpha, gamma = jax.tree.map(
        jax.lax.stop_gradient, (x, beta, alpha, gamma))
    centre = jnp.mean(x)
    y = x - centre

    def per_bin(b, a, g):
        a = a * jnp.exp(-1j * b * centre)
        g = g * jnp.exp(1j * b * centre)
        phase = b * y
        cosine = jnp.cos(phase)
        sinc = jnp.sinc(phase / jnp.pi)
        column = (-(a + g) * y * jnp.sin(phase)
                  + 1j * (g - a) * y * (cosine - sinc))
        basis = jnp.stack((cosine, y * sinc), axis=-1)
        q, _ = jnp.linalg.qr(basis, mode="reduced")
        residual = column - q @ (q.conj().T @ column)
        sigma = jnp.linalg.norm(residual)
        v_fit = a * jnp.exp(-1j * phase) + g * jnp.exp(1j * phase)
        return jnp.linalg.norm(v_fit) / (jnp.real(b) * sigma)

    return (jnp.real(beta) * jnp.abs(x[-1] - x[0]),
            jax.vmap(per_bin)(beta, alpha, gamma))


def extract_msl_nprobe(
    v: jnp.ndarray,
    x: jnp.ndarray,
    i1: jnp.ndarray,
    beta0: jnp.ndarray,
    *,
    z0_hj: float | jnp.ndarray | None = None,
) -> dict:
    """N-probe least-squares MSL wave decomposition.

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
        ``beta_railed`` : (n_freqs,) bool — True where the β scan failed
                     to bracket its optimum (raw argmin at a window-edge
                     node, or refined β within half a grid step of a
                     window limit).  The returned ``beta``/``z0`` at such
                     a bin are the scan-window limit, NOT a measurement; do not quote them.
        ``fit_span_phase`` : (n_freqs,) real — the line's phase across the
                     probes at that bin, the quantity the known-limitations
                     entry conditions on. Detached from AD.
        ``fit_amplification`` : (n_freqs,) real — the fit's sensitivity of β
                     (and of Z0) to a departure of the probe voltages from one
                     forward-plus-backward wave, per probe (the fit's
                     conditioning). It is not an error estimate. Computed as
                     ‖V_fit‖/(Re β·σ⊥), where V_fit is the fitted wave at
                     the probes and σ⊥ is the norm of the beta derivative
                     at fitted alpha/gamma projected off both wave columns.
                     Detached from AD.
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
    x = x - x[0]
    i1 = jnp.asarray(i1, dtype=jnp.complex64)
    beta0 = jnp.asarray(beta0)
    if beta0.ndim == 0:
        beta0 = jnp.broadcast_to(beta0, (v.shape[0],))

    def _per_freq(v_row, b0):
        beta, railed = _estimate_beta(v_row, x, b0)
        alpha, gamma, residual = _lstsq_alpha_gamma(v_row, x, beta)
        return alpha, gamma, beta, residual, railed

    alpha, gamma, beta, residual, beta_railed = jax.vmap(_per_freq)(v, beta0)
    fit_span_phase, fit_amplification = _msl_fit_conditioning(x, beta, alpha, gamma)

    z0 = (alpha - gamma) / (i1 + eps)
    s11 = gamma / (alpha + eps)
    delta = x[1] - x[0]
    q = exp_j(beta, delta.astype(jnp.complex64), unit=-1j, xp=jnp)

    return dict(
        s11=s11,
        z0=z0,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        q=q,
        residual=residual,
        beta_railed=beta_railed,
        fit_span_phase=fit_span_phase,
        fit_amplification=fit_amplification,
        z0_hj=z0_hj,
    )
