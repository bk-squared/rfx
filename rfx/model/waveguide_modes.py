"""Rectangular waveguide geometry and analytical and discrete mode profiles."""

from __future__ import annotations

from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from rfx.boundaries.axes import hi_face_is_wall
from rfx.core.yee import EPS_0, MU_0
from rfx.model.waveguide_waveform import waveguide_waveform_error
from rfx.model.waveguide_mode_solver import (
    _aperture_area,
    _cell_centred_gradient,
    _galerkin_eigh_separable_laplacian_2d,
    _galerkin_stiffness_mass_1d,
    _orthonormalize_profile_arrays,
    _pick_eigenmode_by_overlap,
    _pick_eigenmode_by_target_then_overlap,
    _scale_h_to_unit_cross,
    _shift_profile_to_dual,
)


C0_LOCAL = 1.0 / np.sqrt(EPS_0 * MU_0)


class WaveguidePort(NamedTuple):
    """Waveguide port definition.

    x_index : int
        Plane index along the port normal axis (legacy name kept for
        backwards compatibility).
    y_slice : (y_lo, y_hi) or None
        Legacy x-normal first transverse slice (y) when `normal_axis='x'`.
    z_slice : (z_lo, z_hi) or None
        Legacy x-normal second transverse slice (z) when `normal_axis='x'`.
    a : float
        Waveguide width in meters along the first local transverse axis.
    b : float
        Waveguide height in meters along the second local transverse axis.
    mode : (m, n)
        Mode indices. (1, 0) for TE10 dominant mode.
    mode_type : str
        "TE" or "TM". Default "TE".
    direction : str
        Propagation direction when the port is driven. ``"+x"`` for a left
        port launching into the guide, ``"-x"`` for a right port launching
        back toward the guide interior.
    x_position : float | None
        Physical location (metres) of the source plane along the port normal
        axis. If omitted, helpers fall back to `x_index * dx`.
    normal_axis : {"x","y","z"}
        Port-normal axis. Default `"x"` for the original straight-guide model.
    u_slice, v_slice : tuple[int, int] or None
        Generic transverse aperture slices used for non-x-normal ports.

        CELL spans, not node spans: these index the FIELD ARRAYS through
        :func:`_plane_indexer`, so ``hi - lo`` is the number of aperture cells
        the transverse mode operator is built on and ``sum(widths)`` over the
        slice must equal ``a`` (resp. ``b``). A node span — which is what
        ``rfx/api/_compile.py::_range_to_slice`` and
        ``rfx/runners/nonuniform.py::_range_to_slice_nu`` produce, both
        reporting the physical aperture as ``(hi - lo - 1) * d`` — has ONE
        ENTRY TOO MANY and must be converted at the call site (unconverted, it puts ``f_cutoff`` on
        a guide one cell wider than the
        walls make, 10 % low at ``dx = a/9`` and 2.7 % low at ``dx = a/36``).
    """
    x_index: int
    y_slice: tuple[int, int] | None
    z_slice: tuple[int, int] | None
    a: float
    b: float
    mode: tuple[int, int] = (1, 0)
    mode_type: str = "TE"
    direction: str = "+x"
    x_position: float | None = None
    normal_axis: str = "x"
    u_slice: tuple[int, int] | None = None
    v_slice: tuple[int, int] | None = None


class WaveguidePortConfig(NamedTuple):
    """Compiled waveguide port config for time-stepping."""
    # Port geometry
    x_index: int       # Source injection plane
    ref_x: int         # Reference probe (near source, downstream)
    probe_x: int       # Measurement probe (further downstream)
    y_lo: int
    y_hi: int
    z_lo: int
    z_hi: int
    normal_axis: str
    u_lo: int
    u_hi: int
    v_lo: int
    v_hi: int
    e_u_component: str
    e_v_component: str
    h_u_component: str
    h_v_component: str

    # Normalized mode profiles on the aperture (ny_port, nz_port)
    ey_profile: jnp.ndarray
    ez_profile: jnp.ndarray
    hy_profile: jnp.ndarray
    hz_profile: jnp.ndarray

    # Waveguide parameters
    mode_type: str
    direction: str
    f_cutoff: float
    a: float
    b: float
    dx: float
    dt: float
    # Per-axis transverse cell widths covering the port aperture.
    # Shape (nu_port,) and (nv_port,) respectively. For uniform Yee grids
    # both arrays are constant ``dx``; on a NonUniformGrid these are the
    # slice of ``dx_arr`` / ``dy_arr`` / ``dz`` covering the aperture.
    u_widths: jnp.ndarray
    v_widths: jnp.ndarray
    source_x_m: float
    reference_x_m: float
    probe_x_m: float
    dft_total_steps: int

    # Source waveform parameters. ``waveform`` selects the pulse shape:
    # ``"differentiated_gaussian"`` is refused (circular precursor, #1942);
    # ``"modulated_gaussian"`` (Meep-style, src = cos(2π·f₀·(t-t₀))·exp(-arg²)).
    # ``src_fcen`` is only used by the modulated-Gaussian dispatch.
    src_amp: float
    src_t0: float
    src_tau: float
    src_fcen: float
    waveform: str

    freqs: jnp.ndarray         # (n_freqs,) float

    # Per-step time-series (OpenEMS-style raw record).
    # Modal V and I are real-valued per step (real Yee fields × real mode
    # profiles × cell areas). Record precision follows fields/frequencies.
    # OpenEMS writes the same on disk as two floats per row.
    # Allocated to length ``dft_total_steps``; ``n_steps_recorded`` tracks
    # how many slots were filled by ``update_waveguide_port_probe``.
    # The S-parameter spectra are computed POST-SCAN by
    # ``_extract_global_waves_from_time_series`` via a rectangular
    # full-record DFT on these arrays — no in-scan DFT accumulators
    # are kept (deleted Phase 2 cleanup, 2026-04-25).
    v_probe_t: jnp.ndarray     # (n_steps,) float32 — modal V at probe plane
    v_ref_t: jnp.ndarray       # (n_steps,) float32 — modal V at ref plane
    i_probe_t: jnp.ndarray     # (n_steps,) float32 — modal I at probe plane
    i_ref_t: jnp.ndarray       # (n_steps,) float32 — modal I at ref plane
    v_inc_t: jnp.ndarray       # (n_steps,) float32 — source amplitude record
    n_steps_recorded: jnp.ndarray  # scalar int32 — fill count

    # Precomputed TFSF-style injection waveforms. Both are the source
    # pulse bandpass-filtered to the propagating band |ω| ≥ ω_c — the
    # raw differentiated Gaussian has large sub-cutoff DC content that
    # would only excite evanescent modes, so the filter is essential.
    #
    # e_inc_table[n] is the filtered E-amplitude scalar at (x_src, n·dt);
    # h_inc_table[n] is the companion H-amplitude scalar at (x_src − 0.5·dx,
    # (n+0.5)·dt). Used by ``apply_waveguide_port_h`` and
    # ``apply_waveguide_port_e`` to inject a unidirectional mode wave.
    e_inc_table: jnp.ndarray   # (n_steps,) float
    h_inc_table: jnp.ndarray   # (n_steps,) float

    # Per-cell aperture area for modal V/I integration. Shape (nu, nv).
    # Equals u_widths × v_widths for interior ports, but cells touching a
    # PEC +face boundary at the array edge are weighted 0 to exclude
    # ghost cells whose centre lies inside the conductor (where Ez is
    # not zeroed by apply_pec_faces — Ez is "normal" by convention).
    # Sentinel `(0, 0)` triggers a fallback in `_aperture_dA` for low-
    # level callers that bypass `init_waveguide_port`.
    aperture_dA: jnp.ndarray = jnp.zeros((0, 0), dtype=jnp.float32)
    h_offset: tuple[float, float] = (0.0, 0.0)
    mode_indices: tuple[int, int] = (0, 0)
    mode_profile: str = "discrete"


def _te_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      *,
                      u_widths: np.ndarray | None = None,
                      v_widths: np.ndarray | None = None,
                      aperture_dA: np.ndarray | None = None,
                      h_offset: tuple[float, float] = (0.0, 0.0),
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TE_mn E and H transverse mode profiles.

    Returns (ey, ez, hy, hz) each of shape (ny, nz), normalized so that
    integral(Ey² + Ez²) dA = 1.

    Derivation: TE_mn eigenfunction Hx = cos(mπy/a) cos(nπz/b).
    Transverse E from Maxwell (propagation along +x):
        Ey = -(nπ/b) cos(mπy/a) sin(nπz/b)
        Ez =  (mπ/a) sin(mπy/a) cos(nπz/b)

    The (mπ/a) and (nπ/b) derivative weights are essential for correct
    relative amplitudes in higher-order modes (e.g., TE11 with a != b).
    """
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = -(n * np.pi / b) * np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b) if n > 0 else np.zeros_like(Y)
    ez = (m * np.pi / a) * np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b) if m > 0 else np.zeros_like(Y)

    # H for forward +x propagation is the rotated E template
    # (hy = -ez, hz = ey).  When requested, evaluate H on the Yee dual
    # mesh rather than reusing the E-grid samples; modal_current samples
    # the simulated H field with the same offset.
    if h_offset != (0.0, 0.0):
        if u_widths is None or v_widths is None:
            raise ValueError("h_offset requires u_widths and v_widths")
        u_h = np.asarray(y_coords) + float(h_offset[0]) * np.asarray(u_widths)
        v_h = np.asarray(z_coords) + float(h_offset[1]) * np.asarray(v_widths)
        Yh, Zh = np.meshgrid(u_h, v_h, indexing='ij')
        hy = (
            -(m * np.pi / a)
            * np.sin(m * np.pi * Yh / a)
            * np.cos(n * np.pi * Zh / b)
            if m > 0 else np.zeros_like(Yh)
        )
        hz = (
            -(n * np.pi / b)
            * np.cos(m * np.pi * Yh / a)
            * np.sin(n * np.pi * Zh / b)
            if n > 0 else np.zeros_like(Yh)
        )
    else:
        hy = -ez.copy()
        hz = ey.copy()

    # Normalize: integral(Ey² + Ez²) dA = 1
    if aperture_dA is not None:
        dA = np.asarray(aperture_dA, dtype=np.float64)
        power = float(np.sum((ey**2 + ez**2) * dA))
    elif u_widths is not None and v_widths is not None:
        dA = np.asarray(u_widths)[:, None] * np.asarray(v_widths)[None, :]
        power = float(np.sum((ey**2 + ez**2) * dA))
    else:
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
        dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
        dA = np.full_like(ey, dy * dz, dtype=np.float64)
        power = float(np.sum((ey**2 + ez**2) * dA))
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm
        hy, hz = _scale_h_to_unit_cross(ey, ez, hy, hz, dA)

    return ey, ez, hy, hz


def _tm_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      *,
                      u_widths: np.ndarray | None = None,
                      v_widths: np.ndarray | None = None,
                      aperture_dA: np.ndarray | None = None,
                      h_offset: tuple[float, float] = (0.0, 0.0),
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TM_mn E and H transverse mode profiles.

    TM modes require both m >= 1 and n >= 1.
    Eigenfunction: Ex_z = sin(mπy/a) sin(nπz/b).
    Transverse E from grad_t:
        Ey = (mπ/a) cos(mπy/a) sin(nπz/b)
        Ez = (nπ/b) sin(mπy/a) cos(nπz/b)
    """
    if m < 1 or n < 1:
        raise ValueError(f"TM modes require m >= 1 and n >= 1, got ({m}, {n})")

    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = (m * np.pi / a) * np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b)
    ez = (n * np.pi / b) * np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b)

    if h_offset != (0.0, 0.0):
        if u_widths is None or v_widths is None:
            raise ValueError("h_offset requires u_widths and v_widths")
        u_h = np.asarray(y_coords) + float(h_offset[0]) * np.asarray(u_widths)
        v_h = np.asarray(z_coords) + float(h_offset[1]) * np.asarray(v_widths)
        Yh, Zh = np.meshgrid(u_h, v_h, indexing='ij')
        ey_h = (m * np.pi / a) * np.cos(m * np.pi * Yh / a) * np.sin(n * np.pi * Zh / b)
        ez_h = (n * np.pi / b) * np.sin(m * np.pi * Yh / a) * np.cos(n * np.pi * Zh / b)
        hy = -ez_h
        hz = ey_h
    else:
        hy = -ez.copy()
        hz = ey.copy()

    if aperture_dA is not None:
        dA = np.asarray(aperture_dA, dtype=np.float64)
        power = float(np.sum((ey**2 + ez**2) * dA))
    elif u_widths is not None and v_widths is not None:
        dA = np.asarray(u_widths)[:, None] * np.asarray(v_widths)[None, :]
        power = float(np.sum((ey**2 + ez**2) * dA))
    else:
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
        dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
        dA = np.full_like(ey, dy * dz, dtype=np.float64)
        power = float(np.sum((ey**2 + ez**2) * dA))
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm
        hy, hz = _scale_h_to_unit_cross(ey, ez, hy, hz, dA)

    return ey, ez, hy, hz


def cutoff_frequency(a: float, b: float, m: int, n: int) -> float:
    """TE_mn or TM_mn cutoff frequency for rectangular waveguide."""
    kc = np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
    return kc * C0_LOCAL / (2 * np.pi)


def _discrete_te_mode_profiles(a: float, b: float, m: int, n: int,
                                u_widths: np.ndarray,
                                v_widths: np.ndarray,
                                *,
                                aperture_dA: np.ndarray | None = None,
                                h_offset: tuple[float, float] = (0.0, 0.0),
                                _orthogonalize: bool = True,
                                ) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray, float]:
    """Discrete Yee-grid TE_mn mode profile.

    Solves (∇_t² + kc²) Hx = 0 on the cell-centred transverse grid with
    Neumann BC (PEC walls, ∂Hx/∂n = 0). Transverse E is obtained by
    cell-centred FD of the discrete Hx, giving profiles that are
    eigenvectors of the *exact* Yee finite-difference operator — the
    residual directional leakage from analytic-vs-discrete mode mismatch
    vanishes by construction.

    Returns `(ey, ez, hy, hz, kc_num)` matching `_te_mode_profiles`'s
    shape, amplitude-normalized so that `∫(ey²+ez²) dA = 1`.
    """
    nu = len(u_widths)
    nv = len(v_widths)
    Ku, mu = _galerkin_stiffness_mass_1d(u_widths, bc="neumann")
    Kv, mv = _galerkin_stiffness_mass_1d(v_widths, bc="neumann")

    dA = _aperture_area(u_widths, v_widths, aperture_dA)
    # Keep the physical Yee eigenfunctions/cutoffs on the full cell-centred
    # operator, then normalize and orthogonalize the stored profiles under the
    # extractor aperture. A reduced zero-dA subspace imposes an unintended
    # Dirichlet-like wall at the dropped row/column and breaks source matching.
    evals, evecs = _galerkin_eigh_separable_laplacian_2d(Ku, mu, Kv, mv)

    # Pick eigenvector by overlap with analytic Hx = cos(mπy/a)·cos(nπz/b).
    # Rank picking fails when discrete kc² ordering swaps nearly-degenerate
    # analytic modes (e.g. TE30 ↔ TE21 on WR-90).
    u_c = np.cumsum(u_widths) - 0.5 * np.asarray(u_widths)
    v_c = np.cumsum(v_widths) - 0.5 * np.asarray(v_widths)
    hx_ana = (np.cos(m * np.pi * u_c / a)[:, None]
              * np.cos(n * np.pi * v_c / b)[None, :])
    if aperture_dA is not None:
        target_kc2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
        rank = _pick_eigenmode_by_target_then_overlap(
            evecs, evals, hx_ana.ravel(), target_kc2,
        )
        kc2 = float(max(evals[rank], 0.0))
    else:
        rank = _pick_eigenmode_by_overlap(
            evecs, evals, hx_ana.ravel(),
        )
        kc2 = float(max(evals[rank], 0.0))
    hx = evecs[:, rank].reshape(nu, nv)

    # Align sign (eigh returns arbitrary sign).
    if float(np.sum(hx * hx_ana * dA)) < 0.0:
        hx = -hx

    dHxdu = _cell_centred_gradient(hx, u_widths, axis=0, bc="neumann")
    dHxdv = _cell_centred_gradient(hx, v_widths, axis=1, bc="neumann")

    # Match _te_mode_profiles sign convention for Hx = cos(mπy/a)·cos(nπz/b):
    #   Analytic ey = -(nπ/b) cos(mπy/a) sin(nπz/b).
    #   Since ∂Hx/∂v = -(nπ/b) cos(mπy/a) sin(nπz/b), we have ey = +∂Hx/∂v.
    #   Analytic ez = +(mπ/a) sin(mπy/a) cos(nπz/b) = -∂Hx/∂u.
    ey = dHxdv
    ez = -dHxdu
    # For forward +x propagation and Poynting x̂·P = Ey·Hz − Ez·Hy > 0:
    hy = -_shift_profile_to_dual(ez, h_offset)
    hz = _shift_profile_to_dual(ey, h_offset)

    if aperture_dA is not None:
        dropped = dA <= 0.0
        ey = np.where(dropped, 0.0, ey)
        ez = np.where(dropped, 0.0, ez)
        hy = np.where(dropped, 0.0, hy)
        hz = np.where(dropped, 0.0, hz)
    power = float(np.sum((ey ** 2 + ez ** 2) * dA))
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm
        hy, hz = _scale_h_to_unit_cross(ey, ez, hy, hz, dA)
    if aperture_dA is not None and _orthogonalize:
        previous: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        if n == 0 and m > 1:
            lower_modes = [(k, 0) for k in range(1, m)]
        elif m == 0 and n > 1:
            lower_modes = [(0, k) for k in range(1, n)]
        else:
            lower_modes = []
        for lm, ln in lower_modes:
            ley, lez, lhy, lhz, _ = _discrete_te_mode_profiles(
                a, b, lm, ln, u_widths, v_widths,
                aperture_dA=aperture_dA,
                h_offset=h_offset,
                _orthogonalize=True,
            )
            previous.append((ley, lez, lhy, lhz))
        if previous:
            ey, ez, hy, hz = _orthonormalize_profile_arrays(
                ey, ez, hy, hz, dA, previous,
            )
            if aperture_dA is not None:
                dropped = dA <= 0.0
                ey = np.where(dropped, 0.0, ey)
                ez = np.where(dropped, 0.0, ez)
                hy = np.where(dropped, 0.0, hy)
                hz = np.where(dropped, 0.0, hz)
    return ey, ez, hy, hz, float(np.sqrt(kc2))


def _discrete_tm_mode_profiles(a: float, b: float, m: int, n: int,
                                u_widths: np.ndarray,
                                v_widths: np.ndarray,
                                *,
                                aperture_dA: np.ndarray | None = None,
                                h_offset: tuple[float, float] = (0.0, 0.0),
                                ) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray, float]:
    """Discrete Yee-grid TM_mn mode profile.

    Solves (∇_t² + kc²) Ex = 0 on the cell-centred transverse grid with
    Dirichlet BC (PEC walls, Ex = 0). TM requires m,n ≥ 1.
    """
    if m < 1 or n < 1:
        raise ValueError(f"TM modes require m >= 1 and n >= 1, got ({m}, {n})")
    nu = len(u_widths)
    nv = len(v_widths)
    Ku, mu = _galerkin_stiffness_mass_1d(u_widths, bc="dirichlet")
    Kv, mv = _galerkin_stiffness_mass_1d(v_widths, bc="dirichlet")

    dA = _aperture_area(u_widths, v_widths, aperture_dA)
    evals, evecs = _galerkin_eigh_separable_laplacian_2d(Ku, mu, Kv, mv)

    # Overlap-pick eigenvector against analytic Ex = sin(mπy/a)·sin(nπz/b).
    u_c = np.cumsum(u_widths) - 0.5 * np.asarray(u_widths)
    v_c = np.cumsum(v_widths) - 0.5 * np.asarray(v_widths)
    ex_ana = (np.sin(m * np.pi * u_c / a)[:, None]
              * np.sin(n * np.pi * v_c / b)[None, :])
    if aperture_dA is not None:
        target_kc2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
        rank = _pick_eigenmode_by_target_then_overlap(
            evecs, evals, ex_ana.ravel(), target_kc2,
        )
        kc2 = float(max(evals[rank], 0.0))
    else:
        rank = _pick_eigenmode_by_overlap(
            evecs, evals, ex_ana.ravel(),
        )
        kc2 = float(max(evals[rank], 0.0))
    ex = evecs[:, rank].reshape(nu, nv)

    if float(np.sum(ex * ex_ana * dA)) < 0.0:
        ex = -ex

    dExdu = _cell_centred_gradient(ex, u_widths, axis=0, bc="dirichlet")
    dExdv = _cell_centred_gradient(ex, v_widths, axis=1, bc="dirichlet")

    # Analytic convention: ey = (mπ/a)·cos(mπy/a)·sin(nπz/b) = ∂Ex/∂u
    #                     ez = (nπ/b)·sin(mπy/a)·cos(nπz/b) = ∂Ex/∂v
    ey = dExdu
    ez = dExdv
    hy = -_shift_profile_to_dual(ez, h_offset)
    hz = _shift_profile_to_dual(ey, h_offset)

    if aperture_dA is not None:
        dropped = dA <= 0.0
        ey = np.where(dropped, 0.0, ey)
        ez = np.where(dropped, 0.0, ez)
        hy = np.where(dropped, 0.0, hy)
        hz = np.where(dropped, 0.0, hz)
    power = float(np.sum((ey ** 2 + ez ** 2) * dA))
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm
        hy, hz = _scale_h_to_unit_cross(ey, ez, hy, hz, dA)
    return ey, ez, hy, hz, float(np.sqrt(kc2))


def _node_span_to_cell_span(span: tuple[int, int]) -> tuple[int, int]:
    """NODE-index span -> the CELL span it encloses.

    ``_range_to_slice`` / ``_range_to_slice_nu`` return NODE indices: they
    report the physical aperture as ``(hi - lo - 1) * d``, so the ``hi - lo``
    nodes bracket ``hi - lo - 1`` cells (cell k lies between node k and node
    k+1). ``WaveguidePort.u_slice`` / ``.v_slice`` are FIELD-ARRAY slices —
    ``hi - lo`` is the number of aperture cells the transverse mode operator
    is built on — so the span has to be narrowed by one before it gets there.

    Passing the node span straight through solved the port's Neumann/Dirichlet
    eigenproblem on a guide ONE CELL WIDER than the walls make, which put
    ``WaveguidePortConfig.f_cutoff`` — and with it every ``_compute_beta``
    reference-plane rotation and every ``_compute_mode_impedance`` Z_TE — on
    the wrong guide (issue #868): 5.877 instead of 6.524 GHz on WR-90 at
    dx = a/9, 6.378 instead of 6.555 GHz at dx = a/36. Corrected, the port's
    cutoff is the discrete TE10 cutoff of the N-cell guide,
    ``(2/dx)·sin(π/(2N))·c/2π``, which is what the guide propagates with.

    The invariant it restores, and the cheapest check that it holds:
    ``sum(u_widths) == a`` (``_range_to_slice``'s own reported span).
    """
    lo, hi = int(span[0]), int(span[1])
    if hi - lo < 2:
        raise ValueError(
            f"transverse aperture node span {span!r} encloses no guide cell")
    return (lo, hi - 1)


def init_waveguide_port(
    port: WaveguidePort,
    dx,
    freqs: jnp.ndarray,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    probe_offset: int = 10,
    ref_offset: int = 3,
    dft_total_steps: int = 0,
    dt: float = 0.0,
    waveform: str = "modulated_gaussian",
    mode_profile: str = "discrete",
    grid=None,
    h_offset: tuple[float, float] = (0.5, 0.5), field_dtype=jnp.float32,
) -> WaveguidePortConfig:
    """Initialize a waveguide port with precomputed mode profiles.

    Parameters
    ----------
    dx : float or grid
        Either a scalar Yee cell size (uniform grid path) or a Grid-like
        object exposing per-axis cell-width arrays. NonUniformGrid is
        supported via duck-typing on ``dx_arr``/``dy_arr``/``dz``.
    probe_offset : int
        Cells downstream from source for measurement probe.
    ref_offset : int
        Cells downstream from source for reference probe.
    grid : Grid or None
        Optional Grid object for face-aware aperture pruning. When passed,
        the modal V/I aperture excludes "ghost" cells whose centre lies
        inside a PEC conductor (the cell at the array boundary on a +face
        marked PEC). Without this, uniform-Grid callers integrate over
        the ghost row, producing an ~8% PEC-short |S11| deficit.
    """
    # Explicit grid supplies uniform boundary detection.
    grid_obj = None
    if hasattr(dx, "dx_arr") and hasattr(dx, "dz"):
        grid_obj = dx
        dx = float(grid_obj.dx)
    boundary_grid = grid_obj if grid_obj is not None else grid
    m, n = port.mode
    normal_axis = port.normal_axis
    if normal_axis not in ("x", "y", "z"):
        raise ValueError(f"normal_axis must be 'x', 'y', or 'z', got {normal_axis!r}")
    if port.direction not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError(f"direction must be one of '+x', '-x', '+y', '-y', '+z', '-z', got {port.direction!r}")
    if port.direction[1] != normal_axis:
        raise ValueError(
            f"direction {port.direction!r} is inconsistent with normal_axis {normal_axis!r}"
        )

    if port.u_slice is not None and port.v_slice is not None:
        u_lo, u_hi = port.u_slice
        v_lo, v_hi = port.v_slice
    elif normal_axis == "x":
        if port.y_slice is None or port.z_slice is None:
            raise ValueError("x-normal legacy ports require y_slice and z_slice")
        u_lo, u_hi = port.y_slice
        v_lo, v_hi = port.z_slice
    else:
        raise ValueError(
            "non-x-normal ports require u_slice and v_slice explicit transverse aperture slices"
        )

    nu_port = u_hi - u_lo
    nv_port = v_hi - v_lo

    # --- Per-axis cell widths covering the transverse aperture ---
    # The mapping (u,v) → (x,y,z) depends on normal_axis and is the same
    # one used by `_plane_indexer`.
    # (u, v) → (axis name, grid size along that axis) mapping for the
    # transverse-aperture face-PEC heuristic below. Filled from the grid
    # object when one is available (NU duck-typed grid_obj, or explicit
    # uniform Grid via the `grid=` kwarg).
    u_axis_name = v_axis_name = None
    u_grid_size = v_grid_size = -1
    if grid_obj is not None:
        dx_arr_np = np.asarray(grid_obj.dx_arr)
        dy_arr_np = np.asarray(grid_obj.dy_arr)
        dz_arr_np = np.asarray(grid_obj.dz)
        if normal_axis == "x":
            u_widths_np = dy_arr_np[u_lo:u_hi]
            v_widths_np = dz_arr_np[v_lo:v_hi]
            u_axis_name, v_axis_name = "y", "z"
            u_grid_size, v_grid_size = dy_arr_np.shape[0], dz_arr_np.shape[0]
        elif normal_axis == "y":
            u_widths_np = dx_arr_np[u_lo:u_hi]
            v_widths_np = dz_arr_np[v_lo:v_hi]
            u_axis_name, v_axis_name = "x", "z"
            u_grid_size, v_grid_size = dx_arr_np.shape[0], dz_arr_np.shape[0]
        else:  # z-normal
            u_widths_np = dx_arr_np[u_lo:u_hi]
            v_widths_np = dy_arr_np[u_lo:u_hi] if False else dy_arr_np[v_lo:v_hi]
            u_axis_name, v_axis_name = "x", "y"
            u_grid_size, v_grid_size = dx_arr_np.shape[0], dy_arr_np.shape[0]
    else:
        u_widths_np = np.full(nu_port, float(dx))
        v_widths_np = np.full(nv_port, float(dx))
        if grid is not None:
            # Uniform Grid: cell widths are scalar dx but boundary
            # information is still on the Grid object.
            if normal_axis == "x":
                u_axis_name, v_axis_name = "y", "z"
                u_grid_size, v_grid_size = grid.ny, grid.nz
            elif normal_axis == "y":
                u_axis_name, v_axis_name = "x", "z"
                u_grid_size, v_grid_size = grid.nx, grid.nz
            else:
                u_axis_name, v_axis_name = "x", "y"
                u_grid_size, v_grid_size = grid.nx, grid.ny

    # Cell-centre coordinates (cumulative-sum midpoints). For uniform
    # widths this collapses to the original `np.linspace(0.5*dx, ...)`.
    u_coords = np.cumsum(u_widths_np) - 0.5 * u_widths_np
    v_coords = np.cumsum(v_widths_np) - 0.5 * v_widths_np

    # ----- Aperture dA with face-aware boundary-cell weighting -----
    # Why: when a transverse face is PEC and the port slice reaches the array
    # boundary on that side, the last Yee cell at index Nu-1 (or Nv-1) sits
    # at-or-past the conductor surface. Including it with full weight in the
    # modal V/I integral lets a spurious Ez (which apply_pec_faces does NOT
    # zero — Ez is "normal" at z_hi by staircase convention) contaminate the
    # extraction. Empirically this drops PEC-short WR-90 |S11| from 1.0 to
    # 0.91 (see scripts/_aperture_trim_test.py and ../_aperture_trapezoidal_test.py).
    #
    # Convention: DROP PEC +face cells (weight=0.0). This matches the
    # OpenEMS convention — its mode-function probe integration box is
    # specified to the physical aperture, naturally excluding ghost
    # cells that sit past the conductor (see /usr/lib/python3/dist-packages
    # /openEMS/ports.py:347-360 for the WaveguidePort probe construction).
    # Empirically: weight 0.5 (PROD half-weight kludge, used 2026-04-23..27)
    # gives PEC-short min |S11| ≈ 0.94, weight 0.0 (DROP, OpenEMS-equivalent)
    # gives min |S11| = 0.9997. The half-weight was a 2026-04-26 staging
    # compromise (Phase 2 dead-end) that protected mesh-conv and asymmetric-obstacle
    # reciprocity at the cost of capping PEC-short closure.
    #
    # 2026-04-29 update: test_mesh_convergence_s21_scaled_cpml in
    # tests/oracle/test_waveguide_port_validation_battery.py is a LIVE, currently
    # green committed gate under DROP, not xfail. The 2026-04-22..04-29 jitter
    # that briefly locked it as xfail was root-caused to Box.mask_on_coords
    # closed-closed semantics (issue #75), not to the DROP weight; see the
    # block comment directly above that test ("Test 6 — Mesh convergence of
    # |S21| ..."). The receive-side multi-mode extractor (per the
    # dead-end note's recommendation) remains the path to recovering
    # PEC-short closure without capping it via the half-weight kludge.
    #
    # The hi-face drop (``hi_face_is_wall``) does NOT fire for a port built by ``Simulation``/``run_nonuniform``: those
    # hand ``WaveguidePort`` the guide's CELL span (issue #868), which ends one
    # index short of the array edge, so there is no cell past the wall left to
    # drop. It still fires for a low-level caller that passes the whole node
    # plane (``y_slice=(0, grid.ny)``), which is what it was written for.
    u_aperture_weights = np.ones_like(u_widths_np)
    v_aperture_weights = np.ones_like(v_widths_np)
    if boundary_grid is not None and u_axis_name is not None:
        # Stage 1 conformal PEC: when a +face is conformal, the
        # Dey-Mittra eps_correction handles the staircase shift at the
        # boundary cell. Zeroing the same cell here in the modal V/I
        # integral would be a second copy of that correction — the
        # 2026-04-29 first-attempt failure mode where PEC-short |S11|
        # diff vs OpenEMS regressed 0.025 → 0.094.
        conformal_faces = getattr(boundary_grid, "conformal_faces", set()) or set()
        u_hi_face_pec = hi_face_is_wall(boundary_grid, u_axis_name)
        v_hi_face_pec = hi_face_is_wall(boundary_grid, v_axis_name)
        u_hi_face_conformal = f"{u_axis_name}_hi" in conformal_faces
        v_hi_face_conformal = f"{v_axis_name}_hi" in conformal_faces
        if (u_hi == u_grid_size and u_hi_face_pec
                and not u_hi_face_conformal and u_widths_np.size > 0):
            u_aperture_weights[-1] = 0.0
        if (v_hi == v_grid_size and v_hi_face_pec
                and not v_hi_face_conformal and v_widths_np.size > 0):
            v_aperture_weights[-1] = 0.0
    aperture_dA_np = (
        (u_widths_np * u_aperture_weights)[:, None]
        * (v_widths_np * v_aperture_weights)[None, :]
    )

    if mode_profile not in ("analytic", "discrete"):
        raise ValueError(
            "mode_profile must be 'analytic' or 'discrete', "
            f"got {mode_profile!r}"
        )
    if mode_profile == "discrete":
        if port.mode_type == "TE":
            ey, ez, hy, hz, kc_num = _discrete_te_mode_profiles(
                port.a, port.b, m, n, u_widths_np, v_widths_np,
                aperture_dA=aperture_dA_np,
                h_offset=h_offset,
            )
        else:
            ey, ez, hy, hz, kc_num = _discrete_tm_mode_profiles(
                port.a, port.b, m, n, u_widths_np, v_widths_np,
                aperture_dA=aperture_dA_np,
                h_offset=h_offset,
            )
        # Use the discrete eigenvalue's kc for f_c so h_inc_table's
        # β(ω) = √(k² − kc_num²) matches the Yee-grid mode exactly.
        f_c = kc_num * C0_LOCAL / (2.0 * np.pi)
    else:
        if port.mode_type == "TE":
            ey, ez, hy, hz = _te_mode_profiles(
                port.a, port.b, m, n, u_coords, v_coords,
                u_widths=u_widths_np, v_widths=v_widths_np,
                aperture_dA=aperture_dA_np,
                h_offset=h_offset,
            )
        else:
            ey, ez, hy, hz = _tm_mode_profiles(
                port.a, port.b, m, n, u_coords, v_coords,
                u_widths=u_widths_np, v_widths=v_widths_np,
                aperture_dA=aperture_dA_np,
                h_offset=h_offset,
            )
        f_c = cutoff_frequency(port.a, port.b, m, n)

    if waveform != "modulated_gaussian":
        raise ValueError(
            waveguide_waveform_error(waveform)
        )
    if waveform == "modulated_gaussian":
        # Meep gaussian_src_time convention: width = 1/fwidth,
        # peak_time = cutoff = 5·width. In rfx's exp(-(tt/τ)²) form this
        # is τ = √2·width = √2/fwidth (so exp(-tt²/τ²) matches Meep's
        # exp(-tt²/(2·width²))), and t0 = 5·width = 5/fwidth. Using the
        # π·fwidth convention would make the pulse π× narrower in time
        # (π× broader in spectrum) — the spectrum then straddles the
        # TE-mode cutoff and drives excess directional leakage.
        fwidth = f0 * bandwidth
        width = 1.0 / fwidth
        tau = np.sqrt(2.0) * width
        t0 = 5.0 * width
    else:
        tau = 1.0 / (f0 * bandwidth * np.pi)
        t0 = 3.0 * tau

    step_sign = 1 if port.direction.startswith("+") else -1
    ref_x = port.x_index + step_sign * ref_offset
    probe_x = port.x_index + step_sign * probe_offset
    source_x_m = float(port.x_position) if port.x_position is not None else float(port.x_index * dx)
    # For NU grids, integrate the actual per-cell spacing along the port-normal
    # axis from the source plane to the reference / probe planes. For the
    # uniform path this reduces to offset * dx bit-identically.
    if grid_obj is not None:
        if normal_axis == "x":
            d_axis = np.asarray(grid_obj.dx_arr)
        elif normal_axis == "y":
            d_axis = np.asarray(grid_obj.dy_arr)
        else:
            d_axis = np.asarray(grid_obj.dz)

        def _cum_offset(from_idx: int, to_idx: int) -> float:
            if to_idx == from_idx:
                return 0.0
            lo = min(from_idx, to_idx)
            hi = max(from_idx, to_idx)
            return float(np.sum(d_axis[lo:hi])) * (1.0 if to_idx > from_idx else -1.0)

        reference_x_m = source_x_m + _cum_offset(port.x_index, ref_x)
        probe_x_m = source_x_m + _cum_offset(port.x_index, probe_x)
    else:
        reference_x_m = source_x_m + step_sign * ref_offset * dx
        probe_x_m = source_x_m + step_sign * probe_offset * dx

    if normal_axis == "x":
        e_u_component, e_v_component = "ey", "ez"
        h_u_component, h_v_component = "hy", "hz"
        y_lo, y_hi = u_lo, u_hi
        z_lo, z_hi = v_lo, v_hi
    elif normal_axis == "y":
        # The tangential plane is indexed in physical (x, z) order, which is
        # left-handed with respect to +y.  Flip the stored H profiles so the
        # modal-current inner product still measures +y-directed power.
        ey = ey.copy()
        ez = ez.copy()
        hy = -hy
        hz = -hz
        e_u_component, e_v_component = "ex", "ez"
        h_u_component, h_v_component = "hx", "hz"
        y_lo, y_hi = 0, 0
        z_lo, z_hi = 0, 0
    else:
        e_u_component, e_v_component = "ex", "ey"
        h_u_component, h_v_component = "hx", "hy"
        y_lo, y_hi = 0, 0
        z_lo, z_hi = 0, 0

    # Time-series carry. Shape must be static at trace time. When the
    # caller did not supply ``dft_total_steps`` (legacy low-level path),
    # use length 1 so the carry has a valid shape; ``update_*_probe``
    # then writes only into slot 0 which is harmless for those callers
    # who never use the post-scan extractor.
    n_t = max(int(dft_total_steps), 1)
    # Match cfg.freqs precision so the post-scan rect-DFT (which builds a
    # complex phase from cfg.freqs) has consistent real-side dtype.
    real_dtype = jnp.zeros((), dtype=freqs.dtype).dtype if hasattr(freqs, 'dtype') else jnp.float32
    zeros_t = jnp.zeros((n_t,), dtype=real_dtype)

    # Precompute companion H waveform for TFSF-style E correction.
    # For a forward-only wave whose E-amplitude at x_src matches the source
    # pulse, we need h_inc(t + dt/2) at (x_src − 0.5·dx). Computed in the
    # frequency domain: h_spec(ω) = e_spec(ω) · e^{−jβ(ω)·dx/2 − jω·dt/2} /
    # Z_mode(ω), filtered to propagating frequencies |ω| ≥ ω_c.
    n_steps_aux = int(dft_total_steps) if dft_total_steps > 0 else 0
    if n_steps_aux > 1 and dt > 0.0:
        t_arr = np.arange(n_steps_aux, dtype=np.float64) * float(dt)
        arg_arr = (t_arr - t0) / tau
        # Compute tables with unit amplitude; ``apply_waveguide_port_h/e``
        # scale by ``cfg.src_amp`` at runtime so ``_reset_cfg(drive=False)``
        # zeros both corrections together.
        if waveform == "modulated_gaussian":
            # Meep-style carrier-centered shape. Its low-frequency content is
            # negligible for this configured source, so no sub-cutoff filter is
            # applied; this eliminates the raised-cosine filter tail
            # that drives directional leakage in the differentiated-Gaussian
            # path. The hard ±5·width cutoff (= ±t0) matches Meep's
            # gaussian_src_time::dipole and kills any residual early/late
            # envelope numerics.
            tt = t_arr - t0
            e_inc_raw = np.cos(2.0 * np.pi * f0 * tt) * np.exp(-(arg_arr ** 2))
            e_inc_raw = np.where(np.abs(tt) > t0, 0.0, e_inc_raw)
        else:
            e_inc_raw = (-2.0 * arg_arr) * np.exp(-(arg_arr ** 2))
        e_spec_raw = np.fft.rfft(e_inc_raw)
        freqs_spec = np.fft.rfftfreq(n_steps_aux, d=float(dt))
        omega = 2.0 * np.pi * freqs_spec
        kc = 2.0 * np.pi * f_c / C0_LOCAL
        k_arr = omega / C0_LOCAL
        propagating = omega > (2.0 * np.pi * f_c)
        if mode_profile == "discrete":
            # Invert the 3D Yee dispersion relation along the port-normal axis
            # for the given transverse kc_num:
            #   (sin(ω·dt/2)/(c·dt/2))² = (sin(β·dx/2)/(dx/2))² + kc_num².
            # Using the discrete β in the half-cell phase shift eliminates the
            # O((β·dx)²) continuous-vs-Yee dispersion residue in h_inc_table.
            s_t = np.sin(omega * 0.5 * float(dt)) / (0.5 * float(dt))
            s_x_sq = np.maximum(
                (s_t / C0_LOCAL) ** 2 - kc ** 2, 0.0,
            ) * (0.5 * float(dx)) ** 2
            s_x = np.sqrt(np.minimum(s_x_sq, 1.0))
            beta_arr = np.where(propagating,
                                (2.0 / float(dx)) * np.arcsin(s_x), 0.0)
        else:
            beta_arr = np.where(
                propagating,
                np.sqrt(np.maximum(k_arr ** 2 - kc ** 2, 0.0)),
                0.0,
            )
        if mode_profile == "discrete":
            # Yee-discrete modal impedance derived from the 3D Yee Faraday
            # update applied to a plane wave exp(j(ωt − β·x)):
            #   H₀ · sin(ω·dt/2) = (dt/(μ·dx)) · E₀ · sin(β·dx/2)
            # → Z_TE_disc = μ·dx·sin(ω·dt/2) / (dt·sin(β·dx/2)).
            # In the continuous limit this collapses to ω·μ/β. Using the
            # analytic continuous form against the discrete β leaves an
            # O((β·dx)²) broadband impedance mismatch — the dominant
            # backward-emission driver once the mode profile and β are
            # already discrete-correct.
            s_w = np.sin(omega * 0.5 * float(dt))
            s_b = np.sin(beta_arr * 0.5 * float(dx))
            if port.mode_type == "TE":
                z_mode_arr = np.where(
                    propagating,
                    MU_0 * float(dx) * s_w
                    / np.maximum(float(dt) * s_b, 1e-30),
                    np.inf,
                )
            else:  # TM
                z_mode_arr = np.where(
                    propagating,
                    float(dt) * s_b
                    / np.maximum(EPS_0 * float(dx) * s_w, 1e-30),
                    0.0,
                )
        elif port.mode_type == "TE":
            z_mode_arr = np.where(
                propagating,
                omega * MU_0 / np.maximum(beta_arr, 1e-30),
                np.inf,
            )
        else:  # TM
            z_mode_arr = np.where(
                propagating,
                beta_arr / np.maximum(omega * EPS_0, 1e-30),
                0.0,
            )
        # Bandpass-filter the source to propagating-mode content only.
        # The raw differentiated-Gaussian spectrum spans DC → ~3·f0; in a
        # guided structure, sub-cutoff content cannot propagate and just
        # excites evanescent near-field if injected. A smooth raised-
        # cosine taper across [f_c, 1.2·f_c] avoids Gibbs ringing that a
        # hard cutoff would leave in e_inc_time.
        #
        # The modulated-Gaussian pulse has effectively zero DC content and
        # its spectrum is naturally bandpassed around f₀, so the taper is
        # skipped — the raised-cosine tail is the dominant directional
        # leakage driver for the differentiated source.
        if waveform == "modulated_gaussian":
            e_spec_filt = e_spec_raw
        else:
            # Unreachable for waveguide ports since the #1942 refusal; kept for the causal table (#1942).
            f_c_arr = np.full_like(freqs_spec, f_c)
            f_taper_hi = 1.2 * f_c
            taper_ratio = np.clip(
                (freqs_spec - f_c_arr) / (f_taper_hi - f_c), 0.0, 1.0,
            )
            filt_weight = 0.5 * (1.0 - np.cos(np.pi * taper_ratio))
            e_spec_filt = e_spec_raw * filt_weight
        e_inc_time = np.fft.irfft(e_spec_filt, n_steps_aux)
        # Forward wave in rfx's engineering convention exp(j(ωt − βx)):
        # shifting from (x_src, t) to (x_src − dx/2, t + dt/2) multiplies
        # the spectrum by exp(+jβ·dx/2 + jω·dt/2).
        phase_shift = np.exp(+1j * beta_arr * 0.5 * float(dx)
                             + 1j * omega * 0.5 * float(dt))
        h_spec = np.where(
            propagating,
            e_spec_filt / z_mode_arr * phase_shift,
            0.0,
        )
        h_inc_time = np.fft.irfft(h_spec, n_steps_aux)
        e_inc_table = jnp.asarray(e_inc_time, dtype=field_dtype)
        h_inc_table = jnp.asarray(h_inc_time, dtype=field_dtype)
    else:
        # Tables unavailable (dt=0 at init). Size-1 sentinel triggers the
        # legacy soft-E source fallback inside apply_waveguide_port_e.
        e_inc_table = jnp.zeros((1,), dtype=field_dtype)
        h_inc_table = jnp.zeros((1,), dtype=field_dtype)

    return WaveguidePortConfig(
        x_index=port.x_index,
        ref_x=ref_x,
        probe_x=probe_x,
        y_lo=y_lo, y_hi=y_hi,
        z_lo=z_lo, z_hi=z_hi,
        normal_axis=normal_axis,
        u_lo=u_lo,
        u_hi=u_hi,
        v_lo=v_lo,
        v_hi=v_hi,
        e_u_component=e_u_component,
        e_v_component=e_v_component,
        h_u_component=h_u_component,
        h_v_component=h_v_component,
        ey_profile=jnp.array(ey, dtype=field_dtype),
        ez_profile=jnp.array(ez, dtype=field_dtype),
        hy_profile=jnp.array(hy, dtype=field_dtype),
        hz_profile=jnp.array(hz, dtype=field_dtype),
        mode_type=port.mode_type,
        direction=port.direction,
        f_cutoff=float(f_c),
        a=port.a, b=port.b,
        dx=float(dx),
        dt=float(dt),
        u_widths=jnp.asarray(u_widths_np, dtype=field_dtype),
        v_widths=jnp.asarray(v_widths_np, dtype=field_dtype),
        source_x_m=float(source_x_m),
        reference_x_m=float(reference_x_m),
        probe_x_m=float(probe_x_m),
        dft_total_steps=int(dft_total_steps),
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        src_fcen=float(f0),
        waveform=str(waveform),
        freqs=freqs,
        v_probe_t=zeros_t,
        v_ref_t=zeros_t,
        i_probe_t=zeros_t,
        i_ref_t=zeros_t,
        v_inc_t=zeros_t,
        n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        e_inc_table=e_inc_table,
        h_inc_table=h_inc_table,
        aperture_dA=jnp.asarray(aperture_dA_np, dtype=field_dtype),
        h_offset=(float(h_offset[0]), float(h_offset[1])),
        mode_indices=(int(m), int(n)),
        mode_profile=str(mode_profile),
    )


def _aperture_dA(cfg: WaveguidePortConfig) -> jnp.ndarray:
    """Per-cell area element (nu, nv) on the port aperture.

    Returns the precomputed `cfg.aperture_dA` when populated by
    `init_waveguide_port` (which applies face-aware ghost-cell pruning at
    PEC boundaries). Falls back to the legacy `u_widths × v_widths`
    product when called on a cfg built without the precomputed area
    (low-level test fixtures predating this field).
    """
    if cfg.aperture_dA.shape[0] > 0 and cfg.aperture_dA.shape[1] > 0:
        return cfg.aperture_dA
    return cfg.u_widths[:, None] * cfg.v_widths[None, :]


def _compute_beta(
    freqs: jnp.ndarray,
    f_cutoff: float,
    *,
    dt: float = 0.0,
    dx: float = 0.0,
) -> jnp.ndarray:
    """Guided propagation constant β(f) for a vacuum-filled rectangular guide.

    With ``dt > 0`` and ``dx > 0`` the **Yee-discrete** dispersion relation
    is used::

        (sin(ω·dt/2) / (c·dt/2))² = (sin(β·dx/2) / (dx/2))² + kc²

    This matches the numerical phase velocity the FDTD simulation actually
    propagates — critical when shifting modal waves across a plane by
    ``exp(∓jβ·Δz)``. With ``dt == 0`` or ``dx == 0`` the legacy analytic
    continuous-medium form ``β = sqrt(k² − kc²)`` is returned for
    backwards compatibility. Evanescent (sub-cutoff) branch falls back to
    analytic imaginary β in both modes — those waves are not propagated
    by ``_shift_modal_waves`` and their accuracy does not affect S-params.
    """
    omega = 2 * jnp.pi * freqs
    kc = 2 * jnp.pi * f_cutoff / C0_LOCAL

    if dt > 0.0 and dx > 0.0:
        # Yee 3D dispersion with transverse cutoff kc (2D-FD eigenvalue).
        # Temporal term: sin(ωdt/2) / (c·dt/2).
        s_t_over_c = jnp.sin(omega * 0.5 * dt) / (C0_LOCAL * 0.5 * dt)
        s_x_sq = s_t_over_c ** 2 - kc ** 2
        # Propagating branch: clip arcsin argument to (-1, 1) to protect
        # against grid-dispersion-induced s_x·dx/2 > 1 numerical noise.
        arg = jnp.clip(0.5 * dx * jnp.sqrt(jnp.maximum(s_x_sq, 0.0)),
                       -1.0 + 1e-12, 1.0 - 1e-12)
        beta_prop = (2.0 / dx) * jnp.arcsin(arg)
        # Evanescent branch: β = j·sqrt(-s_x_sq), units m⁻¹ (s_x_sq is m⁻²). The prior
        # form multiplied by C0_LOCAL, giving s⁻¹ (~c× too large) and inconsistent with both
        # beta_prop above and the non-dispersive fallback below — a spurious factor of c
        # (RF-audit 2026-07-23, Codex pass). Below-cutoff bins are preflight-flagged invalid,
        # but a c×-wrong |β| in the plane-shift exp(-jβ·d) would blow up to inf/nan.
        beta_evan = 1j * jnp.sqrt(jnp.maximum(-s_x_sq, 0.0))
        return jnp.where(s_x_sq >= 0, beta_prop, beta_evan)

    k = omega / C0_LOCAL
    beta_sq = k**2 - kc**2
    return jnp.where(
        beta_sq >= 0,
        jnp.sqrt(jnp.maximum(beta_sq, 0.0)),
        1j * jnp.sqrt(jnp.maximum(-beta_sq, 0.0)),
    )


def _compute_mode_impedance(
    freqs: jnp.ndarray,
    f_cutoff: float,
    mode_type: str,
    *,
    dt: float = 0.0,
    dx: float = 0.0,
) -> jnp.ndarray:
    """Rectangular-waveguide modal impedance for TE/TM modes.

    TE: Z = ωμ / β
    TM: Z = β / (ωε)
    """
    omega = 2 * jnp.pi * freqs
    beta = _compute_beta(freqs, f_cutoff, dt=dt, dx=dx)
    safe_beta = jnp.where(jnp.abs(beta) > 1e-30, beta,
                          1e-30 * jnp.ones_like(beta))
    safe_omega = jnp.where(jnp.abs(omega) > 1e-30, omega,
                           1e-30 * jnp.ones_like(omega))
    if dt > 0.0 and dx > 0.0:
        propagating = freqs > f_cutoff
        s_w = jnp.sin(omega * 0.5 * dt)
        s_b = jnp.sin(beta * 0.5 * dx)
        # Evanescent sentinel: use a LARGE FINITE scalar (not jnp.inf).
        # inf × complex(i, 0) numerically yields nan in the imaginary component
        # on most NumPy/JAX implementations — see test_multimode cutoff cases.
        # A finite stand-in keeps downstream `(V ± Z·I)` arithmetic defined.
        te_evanescent = jnp.asarray(1e30, dtype=jnp.float32)
        tm_evanescent = jnp.asarray(0.0, dtype=jnp.float32)
        if mode_type == "TE":
            z_disc = MU_0 * dx * s_w / jnp.maximum(dt * s_b, 1e-30)
            return jnp.where(propagating, z_disc, te_evanescent)
        if mode_type == "TM":
            z_disc = dt * s_b / jnp.maximum(EPS_0 * dx * s_w, 1e-30)
            return jnp.where(propagating, z_disc, tm_evanescent)
    if mode_type == "TE":
        return omega * MU_0 / safe_beta
    if mode_type == "TM":
        return safe_beta / (safe_omega * EPS_0)
    raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")


def solve_rectangular_modes(
    a: float,
    b: float,
    freq_max: float,
    n_modes: int = 3,
) -> list[dict]:
    """Compute TE and TM modes for a rectangular waveguide.

    Enumerates analytical modes sorted by cutoff frequency and returns the
    lowest ``n_modes`` that have cutoff below ``freq_max``.

    Parameters
    ----------
    a, b : float
        Waveguide cross-section dimensions (metres). By convention ``a >= b``.
    freq_max : float
        Maximum frequency in Hz.  Only modes with cutoff below this value
        are included.
    n_modes : int
        Maximum number of modes to return.

    Returns
    -------
    list of dict
        Each dict has keys: ``mode_type`` (``"TE"`` or ``"TM"``), ``m``, ``n``,
        ``f_cutoff``.
    """
    candidates: list[dict] = []

    # Search range: at least 2*n_modes for each index to be safe
    max_idx = max(2 * n_modes, 6)
    for m in range(0, max_idx + 1):
        for n in range(0, max_idx + 1):
            if m == 0 and n == 0:
                continue
            fc = cutoff_frequency(a, b, m, n)
            if fc <= freq_max:
                candidates.append({
                    "mode_type": "TE", "m": m, "n": n, "f_cutoff": fc,
                })

    for m in range(1, max_idx + 1):
        for n in range(1, max_idx + 1):
            fc = cutoff_frequency(a, b, m, n)
            if fc <= freq_max:
                candidates.append({
                    "mode_type": "TM", "m": m, "n": n, "f_cutoff": fc,
                })

    candidates.sort(key=lambda x: x["f_cutoff"])
    return candidates[:n_modes]


def init_multimode_waveguide_port(
    port: WaveguidePort,
    dx: float,
    freqs: jnp.ndarray,
    n_modes: int = 1,
    *,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    probe_offset: int = 10,
    ref_offset: int = 3,
    dft_total_steps: int = 0,
    dt: float = 0.0,
    waveform: str = "modulated_gaussian",
    mode_profile: str = "discrete",
    grid=None,
    h_offset: tuple[float, float] = (0.5, 0.5), field_dtype=jnp.float32,
) -> list[WaveguidePortConfig]:
    """Initialize a multi-mode waveguide port.

    Expands a single physical port into ``n_modes`` independent
    ``WaveguidePortConfig`` objects, one per analytical eigenmode.
    Each config has the profile of a different waveguide mode, but shares
    the same physical location and aperture.

    Only the first mode (lowest cutoff, typically TE10) is driven with a
    source; higher-order modes are passive listeners that extract modal
    amplitudes via overlap integrals.

    Parameters
    ----------
    port : WaveguidePort
        Physical port definition (location, aperture, direction).
        The ``mode`` and ``mode_type`` fields are ignored when
        ``n_modes > 1``; modes are auto-enumerated by cutoff.
    dx : float
        Cell size (metres).
    freqs : jnp.ndarray
        Frequency array for DFT extraction.
    n_modes : int
        Number of lowest-cutoff modes to include.
    f0, bandwidth, amplitude, probe_offset, ref_offset : ...
        Passed to ``init_waveguide_port`` for each mode config.
    dft_total_steps : int
        Total simulation steps; sets the time-series record length used
        by the post-scan rectangular full-record DFT.

    Returns
    -------
    list[WaveguidePortConfig]
        One config per mode, sorted by cutoff frequency.
        Only the first config has a nonzero source amplitude.
    """
    if n_modes == 1:
        cfg = init_waveguide_port(
            port, dx, freqs, f0=f0, bandwidth=bandwidth,
            amplitude=amplitude, probe_offset=probe_offset,
            ref_offset=ref_offset, dft_total_steps=dft_total_steps,
            dt=dt, waveform=waveform, mode_profile=mode_profile,
            grid=grid, h_offset=h_offset, field_dtype=field_dtype,
        )
        return [cfg]

    # Enumerate modes analytically
    freq_max = float(jnp.max(freqs)) * 2.0  # generous upper bound
    mode_list = solve_rectangular_modes(port.a, port.b, freq_max, n_modes)
    if len(mode_list) == 0:
        raise ValueError(
            f"No waveguide modes found below freq_max={freq_max:.3e} Hz "
            f"for aperture a={port.a}, b={port.b}"
        )

    cfgs = []
    for mode_idx, mode_info in enumerate(mode_list):
        m, n = mode_info["m"], mode_info["n"]
        mode_type = mode_info["mode_type"]

        # Create a port variant for this mode
        mode_port = port._replace(
            mode=(m, n),
            mode_type=mode_type,
        )

        # Only the first (dominant) mode is actively driven
        mode_amplitude = amplitude if mode_idx == 0 else 0.0

        cfg = init_waveguide_port(
            mode_port, dx, freqs, f0=f0, bandwidth=bandwidth,
            amplitude=mode_amplitude, probe_offset=probe_offset,
            ref_offset=ref_offset, dft_total_steps=dft_total_steps,
            dt=dt, waveform=waveform, mode_profile=mode_profile,
            grid=grid, h_offset=h_offset, field_dtype=field_dtype,
        )
        cfgs.append(cfg)

    return _gram_schmidt_modes(cfgs)


def _gram_schmidt_modes(
    cfgs: list[WaveguidePortConfig],
) -> list[WaveguidePortConfig]:
    """Jointly orthonormalise stored modal profiles under shared dA."""
    if len(cfgs) <= 1:
        return cfgs
    dA = np.asarray(_aperture_dA(cfgs[0]), dtype=np.float64)
    out: list[WaveguidePortConfig] = []
    for cfg in cfgs:
        ey = np.asarray(cfg.ey_profile, dtype=np.float64).copy()
        ez = np.asarray(cfg.ez_profile, dtype=np.float64).copy()
        hy = np.asarray(cfg.hy_profile, dtype=np.float64).copy()
        hz = np.asarray(cfg.hz_profile, dtype=np.float64).copy()

        for prev in out:
            pey = np.asarray(prev.ey_profile, dtype=np.float64)
            pez = np.asarray(prev.ez_profile, dtype=np.float64)
            phy = np.asarray(prev.hy_profile, dtype=np.float64)
            phz = np.asarray(prev.hz_profile, dtype=np.float64)
            c_prev = float(np.sum((pey * phz - pez * phy) * dA))
            if abs(c_prev) <= 1e-30:
                continue
            proj = float(np.sum((ey * phz - ez * phy) * dA)) / c_prev
            ey -= proj * pey
            ez -= proj * pez
            hy -= proj * phy
            hz -= proj * phz

        c_self = float(np.sum((ey * hz - ez * hy) * dA))
        if abs(c_self) <= 1e-30:
            raise ValueError(
                "Cannot orthonormalize waveguide modes: zero self-overlap"
            )
        scale = np.sqrt(abs(c_self))
        out.append(cfg._replace(
            ey_profile=jnp.asarray(ey / scale, dtype=cfg.ey_profile.dtype),
            ez_profile=jnp.asarray(ez / scale, dtype=cfg.ez_profile.dtype),
            hy_profile=jnp.asarray(hy / scale, dtype=cfg.hy_profile.dtype),
            hz_profile=jnp.asarray(hz / scale, dtype=cfg.hz_profile.dtype),
        ))
    return out
