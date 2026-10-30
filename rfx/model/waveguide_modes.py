"""Rectangular waveguide geometry and analytical and discrete mode profiles."""

from __future__ import annotations

from rfx.model.graded_admission import host_cells

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


class PortMedium(NamedTuple):
    """Realized homogeneous, isotropic, lossless section medium."""
    eps_r: float = 1.0
    mu_r: float = 1.0


def port_light_speed(cfg) -> float:
    """Speed in a port config (or its medium during compilation)."""
    medium = cfg.medium if hasattr(cfg, "medium") else cfg
    return C0_LOCAL / np.sqrt(medium.eps_r * medium.mu_r)


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

    # Precomputed TFSF-style injection waveforms, built independently of
    # record length with the causal admittance, including below cutoff.
    # e_inc_table[n] is the E-amplitude scalar at (x_src, n·dt);
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
    segment_planes: tuple[float, ...] = ()
    port_name: str = ""
    medium: PortMedium = PortMedium()
    probe_medium_matches: bool = True
    probe_medium_description: str = ""
    multimode: bool = False


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
    medium: PortMedium = PortMedium(),
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
        dx_arr_np = host_cells(grid_obj, "x", store=True)
        dy_arr_np = host_cells(grid_obj, "y", store=True)
        dz_arr_np = host_cells(grid_obj, "z", store=True)
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
        f_c = kc_num * port_light_speed(medium) / (2.0 * np.pi)
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
        f_c = cutoff_frequency(port.a, port.b, m, n) / np.sqrt(medium.eps_r * medium.mu_r)

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
            d_axis = host_cells(grid_obj, "x", store=True)
        elif normal_axis == "y":
            d_axis = host_cells(grid_obj, "y", store=True)
        else:
            d_axis = host_cells(grid_obj, "z", store=True)

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

    e_inc_table, h_inc_table = _incident_tables(
        f_c=f_c, medium=medium, dt=dt, dx=dx, dft_total_steps=dft_total_steps,
        t0=t0, tau=tau, f0=f0, waveform=waveform, mode_profile=mode_profile,
        mode_type=port.mode_type, field_dtype=field_dtype)

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
        medium=medium,
    )


def _incident_table_build(*, length, f_c, medium, dt, dx, t0, tau,
                          f0, waveform, mode_profile, mode_type):
    """One periodic host build, including its negative-time (discarded) half."""
    tt = np.arange(length, dtype=np.float64) * float(dt) - t0
    arg = tt / tau
    if waveform == "modulated_gaussian":
        e_raw = np.cos(2.0 * np.pi * f0 * tt) * np.exp(-arg ** 2)
        e_raw = np.where(np.abs(tt) > t0, 0.0, e_raw)
    else:
        e_raw = -2.0 * arg * np.exp(-arg ** 2)
    e_spec = np.fft.rfft(e_raw)
    freqs = np.fft.rfftfreq(length, d=float(dt))
    if waveform != "modulated_gaussian":
        # Unreachable for waveguide ports since the #1942 refusal.
        ratio = np.clip((freqs - f_c) / (0.2 * f_c), 0.0, 1.0)
        e_spec *= 0.5 * (1.0 - np.cos(np.pi * ratio))
    kw = dict(dt=dt if mode_profile == "discrete" else 0.0, dx=dx, _table=True)
    beta = _compute_beta(freqs, f_c, medium=medium, **kw)
    impedance = _compute_mode_impedance(freqs, f_c, mode_type, medium=medium, **kw)
    if np.any(impedance == 0):
        # A TM bin exactly on cutoff has infinite admittance; e/Z would turn the whole table into NaN.
        raise ValueError(
            f"waveguide incident table: a bin of the length-{length} build falls exactly on the "
            f"{mode_type} cutoff (f0={f0}, f_c={f_c}), where the admittance is infinite; "
            "move f0 so the source spectrum at f_c is below about 1e-3 of its peak")
    # exp(j(omega*t - beta*x)): H is upstream by dx/2 and later by dt/2.
    # The evanescent continuation is essential to a causal response at cutoff.
    phase = np.exp(0.5j * beta * float(dx) + 1j * np.pi * freqs * float(dt))
    h_spec = e_spec / impedance * phase
    h_spec[0] = 0.0
    return np.fft.irfft(e_spec, length), np.fft.irfft(h_spec, length)


def _incident_tables(*, f_c, medium, dt, dx, dft_total_steps, t0, tau,
                     f0, waveform, mode_profile, mode_type, field_dtype):
    """Build a causal pulse independently of the requested recorder length."""
    n = int(dft_total_steps)
    if n <= 1 or dt <= 0.0:
        # Size-1 sentinel selects the legacy source when tables are unavailable.
        return (jnp.zeros((1,), dtype=field_dtype),
                jnp.zeros((1,), dtype=field_dtype))
    e, h = _converged_incident_tables(
        f_c=f_c, medium=medium, dt=dt, dx=dx, t0=t0, tau=tau,
        f0=f0, waveform=waveform, mode_profile=mode_profile, mode_type=mode_type)

    def table(values):
        result = np.zeros(n, dtype=np.float64)
        count = min(n, len(values))
        result[:count] = values[:count]
        return jnp.asarray(result, dtype=field_dtype)
    return table(e), table(h)


def incident_tables_uncut(cfg):
    """Host E/H samples through the converged build's causal half, or no tables.

    Traced configurations and the size-1 legacy sentinel are left untouched.
    The rebuild must reproduce the tables the port applies: a config whose
    timing fields were changed without rebuilding its tables (rfx/sparams/
    waveguide.py copies port 0's t0 and tau to every port) raises ValueError,
    so the record check reads "cannot judge" rather than a drive never applied.
    """
    import jax
    from rfx.core.jax_utils import is_tracer

    values = cfg._asdict() if hasattr(cfg, '_asdict') else vars(cfg)
    if any(is_tracer(x) for x in jax.tree_util.tree_leaves(values)):
        return ()
    if (getattr(cfg, 'e_inc_table', np.zeros(1)).size <= 1
            or getattr(cfg, 'h_inc_table', np.zeros(1)).size <= 1 or not cfg.src_amp):
        return ()
    tables = _converged_incident_tables(
        f_c=cfg.f_cutoff, medium=cfg.medium, dt=cfg.dt, dx=cfg.dx,
        t0=cfg.src_t0, tau=cfg.src_tau, f0=cfg.src_fcen,
        waveform=cfg.waveform, mode_profile=cfg.mode_profile, mode_type=cfg.mode_type)
    for name, uncut in zip(("e_inc_table", "h_inc_table"), tables):
        applied = np.asarray(getattr(cfg, name))
        count = min(len(applied), len(uncut))
        if (not np.array_equal(np.asarray(uncut[:count], dtype=applied.dtype), applied[:count])
                or np.any(applied[count:])):
            raise ValueError(f"waveguide port {getattr(cfg, 'port_name', '')!r}: its {name} is not the table its "
                             "configuration rebuilds, so its known drive beyond the record cannot be judged")
    return tables


def _converged_incident_tables(*, f_c, medium, dt, dx, t0, tau,
                               f0, waveform, mode_profile, mode_type):
    """The shared float64 build and stop rule, independent of record length."""
    previous = None
    for exponent in range(14, 25):
        length = 1 << exponent
        e, h = _incident_table_build(
            length=length, f_c=f_c, medium=medium, dt=dt, dx=dx,
            t0=t0, tau=tau, f0=f0, waveform=waveform,
            mode_profile=mode_profile, mode_type=mode_type)
        keep = length // 2
        peak = np.max(np.abs(h))
        tail = np.max(np.abs(h[keep - keep // 16:keep]))
        change = (0.0 if previous is None else
                  np.max(np.abs(h[:len(previous)] - previous)))
        discarded = np.max(np.abs(h[keep:]))
        if tail <= 1e-5 * peak and change <= 1e-5 * peak and discarded <= 2e-5 * peak:
            return e[:keep].copy(), h[:keep].copy()
        previous = h[:keep].copy()
    bandwidth = np.sqrt(2.0) / (tau * f0)
    raise ValueError(
        f"waveguide incident table did not converge by L=2^24: "
        f"f0={f0}, bandwidth={bandwidth}, f_c={f_c}. The source spectrum reaches the "
        f"{mode_type} cutoff, where the forward wave's H decays too slowly to tabulate (a TM "
        "admittance diverges at cutoff; a TE one at DC); raise f0 or narrow the bandwidth so "
        "the source spectrum at f_c is below about 1e-3 of its peak")


def with_port_medium(cfg, medium):
    """Resolve the cutoff and source tables without rebuilding geometric profiles."""
    if cfg.medium == medium:
        return cfg
    f_c = cfg.f_cutoff * port_light_speed(medium) / port_light_speed(cfg)
    e, h = _incident_tables(
        f_c=f_c, medium=medium, dt=cfg.dt, dx=cfg.dx,
        dft_total_steps=cfg.dft_total_steps, t0=cfg.src_t0, tau=cfg.src_tau,
        f0=cfg.src_fcen, waveform=cfg.waveform, mode_profile=cfg.mode_profile,
        mode_type=cfg.mode_type, field_dtype=cfg.e_inc_table.dtype)
    return cfg._replace(medium=medium, f_cutoff=float(f_c), e_inc_table=e, h_inc_table=h)


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


def _compute_beta(freqs, f_cutoff, *, medium, dt=0.0, dx=0.0, _table=False):
    """Guided beta from the section medium and geometric transverse eigenvalue.

    Source tables use host float64, preserving their historical arithmetic;
    measurement uses JAX and retains the evanescent branch for diagnostics.
    """
    xp = np if _table else jnp
    omega = 2 * xp.pi * freqs
    speed = port_light_speed(medium)
    kc = 2 * xp.pi * f_cutoff / speed
    if dt > 0.0 and dx > 0.0:
        if _table:
            s_t = xp.sin(omega * 0.5 * dt) / (0.5 * dt)
            s_x_sq = (s_t / speed) ** 2 - kc ** 2
            arg = xp.sqrt(xp.minimum(xp.maximum(s_x_sq, 0.0) * (0.5 * dx) ** 2, 1.0))
        else:
            s_t_over_c = xp.sin(omega * 0.5 * dt) / (speed * 0.5 * dt)
            s_x_sq = s_t_over_c ** 2 - kc ** 2
            arg = xp.clip(0.5 * dx * xp.sqrt(xp.maximum(s_x_sq, 0.0)),
                          -1.0 + 1e-12, 1.0 - 1e-12)
        prop = (2.0 / dx) * xp.arcsin(arg)
    else:
        s_x_sq = (omega / speed) ** 2 - kc ** 2
        prop = xp.sqrt(xp.maximum(s_x_sq, 0.0))
    if _table:
        alpha = xp.sqrt(xp.maximum(-s_x_sq, 0.0))
        if dt > 0.0 and dx > 0.0:
            alpha = (2.0 / dx) * xp.arcsinh(alpha * dx * 0.5)
        return xp.where(s_x_sq >= 0, prop, -1j * alpha)
    return xp.where(s_x_sq >= 0, prop, 1j * xp.sqrt(xp.maximum(-s_x_sq, 0.0)))


def _compute_mode_impedance(freqs, f_cutoff, mode_type, *, medium,
                            dt=0.0, dx=0.0, _table=False):
    """TE/TM impedance using the same discrete beta as injection and extraction."""
    xp = np if _table else jnp
    omega = 2 * xp.pi * freqs
    beta = _compute_beta(freqs, f_cutoff, medium=medium, dt=dt, dx=dx, _table=_table)
    mu, eps = MU_0 * medium.mu_r, EPS_0 * medium.eps_r
    if _table:
        # Continue the same impedance through cutoff, preserving complex beta.
        if dt > 0.0 and dx > 0.0:
            w = 2.0 / dt * np.sin(omega * dt * 0.5)
            b = 2.0 / dx * np.sin(beta * dx * 0.5)
        else:
            w, b = omega, beta
        if mode_type == "TE":
            numerator, denominator = mu * w, b
        elif mode_type == "TM":
            numerator, denominator = b, eps * w
        else:
            raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")
        z = np.full(beta.shape, complex(np.inf), dtype=np.complex128)
        np.divide(numerator, denominator, out=z, where=denominator != 0)
        # Y(0) = 0: do not inject the truncated pulse's residual DC.
        return np.where(omega == 0, complex(np.inf), z)
    safe_beta = xp.where(xp.abs(beta) > 1e-30, beta, 1e-30 * xp.ones_like(beta))
    safe_omega = xp.where(xp.abs(omega) > 1e-30, omega, 1e-30 * xp.ones_like(omega))
    if dt > 0.0 and dx > 0.0:
        s_w, s_b = xp.sin(omega * 0.5 * dt), xp.sin(beta * 0.5 * dx)
        if mode_type == "TE":
            z = mu * dx * s_w / xp.maximum(dt * s_b, 1e-30)
        elif mode_type == "TM":
            z = dt * s_b / xp.maximum(eps * dx * s_w, 1e-30)
        else:
            raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")
        sentinel = jnp.asarray(1e30, dtype=jnp.float32) if mode_type == "TE" else 0.0
        return xp.where(freqs > f_cutoff, z, sentinel)
    if mode_type == "TE":
        z = omega * mu / safe_beta
    elif mode_type == "TM":
        z = beta / (safe_omega * eps)
    else:
        raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")
    return z


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
        cfgs.append(cfg._replace(multimode=True))

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
