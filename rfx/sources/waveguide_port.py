"""Rectangular waveguide port: analytical TE/TM mode profiles.

Excites and extracts waveguide modes using precomputed analytical profiles
on the port cross-section. Supports TE_mn and TM_mn modes for rectangular
waveguides with PEC walls.

The port sits on a y-z plane at a fixed x index. Mode propagation is along +x.

TE_mn transverse E-field profiles (Pozar, mapped to prop-along-x):
    Ey(y,z) = -(nπ/b) cos(mπy/a) sin(nπz/b)
    Ez(y,z) =  (mπ/a) sin(mπy/a) cos(nπz/b)

where k_c² = (mπ/a)² + (nπ/b)², a = waveguide width (y),
b = waveguide height (z).

Key examples:
    TE10: Ey = 0,  Ez = (π/a) sin(πy/a)
    TE01: Ey = -(π/b) sin(πz/b),  Ez = 0

S21 is extracted using V/I forward-wave decomposition at two probe planes:
    a_fwd(f) = (V(f) + Z_TE(f) * I(f)) / 2
    S21(f)   = a_fwd_probe(f) / a_fwd_ref(f)
This removes the worst standing-wave inflation of voltage-only ratios, though
individual points can still deviate under finite-window/CPML error.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


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
    dft_window: str
    dft_window_alpha: float

    # Source waveform parameters. ``waveform`` selects the pulse shape:
    # ``"differentiated_gaussian"`` (legacy, src = -2·arg·exp(-arg²)) or
    # ``"modulated_gaussian"`` (Meep-style, src = cos(2π·f₀·(t-t₀))·exp(-arg²)).
    # ``src_fcen`` is only used by the modulated-Gaussian dispatch.
    src_amp: float
    src_t0: float
    src_tau: float
    src_fcen: float
    waveform: str

    # DFT accumulators for S-parameter extraction
    v_probe_dft: jnp.ndarray   # (n_freqs,) complex — modal voltage at probe
    v_ref_dft: jnp.ndarray     # (n_freqs,) complex — modal voltage at ref
    i_probe_dft: jnp.ndarray   # (n_freqs,) complex — modal current at probe
    i_ref_dft: jnp.ndarray     # (n_freqs,) complex — modal current at ref
    v_inc_dft: jnp.ndarray     # (n_freqs,) complex — source waveform DFT
    freqs: jnp.ndarray         # (n_freqs,) float

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

    # Optional 1D Klein-Gordon auxiliary grid (P4). When ``aux_enabled`` is
    # true, apply_waveguide_port_h/e read (e, h) directly from a 1D FDTD
    # threaded through the scan body instead of from the FFT tables above —
    # eliminating the precompute-vs-Yee inconsistency floor (~5.6% backward
    # leakage in the table path).
    aux_enabled: bool
    aux_config: object         # WaveguidePortAuxConfig (opaque here to avoid cyclic import)


def _te_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      *,
                      u_widths: np.ndarray | None = None,
                      v_widths: np.ndarray | None = None,
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

    # H for forward +x propagation: hy = -ez, hz = ey (unnormalized)
    # gives Poynting P_x = Ey*Hz - Ez*Hy = Ey² + Ez² > 0
    hy = -ez.copy()
    hz = ey.copy()

    # Normalize: integral(Ey² + Ez²) dA = 1
    if u_widths is not None and v_widths is not None:
        dA = np.asarray(u_widths)[:, None] * np.asarray(v_widths)[None, :]
        power = float(np.sum((ey**2 + ez**2) * dA))
    else:
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
        dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
        power = np.sum(ey**2 + ez**2) * dy * dz
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm

    return ey, ez, hy, hz


def _tm_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      *,
                      u_widths: np.ndarray | None = None,
                      v_widths: np.ndarray | None = None,
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

    hy = -ez.copy()
    hz = ey.copy()

    if u_widths is not None and v_widths is not None:
        dA = np.asarray(u_widths)[:, None] * np.asarray(v_widths)[None, :]
        power = float(np.sum((ey**2 + ez**2) * dA))
    else:
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
        dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
        power = np.sum(ey**2 + ez**2) * dy * dz
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm

    return ey, ez, hy, hz


def cutoff_frequency(a: float, b: float, m: int, n: int) -> float:
    """TE_mn or TM_mn cutoff frequency for rectangular waveguide."""
    kc = np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
    return kc * C0_LOCAL / (2 * np.pi)


def _second_diff_1d(widths: np.ndarray, *, bc: str) -> np.ndarray:
    """Cell-centred finite-volume ∂²/∂u² operator with PEC-like BC.

    bc="neumann" uses ghost = same-as-boundary reflection (TE Hx, ∂Hx/∂n = 0).
    bc="dirichlet" uses ghost = negative-of-boundary (TM Ex / TE Ez tangential,
    field = 0 at wall, so for cell-centred values ghost[-1] = -cell[0] about
    a wall at the cell-face at u=0).

    The operator is symmetric-negative-semidefinite (Neumann) or symmetric-
    negative-definite (Dirichlet) so `eigh(-L)` yields non-negative kc².
    """
    n = len(widths)
    D = np.zeros((n, n), dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    for j in range(n):
        if j > 0:
            d_left = 0.5 * (w[j - 1] + w[j])
            coeff = 1.0 / (d_left * w[j])
            D[j, j - 1] += coeff
            D[j, j] -= coeff
        elif bc == "dirichlet":
            # Ghost cell beyond the wall reflects with opposite sign; the
            # wall lives at the outer face (u = 0). Face-spacing to the
            # mirrored ghost is w[0], and ghost value = -cell[0]:
            #   flux = (cell[0] - ghost) / w[0] = 2·cell[0] / w[0]
            coeff = 2.0 / (w[0] * w[0])
            D[j, j] -= coeff
        if j < n - 1:
            d_right = 0.5 * (w[j] + w[j + 1])
            coeff = 1.0 / (d_right * w[j])
            D[j, j + 1] += coeff
            D[j, j] -= coeff
        elif bc == "dirichlet":
            coeff = 2.0 / (w[-1] * w[-1])
            D[j, j] -= coeff
    return D


def _cell_centred_gradient(field: np.ndarray, widths: np.ndarray,
                            axis: int, *, bc: str) -> np.ndarray:
    """∂field/∂u at cell centres for cell-centred values with PEC-like BC.

    Uses face-derivatives averaged back to cell centres. Boundary handling:
      bc="neumann"   → face-derivative at the wall is 0 (mirrored-equal ghost)
      bc="dirichlet" → ghost value is -cell[0], face-derivative is
                       2·cell[0]/w[0] (signed)

    Returns an array the same shape as `field`.
    """
    n = field.shape[axis]
    w = np.asarray(widths, dtype=np.float64)
    field64 = np.asarray(field, dtype=np.float64)
    field_swap = np.moveaxis(field64, axis, 0)
    face = np.zeros_like(field_swap)
    face_spacing = 0.5 * (w[:-1] + w[1:])  # (n-1,)
    face[1:n, ...] = (field_swap[1:n, ...] - field_swap[:n - 1, ...]) \
        / face_spacing.reshape((-1,) + (1,) * (field_swap.ndim - 1))
    # Boundary faces (j = 0 and j = n) lie on the PEC walls.
    if bc == "dirichlet":
        # Ghost = -cell[0], face derivative = (cell[0] - (-cell[0])) / w[0]
        #                                    = 2·cell[0] / w[0]
        face0 = 2.0 * field_swap[0, ...] / w[0]
        face_n = -2.0 * field_swap[-1, ...] / w[-1]
    else:  # neumann
        face0 = np.zeros_like(field_swap[0, ...])
        face_n = np.zeros_like(field_swap[-1, ...])
    # Cell-centre gradient = half-width-weighted mean of two face values.
    grad_swap = np.zeros_like(field_swap)
    grad_swap[0, ...] = 0.5 * (face0 + face[1, ...])
    grad_swap[-1, ...] = 0.5 * (face[-1, ...] + face_n)
    if n > 2:
        grad_swap[1:n - 1, ...] = 0.5 * (face[1:n - 1, ...] + face[2:n, ...])
    return np.moveaxis(grad_swap, 0, axis)


def _pick_eigenmode_by_overlap(evecs: np.ndarray,
                                evals: np.ndarray,
                                analytic_flat: np.ndarray,
                                num_candidates: int = 40,
                                ) -> int:
    """Pick the eigenvector with the largest |overlap| with an analytic shape.

    Rank-based picking breaks when discrete kc² ordering swaps nearly-
    degenerate analytic modes (e.g. TE30 vs TE21 on a WR-90 grid). Overlap
    picks from the lowest `num_candidates` eigenvectors and returns the
    index with the largest absolute projection onto `analytic_flat`.
    """
    n = min(num_candidates, evecs.shape[1])
    analytic_unit = analytic_flat / max(float(np.linalg.norm(analytic_flat)),
                                         1e-30)
    overlaps = np.abs(evecs[:, :n].T @ analytic_unit)
    return int(np.argmax(overlaps))


def _discrete_te_mode_profiles(a: float, b: float, m: int, n: int,
                                u_widths: np.ndarray,
                                v_widths: np.ndarray,
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
    D_uu = _second_diff_1d(u_widths, bc="neumann")
    D_vv = _second_diff_1d(v_widths, bc="neumann")
    lap = np.kron(D_uu, np.eye(nv)) + np.kron(np.eye(nu), D_vv)

    evals, evecs = np.linalg.eigh(-lap)  # eigenvalues = kc² ≥ 0, ascending

    # Pick eigenvector by overlap with analytic Hx = cos(mπy/a)·cos(nπz/b).
    # Rank picking fails when discrete kc² ordering swaps nearly-degenerate
    # analytic modes (e.g. TE30 ↔ TE21 on WR-90).
    u_c = np.cumsum(u_widths) - 0.5 * np.asarray(u_widths)
    v_c = np.cumsum(v_widths) - 0.5 * np.asarray(v_widths)
    hx_ana = (np.cos(m * np.pi * u_c / a)[:, None]
              * np.cos(n * np.pi * v_c / b)[None, :])
    rank = _pick_eigenmode_by_overlap(evecs, evals, hx_ana.ravel())
    kc2 = float(max(evals[rank], 0.0))
    hx = evecs[:, rank].reshape(nu, nv)

    # Align sign (eigh returns arbitrary sign).
    if float(np.sum(hx * hx_ana)) < 0.0:
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
    hy = -ez.copy()
    hz = ey.copy()

    dA = np.asarray(u_widths)[:, None] * np.asarray(v_widths)[None, :]
    power = float(np.sum((ey ** 2 + ez ** 2) * dA))
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm
    return ey, ez, hy, hz, float(np.sqrt(kc2))


def _discrete_tm_mode_profiles(a: float, b: float, m: int, n: int,
                                u_widths: np.ndarray,
                                v_widths: np.ndarray,
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
    D_uu = _second_diff_1d(u_widths, bc="dirichlet")
    D_vv = _second_diff_1d(v_widths, bc="dirichlet")
    lap = np.kron(D_uu, np.eye(nv)) + np.kron(np.eye(nu), D_vv)

    evals, evecs = np.linalg.eigh(-lap)

    # Overlap-pick eigenvector against analytic Ex = sin(mπy/a)·sin(nπz/b).
    u_c = np.cumsum(u_widths) - 0.5 * np.asarray(u_widths)
    v_c = np.cumsum(v_widths) - 0.5 * np.asarray(v_widths)
    ex_ana = (np.sin(m * np.pi * u_c / a)[:, None]
              * np.sin(n * np.pi * v_c / b)[None, :])
    rank = _pick_eigenmode_by_overlap(evecs, evals, ex_ana.ravel())
    kc2 = float(max(evals[rank], 0.0))
    ex = evecs[:, rank].reshape(nu, nv)

    if float(np.sum(ex * ex_ana)) < 0.0:
        ex = -ex

    dExdu = _cell_centred_gradient(ex, u_widths, axis=0, bc="dirichlet")
    dExdv = _cell_centred_gradient(ex, v_widths, axis=1, bc="dirichlet")

    # Analytic convention: ey = (mπ/a)·cos(mπy/a)·sin(nπz/b) = ∂Ex/∂u
    #                     ez = (nπ/b)·sin(mπy/a)·cos(nπz/b) = ∂Ex/∂v
    ey = dExdu
    ez = dExdv
    hy = -ez.copy()
    hz = ey.copy()

    dA = np.asarray(u_widths)[:, None] * np.asarray(v_widths)[None, :]
    power = float(np.sum((ey ** 2 + ez ** 2) * dA))
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm
    return ey, ez, hy, hz, float(np.sqrt(kc2))


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
    dft_window: str = "tukey",
    dft_window_alpha: float = 0.25,
    dt: float = 0.0,
    waveform: str = "differentiated_gaussian",
    mode_profile: str = "analytic",
    use_aux_grid: bool = False,
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
    """
    # Duck-type: if `dx` looks like a grid (has dx_arr / dz arrays),
    # use per-axis widths slicing the aperture; else assume scalar dx.
    grid_obj = None
    if hasattr(dx, "dx_arr") and hasattr(dx, "dz"):
        grid_obj = dx
        dx = float(grid_obj.dx)
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
    if grid_obj is not None:
        dx_arr_np = np.asarray(grid_obj.dx_arr)
        dy_arr_np = np.asarray(grid_obj.dy_arr)
        dz_arr_np = np.asarray(grid_obj.dz)
        if normal_axis == "x":
            u_widths_np = dy_arr_np[u_lo:u_hi]
            v_widths_np = dz_arr_np[v_lo:v_hi]
        elif normal_axis == "y":
            u_widths_np = dx_arr_np[u_lo:u_hi]
            v_widths_np = dz_arr_np[v_lo:v_hi]
        else:  # z-normal
            u_widths_np = dx_arr_np[u_lo:u_hi]
            v_widths_np = dy_arr_np[u_lo:u_hi] if False else dy_arr_np[v_lo:v_hi]
    else:
        u_widths_np = np.full(nu_port, float(dx))
        v_widths_np = np.full(nv_port, float(dx))

    # Cell-centre coordinates (cumulative-sum midpoints). For uniform
    # widths this collapses to the original `np.linspace(0.5*dx, ...)`.
    u_coords = np.cumsum(u_widths_np) - 0.5 * u_widths_np
    v_coords = np.cumsum(v_widths_np) - 0.5 * v_widths_np

    if mode_profile not in ("analytic", "discrete"):
        raise ValueError(
            "mode_profile must be 'analytic' or 'discrete', "
            f"got {mode_profile!r}"
        )
    if mode_profile == "discrete":
        if port.mode_type == "TE":
            ey, ez, hy, hz, kc_num = _discrete_te_mode_profiles(
                port.a, port.b, m, n, u_widths_np, v_widths_np,
            )
        else:
            ey, ez, hy, hz, kc_num = _discrete_tm_mode_profiles(
                port.a, port.b, m, n, u_widths_np, v_widths_np,
            )
        # Use the discrete eigenvalue's kc for f_c so h_inc_table's
        # β(ω) = √(k² − kc_num²) matches the Yee-grid mode exactly.
        f_c = kc_num * C0_LOCAL / (2.0 * np.pi)
    else:
        if port.mode_type == "TE":
            ey, ez, hy, hz = _te_mode_profiles(
                port.a, port.b, m, n, u_coords, v_coords,
                u_widths=u_widths_np, v_widths=v_widths_np,
            )
        else:
            ey, ez, hy, hz = _tm_mode_profiles(
                port.a, port.b, m, n, u_coords, v_coords,
                u_widths=u_widths_np, v_widths=v_widths_np,
            )
        f_c = cutoff_frequency(port.a, port.b, m, n)

    if waveform not in ("differentiated_gaussian", "modulated_gaussian"):
        raise ValueError(
            "waveform must be 'differentiated_gaussian' or 'modulated_gaussian', "
            f"got {waveform!r}"
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

    nf = len(freqs)
    zeros_c = jnp.zeros(nf, dtype=jnp.complex64)

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
            # Meep's GaussianSource shape. No DC content — no sub-cutoff
            # filter needed, which eliminates the raised-cosine filter tail
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
        e_inc_table = jnp.asarray(e_inc_time, dtype=jnp.float32)
        h_inc_table = jnp.asarray(h_inc_time, dtype=jnp.float32)
    else:
        # Tables unavailable (dt=0 at init). Size-1 sentinel triggers the
        # legacy soft-E source fallback inside apply_waveguide_port_e.
        e_inc_table = jnp.zeros((1,), dtype=jnp.float32)
        h_inc_table = jnp.zeros((1,), dtype=jnp.float32)

    # --- P4: 1D Klein-Gordon auxiliary grid (optional) ---
    aux_enabled = bool(use_aux_grid) and dt > 0.0
    if aux_enabled:
        from rfx.sources.waveguide_port_aux import init_wg_aux
        aux_cfg, aux_state0 = init_wg_aux(
            f_cutoff=f_c,
            dx=float(dx),
            dt=float(dt),
            direction=port.direction,
            src_amp=float(amplitude),
            src_t0=float(t0),
            src_tau=float(tau),
            src_fcen=float(f0),
            src_waveform=str(waveform),
        )
    else:
        aux_cfg = None
        aux_state0 = None

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
        ey_profile=jnp.array(ey, dtype=jnp.float32),
        ez_profile=jnp.array(ez, dtype=jnp.float32),
        hy_profile=jnp.array(hy, dtype=jnp.float32),
        hz_profile=jnp.array(hz, dtype=jnp.float32),
        mode_type=port.mode_type,
        direction=port.direction,
        f_cutoff=float(f_c),
        a=port.a, b=port.b,
        dx=float(dx),
        u_widths=jnp.asarray(u_widths_np, dtype=jnp.float32),
        v_widths=jnp.asarray(v_widths_np, dtype=jnp.float32),
        source_x_m=float(source_x_m),
        reference_x_m=float(reference_x_m),
        probe_x_m=float(probe_x_m),
        dft_total_steps=int(dft_total_steps),
        dft_window=dft_window,
        dft_window_alpha=float(dft_window_alpha),
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        src_fcen=float(f0),
        waveform=str(waveform),
        v_probe_dft=zeros_c,
        v_ref_dft=zeros_c,
        i_probe_dft=zeros_c,
        i_ref_dft=zeros_c,
        v_inc_dft=zeros_c,
        freqs=freqs,
        e_inc_table=e_inc_table,
        h_inc_table=h_inc_table,
        aux_enabled=aux_enabled,
        aux_config=aux_cfg,
    )


def _plane_indexer(cfg: WaveguidePortConfig, plane_index: int | None = None):
    """Return an indexer for the tangential aperture plane."""
    idx = cfg.x_index if plane_index is None else plane_index
    if cfg.normal_axis == "x":
        return (idx, slice(cfg.u_lo, cfg.u_hi), slice(cfg.v_lo, cfg.v_hi))
    if cfg.normal_axis == "y":
        return (slice(cfg.u_lo, cfg.u_hi), idx, slice(cfg.v_lo, cfg.v_hi))
    if cfg.normal_axis == "z":
        return (slice(cfg.u_lo, cfg.u_hi), slice(cfg.v_lo, cfg.v_hi), idx)
    raise ValueError(f"normal_axis must be 'x', 'y', or 'z', got {cfg.normal_axis!r}")


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


def inject_waveguide_port(state, cfg: WaveguidePortConfig,
                          t: float, dt: float, dx: float):
    """Legacy soft-E source (superseded by TFSF-style pair in scan body).

    Retained for backwards compatibility — direct callers still get a soft
    E source. The simulation scan body no longer invokes this function;
    it now uses ``apply_waveguide_port_h`` and ``apply_waveguide_port_e``
    together, which form a one-sided TFSF boundary and launch a
    unidirectional mode wave (no spurious backward emission).

    Dispatches on ``cfg.waveform`` so a user who built the port with
    ``waveform="modulated_gaussian"`` sees a consistent pulse shape
    whether they call this helper manually or rely on the scan-body pair.
    """
    arg = (t - cfg.src_t0) / cfg.src_tau
    if cfg.waveform == "modulated_gaussian":
        tt = t - cfg.src_t0
        src_val = cfg.src_amp * jnp.cos(
            2.0 * jnp.pi * cfg.src_fcen * tt
        ) * jnp.exp(-(arg ** 2))
        # Meep-style hard cutoff at ±5·width (= ±t0 for s=5).
        src_val = jnp.where(jnp.abs(tt) > cfg.src_t0, 0.0, src_val)
    else:
        src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    field_u = getattr(state, cfg.e_u_component)
    field_v = getattr(state, cfg.e_v_component)
    indexer = _plane_indexer(cfg)

    field_u = field_u.at[indexer].add(src_val * cfg.ey_profile)
    field_v = field_v.at[indexer].add(src_val * cfg.ez_profile)

    return state._replace(**{cfg.e_u_component: field_u, cfg.e_v_component: field_v})


def apply_waveguide_port_h(state, cfg: WaveguidePortConfig,
                            step, dt: float, dx: float,
                            aux_state=None):
    """TFSF-style H correction at the port's upstream half-cell.

    Two source-value paths:

    - **1D aux path** (``cfg.aux_enabled`` + ``aux_state`` supplied): read
      the E amplitude at the source plane straight from the 1D Klein-
      Gordon auxiliary FDTD. Values are Yee-discrete-exact by
      construction, so backward cancellation with ``apply_waveguide_port_e``
      reaches the same <0.1 % floor as ``tfsf.py``'s plane-wave pair.

    - **FFT-table path** (``cfg.e_inc_table`` populated): legacy behaviour.
      Uses the bandpass-filtered scalar pre-computed at init. Floors at
      ~5.6 % backward leakage because continuous-wave samples can't match
      Yee-grid running-wave values (see 2026-04-21 A-phase investigation).

    When both tables are empty (low-level callers that never pass ``dt``),
    this helper is a no-op and emission falls through to the legacy soft-E
    source inside ``apply_waveguide_port_e``.
    """
    if cfg.aux_enabled and aux_state is not None:
        # 1D aux: e1d[i0] is the E amplitude at the 1D index that maps to
        # the 3D source plane x_src at the current (integer) time level.
        src_val = aux_state.e1d[cfg.aux_config.i0]
    else:
        table = cfg.e_inc_table
        table_size = table.shape[0]
        if table_size <= 1:
            return state
        safe_step = jnp.clip(jnp.asarray(step, dtype=jnp.int32),
                             0, table_size - 1)
        src_val = cfg.src_amp * table[safe_step]
    coeff = dt / (MU_0 * dx)

    # For a "+axis" port the source's backward emission travels in "−axis";
    # the Yee H-curl stencil that reads the injected E lives one half-cell
    # on the −axis side of the source plane, at H-index x_src − 1.
    # For a "−axis" port the mirrored H-plane is at x_src.
    if cfg.direction.startswith("+"):
        h_plane_index = cfg.x_index - 1
        sign = -1.0
    else:
        h_plane_index = cfg.x_index
        sign = +1.0

    indexer_h = _plane_indexer(cfg, h_plane_index)

    h_u_field = getattr(state, cfg.h_u_component)
    h_v_field = getattr(state, cfg.h_v_component)

    # H_u is driven by E_v (ez); H_v by E_u (ey) with opposite curl sign.
    h_u_field = h_u_field.at[indexer_h].add(sign * coeff * src_val * cfg.ez_profile)
    h_v_field = h_v_field.at[indexer_h].add(-sign * coeff * src_val * cfg.ey_profile)

    return state._replace(**{
        cfg.h_u_component: h_u_field,
        cfg.h_v_component: h_v_field,
    })


def apply_waveguide_port_e(state, cfg: WaveguidePortConfig,
                            step, dt: float, dx: float,
                            aux_state=None):
    """TFSF-style E correction at the port source plane.

    Other half of the one-sided TFSF boundary. Adds the incident H-side
    contribution that the Yee curl-H stencil at ``x_src`` expects from a
    forward-only wave; reading from the precomputed ``cfg.h_inc_table``
    so the correction is broadband-accurate with only an O((β·dx)²)
    numerical-dispersion residue.

    Same pattern as ``apply_tfsf_e`` in ``rfx/sources/tfsf.py``. Call AFTER
    ``update_e``, along with the paired ``apply_waveguide_port_h``.
    """
    coeff = dt / (EPS_0 * dx)

    if cfg.aux_enabled and aux_state is not None:
        # 1D aux: h1d[i0-1] is the H at the 1D half-cell upstream of the
        # source plane mapping — exactly the Yee location the 3D E-update
        # stencil at x_src expects from a forward-only wave. Using the
        # 1D aux value avoids the continuous-wave sampling residue that
        # floors the FFT table path at ~5.6 %.
        h_inc_scalar = aux_state.h1d[cfg.aux_config.i0 - 1]
        if cfg.direction.startswith("+"):
            e_plane_index = cfg.x_index
            sign = -1.0
        else:
            e_plane_index = cfg.x_index + 1
            sign = +1.0
        indexer_e = _plane_indexer(cfg, e_plane_index)
        e_u_field = getattr(state, cfg.e_u_component)
        e_v_field = getattr(state, cfg.e_v_component)
        e_v_field = e_v_field.at[indexer_e].add(
            sign * coeff * h_inc_scalar * cfg.hy_profile)
        e_u_field = e_u_field.at[indexer_e].add(
            -sign * coeff * h_inc_scalar * cfg.hz_profile)
        return state._replace(**{
            cfg.e_u_component: e_u_field,
            cfg.e_v_component: e_v_field,
        })

    # h_inc_table is empty (shape (1,)) when dt or dft_total_steps was not
    # supplied at init time. In that case the paired H correction is also
    # disabled (see apply_waveguide_port_h) and we emit via the legacy
    # soft-E source so low-level callers that never knew about the TFSF
    # pair keep working unchanged.
    table = cfg.h_inc_table
    table_size = table.shape[0]
    if table_size <= 1:
        t = jnp.asarray(step, dtype=jnp.float32) * dt
        arg = (t - cfg.src_t0) / cfg.src_tau
        if cfg.waveform == "modulated_gaussian":
            tt = t - cfg.src_t0
            src_val = cfg.src_amp * jnp.cos(
                2.0 * jnp.pi * cfg.src_fcen * tt
            ) * jnp.exp(-(arg ** 2))
            src_val = jnp.where(jnp.abs(tt) > cfg.src_t0, 0.0, src_val)
        else:
            src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
        field_u = getattr(state, cfg.e_u_component)
        field_v = getattr(state, cfg.e_v_component)
        indexer = _plane_indexer(cfg)
        field_u = field_u.at[indexer].add(src_val * cfg.ey_profile)
        field_v = field_v.at[indexer].add(src_val * cfg.ez_profile)
        return state._replace(**{
            cfg.e_u_component: field_u,
            cfg.e_v_component: field_v,
        })

    safe_step = jnp.clip(jnp.asarray(step, dtype=jnp.int32), 0, table_size - 1)
    # Table stores the unit-amplitude response; scale by cfg.src_amp so the
    # passive-port convention (``src_amp=0``) cleanly disables this
    # correction alongside ``apply_waveguide_port_h``.
    h_inc_scalar = cfg.src_amp * table[safe_step]

    if cfg.direction.startswith("+"):
        e_plane_index = cfg.x_index
        sign = -1.0
    else:
        # "−axis" port emits toward −axis; TFSF boundary sits at x_src + 1.
        e_plane_index = cfg.x_index + 1
        sign = +1.0

    indexer_e = _plane_indexer(cfg, e_plane_index)

    e_u_field = getattr(state, cfg.e_u_component)
    e_v_field = getattr(state, cfg.e_v_component)

    # E_v is corrected by H_u (hy_profile); E_u by H_v (hz_profile) with
    # opposite curl sign. The modal H amplitude (positive for positive
    # forward E) multiplies the stored h*_profile which already carries
    # the mode-specific sign relative to its e_*_profile counterpart.
    e_v_field = e_v_field.at[indexer_e].add(sign * coeff * h_inc_scalar * cfg.hy_profile)
    e_u_field = e_u_field.at[indexer_e].add(-sign * coeff * h_inc_scalar * cfg.hz_profile)

    return state._replace(**{
        cfg.e_u_component: e_u_field,
        cfg.e_v_component: e_v_field,
    })


def _aperture_dA(cfg: WaveguidePortConfig) -> jnp.ndarray:
    """Per-cell area element (nu, nv) on the port aperture."""
    return cfg.u_widths[:, None] * cfg.v_widths[None, :]


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
    h_u_sim = _plane_h_field(getattr(state, cfg.h_u_component), cfg, x_idx)
    h_v_sim = _plane_h_field(getattr(state, cfg.h_v_component), cfg, x_idx)
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
    h_u_sim = _plane_h_field(getattr(state, cfg.h_u_component), cfg, x_idx)
    h_v_sim = _plane_h_field(getattr(state, cfg.h_v_component), cfg, x_idx)

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
    """Accumulate DFT of modal V and I at ref and probe planes."""
    t = state.step * dt

    v_ref = modal_voltage(state, cfg, cfg.ref_x, dx)
    v_probe = modal_voltage(state, cfg, cfg.probe_x, dx)
    i_ref = modal_current(state, cfg, cfg.ref_x, dx)
    i_probe = modal_current(state, cfg, cfg.probe_x, dx)

    arg = (t - cfg.src_t0) / cfg.src_tau
    v_inc = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)
    weight = _dft_window_weight(state.step, cfg.dft_total_steps, cfg.dft_window, cfg.dft_window_alpha)

    return cfg._replace(
        v_probe_dft=cfg.v_probe_dft + v_probe * phase * dt * weight,
        v_ref_dft=cfg.v_ref_dft + v_ref * phase * dt * weight,
        i_probe_dft=cfg.i_probe_dft + i_probe * phase * dt * weight,
        i_ref_dft=cfg.i_ref_dft + i_ref * phase * dt * weight,
        v_inc_dft=cfg.v_inc_dft + v_inc * phase * dt * weight,
    )


def _compute_beta(freqs: jnp.ndarray, f_cutoff: float) -> jnp.ndarray:
    """Guided propagation constant β(f) for a vacuum-filled rectangular guide."""
    omega = 2 * jnp.pi * freqs
    k = omega / C0_LOCAL
    kc = 2 * jnp.pi * f_cutoff / C0_LOCAL

    beta_sq = k**2 - kc**2
    return jnp.where(
        beta_sq >= 0,
        jnp.sqrt(jnp.maximum(beta_sq, 0.0)),
        1j * jnp.sqrt(jnp.maximum(-beta_sq, 0.0)),
    )


from rfx.core.dft_utils import dft_window_weight as _dft_window_weight

def _compute_mode_impedance(
    freqs: jnp.ndarray,
    f_cutoff: float,
    mode_type: str,
) -> jnp.ndarray:
    """Rectangular-waveguide modal impedance for TE/TM modes.

    TE: Z = ωμ / β
    TM: Z = β / (ωε)
    """
    omega = 2 * jnp.pi * freqs
    beta = _compute_beta(freqs, f_cutoff)
    safe_beta = jnp.where(jnp.abs(beta) > 1e-30, beta,
                          1e-30 * jnp.ones_like(beta))
    safe_omega = jnp.where(jnp.abs(omega) > 1e-30, omega,
                           1e-30 * jnp.ones_like(omega))
    if mode_type == "TE":
        return omega * MU_0 / safe_beta
    if mode_type == "TM":
        return safe_beta / (safe_omega * EPS_0)
    raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")


def _extract_global_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return global (+x/-x) modal waves from colocated modal V/I spectra."""
    z_mode = _compute_mode_impedance(cfg.freqs, cfg.f_cutoff, cfg.mode_type)
    forward = 0.5 * (voltage_dft + z_mode * current_dft)
    backward = 0.5 * (voltage_dft - z_mode * current_dft)
    return forward, backward


def _extract_port_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return port-local (incident, outgoing) waves.

    For a positive-direction port (`+x`, `+y`, `+z`), incident is the local
    forward wave and outgoing is the local backward wave. For a negative
    direction port, the mapping is reversed.
    """
    forward, backward = _extract_global_waves(cfg, voltage_dft, current_dft)
    if cfg.direction.startswith("+"):
        return forward, backward
    return backward, forward


def waveguide_plane_positions(cfg: WaveguidePortConfig) -> dict[str, float]:
    """Physical source/reference/probe positions along the port normal axis.

    When `cfg` comes from `rfx.api.Simulation`, these are domain-relative
    physical coordinates along the active port-normal axis. For lower-level
    manual setups without `x_position`,
    they fall back to the padded-grid coordinate origin.
    """
    return {
        "source": cfg.source_x_m,
        "reference": cfg.reference_x_m,
        "probe": cfg.probe_x_m,
    }


def _shift_modal_waves(
    forward: jnp.ndarray,
    backward: jnp.ndarray,
    beta: jnp.ndarray,
    shift_m: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shift modal waves to a new reference plane.

    `shift_m > 0` means shifting the reporting plane downstream along the
    positive normal axis of the port.
    """
    if shift_m == 0.0:
        return forward, backward
    shift = jnp.asarray(shift_m, dtype=beta.dtype)
    forward_shifted = forward * jnp.exp(-1j * beta * shift)
    backward_shifted = backward * jnp.exp(+1j * beta * shift)
    return forward_shifted, backward_shifted


def extract_waveguide_sparams(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
    probe_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (S11, S21) with optional reference-plane shifts.

    Parameters
    ----------
    ref_shift : float
        Metres to shift the reference-plane reporting location relative to the
        stored reference probe. Positive is downstream (+x), negative upstream.
    probe_shift : float
        Metres to shift the probe-plane reporting location relative to the
        stored probe plane. Positive is downstream (+x), negative upstream.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff)
    a_ref, b_ref = _extract_port_waves(cfg, cfg.v_ref_dft, cfg.i_ref_dft)
    a_probe, b_probe = _extract_port_waves(cfg, cfg.v_probe_dft, cfg.i_probe_dft)
    a_ref, b_ref = _shift_modal_waves(a_ref, b_ref, beta, ref_shift)
    a_probe, b_probe = _shift_modal_waves(a_probe, b_probe, beta, probe_shift)
    safe_ref = jnp.where(jnp.abs(a_ref) > 0, a_ref, jnp.ones_like(a_ref))
    s11 = b_ref / safe_ref
    s21 = a_probe / safe_ref
    return s11, s21


def extract_waveguide_s21(cfg: WaveguidePortConfig,
                          dt: float = 0.0,
                          *,
                          ref_shift: float = 0.0,
                          probe_shift: float = 0.0) -> jnp.ndarray:
    """Extract S21 from forward-wave modal amplitudes with optional de-embedding."""
    _, s21 = extract_waveguide_sparams(
        cfg,
        ref_shift=ref_shift,
        probe_shift=probe_shift,
    )
    return s21


def extract_waveguide_s11(cfg: WaveguidePortConfig,
                          *,
                          ref_shift: float = 0.0) -> jnp.ndarray:
    """Extract S11 from backward/forward modal amplitudes with optional de-embedding."""
    s11, _ = extract_waveguide_sparams(cfg, ref_shift=ref_shift)
    return s11


def extract_waveguide_port_waves(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return port-local (incident, outgoing) waves at a shifted reference plane."""
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff)
    a_ref, b_ref = _extract_port_waves(cfg, cfg.v_ref_dft, cfg.i_ref_dft)
    return _shift_modal_waves(a_ref, b_ref, beta, ref_shift)


def extract_waveguide_s_matrix(
    grid,
    materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str = "x",
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Assemble an x-directed waveguide S-matrix via one-driven-port-at-a-time runs."""
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_matrix requires at least two waveguide ports"
        )
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in port_cfgs)
    if len(ref_shifts) != len(port_cfgs):
        raise ValueError("ref_shifts must match the number of waveguide ports when provided")

    from rfx.simulation import run as run_simulation

    template_cfgs = tuple(port_cfgs)
    n_ports = len(template_cfgs)
    n_freqs = len(template_cfgs[0].freqs)
    s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros = jnp.zeros_like(cfg.v_probe_dft)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_dft=zeros,
            v_ref_dft=zeros,
            i_probe_dft=zeros,
            i_ref_dft=zeros,
            v_inc_dft=zeros,
        )

    for drive_idx in range(n_ports):
        driven_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        result = run_simulation(
            grid,
            materials,
            n_steps,
            boundary=boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            waveguide_ports=driven_cfgs,
        )
        final_cfgs = result.waveguide_ports or ()
        if len(final_cfgs) != n_ports:
            raise RuntimeError("waveguide S-matrix extraction expected one final config per port")

        a_drive, b_drive = extract_waveguide_port_waves(
            final_cfgs[drive_idx],
            ref_shift=ref_shifts[drive_idx],
        )
        safe_a = jnp.where(jnp.abs(a_drive) > 0, a_drive, jnp.ones_like(a_drive))
        for recv_idx, cfg in enumerate(final_cfgs):
            _a_recv, b_recv = extract_waveguide_port_waves(
                cfg,
                ref_shift=ref_shifts[recv_idx],
            )
            s_matrix[recv_idx, drive_idx, :] = np.array(b_recv / safe_a)

    return jnp.asarray(s_matrix)



def extract_waveguide_s_params_normalized(
    grid,
    materials,
    ref_materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str = "x",
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_debye: tuple | None = None,
    ref_lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, ...] | None = None,
) -> jnp.ndarray:
    """Two-run normalized waveguide S-matrix.

    Cancels Yee-grid numerical dispersion by normalizing device outgoing
    waves against reference-run waves measured at the **same** port
    location.

    For each driven port j:
      1. Run reference (empty guide) with port j driven.
      2. Run device with port j driven.
      3. Off-diagonal (i != j):
           S_ij = b_out_device[i] / b_out_reference[i]
         The reference b_out[i] captures the wave as it arrives at port i,
         including all numerical dispersion, so dividing cancels the bias.
      4. Diagonal (i == j):
           S_jj = b_out_device[j] / a_inc_reference[j]
         The reference reflection is near zero for an empty guide, so we
         normalize the device reflection by the incident wave instead.

    This avoids the small/small blow-up of element-wise S_dev/S_ref for
    reflection terms while still cancelling dispersion for transmission.

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    materials : MaterialArrays
        Device material arrays.
    ref_materials : MaterialArrays
        Reference (empty waveguide) material arrays.
    port_cfgs : list[WaveguidePortConfig]
        Port configurations (same for both runs).
    n_steps : int
        Number of timesteps for each run.
    boundary, cpml_axes, pec_axes, periodic : str / tuple
        Boundary conditions (same for both runs).
    debye, lorentz : tuple or None
        Dispersion specs for the device run.
    ref_debye, ref_lorentz : tuple or None
        Dispersion specs for the reference run.
    ref_shifts : list[float] or None
        Optional reference-plane shifts per port.

    Returns
    -------
    jnp.ndarray
        Normalized S-matrix of shape (n_ports, n_ports, n_freqs), complex.
    """
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_params_normalized requires at least two waveguide ports"
        )
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in port_cfgs)
    if len(ref_shifts) != len(port_cfgs):
        raise ValueError("ref_shifts must match the number of waveguide ports when provided")

    from rfx.simulation import run as run_simulation

    template_cfgs = tuple(port_cfgs)
    n_ports = len(template_cfgs)
    n_freqs = len(template_cfgs[0].freqs)
    s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros = jnp.zeros_like(cfg.v_probe_dft)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_dft=zeros,
            v_ref_dft=zeros,
            i_probe_dft=zeros,
            i_ref_dft=zeros,
            v_inc_dft=zeros,
        )

    common_run_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
    )

    for drive_idx in range(n_ports):
        # --- Reference run: extract waves at all ports ---
        ref_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        ref_result = run_simulation(
            grid, ref_materials, n_steps,
            debye=ref_debye, lorentz=ref_lorentz,
            waveguide_ports=ref_cfgs, **common_run_kw,
        )
        ref_final_cfgs = ref_result.waveguide_ports or ()
        if len(ref_final_cfgs) != n_ports:
            raise RuntimeError(
                "waveguide S-matrix extraction expected one final config per port"
            )

        # Incident wave at the driven port (for diagonal normalization)
        a_inc_ref, _ = extract_waveguide_port_waves(
            ref_final_cfgs[drive_idx],
            ref_shift=ref_shifts[drive_idx],
        )
        a_inc_ref_np = np.array(a_inc_ref)
        safe_a_inc = np.where(
            np.abs(a_inc_ref_np) > 1e-30,
            a_inc_ref_np,
            np.ones_like(a_inc_ref_np),
        )

        # Outgoing waves at all ports (for off-diagonal normalization)
        b_out_ref = []
        for recv_idx in range(n_ports):
            _, b_ref_i = extract_waveguide_port_waves(
                ref_final_cfgs[recv_idx],
                ref_shift=ref_shifts[recv_idx],
            )
            b_out_ref.append(np.array(b_ref_i))

        # --- Device run: extract outgoing waves at every port ---
        dev_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        dev_result = run_simulation(
            grid, materials, n_steps,
            debye=debye, lorentz=lorentz,
            waveguide_ports=dev_cfgs, **common_run_kw,
        )
        dev_final_cfgs = dev_result.waveguide_ports or ()
        if len(dev_final_cfgs) != n_ports:
            raise RuntimeError(
                "waveguide S-matrix extraction expected one final config per port"
            )

        for recv_idx, cfg in enumerate(dev_final_cfgs):
            _, b_recv_dev = extract_waveguide_port_waves(
                cfg,
                ref_shift=ref_shifts[recv_idx],
            )
            b_recv_dev_np = np.array(b_recv_dev)

            if recv_idx == drive_idx:
                # Diagonal: normalize reflection by incident wave
                s_matrix[recv_idx, drive_idx, :] = b_recv_dev_np / safe_a_inc
            else:
                # Off-diagonal: normalize by reference outgoing at same port
                safe_b_ref = np.where(
                    np.abs(b_out_ref[recv_idx]) > 1e-30,
                    b_out_ref[recv_idx],
                    np.ones_like(b_out_ref[recv_idx]),
                )
                s_matrix[recv_idx, drive_idx, :] = b_recv_dev_np / safe_b_ref

    return jnp.asarray(s_matrix)



# ---------------------------------------------------------------------------
# Overlap integral modal extraction — S-parameter extraction (Spec 6E4)
#
# Time-domain overlap helpers ``mode_self_overlap`` and
# ``overlap_modal_amplitude`` live above (near ``modal_voltage``).
#
# For DFT-based extraction the overlap integral reuses the existing V/I
# DFT accumulators (``v_ref_dft``, ``i_ref_dft``, etc.) because the two
# cross-product terms in the overlap decompose into modal voltage and
# modal current when the stored mode profiles satisfy h_mode = n̂ × e_mode:
#
#   P1 = ∫(E_sim × h_stored) · n̂ dA  =  V   (modal voltage)
#   P2 = ∫(e_stored × H_sim) · n̂ dA  =  I   (modal current)
#
# The physical mode H-field is H_mode = h_stored / Z_mode(f), so the
# frequency-dependent impedance must be folded in at extraction time:
#
#   a±(f) = 0.5 * (V(f) ± Z_mode(f) * I(f)) / C_stored
#
# where C_stored = ∫(e_stored × h_stored) · n̂ dA  ≈ 1 for normalized
# profiles.  This is mathematically equivalent to the V/I wave
# decomposition with an additional mode-overlap normalization factor.
# ---------------------------------------------------------------------------


class OverlapDFTAccumulators(NamedTuple):
    """DFT accumulators for overlap-based modal extraction.

    Stores the two raw cross-product DFTs (P1 = modal voltage, P2 = modal
    current) at the reference and probe planes.  These reuse the same
    time-domain quantities as the V/I accumulators but are combined
    differently at extraction time.
    """
    p1_ref_dft: jnp.ndarray     # (n_freqs,) complex — ∫(E_sim × h_mode)·n̂ dA at ref
    p2_ref_dft: jnp.ndarray     # (n_freqs,) complex — ∫(e_mode × H_sim)·n̂ dA at ref
    p1_probe_dft: jnp.ndarray   # (n_freqs,) complex — ∫(E_sim × h_mode)·n̂ dA at probe
    p2_probe_dft: jnp.ndarray   # (n_freqs,) complex — ∫(e_mode × H_sim)·n̂ dA at probe


def init_overlap_dft(freqs: jnp.ndarray) -> OverlapDFTAccumulators:
    """Initialize zero-valued overlap DFT accumulators."""
    nf = len(freqs)
    zeros_c = jnp.zeros(nf, dtype=jnp.complex64)
    return OverlapDFTAccumulators(
        p1_ref_dft=zeros_c,
        p2_ref_dft=zeros_c,
        p1_probe_dft=zeros_c,
        p2_probe_dft=zeros_c,
    )


# ---------------------------------------------------------------------------
# Multi-mode waveguide port support
# ---------------------------------------------------------------------------

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
    dft_window: str = "tukey",
    dft_window_alpha: float = 0.25,
    dt: float = 0.0,
    waveform: str = "differentiated_gaussian",
    mode_profile: str = "analytic",
    use_aux_grid: bool = False,
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
        Total simulation steps (for DFT windowing).
    dft_window : str
        DFT window type.
    dft_window_alpha : float
        DFT window parameter.

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
            dft_window=dft_window, dft_window_alpha=dft_window_alpha,
            dt=dt, waveform=waveform, mode_profile=mode_profile,
            use_aux_grid=use_aux_grid,
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
            dft_window=dft_window, dft_window_alpha=dft_window_alpha,
            dt=dt, waveform=waveform, mode_profile=mode_profile,
            use_aux_grid=use_aux_grid,
        )
        cfgs.append(cfg)

    return cfgs


def extract_multimode_s_matrix(
    grid,
    materials,
    port_mode_cfgs: list[list[WaveguidePortConfig]],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str = "x",
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_shifts: list[float] | None = None,
) -> tuple[jnp.ndarray, list[tuple[int, int, str, tuple[int, int]]]]:
    """Assemble a multi-mode waveguide S-matrix.

    Each physical port may have multiple modes.  The S-matrix indices
    enumerate (port_index, mode_index) pairs.

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    materials : MaterialArrays
        Material arrays.
    port_mode_cfgs : list of list of WaveguidePortConfig
        ``port_mode_cfgs[p]`` is a list of configs for physical port ``p``,
        one per mode.
    n_steps : int
        Number of timesteps per run.
    ref_shifts : list[float] or None
        Reference-plane shifts per physical port.  Applied to all modes
        of each port.

    Returns
    -------
    s_matrix : jnp.ndarray, shape (N, N, n_freqs)
        S-matrix where N = sum of modes across all ports.
    mode_map : list of (port_idx, mode_idx, mode_type, (m, n))
        Ordering of rows/columns in the S-matrix.
    """
    from rfx.simulation import run as run_simulation

    # Flatten to a linear list, keeping track of (port_idx, mode_within_port)
    flat_cfgs: list[WaveguidePortConfig] = []
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []
    port_of_flat: list[int] = []

    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(mode_cfgs):
            flat_cfgs.append(cfg)
            # Extract mode info from the config
            m_idx = _mode_indices_from_config(cfg)
            mode_map.append((port_idx, mode_within, cfg.mode_type, m_idx))
            port_of_flat.append(port_idx)

    n_total = len(flat_cfgs)
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=np.complex64)

    n_ports = len(port_mode_cfgs)
    if ref_shifts is None:
        ref_shifts = [0.0] * n_ports
    if len(ref_shifts) != n_ports:
        raise ValueError("ref_shifts must have one entry per physical port")

    # Expand ref_shifts to flat indexing
    flat_ref_shifts = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for _ in mode_cfgs:
            flat_ref_shifts.append(ref_shifts[port_idx])

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros = jnp.zeros_like(cfg.v_probe_dft)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_dft=zeros,
            v_ref_dft=zeros,
            i_probe_dft=zeros,
            i_ref_dft=zeros,
            v_inc_dft=zeros,
        )

    # Drive each mode one at a time
    for drive_flat_idx in range(n_total):
        driven_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        # Set the driven mode amplitude to the original first-mode amplitude
        orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0
        driven_cfgs[drive_flat_idx] = driven_cfgs[drive_flat_idx]._replace(
            src_amp=orig_amp,
        )

        result = run_simulation(
            grid, materials, n_steps,
            boundary=boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            waveguide_ports=driven_cfgs,
        )
        final_cfgs = result.waveguide_ports or ()
        if len(final_cfgs) != n_total:
            raise RuntimeError(
                f"Expected {n_total} final waveguide configs, got {len(final_cfgs)}"
            )

        a_drive, _ = extract_waveguide_port_waves(
            final_cfgs[drive_flat_idx],
            ref_shift=flat_ref_shifts[drive_flat_idx],
        )
        safe_a = jnp.where(jnp.abs(a_drive) > 0, a_drive, jnp.ones_like(a_drive))

        for recv_idx, cfg in enumerate(final_cfgs):
            _, b_recv = extract_waveguide_port_waves(
                cfg,
                ref_shift=flat_ref_shifts[recv_idx],
            )
            s_matrix[recv_idx, drive_flat_idx, :] = np.array(b_recv / safe_a)

    return jnp.asarray(s_matrix), mode_map


def _mode_indices_from_config(cfg: WaveguidePortConfig) -> tuple[int, int]:
    """Best-effort extraction of (m, n) mode indices from a config.

    Uses the cutoff frequency and aperture dimensions to identify the mode.
    """
    a, b = cfg.a, cfg.b
    fc = cfg.f_cutoff
    kc = 2 * np.pi * fc / C0_LOCAL
    kc_sq = kc ** 2

    best_err = float('inf')
    best_mn = (0, 0)
    for m in range(0, 8):
        for n in range(0, 8):
            if m == 0 and n == 0:
                continue
            kc_sq_an = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
            err = abs(kc_sq - kc_sq_an) / max(kc_sq_an, 1e-30)
            if err < best_err:
                best_err = err
                best_mn = (m, n)
    if best_err < 0.1:
        return best_mn
    return (0, 0)


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
    h_u_sim = _plane_h_field(getattr(state, cfg.h_u_component), cfg, x_idx)
    h_v_sim = _plane_h_field(getattr(state, cfg.h_v_component), cfg, x_idx)

    # P1 = ∫(eu_sim * hv_mode - ev_sim * hu_mode) dA
    p1 = jnp.sum((e_u_sim * cfg.hz_profile - e_v_sim * cfg.hy_profile) * dA)
    # P2 = ∫(eu_mode * hv_sim - ev_mode * hu_sim) dA
    p2 = jnp.sum((cfg.ey_profile * h_v_sim - cfg.ez_profile * h_u_sim) * dA)

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
    t = state.step * dt

    p1_ref, p2_ref = _overlap_cross_products(state, cfg, cfg.ref_x, dx)
    p1_probe, p2_probe = _overlap_cross_products(state, cfg, cfg.probe_x, dx)

    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)
    weight = _dft_window_weight(
        state.step, cfg.dft_total_steps, cfg.dft_window, cfg.dft_window_alpha
    )

    return OverlapDFTAccumulators(
        p1_ref_dft=acc.p1_ref_dft + p1_ref * phase * dt * weight,
        p2_ref_dft=acc.p2_ref_dft + p2_ref * phase * dt * weight,
        p1_probe_dft=acc.p1_probe_dft + p1_probe * phase * dt * weight,
        p2_probe_dft=acc.p2_probe_dft + p2_probe * phase * dt * weight,
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
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff)
    z_mode = _compute_mode_impedance(cfg.freqs, cfg.f_cutoff, cfg.mode_type)
    c_stored = mode_self_overlap(cfg, cfg.dx)

    # Convert cross-product DFTs to global forward/backward waves
    fwd_ref, bwd_ref = _overlap_to_waves(
        acc.p1_ref_dft, acc.p2_ref_dft, z_mode, c_stored
    )
    fwd_probe, bwd_probe = _overlap_to_waves(
        acc.p1_probe_dft, acc.p2_probe_dft, z_mode, c_stored
    )

    # Port-local wave mapping
    if cfg.direction.startswith("+"):
        a_inc_ref, a_out_ref = fwd_ref, bwd_ref
        a_inc_probe = fwd_probe
    else:
        a_inc_ref, a_out_ref = bwd_ref, fwd_ref
        a_inc_probe = bwd_probe

    # Apply reference-plane shifts
    a_inc_ref, a_out_ref = _shift_modal_waves(a_inc_ref, a_out_ref, beta, ref_shift)
    a_inc_probe, _ = _shift_modal_waves(
        a_inc_probe,
        jnp.zeros_like(a_inc_probe),
        beta,
        probe_shift,
    )

    safe_ref = jnp.where(
        jnp.abs(a_inc_ref) > 0, a_inc_ref, jnp.ones_like(a_inc_ref)
    )
    s11 = a_out_ref / safe_ref
    s21 = a_inc_probe / safe_ref

    return s11, s21
