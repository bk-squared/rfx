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
    # ``"differentiated_gaussian"`` (legacy, src = -2·arg·exp(-arg²)) or
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
    # profiles × real cell areas). Stored as float32 to halve memory
    # vs complex; OpenEMS writes the same on disk as two floats per row.
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


def _pick_eigenmode_by_target_then_overlap(
    evecs: np.ndarray,
    evals: np.ndarray,
    analytic_flat: np.ndarray,
    target_kc2: float,
    num_candidates: int = 80,
) -> int:
    """Pick a mode by cutoff proximity, using overlap to break ties."""
    n = min(num_candidates, evecs.shape[1])
    target = max(float(target_kc2), 1e-30)
    rel = np.abs(evals[:n] - target) / target
    min_rel = float(np.min(rel))
    candidates = np.where(rel <= min_rel + 0.05)[0]
    if candidates.size == 1:
        return int(candidates[0])

    analytic_unit = analytic_flat / max(float(np.linalg.norm(analytic_flat)),
                                         1e-30)
    overlaps = np.abs(evecs[:, candidates].T @ analytic_unit)
    return int(candidates[int(np.argmax(overlaps))])


def _aperture_area(
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    aperture_dA: np.ndarray | None,
) -> np.ndarray:
    """Return the physical aperture integration measure used by extraction."""
    if aperture_dA is None:
        dA = np.asarray(u_widths, dtype=np.float64)[:, None] * np.asarray(
            v_widths, dtype=np.float64
        )[None, :]
    else:
        dA = np.asarray(aperture_dA, dtype=np.float64)
    if dA.shape != (len(u_widths), len(v_widths)):
        raise ValueError(
            "aperture_dA must have shape "
            f"({len(u_widths)}, {len(v_widths)}), got {dA.shape}"
        )
    flat = dA.ravel()
    positive = flat[flat > 0.0]
    if positive.size == 0:
        raise ValueError("aperture_dA must contain at least one positive cell")
    return dA


def _shift_profile_to_dual(
    profile: np.ndarray,
    h_offset: tuple[float, float],
) -> np.ndarray:
    """Linearly shift a transverse template by supported half-cell offsets."""
    out = np.asarray(profile, dtype=np.float64).copy()
    for axis, offset in enumerate(h_offset):
        if offset == 0.0:
            continue
        if offset != 0.5:
            raise ValueError(
                "h_offset currently supports only 0.0 or 0.5 per axis, "
                f"got {h_offset!r}"
            )
        rolled = np.roll(out, -1, axis=axis)
        out = 0.5 * (out + rolled)
    return out


def _orthonormalize_profile_arrays(
    ey: np.ndarray,
    ez: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    dA: np.ndarray,
    previous: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormalize one stored profile against previous profiles."""
    ey = np.asarray(ey, dtype=np.float64).copy()
    ez = np.asarray(ez, dtype=np.float64).copy()
    hy = np.asarray(hy, dtype=np.float64).copy()
    hz = np.asarray(hz, dtype=np.float64).copy()
    for pey, pez, phy, phz in previous:
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
        raise ValueError("Cannot normalize waveguide mode with zero self-overlap")
    scale = np.sqrt(abs(c_self))
    return ey / scale, ez / scale, hy / scale, hz / scale


def _scale_h_to_unit_cross(
    ey: np.ndarray,
    ez: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    dA: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale stored H templates so ∫(e×h)·n dA is unity."""
    c_self = float(np.sum((ey * hz - ez * hy) * dA))
    if abs(c_self) <= 1e-30:
        return hy, hz
    return hy / c_self, hz / c_self


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
    D_uu = _second_diff_1d(u_widths, bc="neumann")
    D_vv = _second_diff_1d(v_widths, bc="neumann")
    lap = np.kron(D_uu, np.eye(nv)) + np.kron(np.eye(nu), D_vv)

    dA = _aperture_area(u_widths, v_widths, aperture_dA)
    # Keep the physical Yee eigenfunctions/cutoffs on the full cell-centred
    # operator, then normalize and orthogonalize the stored profiles under the
    # extractor aperture. A reduced zero-dA subspace imposes an unintended
    # Dirichlet-like wall at the dropped row/column and breaks source matching.
    evals, evecs = np.linalg.eigh(-lap)

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
    D_uu = _second_diff_1d(u_widths, bc="dirichlet")
    D_vv = _second_diff_1d(v_widths, bc="dirichlet")
    lap = np.kron(D_uu, np.eye(nv)) + np.kron(np.eye(nu), D_vv)

    dA = _aperture_area(u_widths, v_widths, aperture_dA)
    evals, evecs = np.linalg.eigh(-lap)

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
    h_offset: tuple[float, float] = (0.5, 0.5),
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
    # Duck-type: if `dx` looks like a NU grid (has dx_arr / dz arrays),
    # use per-axis widths slicing the aperture; else assume scalar dx.
    # When the explicit ``grid=`` kwarg is supplied (uniform Grid path),
    # use it for boundary detection even though dx is still a scalar.
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
    # compromise (docs/research_notes/2026-04-26_phase2_aperture_weight_dead_end.md
    # Phase 2 dead-end) that protected mesh-conv and asymmetric-obstacle
    # reciprocity at the cost of capping PEC-short closure. With DROP,
    # one mesh-conv |S21| test currently regresses (xfail-locked in
    # tests/test_waveguide_port_validation_battery.py); the receive-side
    # multi-mode extractor (per the dead-end note's recommendation) is the
    # path to recovering that without re-introducing the PEC-short cap.
    #
    # Detection: +face is PEC iff axis fully PEC (not in cpml_axes) OR per-
    # face spec marks it PEC. Only fires when slice reaches the grid edge.
    u_aperture_weights = np.ones_like(u_widths_np)
    v_aperture_weights = np.ones_like(v_widths_np)
    if boundary_grid is not None and u_axis_name is not None:
        cpml_axes = getattr(boundary_grid, "cpml_axes", "")
        pec_faces = getattr(boundary_grid, "pec_faces", set()) or set()
        u_hi_face_pec = (u_axis_name not in cpml_axes) or (
            f"{u_axis_name}_hi" in pec_faces)
        v_hi_face_pec = (v_axis_name not in cpml_axes) or (
            f"{v_axis_name}_hi" in pec_faces)
        if u_hi == u_grid_size and u_hi_face_pec and u_widths_np.size > 0:
            u_aperture_weights[-1] = 0.0
        if v_hi == v_grid_size and v_hi_face_pec and v_widths_np.size > 0:
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
        dt=float(dt),
        u_widths=jnp.asarray(u_widths_np, dtype=jnp.float32),
        v_widths=jnp.asarray(v_widths_np, dtype=jnp.float32),
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
        aperture_dA=jnp.asarray(aperture_dA_np, dtype=jnp.float32),
        h_offset=(float(h_offset[0]), float(h_offset[1])),
        mode_indices=(int(m), int(n)),
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


def _plane_h_field_at_dual(
    field,
    cfg: WaveguidePortConfig,
    plane_index: int,
    h_offset: tuple[float, float],
):
    """Extract H on the normal-averaged plane and optional transverse dual mesh."""
    base = _plane_h_field(field, cfg, plane_index)
    if h_offset == (0.0, 0.0):
        return base
    out = base
    for axis, offset in enumerate(h_offset):
        if offset == 0.0:
            continue
        if offset != 0.5:
            raise ValueError(
                "h_offset currently supports only 0.0 or 0.5 per axis, "
                f"got {h_offset!r}"
            )
        out = 0.5 * (out + jnp.roll(out, -1, axis=axis))
    return out


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
                            step, dt: float, dx: float):
    """TFSF-style H correction at the port's upstream half-cell.

    Half of the one-sided TFSF boundary for a waveguide-mode source.
    Reads the bandpass-filtered source amplitude from ``cfg.e_inc_table``
    (not the raw differentiated Gaussian) so sub-cutoff DC content — which
    cannot propagate in the guide anyway — never reaches the 3D grid.

    Same pattern as ``apply_tfsf_h`` in ``rfx/sources/tfsf.py``.

    When ``cfg.e_inc_table`` is empty (init was called without ``dt``),
    the TFSF pair cannot be computed — this function is a no-op and
    ``apply_waveguide_port_e`` handles source emission via the legacy
    soft-E path instead. This keeps low-level callers that predate the
    TFSF pair working without API changes.
    """
    table = cfg.e_inc_table
    table_size = table.shape[0]
    if table_size <= 1:
        # Tables unavailable — delegate source emission to the legacy
        # soft-E fallback inside apply_waveguide_port_e. No-op here.
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
                            step, dt: float, dx: float):
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
    h_u_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset,
    )
    h_v_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset,
    )
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
    h_u_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset,
    )
    h_v_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset,
    )

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
    """Record per-step modal V and I at ref and probe planes.

    The S-parameter spectra are computed POST-SCAN by
    ``_extract_global_waves_from_time_series`` via a rectangular
    full-record DFT on the recorded arrays. No in-scan windowed DFT
    accumulators are kept (Phase 2 cleanup, 2026-04-25). The previous
    in-scan windowed/gated path silently integrated partially-decayed
    standing waves and produced ``|S11| → ∞`` as the gate widened.
    """
    t = state.step * dt

    v_ref = modal_voltage(state, cfg, cfg.ref_x, dx)
    v_probe = modal_voltage(state, cfg, cfg.probe_x, dx)
    i_ref = modal_current(state, cfg, cfg.ref_x, dx)
    i_probe = modal_current(state, cfg, cfg.probe_x, dx)

    arg = (t - cfg.src_t0) / cfg.src_tau
    v_inc = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    # Write per-step modal V/I into the time-series carry. The carry
    # length is fixed at trace time (= dft_total_steps); ``state.step``
    # is a JAX scalar — clip to avoid out-of-bounds writes if the
    # simulation runs longer than the allocated record buffer (XLA
    # dynamic-update-slice clamps anyway, but be explicit).
    #
    # Also stamp ``cfg.dt`` from the scan's authoritative ``dt`` so the
    # post-scan rect DFT (``_extract_global_waves_from_time_series``)
    # uses the right ``Δt`` even when the caller initialised the cfg
    # via the low-level ``init_waveguide_port`` without passing ``dt=``
    # (legacy path used by ``tests/test_waveguide_port.py`` Python loops).
    n_t = cfg.v_probe_t.shape[0]
    safe_step = jnp.clip(jnp.asarray(state.step, dtype=jnp.int32), 0, n_t - 1)
    return cfg._replace(
        dt=float(dt),
        v_probe_t=cfg.v_probe_t.at[safe_step].set(
            jnp.asarray(v_probe, cfg.v_probe_t.dtype)),
        v_ref_t=cfg.v_ref_t.at[safe_step].set(
            jnp.asarray(v_ref, cfg.v_ref_t.dtype)),
        i_probe_t=cfg.i_probe_t.at[safe_step].set(
            jnp.asarray(i_probe, cfg.i_probe_t.dtype)),
        i_ref_t=cfg.i_ref_t.at[safe_step].set(
            jnp.asarray(i_ref, cfg.i_ref_t.dtype)),
        v_inc_t=cfg.v_inc_t.at[safe_step].set(
            jnp.asarray(v_inc, cfg.v_inc_t.dtype)),
        n_steps_recorded=jnp.maximum(cfg.n_steps_recorded, safe_step + 1),
    )


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
        # Evanescent branch: keep analytic imaginary form — these are not
        # propagated and their magnitude is not used for plane-shift.
        beta_evan = 1j * jnp.sqrt(jnp.maximum(-s_x_sq, 0.0)) * C0_LOCAL
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


def _co_located_current_spectrum(
    cfg: WaveguidePortConfig,
    current_dft: jnp.ndarray,
) -> jnp.ndarray:
    """Correct the H-derived DFT for Yee leapfrog time staggering.

    The waveguide-port probe runs at the end of a scan-body iteration,
    after `update_h` and `update_e`. At that point `state.step = n+1`,
    `E` is at time ``(n+1)·dt`` and `H` is at ``(n+1/2)·dt``. The
    accumulator pairs both samples with phase ``exp(-jω·(n+1)·dt)``, so
    the raw I DFT carries an extra factor ``exp(-jω·dt/2)`` relative to
    the true DFT of H at its own timestamp. To recover the true DFT we
    multiply by ``exp(+jω·dt/2)`` — applied here before any modal
    decomposition so V and I are time-aligned at the E timestep.

    Before 2026-04-22 the sign here was `-jω·dt/2`, which amplified the
    very error it was supposed to cancel. The observable symptom was a
    residual imaginary component of `V/I` on an empty guide (mean
    ``∠(Z_formula / Z_actual)`` ≈ −8°). The current sign drops that to
    ≈ −1° on the same diagnostic (`scripts/isolate_extractor_vs_engine.py`).
    """
    if cfg.dt <= 0.0:
        return current_dft
    omega = 2 * jnp.pi * cfg.freqs
    return current_dft * jnp.exp(+1j * omega * (0.5 * cfg.dt))


def _extract_global_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return global (+x/-x) modal waves from colocated modal V/I spectra."""
    z_mode = _compute_mode_impedance(
        cfg.freqs,
        cfg.f_cutoff,
        cfg.mode_type,
        dt=cfg.dt,
        dx=cfg.dx,
    )
    current_dft = _co_located_current_spectrum(cfg, current_dft)
    forward = 0.5 * (voltage_dft + z_mode * current_dft)
    backward = 0.5 * (voltage_dft - z_mode * current_dft)
    return forward, backward


def _rect_dft(time_series: jnp.ndarray, freqs: jnp.ndarray, dt: float,
              n_valid: jnp.ndarray | int) -> jnp.ndarray:
    """Rectangular full-record DFT matching OpenEMS ``utilities.DFT_time2freq``.

    Computes the single-sided spectrum::

        V(f) = 2 · dt · Σ_n V(t_n) · exp(-j 2π f t_n)

    where ``t_n = n · dt`` and the sum runs over ``n in [0, n_valid)``.
    No window function, no time gate beyond ``n_valid`` — the integrand
    is the literal recorded time series. For a finite-energy waveform
    (transient pulse + decaying ringdown) this Fourier transform is
    finite at every frequency; for a non-decaying standing wave the
    integral grows with N, which is the architectural failure mode the
    in-scan windowed accumulator silently exhibits.

    Reference: ``/usr/lib/python3/dist-packages/openEMS/utilities.py`` lines 21-35.
    """
    n = jnp.arange(time_series.shape[0])
    t = n.astype(time_series.dtype) * jnp.asarray(dt, dtype=time_series.dtype)
    mask = (n < n_valid).astype(time_series.dtype)
    masked_v = time_series * mask
    omega = 2.0 * jnp.pi * freqs.astype(time_series.dtype)
    # Build the DFT outer product: phase shape (n_steps, n_freqs).
    # complex dtype follows the freqs precision (complex64 by default,
    # complex128 under JAX_ENABLE_X64).
    phase = jnp.exp(-1j * omega[None, :] * t[:, None])
    return 2.0 * jnp.asarray(dt, dtype=phase.dtype) * jnp.einsum(
        "n,nf->f", masked_v.astype(phase.dtype), phase
    )


def _extract_global_waves_from_time_series(
    cfg: WaveguidePortConfig,
    voltage_t: jnp.ndarray,
    current_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Post-scan: rect-DFT the time series, then run the standard wave decomp.

    Mirrors ``_extract_global_waves`` but takes raw time-series inputs
    and computes the DFT in post-processing — eliminating the in-scan
    windowed/gated accumulator path that integrates partially-decayed
    standing waves and produces ``|S11| → ∞`` as the gate widens.
    """
    n_valid = cfg.n_steps_recorded
    voltage_dft = _rect_dft(voltage_t, cfg.freqs, cfg.dt, n_valid)
    current_dft = _rect_dft(current_t, cfg.freqs, cfg.dt, n_valid)
    return _extract_global_waves(cfg, voltage_dft, current_dft)


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


def _extract_port_waves_from_time_series(
    cfg: WaveguidePortConfig,
    voltage_t: jnp.ndarray,
    current_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time-series variant of ``_extract_port_waves`` (post-scan rect DFT)."""
    forward, backward = _extract_global_waves_from_time_series(
        cfg, voltage_t, current_t
    )
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
    step_sign: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shift modal waves to a new reference plane.

    `shift_m` is in GLOBAL coordinates along the port normal axis. For a
    "+x" port (``step_sign=+1``) `shift_m > 0` moves the reporting plane
    downstream (local-forward). For a "-x" port (``step_sign=-1``) the
    local-forward direction is -x in global coords, so the incident-wave
    phase flips: we convert to local via ``local_shift = shift_m *
    step_sign`` and apply the same formula.

    Before 2026-04-22 the function ignored ``step_sign`` and applied
    the "+x" formula unconditionally — this silently gave the wrong sign
    on any negative-direction port, which is why user-specified
    ``reference_plane`` shifts produced only half-corrected phases for
    "-x"/"-y"/"-z" ports and the crossval/11 phase offset never closed.
    """
    if shift_m == 0.0:
        return forward, backward
    local_shift = shift_m * step_sign
    shift = jnp.asarray(local_shift, dtype=beta.dtype)
    forward_shifted = forward * jnp.exp(-1j * beta * shift)
    backward_shifted = backward * jnp.exp(+1j * beta * shift)
    return forward_shifted, backward_shifted


def extract_waveguide_sparams(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
    probe_shift: float = 0.0,
    v_ref_dft: jnp.ndarray | None = None,
    i_ref_dft: jnp.ndarray | None = None,
    v_probe_dft: jnp.ndarray | None = None,
    i_probe_dft: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (S11, S21) with optional reference-plane shifts.

    The Phase 2 cleanup (2026-04-25) removed the in-scan DFT accumulators
    (``cfg.v_ref_dft`` etc.) from ``WaveguidePortConfig``. Callers that
    used to read those fields off the config now must either:

    1. Pass V/I DFT spectra explicitly via the ``*_dft`` keyword args
       (used by direct unit-test mock paths and by callers that have
       precomputed the spectra outside the FDTD scan); OR
    2. Use ``extract_waveguide_port_waves`` /
       ``extract_waveguide_s_matrix`` which run the canonical post-scan
       rectangular full-record DFT on ``cfg.v_ref_t`` / ``cfg.i_ref_t``.

    When the explicit DFTs are not provided this function falls back to
    the post-scan rect-DFT path on the recorded time series.

    Parameters
    ----------
    ref_shift : float
        Metres to shift the reference-plane reporting location relative to the
        stored reference probe. Positive is downstream (+x), negative upstream.
    probe_shift : float
        Metres to shift the probe-plane reporting location relative to the
        stored probe plane. Positive is downstream (+x), negative upstream.
    v_ref_dft, i_ref_dft, v_probe_dft, i_probe_dft : array or None
        Optional precomputed V/I spectra at the reference and probe
        planes. When all are given the wave decomposition runs against
        them directly; otherwise the recorded time-series on ``cfg`` is
        used.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx)
    step_sign = 1 if cfg.direction.startswith("+") else -1
    explicit = (
        v_ref_dft is not None and i_ref_dft is not None
        and v_probe_dft is not None and i_probe_dft is not None
    )
    if explicit:
        a_ref, b_ref = _extract_port_waves(cfg, v_ref_dft, i_ref_dft)
        a_probe, b_probe = _extract_port_waves(cfg, v_probe_dft, i_probe_dft)
    else:
        a_ref, b_ref = _extract_port_waves_from_time_series(
            cfg, cfg.v_ref_t, cfg.i_ref_t,
        )
        a_probe, b_probe = _extract_port_waves_from_time_series(
            cfg, cfg.v_probe_t, cfg.i_probe_t,
        )
    a_ref, b_ref = _shift_modal_waves(a_ref, b_ref, beta, ref_shift, step_sign)
    a_probe, b_probe = _shift_modal_waves(a_probe, b_probe, beta, probe_shift, step_sign)
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
    """Return port-local (incident, outgoing) waves at a shifted reference plane.

    The spectra are recomputed POST-SCAN from the per-step modal V/I
    time series via a rectangular full-record DFT (matching OpenEMS's
    ``utilities.DFT_time2freq``). The legacy in-scan windowed/gated
    DFT accumulators were removed in the Phase 2 cleanup
    (2026-04-25): they integrated partially-decayed standing waves
    and produced ``|S11|`` that grew without bound as the gate widened.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx)
    step_sign = 1 if cfg.direction.startswith("+") else -1
    a_ref, b_ref = _extract_port_waves_from_time_series(
        cfg, cfg.v_ref_t, cfg.i_ref_t,
    )
    return _shift_modal_waves(a_ref, b_ref, beta, ref_shift, step_sign)


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
    aniso_eps: tuple | None = None,
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
        zeros_t = jnp.zeros_like(cfg.v_probe_t)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
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
            aniso_eps=aniso_eps,
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
    aniso_eps: tuple | None = None,
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
           S_jj = (b_out_device[j] - b_out_reference[j]) / a_inc_reference[j]
         The subtraction removes the empty-guide CPML/discretization
         contribution at the driven port so S_jj measures only the
         device-induced reflection. Spectra are computed POST-SCAN by
         a rectangular full-record DFT on the recorded modal V/I
         time series (Phase 2 cleanup, 2026-04-25); strong-reflector
         cases no longer suffer the |S11| → ∞ growth that the legacy
         in-scan windowed accumulator produced as the gate widened.

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
        zeros_t = jnp.zeros_like(cfg.v_probe_t)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    common_run_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
    )
    # ``aniso_eps`` is per-component smoothed permittivity for the device
    # geometry. The reference run is vacuum and has no ε interfaces, so
    # it always passes aniso_eps=None — the two-run cancellation only
    # benefits if the smoothed ε is applied to the device run.

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
            waveguide_ports=dev_cfgs, aniso_eps=aniso_eps,
            **common_run_kw,
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
                # Diagonal: subtract the reference-run reflection so the
                # result captures only device-induced reflection.
                s_matrix[recv_idx, drive_idx, :] = (
                    b_recv_dev_np - b_out_ref[recv_idx]
                ) / safe_a_inc
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
# For DFT-based extraction the overlap integral computes its own
# cross-product DFTs (``OverlapDFTAccumulators``) because the two
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
    current) at the reference and probe planes. ``update_overlap_dft``
    accumulates a rect-window streaming DFT every step (Phase 2 cleanup
    removed the windowed/gated path that previously lived on
    ``WaveguidePortConfig``).
    """
    p1_ref_dft: jnp.ndarray     # (n_freqs,) complex — ∫(E_sim × h_mode)·n̂ dA at ref
    p2_ref_dft: jnp.ndarray     # (n_freqs,) complex — ∫(e_mode × H_sim)·n̂ dA at ref
    p1_probe_dft: jnp.ndarray   # (n_freqs,) complex — ∫(E_sim × h_mode)·n̂ dA at probe
    p2_probe_dft: jnp.ndarray   # (n_freqs,) complex — ∫(e_mode × H_sim)·n̂ dA at probe


def init_overlap_dft(freqs: jnp.ndarray) -> OverlapDFTAccumulators:
    """Initialize zero-valued overlap DFT accumulators."""
    nf = len(freqs)
    # Canonical complex dtype — see init_waveguide_port for rationale.
    zeros_c = jnp.zeros(nf, dtype=jnp.array(0j).dtype)
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
    dt: float = 0.0,
    waveform: str = "modulated_gaussian",
    mode_profile: str = "discrete",
    grid=None,
    h_offset: tuple[float, float] = (0.5, 0.5),
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
            grid=grid, h_offset=h_offset,
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
            grid=grid, h_offset=h_offset,
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

    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(mode_cfgs):
            flat_cfgs.append(cfg)
            # Extract mode info from the config
            m_idx = _mode_indices_from_config(cfg)
            mode_map.append((port_idx, mode_within, cfg.mode_type, m_idx))

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
        zeros_t = jnp.zeros_like(cfg.v_probe_t)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
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


def extract_multimode_s_params_normalized(
    grid,
    materials,
    ref_materials,
    port_mode_cfgs: list[list[WaveguidePortConfig]],
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
    aniso_eps: tuple | None = None,
) -> tuple[jnp.ndarray, list[tuple[int, int, str, tuple[int, int]]]]:
    """Two-run normalized S-matrix for multi-mode waveguide ports.

    The row/column ordering matches :func:`extract_multimode_s_matrix`.
    Each drive mode gets its own empty-guide reference run. Channels with a
    nonzero empty-guide through response are normalized by that reference
    outgoing wave; reflection and mode-conversion channels subtract the
    reference outgoing wave and normalize by the driven incident wave.
    """
    from rfx.simulation import run as run_simulation

    flat_cfgs: list[WaveguidePortConfig] = []
    mode_map: list[tuple[int, int, str, tuple[int, int]]] = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        for mode_within, cfg in enumerate(mode_cfgs):
            flat_cfgs.append(cfg)
            mode_map.append((
                port_idx,
                mode_within,
                cfg.mode_type,
                _mode_indices_from_config(cfg),
            ))

    if len(flat_cfgs) < 2:
        raise ValueError(
            "extract_multimode_s_params_normalized requires at least two modal channels"
        )

    n_total = len(flat_cfgs)
    n_freqs = len(flat_cfgs[0].freqs)
    s_matrix = np.zeros((n_total, n_total, n_freqs), dtype=np.complex64)

    n_ports = len(port_mode_cfgs)
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in range(n_ports))
    if len(ref_shifts) != n_ports:
        raise ValueError("ref_shifts must have one entry per physical port")
    flat_ref_shifts: list[float] = []
    for port_idx, mode_cfgs in enumerate(port_mode_cfgs):
        flat_ref_shifts.extend([float(ref_shifts[port_idx])] * len(mode_cfgs))

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros_t = jnp.zeros_like(cfg.v_probe_t)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_t=zeros_t,
            v_ref_t=zeros_t,
            i_probe_t=zeros_t,
            i_ref_t=zeros_t,
            v_inc_t=zeros_t,
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        )

    common_run_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
    )
    orig_amp = flat_cfgs[0].src_amp if flat_cfgs[0].src_amp != 0.0 else 1.0

    for drive_flat_idx in range(n_total):
        ref_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        ref_cfgs[drive_flat_idx] = ref_cfgs[drive_flat_idx]._replace(
            src_amp=orig_amp,
        )
        ref_result = run_simulation(
            grid, ref_materials, n_steps,
            debye=ref_debye, lorentz=ref_lorentz,
            waveguide_ports=ref_cfgs, **common_run_kw,
        )
        ref_final_cfgs = ref_result.waveguide_ports or ()
        if len(ref_final_cfgs) != n_total:
            raise RuntimeError(
                f"Expected {n_total} reference waveguide configs, got {len(ref_final_cfgs)}"
            )

        a_inc_ref, _ = extract_waveguide_port_waves(
            ref_final_cfgs[drive_flat_idx],
            ref_shift=flat_ref_shifts[drive_flat_idx],
        )
        a_inc_ref_np = np.array(a_inc_ref)
        safe_a_inc = np.where(
            np.abs(a_inc_ref_np) > 1e-30,
            a_inc_ref_np,
            np.ones_like(a_inc_ref_np),
        )

        b_out_ref = []
        for recv_idx, cfg in enumerate(ref_final_cfgs):
            _, b_ref_i = extract_waveguide_port_waves(
                cfg,
                ref_shift=flat_ref_shifts[recv_idx],
            )
            b_out_ref.append(np.array(b_ref_i))

        dev_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_flat_idx))
            for idx, cfg in enumerate(flat_cfgs)
        ]
        dev_cfgs[drive_flat_idx] = dev_cfgs[drive_flat_idx]._replace(
            src_amp=orig_amp,
        )
        dev_result = run_simulation(
            grid, materials, n_steps,
            debye=debye, lorentz=lorentz,
            waveguide_ports=dev_cfgs, aniso_eps=aniso_eps,
            **common_run_kw,
        )
        dev_final_cfgs = dev_result.waveguide_ports or ()
        if len(dev_final_cfgs) != n_total:
            raise RuntimeError(
                f"Expected {n_total} device waveguide configs, got {len(dev_final_cfgs)}"
            )

        for recv_idx, cfg in enumerate(dev_final_cfgs):
            _, b_recv_dev = extract_waveguide_port_waves(
                cfg,
                ref_shift=flat_ref_shifts[recv_idx],
            )
            b_recv_dev_np = np.array(b_recv_dev)
            b_ref_np = b_out_ref[recv_idx]
            ref_nonzero = np.abs(b_ref_np) > 1e-30
            safe_b_ref = np.where(ref_nonzero, b_ref_np, np.ones_like(b_ref_np))

            with np.errstate(divide="ignore", invalid="ignore"):
                if recv_idx == drive_flat_idx:
                    s_matrix[recv_idx, drive_flat_idx, :] = (
                        b_recv_dev_np - b_ref_np
                    ) / safe_a_inc
                else:
                    through_norm = b_recv_dev_np / safe_b_ref
                    conversion_norm = (b_recv_dev_np - b_ref_np) / safe_a_inc
                    s_matrix[recv_idx, drive_flat_idx, :] = np.where(
                        ref_nonzero,
                        through_norm,
                        conversion_norm,
                    )

    return jnp.asarray(s_matrix), mode_map


def _mode_indices_from_config(cfg: WaveguidePortConfig) -> tuple[int, int]:
    """Best-effort extraction of (m, n) mode indices from a config.

    Uses the cutoff frequency and aperture dimensions to identify the mode.
    """
    if getattr(cfg, "mode_indices", (0, 0)) != (0, 0):
        return cfg.mode_indices
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
    if best_err < 0.35:
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
    h_u_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset,
    )
    h_v_sim = _plane_h_field_at_dual(
        getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset,
    )

    # Preserve the production V/I-equivalent DFT path even when the stored H
    # template is sampled on a transverse dual mesh (`h_offset != 0`).  In the
    # unshifted case these are exactly the Lorentz cross-products because
    # h_mode = n̂×e_mode; with dual-mesh H they remain the modal voltage/current
    # quantities consumed by `_overlap_to_waves`.
    p1 = jnp.sum((e_u_sim * cfg.ey_profile + e_v_sim * cfg.ez_profile) * dA)
    p2 = jnp.sum((h_u_sim * cfg.hy_profile + h_v_sim * cfg.hz_profile) * dA)

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

    # Rect-window streaming DFT (constant unit weight). The Phase 2
    # cleanup (2026-04-25) deleted the windowed/gated path on the V/I
    # extractor; the overlap path historically shared the same gating
    # knobs but those fields were also removed from
    # ``WaveguidePortConfig``. The S-param result is unchanged for runs
    # that previously used ``dft_window='rect'`` and no early gate.
    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)

    return OverlapDFTAccumulators(
        p1_ref_dft=acc.p1_ref_dft + p1_ref * phase * dt,
        p2_ref_dft=acc.p2_ref_dft + p2_ref * phase * dt,
        p1_probe_dft=acc.p1_probe_dft + p1_probe * phase * dt,
        p2_probe_dft=acc.p2_probe_dft + p2_probe * phase * dt,
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
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff, dt=cfg.dt, dx=cfg.dx)
    z_mode = _compute_mode_impedance(
        cfg.freqs,
        cfg.f_cutoff,
        cfg.mode_type,
        dt=cfg.dt,
        dx=cfg.dx,
    )
    c_stored = mode_self_overlap(cfg, cfg.dx)
    p2_ref = _co_located_current_spectrum(cfg, acc.p2_ref_dft)
    p2_probe = _co_located_current_spectrum(cfg, acc.p2_probe_dft)

    # Convert cross-product DFTs to global forward/backward waves
    fwd_ref, bwd_ref = _overlap_to_waves(
        acc.p1_ref_dft, p2_ref, z_mode, c_stored
    )
    fwd_probe, bwd_probe = _overlap_to_waves(
        acc.p1_probe_dft, p2_probe, z_mode, c_stored
    )

    # Port-local wave mapping
    if cfg.direction.startswith("+"):
        a_inc_ref, a_out_ref = fwd_ref, bwd_ref
        a_inc_probe = fwd_probe
    else:
        a_inc_ref, a_out_ref = bwd_ref, fwd_ref
        a_inc_probe = bwd_probe

    # Apply reference-plane shifts (global-frame shift; sign handled in _shift_modal_waves)
    step_sign = 1 if cfg.direction.startswith("+") else -1
    a_inc_ref, a_out_ref = _shift_modal_waves(a_inc_ref, a_out_ref, beta, ref_shift, step_sign)
    a_inc_probe, _ = _shift_modal_waves(
        a_inc_probe,
        jnp.zeros_like(a_inc_probe),
        beta,
        probe_shift,
        step_sign,
    )

    safe_ref = jnp.where(
        jnp.abs(a_inc_ref) > 0, a_inc_ref, jnp.ones_like(a_inc_ref)
    )
    s11 = a_out_ref / safe_ref
    s21 = a_inc_probe / safe_ref

    return s11, s21
