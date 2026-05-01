"""Microstrip line (MSL) port: 2D distributed port with 3-probe de-embedding.

Unlike the 1-cell-transverse ``WirePort``, the MSL port covers the full
trace cross-section (y × z under the trace) and distributes the total
port impedance Z0 as conductivity over the cross-section cells. After
the FDTD run, three downstream probe planes are used in an OpenEMS-style
3-probe recurrence to extract the propagation constant β, characteristic
impedance Z0, and the reflection coefficient at the reference plane.

The math is intentionally numpy-only: extraction runs once per port,
post-simulation, on small per-frequency arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0


# ---------------------------------------------------------------------------
# Port description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSLPort:
    """Microstrip line port spanning the full trace cross-section.

    Parameters
    ----------
    feed_x : float
        Feed-plane x-coordinate (metres) where the source and termination
        are placed.
    y_lo, y_hi : float
        Trace lateral extent (metres). ``y_hi - y_lo`` is the trace width.
    z_lo, z_hi : float
        Substrate vertical extent (metres). ``z_lo`` is typically the
        ground plane / substrate bottom; ``z_hi`` the top of the
        substrate (where the trace lies).
    direction : "+x" or "-x"
        Direction the launched wave propagates away from the feed plane.
    impedance : float
        Target characteristic impedance Z0 in ohms (used to set σ).
    excitation : callable or None
        Source waveform ``f(t) -> amplitude`` (e.g. ``GaussianPulse``).
        ``None`` for a passive matched port.
    """

    feed_x: float
    y_lo: float
    y_hi: float
    z_lo: float
    z_hi: float
    direction: str
    impedance: float
    excitation: object = None


# ---------------------------------------------------------------------------
# Cross-section helpers
# ---------------------------------------------------------------------------


def _axis_cell_size(grid, axis: str, idx: int) -> float:
    """Return the cell size at index ``idx`` along ``axis``.

    Supports both uniform Grid and NonUniformGrid via duck typing.
    """
    profile_attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
    profile = getattr(grid, profile_attr, None)
    if profile is not None:
        try:
            n = int(profile.shape[0])
        except Exception:
            return float(getattr(grid, axis if axis != "x" else "dx", grid.dx))
        clamped = max(0, min(idx, n - 1))
        return float(profile[clamped])
    # Fallback for axis-specific scalar (Grid doesn't carry .dy/.dz today)
    return float(getattr(grid, axis if axis != "x" else "dx", grid.dx))


def _msl_yz_cells(grid, port: MSLPort) -> list[tuple[int, int, int]]:
    """Return the (i, j, k) grid indices spanning the MSL cross-section.

    ``i`` is the feed-plane x index. ``j`` ranges over the y-cells from
    ``port.y_lo`` to ``port.y_hi`` (inclusive); ``k`` likewise over z.
    """
    i_feed, j_lo, k_lo = grid.position_to_index((port.feed_x, port.y_lo, port.z_lo))
    _, j_hi, k_hi = grid.position_to_index((port.feed_x, port.y_hi, port.z_hi))
    j_a, j_b = (j_lo, j_hi) if j_lo <= j_hi else (j_hi, j_lo)
    k_a, k_b = (k_lo, k_hi) if k_lo <= k_hi else (k_hi, k_lo)
    cells = []
    for j in range(j_a, j_b + 1):
        for k in range(k_a, k_b + 1):
            cells.append((int(i_feed), int(j), int(k)))
    return cells


# ---------------------------------------------------------------------------
# Material setup: distribute Z0 as σ over the cross-section
# ---------------------------------------------------------------------------


def setup_msl_port(grid, port: MSLPort, materials):
    """Fold port impedance Z0 into σ over the MSL cross-section cells.

    Cells stacked in z are in series (voltage adds), cells in y are in
    parallel (current adds). For total impedance Z0::

        σ_cell = (N_z · dz_cell) / (Z0 · N_y · dx_cell · dy_cell)

    Returns the updated ``materials`` NamedTuple.
    """
    cells = _msl_yz_cells(grid, port)
    if not cells:
        return materials
    j_set = sorted({c[1] for c in cells})
    k_set = sorted({c[2] for c in cells})
    n_y = len(j_set)
    n_z = len(k_set)

    sigma = materials.sigma
    for (i, j, k) in cells:
        dx_cell = _axis_cell_size(grid, "x", i)
        dy_cell = _axis_cell_size(grid, "y", j)
        dz_cell = _axis_cell_size(grid, "z", k)
        sigma_cell = (n_z * dz_cell) / (port.impedance * n_y * dx_cell * dy_cell)
        sigma = sigma.at[i, j, k].add(sigma_cell)
    return materials._replace(sigma=sigma)


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScaledWaveform:
    """Wrap a base waveform and scale its output by a constant."""

    base: object
    scale: float

    def __call__(self, t):
        return self.base(t) * self.scale


def make_msl_port_sources(grid, port: MSLPort, materials, n_steps):
    """Build the SourceSpec list for an MSL feed plane.

    Each cell in the cross-section gets an Ez source with amplitude
    ``V_src / N_z`` (voltage division along z), Cb-normalised so that
    the source enters the Ampere update through the ``Cb·J`` term.

    The port impedance must already be folded into ``materials`` via
    :func:`setup_msl_port`.
    """
    if port.excitation is None:
        return []
    from rfx.simulation import SourceSpec  # local import: avoid cycles

    cells = _msl_yz_cells(grid, port)
    if not cells:
        return []
    n_z = len({c[2] for c in cells})
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    base_wave = jax.vmap(port.excitation)(times)

    specs = []
    for (i, j, k) in cells:
        eps = materials.eps_r[i, j, k] * EPS_0
        sigma = materials.sigma[i, j, k]
        loss = sigma * grid.dt / (2.0 * eps)
        cb = (grid.dt / eps) / (1.0 + loss)
        d_par = _axis_cell_size(grid, "z", k)
        waveform = (cb / d_par) * base_wave / float(n_z)
        specs.append(SourceSpec(i=i, j=j, k=k, component="ez", waveform=waveform))
    return specs


# ---------------------------------------------------------------------------
# Probe-plane locations
# ---------------------------------------------------------------------------


def msl_probe_x_coords(
    grid,
    port: MSLPort,
    n_offset_cells: int = 5,
    n_spacing_cells: int = 3,
) -> tuple[float, float, float]:
    """Return three downstream probe x-coordinates for 3-probe extraction.

    The first probe is ``n_offset_cells`` cells from the feed plane; the
    remaining two are spaced by ``n_spacing_cells`` further along the
    propagation direction.  Indices are clamped into the valid grid
    range so callers always receive in-domain physical coordinates.
    """
    i_feed, _, _ = grid.position_to_index((port.feed_x, port.y_lo, port.z_lo))
    sign = 1 if port.direction == "+x" else -1
    nx = grid.nx

    def _x_for_index(target_i: int) -> float:
        clamped = max(0, min(target_i, nx - 1))
        # User-domain physical x-coord. position_to_index adds pad_x_lo, so
        # subtract it back when rebuilding the coordinate.
        pad = getattr(grid, "pad_x_lo", 0)
        return float((clamped - pad) * grid.dx)

    i1 = i_feed + sign * n_offset_cells
    i2 = i1 + sign * n_spacing_cells
    i3 = i2 + sign * n_spacing_cells
    return _x_for_index(i1), _x_for_index(i2), _x_for_index(i3)


# ---------------------------------------------------------------------------
# 3-probe S-parameter extraction
# ---------------------------------------------------------------------------


def _integrate_v(ez_plane: np.ndarray, j_center: int, z_lo_idx: int, z_hi_idx: int,
                 dz_arr: np.ndarray) -> np.ndarray:
    """V(f) = ∫ Ez dz along the substrate height at y = j_center."""
    total = np.zeros(ez_plane.shape[0], dtype=complex)
    for k in range(z_lo_idx, z_hi_idx + 1):
        total = total + ez_plane[:, j_center, k] * float(dz_arr[k])
    return total


def _integrate_i(hy_plane: np.ndarray, y_lo_idx: int, y_hi_idx: int, z_top_idx: int,
                 dy_arr: np.ndarray) -> np.ndarray:
    """I(f) = ∫ Hy dy across the trace width at z = z_top_idx."""
    total = np.zeros(hy_plane.shape[0], dtype=complex)
    for j in range(y_lo_idx, y_hi_idx + 1):
        total = total + hy_plane[:, j, z_top_idx] * float(dy_arr[j])
    return total


def extract_msl_s_params(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    i1: np.ndarray,
    *,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-probe de-embedding for V/I on a transmission line.

    Given V at three equally-spaced probe planes (separation Δ) and I at
    the first plane, recover the reflection coefficient ``S11`` at probe 1,
    the characteristic impedance ``Z0``, and the per-step phasor
    ``q = exp(-jβΔ)`` (positive root with ``|q| ≤ 1``).

    Parameters
    ----------
    v1, v2, v3 : (n_freqs,) complex
        DFT'd voltage at three probe planes spaced by Δ along the line.
    i1 : (n_freqs,) complex
        DFT'd current at probe plane 1.

    Returns
    -------
    s11, z0, q : (n_freqs,) complex
        ``s11`` is the reflection coefficient at probe 1; ``z0`` the
        extracted characteristic impedance; ``q`` the per-Δ phasor.
    """
    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    v3 = np.asarray(v3, dtype=complex)
    i1 = np.asarray(i1, dtype=complex)

    s11, z0, q = _solve_3probe(v1, v2, v3, i1, eps)
    return s11, z0, q


def _solve_3probe(v1, v2, v3, i1, eps):
    """Closed-form 3-probe solver shared by extract_msl_s_params/forward."""
    # q + 1/q = (V1 + V3) / V2  →  q² − coeff·q + 1 = 0
    coeff = (v1 + v3) / (v2 + eps)
    disc = coeff**2 - 4.0 + 0j
    sqrt_disc = np.sqrt(disc)
    q_plus = (coeff + sqrt_disc) / 2.0
    q_minus = (coeff - sqrt_disc) / 2.0

    # Both roots are reciprocals (q_minus = 1/q_plus), so on a lossless
    # line they have |q|=1 exactly and the |q|≤1 selector becomes
    # ambiguous.  The physical forward root must reproduce the observed
    # forward step ratio V2/V1 in the absence of strong reflection; we
    # therefore pick the root whose phase is closer to V2/V1.
    ratio = v2 / (v1 + eps)
    err_plus = np.abs(q_plus - ratio)
    err_minus = np.abs(q_minus - ratio)
    # Tie-breaker: |q| ≤ 1 (decaying) is preferred when both errors match.
    use_plus = (err_plus < err_minus) | (
        (np.isclose(err_plus, err_minus)) & (np.abs(q_plus) <= np.abs(q_minus))
    )
    q = np.where(use_plus, q_plus, q_minus)

    # Forward (alpha) and backward (gamma) wave amplitudes at probe 1
    denom = (q * q - 1.0) + eps
    alpha = (q * v2 - v1) / denom
    gamma = q * (v1 * q - v2) / denom

    z0 = (alpha - gamma) / (i1 + eps)
    s11 = gamma / (alpha + eps)
    return s11, z0, q


def msl_forward_amplitude(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    *,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha, q)``: forward amplitude at probe 1 and per-Δ phasor.

    Useful on the passive (non-driven) port where only the voltage triple
    is needed to recover the transmitted forward wave.
    """
    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    v3 = np.asarray(v3, dtype=complex)
    # Reuse the shared solver (i1 unused for q/alpha — pass v1 as a
    # placeholder; z0/s11 outputs are discarded).
    _, _, q = _solve_3probe(v1, v2, v3, v1, eps)
    denom = (q * q - 1.0) + eps
    alpha = (q * v2 - v1) / denom
    return alpha, q


def compute_s21(alpha_passive: np.ndarray, alpha_driven: np.ndarray,
                *, eps: float = 1e-30) -> np.ndarray:
    """S21 from forward amplitudes on driven (port 1) and passive (port 2)."""
    return np.asarray(alpha_passive, dtype=complex) / (
        np.asarray(alpha_driven, dtype=complex) + eps
    )
