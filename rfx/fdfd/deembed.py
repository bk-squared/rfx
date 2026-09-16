"""Differentiable network-parameter conversions, de-embedding and inductor
metrics (``jax.numpy`` mirror of :mod:`rfx.deembed`).

Everything here is plain ``jax.numpy`` on complex128 arrays, so every
function can sit between a differentiable field solver (``rfx.fdfd``) and a
scalar objective and be differentiated in forward or reverse mode, jitted
and vmapped. Array layout follows :mod:`rfx.deembed`: an N-port network
parameter matrix is ``(n_ports, n_ports, n_freqs)``, ``freqs`` is ``(n_freqs,)``
in Hz. Functions whose names match :mod:`rfx.deembed` have the same
signature and return the same numbers on the same inputs (the tests check
this to 1e-12 against the NumPy versions).

Contents
--------
* ``s_to_z``, ``z_to_s``, ``s_to_y``, ``y_to_s`` -- general N-port with a
  real reference impedance per port; ``s_to_abcd`` / ``abcd_to_s`` for
  2-ports with a real reference impedance per port.
* ``deembed_port_extension``, ``deembed_thru``, ``deembed_series_impedance``,
  ``deembed_series_inductance``, ``deembed_line_segment`` -- the
  :mod:`rfx.deembed` methods.
* ``open_short_deembed`` -- the two-step open/short pad de-embedding used
  for on-chip inductor test structures.
* ``z_diff``, ``l_diff``, ``q_diff``, ``l_se``, ``q_se``, ``l_from_y11``,
  ``q_from_y11``, ``inductor_metrics`` -- inductance and quality factor of a
  2-port inductor from its Z (or Y) matrix.

Scope fence
-----------
Reference impedances are real (the usual 50 ohm case). Complex reference
impedances need the power-wave (Kurokawa) definition, which is not
implemented. Value-dependent branches that exist in :mod:`rfx.deembed`
(the eigen-decomposition fallback in the thru matrix square root, the
``zc > 0`` check in ``deembed_line_segment``) are not reproduced, because a
traced value cannot be branched on; those degenerate inputs produce
``inf``/``nan`` here rather than an alternative code path or a
``ValueError``. Precision: use x64 -- these routines invert near-singular
2x2 matrices at low frequency and float32 is not meaningful.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "s_to_z", "z_to_s", "s_to_y", "y_to_s", "s_to_abcd", "abcd_to_s",
    "deembed_port_extension", "deembed_thru", "deembed_series_impedance",
    "deembed_series_inductance", "deembed_line_segment",
    "open_short_deembed",
    "z_diff", "l_diff", "q_diff", "l_se", "q_se", "l_from_y11", "q_from_y11",
    "inductor_metrics",
]

# Speed of light in vacuum (m/s), same constant as rfx.deembed.
_C0 = 299_792_458.0


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _batch(m: jax.Array) -> jax.Array:
    """``(n, n, n_freqs)`` -> ``(n_freqs, n, n)`` complex128."""
    m = jnp.asarray(m, dtype=jnp.complex128)
    if m.ndim != 3 or m.shape[0] != m.shape[1]:
        raise ValueError(f"expected shape (n_ports, n_ports, n_freqs), got {m.shape}")
    return jnp.moveaxis(m, 2, 0)


def _unbatch(m: jax.Array) -> jax.Array:
    """``(n_freqs, n, n)`` -> ``(n, n, n_freqs)``."""
    return jnp.moveaxis(m, 0, 2)


def _port_z0(z0, n_ports: int) -> jax.Array:
    """Broadcast a scalar or ``(n_ports,)`` real reference impedance."""
    z = jnp.asarray(z0, dtype=jnp.float64)
    if z.ndim == 0:
        z = jnp.broadcast_to(z, (n_ports,))
    if z.shape != (n_ports,):
        raise ValueError(f"z0 has shape {z.shape}, expected () or ({n_ports},)")
    return z


def _eye_like(n: int) -> jax.Array:
    return jnp.eye(n, dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# N-port conversions
# ---------------------------------------------------------------------------

def s_to_z(s: jax.Array, z0=50.0) -> jax.Array:
    """Impedance matrix from a scattering matrix.

    With ``G = diag(sqrt(z0))`` (real ``z0`` per port) the normalised
    impedance is ``Zn = (I - S)^-1 (I + S)`` and ``Z = G Zn G``. Singular
    for ``S`` with a unit eigenvalue (an open circuit at a port).

    Parameters
    ----------
    s : array, shape (n_ports, n_ports, n_freqs)
    z0 : float or array, shape (n_ports,)
        Real reference impedance per port (ohm).
    """
    sb = _batch(s)
    n = sb.shape[-1]
    g = jnp.sqrt(_port_z0(z0, n)).astype(jnp.complex128)
    eye = _eye_like(n)
    zn = jnp.linalg.solve(eye - sb, eye + sb)
    return _unbatch(g[:, None] * zn * g[None, :])


def z_to_s(z: jax.Array, z0=50.0) -> jax.Array:
    """Scattering matrix from an impedance matrix (inverse of :func:`s_to_z`).

    ``Zn = G^-1 Z G^-1``, ``S = (Zn + I)^-1 (Zn - I)``; the two factors
    commute so the order does not matter.
    """
    zb = _batch(z)
    n = zb.shape[-1]
    gi = (1.0 / jnp.sqrt(_port_z0(z0, n))).astype(jnp.complex128)
    zn = gi[:, None] * zb * gi[None, :]
    eye = _eye_like(n)
    return _unbatch(jnp.linalg.solve(zn + eye, zn - eye))


def s_to_y(s: jax.Array, z0=50.0) -> jax.Array:
    """Admittance matrix from a scattering matrix.

    ``Yn = (I + S)^-1 (I - S)``, ``Y = G^-1 Yn G^-1``. Singular for ``S``
    with eigenvalue -1 (a short circuit at a port). Unlike inverting
    :func:`s_to_z`, this is finite for an open-circuited port.
    """
    sb = _batch(s)
    n = sb.shape[-1]
    gi = (1.0 / jnp.sqrt(_port_z0(z0, n))).astype(jnp.complex128)
    eye = _eye_like(n)
    yn = jnp.linalg.solve(eye + sb, eye - sb)
    return _unbatch(gi[:, None] * yn * gi[None, :])


def y_to_s(y: jax.Array, z0=50.0) -> jax.Array:
    """Scattering matrix from an admittance matrix (inverse of :func:`s_to_y`).

    ``Yn = G Y G``, ``S = (I + Yn)^-1 (I - Yn)``.
    """
    yb = _batch(y)
    n = yb.shape[-1]
    g = jnp.sqrt(_port_z0(z0, n)).astype(jnp.complex128)
    yn = g[:, None] * yb * g[None, :]
    eye = _eye_like(n)
    return _unbatch(jnp.linalg.solve(eye + yn, eye - yn))


def s_to_abcd(s: jax.Array, z0=50.0) -> jax.Array:
    """ABCD (chain) matrix of a 2-port from its S matrix.

    ``[V1; I1] = [[A, B], [C, D]] [V2; -I2]`` (I2 flowing into port 2).
    Real reference impedances ``z01, z02`` (Frickey, IEEE T-MTT 42(2),
    1994, specialised to real ``z0``):

        A = sqrt(z01/z02) [(1+S11)(1-S22) + S12 S21] / (2 S21)
        B = sqrt(z01 z02) [(1+S11)(1+S22) - S12 S21] / (2 S21)
        C = [(1-S11)(1-S22) - S12 S21] / (2 S21 sqrt(z01 z02))
        D = sqrt(z02/z01) [(1-S11)(1+S22) + S12 S21] / (2 S21)

    Undefined (``inf``) for ``S21 = 0``. Returns shape ``(2, 2, n_freqs)``.
    """
    s = jnp.asarray(s, dtype=jnp.complex128)
    if s.shape[:2] != (2, 2):
        raise ValueError("s must be a 2-port S-matrix (2, 2, n_freqs)")
    z = _port_z0(z0, 2)
    z01, z02 = z[0], z[1]
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    two_s21 = 2.0 * s21
    a = jnp.sqrt(z01 / z02) * ((1 + s11) * (1 - s22) + s12 * s21) / two_s21
    b = jnp.sqrt(z01 * z02) * ((1 + s11) * (1 + s22) - s12 * s21) / two_s21
    c = ((1 - s11) * (1 - s22) - s12 * s21) / (two_s21 * jnp.sqrt(z01 * z02))
    d = jnp.sqrt(z02 / z01) * ((1 - s11) * (1 + s22) + s12 * s21) / two_s21
    return jnp.stack([jnp.stack([a, b]), jnp.stack([c, d])])


def abcd_to_s(abcd: jax.Array, z0=50.0) -> jax.Array:
    """S matrix of a 2-port from its ABCD matrix (inverse of :func:`s_to_abcd`).

        den = A z02 + B + C z01 z02 + D z01
        S11 = ( A z02 + B - C z01 z02 - D z01) / den
        S12 = 2 (A D - B C) sqrt(z01 z02) / den
        S21 = 2 sqrt(z01 z02) / den
        S22 = (-A z02 + B - C z01 z02 + D z01) / den
    """
    m = jnp.asarray(abcd, dtype=jnp.complex128)
    if m.shape[:2] != (2, 2):
        raise ValueError("abcd must have shape (2, 2, n_freqs)")
    z = _port_z0(z0, 2)
    z01, z02 = z[0], z[1]
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    root = jnp.sqrt(z01 * z02)
    den = a * z02 + b + c * z01 * z02 + d * z01
    s11 = (a * z02 + b - c * z01 * z02 - d * z01) / den
    s12 = 2.0 * (a * d - b * c) * root / den
    s21 = 2.0 * root / den
    s22 = (-a * z02 + b - c * z01 * z02 + d * z01) / den
    return jnp.stack([jnp.stack([s11, s12]), jnp.stack([s21, s22])])


# ---------------------------------------------------------------------------
# 2-port wave-transfer helpers (batched over frequency, shape (n_freqs, 2, 2))
# ---------------------------------------------------------------------------

def _s_to_t(s: jax.Array) -> jax.Array:
    """Chain-scattering matrix ``[b1; a1] = T [a2; b2]`` of a 2x2 S (batched).

    Same formulas as ``rfx.deembed._s_to_t``: ``T = [[-det S, S11],
    [-S22, 1]] / S21``.
    """
    s11, s12, s21, s22 = s[..., 0, 0], s[..., 0, 1], s[..., 1, 0], s[..., 1, 1]
    det = s11 * s22 - s12 * s21
    row0 = jnp.stack([-det / s21, s11 / s21], axis=-1)
    row1 = jnp.stack([-s22 / s21, 1.0 / s21], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def _t_to_s(t: jax.Array) -> jax.Array:
    """Inverse of :func:`_s_to_t`: ``S = [[T12, det T], [1, -T21]] / T22``."""
    t11, t12, t21, t22 = t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1]
    det = t11 * t22 - t12 * t21
    row0 = jnp.stack([t12 / t22, det / t22], axis=-1)
    row1 = jnp.stack([1.0 / t22, -t21 / t22], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def _matrix_sqrt_2x2(m: jax.Array) -> jax.Array:
    """Principal square root of a batch of 2x2 matrices (Cayley-Hamilton).

    ``sqrt(M) = (M + sqrt(det M) I) / sqrt(tr M + 2 sqrt(det M))``, the same
    closed form and branch (principal ``sqrt``) as ``rfx.deembed``. The
    NumPy version falls back to an eigendecomposition when the denominator
    vanishes; that value-dependent branch is not reproduced here (see the
    module scope fence).
    """
    det = m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
    sqrt_det = jnp.sqrt(det)
    tr = m[..., 0, 0] + m[..., 1, 1]
    s = jnp.sqrt(tr + 2.0 * sqrt_det)
    return (m + sqrt_det[..., None, None] * _eye_like(2)) / s[..., None, None]


def _twoport_batch(s_matrix, freqs=None):
    """Validate a ``(2, 2, n_freqs)`` S matrix (+ optional freqs) and batch it."""
    s = jnp.asarray(s_matrix, dtype=jnp.complex128)
    if s.ndim != 3 or s.shape[:2] != (2, 2):
        raise ValueError("s_matrix must be a 2-port S-matrix (2, 2, n_freqs)")
    n_freqs = s.shape[2]
    if freqs is not None:
        f = jnp.asarray(freqs, dtype=jnp.float64)
        if f.shape != (n_freqs,):
            raise ValueError(f"freqs has shape {f.shape}, expected ({n_freqs},)")
        return _batch(s), f
    return _batch(s), None


def _series_element_s(z: jax.Array, z0) -> jax.Array:
    """S (batched) of a series impedance ``z`` between ports, reference ``z0``."""
    denom = z + 2.0 * z0
    r = z / denom
    t = 2.0 * z0 / denom
    return jnp.stack([jnp.stack([r, t], axis=-1), jnp.stack([t, r], axis=-1)], axis=-2)


def _deembed_both_sides(t_meas: jax.Array, s_left: jax.Array, s_right: jax.Array) -> jax.Array:
    """``T_left^-1 T_meas T_right^-1`` -> S, batched over frequency."""
    inv_l = jnp.linalg.inv(_s_to_t(s_left))
    inv_r = jnp.linalg.inv(_s_to_t(s_right))
    return _unbatch(_t_to_s(inv_l @ t_meas @ inv_r))


# ---------------------------------------------------------------------------
# rfx.deembed mirrors
# ---------------------------------------------------------------------------

def deembed_port_extension(s_matrix, freqs, port_lengths, z0=50.0, eps_eff=1.0) -> jax.Array:
    """Remove lossless matched feed lines of known length at each port.

    ``S'_ij = S_ij exp(j beta (L_i + L_j))`` with
    ``beta = 2 pi f sqrt(eps_eff) / c``: a pure phase correction that moves
    the reference plane inward by ``L_i`` at port ``i``. ``z0`` is unused
    (kept for signature parity with :func:`rfx.deembed.deembed_port_extension`).
    ``eps_eff`` may be a scalar or ``(n_freqs,)`` for a dispersive line.
    N-port, shape ``(n_ports, n_ports, n_freqs)``.
    """
    s = jnp.asarray(s_matrix, dtype=jnp.complex128)
    f = jnp.asarray(freqs, dtype=jnp.float64)
    lengths = jnp.asarray(port_lengths, dtype=jnp.float64)
    eps = jnp.asarray(eps_eff, dtype=jnp.float64)
    n_ports = s.shape[0]
    if lengths.shape != (n_ports,):
        raise ValueError(
            f"port_lengths has {lengths.shape[0] if lengths.ndim else 1} entries but "
            f"s_matrix has {n_ports} ports")
    beta = 2.0 * jnp.pi * f * jnp.sqrt(eps) / _C0                  # (n_freqs,)
    phase = jnp.exp(1j * lengths[:, None] * beta[None, :])           # (n_ports, n_freqs)
    return s * phase[:, None, :] * phase[None, :, :]


def deembed_thru(s_measured, s_thru) -> jax.Array:
    """Thru-only (TRL-lite) de-embedding of a symmetric 2-port fixture.

    ``T_dut = T_half^-1 T_meas T_half^-1`` with ``T_half = sqrt(T_thru)``
    (principal 2x2 matrix square root). Assumes both fixture halves are
    identical and mirror-symmetric, which is what makes the thru's T matrix
    the square of one half's.
    """
    s_m = jnp.asarray(s_measured, dtype=jnp.complex128)
    s_t = jnp.asarray(s_thru, dtype=jnp.complex128)
    if s_m.shape[:2] != (2, 2):
        raise ValueError("s_measured must be a 2-port S-matrix (2, 2, n_freqs)")
    if s_t.shape[:2] != (2, 2):
        raise ValueError("s_thru must be a 2-port S-matrix (2, 2, n_freqs)")
    if s_m.shape[2] != s_t.shape[2]:
        raise ValueError(
            f"Frequency count mismatch: s_measured has {s_m.shape[2]}, "
            f"s_thru has {s_t.shape[2]}")
    t_meas = _s_to_t(_batch(s_m))
    t_half_inv = jnp.linalg.inv(_matrix_sqrt_2x2(_s_to_t(_batch(s_t))))
    return _unbatch(_t_to_s(t_half_inv @ t_meas @ t_half_inv))


def deembed_series_impedance(s_matrix, freqs, series_z, z0=50.0) -> jax.Array:
    """Remove a known series impedance at each port of a 2-port.

    Exact wave-cascade inverse ``T_dut = T_post(z_1)^-1 T_meas T_post(z_2)^-1``
    where the series element has ``S11 = z/(z + 2 z0)``,
    ``S21 = 2 z0/(z + 2 z0)``. Removes both discontinuities exactly,
    including the far-side one seen through the DUT (a per-port
    ``Z_in - z`` subtraction cannot). ``series_z`` has shape ``(2, n_freqs)``
    and is differentiable.
    """
    t_meas, _ = _twoport_batch(s_matrix, freqs)
    zs = jnp.asarray(series_z, dtype=jnp.complex128)
    n_freqs = t_meas.shape[0]
    if zs.shape != (2, n_freqs):
        raise ValueError(f"series_z has shape {zs.shape}, expected (2, {n_freqs})")
    t_meas = _s_to_t(t_meas)
    return _deembed_both_sides(
        t_meas, _series_element_s(zs[0], z0), _series_element_s(zs[1], z0))


def deembed_series_inductance(s_matrix, freqs, inductances, z0=50.0) -> jax.Array:
    """Remove a series inductance ``L_p`` (henry) at each port of a 2-port.

    :func:`deembed_series_impedance` with ``z_p(f) = j 2 pi f L_p``.
    ``inductances`` has shape ``(2,)`` and is differentiable.
    """
    f = jnp.asarray(freqs, dtype=jnp.float64)
    ind = jnp.asarray(inductances, dtype=jnp.float64)
    if ind.shape != (2,):
        raise ValueError(f"inductances has shape {ind.shape}, expected (2,)")
    series_z = 1j * (2.0 * jnp.pi * f)[None, :] * ind[:, None]
    return deembed_series_impedance(s_matrix, f, series_z, z0=z0)


def deembed_line_segment(s_matrix, freqs, segments, z0=50.0) -> jax.Array:
    """Remove a short lossless line segment ``(zc, tau)`` at each port.

    The segment has ABCD ``[[cos th, j zc sin th], [j sin th / zc, cos th]]``,
    ``th = 2 pi f tau``; its S matrix (reference ``z0``) is cascade-inverted
    off both sides exactly as in :func:`deembed_series_impedance`. In the
    ``tau -> 0`` limit with ``zc tau = L`` fixed this is
    :func:`deembed_series_inductance`. ``segments`` has shape ``(2, 2)`` of
    ``(zc, tau)`` rows and is differentiable in both. The NumPy version
    raises for ``zc <= 0``; here that value check is not performed.
    """
    sb, f = _twoport_batch(s_matrix, freqs)
    segs = jnp.asarray(segments, dtype=jnp.float64)
    if segs.shape != (2, 2):
        raise ValueError(
            f"segments has shape {segs.shape}, expected (2, 2) -- two (zc_seg, tau_seg) pairs")
    omega = 2.0 * jnp.pi * f

    def seg_s(zc, tau):
        th = omega * tau
        a = jnp.cos(th).astype(jnp.complex128)
        b = 1j * zc * jnp.sin(th)
        c = 1j * jnp.sin(th) / zc
        delta = a + b / z0 + c * z0 + a
        s11 = (a + b / z0 - c * z0 - a) / delta
        s21 = 2.0 / delta
        s22 = (-a + b / z0 - c * z0 + a) / delta
        return jnp.stack([jnp.stack([s11, s21], axis=-1),
                          jnp.stack([s21, s22], axis=-1)], axis=-2)

    return _deembed_both_sides(
        _s_to_t(sb), seg_s(segs[0, 0], segs[0, 1]), seg_s(segs[1, 0], segs[1, 1]))


# ---------------------------------------------------------------------------
# Open/short pad de-embedding
# ---------------------------------------------------------------------------

def open_short_deembed(s_meas, s_open, s_short, z0=50.0) -> jax.Array:
    """Two-step open/short de-embedding of an on-chip 2-port test structure.

    Returns the DUT impedance matrix ``Z_dut``, shape ``(2, 2, n_freqs)``
    (convert with :func:`z_to_s` if S is wanted). The steps
    (Koolen, Geelen, Versleijen, Proc. BCTM 1991):

        Y_dut'   = Y_meas  - Y_open        (strip the pad shunt admittance)
        Y_short' = Y_short - Y_open        (same, for the short standard)
        Z_dut    = inv(Y_dut') - inv(Y_short')   (strip the series interconnect)

    Assumed fixture topology, outermost first: an arbitrary 2-port shunt
    admittance network at the probe pads (pad capacitance to substrate and
    pad-to-pad coupling; this is what the OPEN standard measures), then an
    UNCOUPLED series impedance in each lead between pad and DUT (this is
    what the SHORT standard, with the DUT terminals tied to ground, measures
    once the pads are removed). Under these assumptions the recovery is
    exact at every frequency (test T3). It is inexact when the interconnect
    has significant shunt admittance of its own or when the two leads are
    mutually coupled -- those need three-step (open/short/thru) methods.
    Reference impedance ``z0`` is that of all three measurements.
    """
    y_meas = s_to_y(s_meas, z0)
    y_open = s_to_y(s_open, z0)
    y_short = s_to_y(s_short, z0)
    y_dut = _batch(y_meas - y_open)
    y_short2 = _batch(y_short - y_open)
    return _unbatch(jnp.linalg.inv(y_dut) - jnp.linalg.inv(y_short2))


# ---------------------------------------------------------------------------
# Inductor metrics
# ---------------------------------------------------------------------------

def _omega(freqs) -> jax.Array:
    return 2.0 * jnp.pi * jnp.asarray(freqs, dtype=jnp.float64)


def _quality(z: jax.Array) -> jax.Array:
    """``Q = Im(z) / Re(z)`` with the lossless case guarded.

    For ``Re(z) == 0`` the result is ``+inf`` for non-negative reactance and
    ``-inf`` otherwise (never ``nan``), so an ideal inductor reports
    ``Q = +inf``. The guard is a ``where`` on a safe denominator, so the
    gradient stays finite off the lossless line.
    """
    re = jnp.real(z)
    im = jnp.imag(z)
    lossless = re == 0.0
    re_safe = jnp.where(lossless, 1.0, re)
    q_inf = jnp.where(im >= 0.0, jnp.inf, -jnp.inf)
    return jnp.where(lossless, q_inf, im / re_safe)


def z_diff(z: jax.Array) -> jax.Array:
    """Differential impedance ``Z11 - Z12 - Z21 + Z22`` of a 2-port, ``(n_freqs,)``.

    The impedance seen between the two ports when driven differentially
    with no ground return through the ports -- the quantity a differential
    LC tank sees. Definition as in the SG13G2 2.4 GHz LC-VCO inductor
    characterisation of arXiv:2607.08852.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    return z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1]


def l_diff(z: jax.Array, freqs) -> jax.Array:
    """Differential inductance ``Im(z_diff) / omega`` (henry), ``(n_freqs,)``."""
    return jnp.imag(z_diff(z)) / _omega(freqs)


def q_diff(z: jax.Array) -> jax.Array:
    """Differential quality factor ``Im(z_diff) / Re(z_diff)`` (``+inf`` if lossless)."""
    return _quality(z_diff(z))


def l_se(z: jax.Array, freqs) -> jax.Array:
    """Single-ended inductance ``Im(Z11) / omega`` (port 2 open), henry."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    return jnp.imag(z[0, 0]) / _omega(freqs)


def q_se(z: jax.Array) -> jax.Array:
    """Single-ended quality factor ``Im(Z11) / Re(Z11)`` (``+inf`` if lossless)."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    return _quality(z[0, 0])


def l_from_y11(y: jax.Array, freqs) -> jax.Array:
    """Single-ended inductance with port 2 shorted: ``Im(1/Y11) / omega``.

    This is the definition used in arXiv:2607.08852 (SG13G2 LC-VCO), where
    the inductor is characterised from the Y-parameters of the de-embedded
    2-port; ``1/Y11`` is the port-1 impedance with port 2 grounded.
    """
    y = jnp.asarray(y, dtype=jnp.complex128)
    return jnp.imag(1.0 / y[0, 0]) / _omega(freqs)


def q_from_y11(y: jax.Array) -> jax.Array:
    """Quality factor from ``1/Y11``: ``Im(1/Y11) / Re(1/Y11)`` (``+inf`` if lossless)."""
    y = jnp.asarray(y, dtype=jnp.complex128)
    return _quality(1.0 / y[0, 0])


def inductor_metrics(z: jax.Array, freqs) -> dict[str, jax.Array]:
    """All inductor figures of merit from a de-embedded 2-port Z matrix.

    Returns ``{"z_diff", "L_diff", "Q_diff", "L_se", "Q_se", "L_y11",
    "Q_y11"}``, each ``(n_freqs,)``; ``L_y11``/``Q_y11`` come from
    ``Y = inv(Z)``. Definitions follow the SG13G2 2.4 GHz LC-VCO work
    (arXiv:2607.08852): ``L_diff = Im(Z11 - Z12 - Z21 + Z22)/omega``,
    ``Q = Im/Re`` of the same impedance; single-ended ``L_se`` from ``Z11``
    and, with the far port grounded, from ``1/Y11``.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    y = _unbatch(jnp.linalg.inv(_batch(z)))
    return {
        "z_diff": z_diff(z),
        "L_diff": l_diff(z, freqs),
        "Q_diff": q_diff(z),
        "L_se": l_se(z, freqs),
        "Q_se": q_se(z),
        "L_y11": l_from_y11(y, freqs),
        "Q_y11": q_from_y11(y),
    }
