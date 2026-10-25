"""The S reference of a line port: its declared ``impedance=`` (#2014).

Every S producer that admits an MSL or coax port forms its power waves at the
analytic line impedance (Hammerstad-Jensen Z0 for MSL, TEM Z0 for coax) and
then refers the matrix to each port's declared ``impedance=`` through
:func:`renormalize_power_waves`, once, before the opt-in passivity projection.

Import contract (as ``rfx.sparams._common``): only external ``rfx.*`` / stdlib
/ jax / numpy, never ``rfx.api``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np


def renormalize_power_waves(S, z_from, z_to):
    """Refer a power-wave S-matrix from one real reference per port to another (#2014).

    Power waves at a real positive reference Z: a = (V + Z I)/(2 sqrt Z),
    b = (V - Z I)/(2 sqrt Z). Changing Z to Z' per port gives
    a' = c (a - r b), b' = c (b - r a) with r = (Z' - Z)/(Z' + Z) and
    c = (Z + Z')/(2 sqrt(Z Z')), so with b = S a

        S' = C (S - R)(I - R S)^-1 C^-1,   R = diag(r), C = diag(c).

    ``S`` is ``(n, n, n_freqs)`` (``S[receiver, driven, f]``) or ``(n, n)``;
    the formula is applied per frequency bin. A jax array (or tracer) is
    transformed with ``jnp`` and stays differentiable; a NumPy array is
    transformed with NumPy at its own precision. ``z_from`` and ``z_to`` are
    concrete ``(n,)`` real positive impedances.

    When ``z_to == z_from`` for every port, ``S`` itself is returned (same
    object, nothing computed). Ports whose reference does not change enter
    the product in block form: the columns driven at a changed port read
    only the columns driven at changed ports, so a NaN column driven at an
    unchanged port (a refused column) stays confined to itself.
    """
    zf = np.asarray(z_from, dtype=np.float64)
    zt = np.asarray(z_to, dtype=np.float64)
    n = int(S.shape[0])
    for name, z in (("z_from", zf), ("z_to", zt)):
        if (z.shape != (n,) or not np.all(np.isfinite(z))
                or not np.all(z > 0)):
            raise ValueError(
                f"renormalize_power_waves: {name} must be {n} finite "
                f"positive real impedances, got {z!r}")
    if S.ndim not in (2, 3) or S.shape[1] != n:
        raise ValueError(
            "renormalize_power_waves: S must be (n, n) or (n, n, n_freqs), "
            f"got shape {tuple(S.shape)}")
    if np.array_equal(zf, zt):
        return S

    xp = np if isinstance(S, np.ndarray) else jnp
    out_dtype = xp.result_type(S.dtype, xp.complex64)
    r = (zt - zf) / (zt + zf)
    c = (zf + zt) / (2.0 * np.sqrt(zf * zt))
    m = np.flatnonzero(zf != zt)     # ports whose reference changes
    k = np.flatnonzero(zf == zt)     # ports whose reference stays
    rm, cm = r[m], c[m]

    s_f = xp.moveaxis(S if S.ndim == 3 else S[..., None], -1, 0)  # (nf, n, n)
    s_mm = s_f[:, m][:, :, m]
    s_km = s_f[:, k][:, :, m]
    eye_m = xp.eye(m.size, dtype=out_dtype)
    # Y = (I - R_M S_MM)^-1; P Y is formed as solve(K^T, P^T)^T.
    kmat = eye_m - rm[:, None] * s_mm

    def _right_y(p):
        return xp.swapaxes(
            xp.linalg.solve(xp.swapaxes(kmat, -1, -2), xp.swapaxes(p, -1, -2)),
            -1, -2)

    s_mm_minus_r = s_mm - rm[:, None] * eye_m
    out_mm = cm[:, None] * _right_y(s_mm_minus_r) / cm[None, :]
    out_km = _right_y(s_km) / cm[None, :]
    if k.size:
        s_mk = s_f[:, m][:, :, k]
        s_kk = s_f[:, k][:, :, k]
        x = xp.linalg.solve(kmat, rm[:, None] * s_mk)    # Y R_M S_MK
        out_mk = cm[:, None] * (s_mk + s_mm_minus_r @ x)
        out_kk = s_kk + s_km @ x
        top = xp.concatenate([out_mm, out_mk], axis=-1)
        bottom = xp.concatenate([out_km, out_kk], axis=-1)
        blocked = xp.concatenate([top, bottom], axis=-2)
    else:
        blocked = out_mm
    inv = np.argsort(np.concatenate([m, k]))
    out = xp.moveaxis(blocked[:, inv][:, :, inv], 0, -1).astype(out_dtype)
    return out if S.ndim == 3 else out[..., 0]


def warn_line_reference_mismatch(extractor, ports, *, rel_tol=0.05):
    """Say once per port when the declared reference is far from the line Z0.

    ``ports`` is ``((label, declared_ohm, line_ohm), ...)``. Since #2014 the
    declared ``impedance=`` is the reference of the returned S (exact for
    real references), so the gap is not an extraction error: a line whose Z0
    is far from the reference reflects at the port plane by
    |Z0 - R|/(Z0 + R), and that reflection is in S as a property of the
    declared line. Moved here from ``compute_coax_msl_transition``'s
    issue-#581 advisory, which said the opposite before #2014.
    """
    for label, declared, line in ports:
        declared, line = float(declared), float(line)
        if line <= 0.0 or abs(declared - line) / line <= rel_tol:
            continue
        warnings.warn(
            f"{extractor}: the registered {label} port impedance "
            f"({declared:.2f} ohm), which is the reference impedance of the "
            f"returned S, differs {abs(declared - line) / line * 100:.1f}% "
            f"from the analytic {label} line Z0 ({line:.2f} ohm, "
            "line_impedances). S therefore includes the line-to-reference "
            f"mismatch at that port (|Gamma| = "
            f"{abs(declared - line) / (declared + line):.3f}). Pass a "
            "matching pin_radius/outer_radius (coax) or width/height/"
            "eps_r_sub (msl), or impedance= equal to the line Z0, if the "
            "line's own reference is wanted.",
            stacklevel=3,
        )
