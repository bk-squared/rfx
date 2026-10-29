"""The S reference of a line port: its declared ``impedance=`` (#2014).

Every S producer that admits an MSL or coax port forms its power waves at the
analytic line impedance (Hammerstad-Jensen Z0 for MSL, TEM Z0 for coax) and
then refers the matrix to each port's declared ``impedance=`` through
:func:`renormalize_power_waves`, once, before the opt-in passivity projection.

Import contract (as ``rfx.sparams._common``): only external ``rfx.*`` / stdlib
/ jax / numpy, never ``rfx.api``.
"""

from __future__ import annotations

from rfx.errors import ConfigurationWarning
import warnings

import jax
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
    ``(n,)`` real positive impedances; either may be traced (a line impedance
    that depends on a design variable), and then every port is transformed
    with ``jnp`` and the value checks are left to the caller; the NaN-column
    confinement below then does not apply.

    When ``z_to == z_from`` for every port, ``S`` itself is returned (same
    object, nothing computed). Ports whose reference does not change enter
    the product in block form: the columns driven at a changed port read
    only the columns driven at changed ports, so a NaN column driven at an
    unchanged port (a refused column) stays confined to itself.
    """
    n = int(S.shape[0])
    if S.ndim not in (2, 3) or S.shape[1] != n:
        raise ValueError(
            "renormalize_power_waves: S must be (n, n) or (n, n, n_freqs), "
            f"got shape {tuple(S.shape)}")
    zf, zt = _concrete(z_from), _concrete(z_to)
    if zf is None or zt is None:
        # A traced reference: every port is transformed, with jnp.
        xp = jnp
        zf, zt = jnp.asarray(z_from), jnp.asarray(z_to)
        if zf.shape != (n,) or zt.shape != (n,):
            raise ValueError(
                f"renormalize_power_waves: z_from and z_to must be ({n},), "
                f"got {zf.shape} and {zt.shape}")
        m, k = np.arange(n), np.arange(0)
    else:
        for name, z in (("z_from", zf), ("z_to", zt)):
            if (z.shape != (n,) or not np.all(np.isfinite(z))
                    or not np.all(z > 0)):
                raise ValueError(
                    f"renormalize_power_waves: {name} must be {n} finite "
                    f"positive real impedances, got {z!r}")
        if np.array_equal(zf, zt):
            return S
        xp = np if isinstance(S, np.ndarray) else jnp
        m = np.flatnonzero(zf != zt)     # ports whose reference changes
        k = np.flatnonzero(zf == zt)     # ports whose reference stays

    out_dtype = xp.result_type(S.dtype, xp.complex64)
    r = (zt - zf) / (zt + zf)
    c = (zf + zt) / (2.0 * (np if isinstance(zf, np.ndarray) else jnp).sqrt(zf * zt))
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


def _concrete(z):
    """``z`` as a float64 NumPy array, or ``None`` when it is traced."""
    try:
        return np.asarray(z, dtype=np.float64)
    except jax.errors.TracerArrayConversionError:
        return None


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
            "line's own reference is wanted.", ConfigurationWarning,
            stacklevel=3,
        )


def coax_two_port_line_waves(S, grid, eps_scale, z_tem, *, center_xy,
                             pin_radius, outer_radius, ladders):
    """The coax two-port's S as power waves at each ladder's line impedance.

    Returns ``(S_line, line_impedances)`` (#2014 review). The wave split is a
    voltage-only ratio, so its waves sit at the impedance of the line under
    each probe ladder. Without ``eps_scale`` that is the stamped fill's
    ``z_tem`` at both ends and ``S`` is returned as it is. ``eps_scale``
    multiplies the fill's permittivity, so the line under ladder ``i`` is
    ``z_tem / sqrt(s_i)``, ``s_i`` the scale over the annulus cells of that
    ladder's planes, and a voltage ratio between ladders on unequal lines
    becomes a power-wave ratio by ``sqrt(Z_driven / Z_receiver)``. A concrete
    scale that is not uniform over a ladder is refused (the ladder fit
    assumes one uniform line there); a traced one is averaged.
    """
    if eps_scale is None:
        return S, np.full(len(ladders), float(z_tem))
    # The fill's own cells: stamp_coaxial_line's Cylinder masks (node
    # coordinates), over the whole z extent.
    from rfx.geometry.csg import Cylinder
    span = 2.0 * float(np.sum(grid.cells(2)))
    centre = (float(center_xy[0]), float(center_xy[1]), 0.0)
    annulus = np.asarray(
        Cylinder(center=centre, radius=float(outer_radius), height=span,
                 axis="z").mask(grid)
        & ~Cylinder(center=centre, radius=float(pin_radius), height=span,
                    axis="z").mask(grid))
    scale = jnp.broadcast_to(jnp.asarray(eps_scale), grid.shape)
    z_line = []
    for i, zs in enumerate(ladders):
        zs = np.asarray(zs, dtype=int)
        cells = scale[:, :, zs][annulus[:, :, zs]]
        concrete = _concrete(cells)
        if concrete is None:
            z_line.append(float(z_tem) / jnp.sqrt(jnp.mean(cells)))
            continue
        if np.ptp(concrete) > 1e-6 * float(np.max(np.abs(concrete))):
            raise ValueError(
                "compute_coaxial_two_port(): eps_scale varies over the "
                f"annulus of probe ladder {i} (from {concrete.min():.6g} to "
                f"{concrete.max():.6g}); the ladder reads one uniform line, "
                "so keep eps_scale uniform over each probe ladder and put "
                "the device between them.")
        z_line.append(float(z_tem) / float(np.sqrt(concrete.mean())))
    if all(isinstance(z, float) for z in z_line):
        z_line = np.asarray(z_line, dtype=np.float64)
        if np.all(z_line == z_line[0]):
            return S, z_line
        root = np.sqrt(z_line)
    else:
        z_line = jnp.stack([jnp.asarray(z) for z in z_line])
        root = jnp.sqrt(z_line)
    return S * (root[None, :, None] / root[:, None, None]), z_line
