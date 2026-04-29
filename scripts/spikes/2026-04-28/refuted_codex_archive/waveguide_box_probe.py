"""Continuous-coordinate waveguide box-probe post-processing.

This module is intentionally NumPy-side: it post-processes frequency-domain
field planes that were already accumulated by a DFT plane probe or loaded from
an external simulator dump.  It mirrors the OpenEMS RectWGPort idea of applying
analytic mode functions over a physical probe box rather than summing
cell-centred samples with a single rectangular cell area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


C0 = 2.998e8
MU_0 = 1.2566370614e-6


@dataclass(frozen=True)
class WaveguideBoxProbe:
    """Continuous-coordinate TE/TM mode projection over a 2-D box.

    Parameters
    ----------
    a, b:
        Physical waveguide aperture dimensions along the two transverse axes.
    mode:
        Rectangular waveguide mode indices.  The WR-90 diagnostic uses TE10.
    mode_type:
        ``"TE"`` or ``"TM"``.
    quad_per_interval:
        Number of sub-intervals per native field-sample interval for the
        trapezoidal quadrature grid.  A value of 4 matches the architectural
        candidate's suggested 4× subpixel refinement.
    """

    a: float
    b: float
    mode: tuple[int, int] = (1, 0)
    mode_type: str = "TE"
    quad_per_interval: int = 4

    def mode_weight(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the tangential TE/TM electric mode weight on ``(u, v)``.

        For TE10 this is simply ``sin(pi*u/a)``; constants such as ``pi/a``
        cancel in the V/I wave decomposition, so the dimensionless shape is
        preferred for numerical conditioning and apples-to-apples dump checks.
        """
        m, n = self.mode
        if self.mode_type == "TE":
            if m <= 0:
                return np.zeros_like(u, dtype=np.float64)
            return np.sin(m * np.pi * u / self.a) * np.cos(n * np.pi * v / self.b)
        if self.mode_type == "TM":
            if m <= 0 or n <= 0:
                raise ValueError(f"TM modes require m,n >= 1, got {(m, n)!r}")
            return np.sin(m * np.pi * u / self.a) * np.cos(n * np.pi * v / self.b)
        raise ValueError(f"mode_type must be 'TE' or 'TM', got {self.mode_type!r}")

    def quadrature_axes(self, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build physical subpixel axes spanning the probe box.

        The input arrays may be centred around zero (OpenEMS/Meep dumps) or
        offset from zero (rfx).  The probe box starts at each array's minimum
        coordinate and spans exactly ``a × b``.
        """
        y = _as_increasing_axis(y, "y")
        z = _as_increasing_axis(z, "z")
        q = int(self.quad_per_interval)
        if q <= 0:
            raise ValueError(f"quad_per_interval must be positive, got {q}")
        ny_q = (y.size - 1) * q + 1
        nz_q = (z.size - 1) * q + 1
        yq = np.linspace(float(y[0]), float(y[0] + self.a), ny_q)
        zq = np.linspace(float(z[0]), float(z[0] + self.b), nz_q)
        return yq, zq

    def modal_vi(
        self,
        e_field: np.ndarray,
        h_field: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        h_y: np.ndarray | None = None,
        h_z: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project frequency-domain field planes to modal ``V`` and ``I``.

        ``e_field`` and ``h_field`` have shape ``(n_freqs, ny, nz)``.  Fields
        are bilinearly interpolated to the quadrature grid before applying a
        trapezoidal integral of ``field × analytic_mode_weight`` over the
        physical aperture.  ``h_y``/``h_z`` may be supplied for Yee-staggered
        H dumps such as OpenEMS' raw H-field probes.
        """
        e_field = np.asarray(e_field)
        h_field = np.asarray(h_field)
        if e_field.shape != h_field.shape or e_field.ndim != 3:
            raise ValueError(
                "e_field and h_field must share shape (n_freqs, ny, nz), "
                f"got {e_field.shape} and {h_field.shape}"
            )
        y = _as_increasing_axis(y, "y")
        z = _as_increasing_axis(z, "z")
        if y.size != e_field.shape[1] or z.size != e_field.shape[2]:
            raise ValueError(
                f"axis size mismatch: fields={e_field.shape}, y={y.size}, z={z.size}"
            )
        h_y_arr = y if h_y is None else _as_increasing_axis(h_y, "h_y")
        h_z_arr = z if h_z is None else _as_increasing_axis(h_z, "h_z")
        if h_y_arr.size != h_field.shape[1] or h_z_arr.size != h_field.shape[2]:
            raise ValueError(
                "H-axis size mismatch: "
                f"h_field={h_field.shape}, h_y={h_y_arr.size}, h_z={h_z_arr.size}"
            )

        yq, zq = self.quadrature_axes(y, z)
        e_interp = _interp_field_plane(e_field, y, z, yq, zq)
        h_interp = _interp_field_plane(h_field, h_y_arr, h_z_arr, yq, zq)

        u = yq - yq[0]
        v = zq - zq[0]
        U, Vv = np.meshgrid(u, v, indexing="ij")
        weight = self.mode_weight(U, Vv)

        v_modal = np.trapezoid(
            np.trapezoid(e_interp * weight[None, :, :], zq, axis=2),
            yq,
            axis=1,
        )
        i_modal = np.trapezoid(
            np.trapezoid(h_interp * weight[None, :, :], zq, axis=2),
            yq,
            axis=1,
        )
        return v_modal, i_modal


def s11_from_box_fields(
    e_field: np.ndarray,
    h_field: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    freqs: np.ndarray,
    *,
    a: float,
    b: float | None = None,
    h_y: np.ndarray | None = None,
    h_z: np.ndarray | None = None,
    mode: tuple[int, int] = (1, 0),
    mode_type: str = "TE",
    quad_per_interval: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute dump-derived ``|S11|`` using a continuous-coordinate box probe."""
    y_arr = _as_increasing_axis(y, "y")
    z_arr = _as_increasing_axis(z, "z")
    b_eff = float(b if b is not None else (z_arr[-1] - z_arr[0]))
    probe = WaveguideBoxProbe(
        a=float(a),
        b=b_eff,
        mode=mode,
        mode_type=mode_type,
        quad_per_interval=int(quad_per_interval),
    )
    v_modal, i_modal = probe.modal_vi(
        e_field, h_field, y_arr, z_arr, h_y=h_y, h_z=h_z
    )

    freqs_arr = np.asarray(freqs, dtype=np.float64)
    omega = 2.0 * np.pi * freqs_arr
    f_c = C0 / (2.0 * float(a))
    k0 = omega / C0
    kc = 2.0 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k0**2 - kc**2, 0.0) + 0j)
    z_te = omega * MU_0 / beta

    a_fwd = 0.5 * (v_modal + i_modal * z_te)
    a_ref = v_modal - a_fwd
    s11 = np.abs(a_ref / a_fwd)
    return s11, v_modal, i_modal, z_te


def _as_increasing_axis(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1-D array with at least two points")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    return arr


def _interp_field_plane(
    field: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    yq: np.ndarray,
    zq: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate ``(nf, ny, nz)`` complex planes to ``(yq,zq)``."""
    nf, _, nz = field.shape
    tmp = np.empty((nf, yq.size, nz), dtype=np.complex128)
    out = np.empty((nf, yq.size, zq.size), dtype=np.complex128)
    for fi in range(nf):
        for kk in range(nz):
            tmp[fi, :, kk] = (
                np.interp(yq, y, field[fi, :, kk].real)
                + 1j * np.interp(yq, y, field[fi, :, kk].imag)
            )
        for jj in range(yq.size):
            out[fi, jj, :] = (
                np.interp(zq, z, tmp[fi, jj, :].real)
                + 1j * np.interp(zq, z, tmp[fi, jj, :].imag)
            )
    return out
