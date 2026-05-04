"""Microstrip line (MSL) eigenmode source — quasi-TEM (Ey, Ez, Hy, Hz, β).

This module returns an :class:`MSLEigenmodeData` container with all four
transverse field components and the dispersion β(ω) curve, ready for use as
a Schelkunoff J+M one-sided source in
:func:`rfx.sources.msl_port.make_msl_port_sources_jm`.

Implementation status (Path B — from-scratch FDFD, in progress):

* Cross-section box (lateral padding, substrate cell range, trace footprint)
  is taken from the existing static-Laplace
  :func:`rfx.sources.msl_port.compute_msl_mode_profile` so cell layout
  matches what :func:`make_msl_port_sources_jm` expects.
* The vectorial 2-D Maxwell eigenproblem is solved by the 2-component
  (Ey, Ez) E-formulation FDFD operator in
  :func:`rfx.sources.msl_fdfd_eigenmode.solve_msl_fdfd_eigenmode`.
  Reference: Rumpf EMPossible Lecture 19, cross-checked against MPB source
  ``mpb/src/maxwell/maxwell_op.c`` (4-component (E_t, H_t) operator with
  divergence constraint built in).
* Transverse H is derived from the FDFD eigenmode via Maxwell curl.
* β(ω) curve: weak-dispersion approximation ``β(ω) = ω · √ε_eff / c`` using
  the eigensolver-derived ``ε_eff`` at ``f_design`` (midband by default).

Failed prior attempts (do not repeat):

* MPB subprocess wrapper (deleted 2026-05-04): boxed-PEC MPB modes do not
  match open-FDTD modes — geometry mismatch, |S11| = 0.61 vs Laplace 0.118.
  See ``docs/research_notes/20260504_msl_eigenmode_handover.md``.
* H-only formulation (deleted): cannot self-consistently couple E/H in
  inhomogeneous ε — gave ε_eff = 1.37 instead of ~3 for microstrip.
* 4-component (Ey,Ez,Hy,Hz) propagation matrix (deleted): redundant DOF +
  BC truncation artefacts on slab waveguide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfx.core.yee import EPS_0, MU_0


_C0 = 1.0 / np.sqrt(MU_0 * EPS_0)
_ETA0 = float(np.sqrt(MU_0 / EPS_0))


# ---------------------------------------------------------------------------
# Hammerstad-Jensen analytic Z0 / ε_eff (kept for callers that already use it)
# ---------------------------------------------------------------------------


def hammerstad_jensen_z0_eps_eff(
    w: float, h: float, eps_r: float
) -> tuple[float, float]:
    """Hammerstad-Jensen closed-form microstrip Z0 and ε_eff.

    Reference: Pozar §3.7, Hammerstad & Jensen 1980. Accurate to ~0.5%
    for standard MSL geometries (0.1 ≤ W/H ≤ 10, 1 ≤ εr ≤ 16).
    """
    u = w / h
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (
        1.0 + 12.0 / u
    ) ** -0.5
    if u <= 1.0:
        z0 = (60.0 / np.sqrt(eps_eff)) * np.log(8.0 / u + u / 4.0)
    else:
        z0 = (
            120.0 * np.pi
            / (np.sqrt(eps_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
        )
    return float(z0), float(eps_eff)


# ---------------------------------------------------------------------------
# MSLEigenmodeData container — public API consumed by msl_port.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSLEigenmodeData:
    """Solved quasi-TEM eigenmode for an MSL cross-section.

    Field arrays are real-valued and live on the FDTD-grid window
    ``[j_grid_lo : j_grid_lo + n_y_grid, k_grid_lo : k_grid_lo + n_z_grid]``,
    with shape ``(n_y_grid, n_z_grid)``. All four components are
    cell-centred so :func:`rfx.sources.msl_port.make_msl_port_sources_jm`
    can deposit J/M directly via the nearest-cell rule.

    Sign convention: ``∫Ez·dz`` along the substrate at the trace y-centre
    equals +1V (sign-flipped from physical Ez to match the FDTD source
    polarity used by the 3-probe extractor in
    :func:`rfx.api.Simulation.compute_msl_s_matrix`).
    """

    ey: np.ndarray
    ez: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    beta: np.ndarray             # (n_freqs,) — β(ω) curve
    z0: float                    # V/I extraction at f_design
    z0_energy: float             # Poynting V²/(2P) extraction at f_design
    eps_eff: float               # design-frequency ε_eff
    f_design: float
    j_grid_lo: int
    k_grid_lo: int
    n_y_grid: int
    n_z_grid: int                # = n_z_sub (substrate cells only)
    dy: float
    dz: float
    cell_indices: list           # FDTD (i, j, k) cells in cross-section
    sigma_support_mask: np.ndarray  # (n_y_grid, n_z_grid) bool
    trace_j_lo_local: int
    trace_j_hi_local: int


# ---------------------------------------------------------------------------
# Cross-section geometry helpers
# ---------------------------------------------------------------------------


def _axis_cell_size(grid, axis: str, idx: int) -> float:
    profile_attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
    profile = getattr(grid, profile_attr, None)
    if profile is not None:
        try:
            n = int(profile.shape[0])
        except Exception:
            return float(getattr(grid, axis if axis != "x" else "dx", grid.dx))
        clamped = max(0, min(idx, n - 1))
        return float(profile[clamped])
    return float(getattr(grid, axis if axis != "x" else "dx", grid.dx))


# ---------------------------------------------------------------------------
# Public API: compute_msl_eigenmode_profile
# ---------------------------------------------------------------------------


def compute_msl_eigenmode_profile(
    grid,
    port,
    eps_r_sub: float,
    freqs: np.ndarray,
    *,
    f_design: float | None = None,
    pad_y_cells: int | None = None,
    pad_z_cells: int | None = None,
    refine: int = 4,
    n_z_air_padding: int | None = None,
) -> MSLEigenmodeData:
    """Solve the vectorial 2-D quasi-TEM eigenmode for an MSL cross-section.

    Returns the transverse (Ey, Ez, Hy, Hz) field shape from the full
    Maxwell vectorial eigenproblem (spec §2.3), β(ω) from the
    eigensolver-derived ε_eff, and Z0 (V/I and Poynting paths) for
    cross-checking.

    Parameters
    ----------
    grid : Grid (uniform)
    port : MSLPort
    eps_r_sub : float
        Substrate relative permittivity.
    freqs : (n_freqs,) array
        Measurement frequency grid for the β(ω) curve.
    f_design : float, optional
        Frequency at which to solve the eigenmode (mode shape and ε_eff).
        Default = midband of ``freqs``. v1 uses a frequency-independent
        ε_eff (weak-dispersion approx, ~1% on RO4350B for ≤10 GHz).
    pad_y_cells, pad_z_cells, refine : forwarded to the static-Laplace
        cross-section box helper for layout consistency with
        ``make_msl_port_sources_jm``.
    n_z_air_padding : int, optional
        Air cells above the trace in the eigensolver box. Default
        ``max(6·n_z_sub, 12)`` per spec §2.4.

    Notes
    -----
    Eigensolver runs in numpy/scipy outside JAX. Result should be cached
    by the caller (the rfx runner already does this implicitly because
    ``add_msl_port`` only calls this function once per port).
    """
    raise NotImplementedError(
        "Path B from-scratch FDFD eigenmode solver "
        "(rfx.sources.msl_fdfd_eigenmode) is not yet wired in. "
        "MPB subprocess approach was removed 2026-05-04 — see module "
        "docstring + docs/research_notes/20260504_msl_eigenmode_handover.md. "
        "Use mode='laplace' for now."
    )
