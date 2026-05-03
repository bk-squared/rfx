"""Microstrip line (MSL) eigenmode solver — quasi-TEM E_t and H_t fields
plus β(ω) and Z0, suitable for a Schelkunoff J+M one-sided source.

Status (v1, per implementer–scientist consult log Q-1 2026-05-03):
the dominant |S11|≈0.10 floor on the integration test is caused by the
MISSING magnetic current J in the legacy Ez-only source (~5% bounce off
the σ-loaded plane), with the static-vs-vectorial mode-SHAPE error
contributing only the smaller ~4% (β·H)² correction (spec §2.2). v1
therefore uses the existing static-Laplace solver for the transverse-E
field shape and derives H_t from the wave-impedance identity for a
+x-propagating quasi-TEM mode. This implements the Schelkunoff J+M pair
that is the spec's primary fix while accepting the ~4% mode-shape
residual for now. A full 2-D Maxwell vectorial eigensolver per spec §2.3
remains a future upgrade if the |S11| floor exceeds the 0.02 target.

Theory:

* Static-Laplace `φ(y,z)` solves `∇·(εr ∇φ) = 0` with φ=1V on trace,
  φ=0 on ground. The transverse E-field is `E_t = -∇φ`. For the
  quasi-TEM MSL mode this is exact in the ω→0 limit and accurate to
  `O((β·H)²)·(εr-1)/εr ≈ 1.6e-3` for RO4350B at 4 GHz (spec §2.2).
* The transverse H-field for a +x̂-propagating quasi-TEM mode is
  `H_t = (1/η_eff) · (x̂ × E_t)`, i.e.
  `Hy = -Ez/η_eff`, `Hz = +Ey/η_eff`, with `η_eff = ω·μ₀/β`,
  `β = ω·√ε_eff/c`, and ε_eff from Hammerstad-Jensen.
* The eigensolver runs ONCE per port at simulation start (numpy/scipy,
  no JAX), result cached on the port entry, then converted to numpy
  floats for the JAX FDTD source build.

The Schelkunoff J+M pair on the source plane (n̂ = direction × x̂):

    J_y = -Hz · waveform(t),   J_z = +Hy · waveform(t)        (electric currents)
    M_y = +Ez · waveform(t),   M_z = -Ey · waveform(t)        (magnetic currents)

with a half-step time offset between J and M (Yee leapfrog). See
``rfx/sources/msl_port.py::make_msl_port_sources`` for the FDTD wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfx.core.yee import EPS_0, MU_0


_C0 = 1.0 / np.sqrt(MU_0 * EPS_0)
_ETA0 = float(np.sqrt(MU_0 / EPS_0))


# ---------------------------------------------------------------------------
# Hammerstad-Jensen analytic Z0 / ε_eff
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
# MSLEigenmodeData container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSLEigenmodeData:
    """Solved quasi-TEM eigenmode for an MSL cross-section.

    All field arrays are real-valued (the dominant quasi-TEM mode is
    real up to a global phase) and live on the FDTD-grid window
    ``[j_grid_lo : j_grid_lo + n_y_grid, k_grid_lo : k_grid_lo + n_z_grid]``.

    Field collocation: in v1 the four components ey, ez, hy, hz are all
    cell-centred on the substrate slab (n_y_grid, n_z_grid). The FDTD
    source code treats this as the source-plane reference profile and
    deposits J/M into the appropriate Yee-staggered cells via the
    nearest-cell rule.

    Normalisation: ``∫Ez·dz`` along the substrate at the trace centre
    equals 1 V; H_t scales accordingly via η_eff.
    """

    ey: np.ndarray
    ez: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    beta: np.ndarray             # (n_freqs,) — β(ω) curve
    z0: float                    # V/I extraction
    z0_energy: float             # √(L'/C') extraction
    eps_eff: float               # design-frequency ε_eff
    f_design: float
    j_grid_lo: int
    k_grid_lo: int
    n_y_grid: int
    n_z_grid: int                # height of the (y, z) box (substrate only)
    dy: float
    dz: float
    cell_indices: list           # FDTD (i, j, k) cells in cross-section
    sigma_support_mask: np.ndarray  # (n_y_grid, n_z_grid) bool — σ-load cells
    trace_j_lo_local: int
    trace_j_hi_local: int


# ---------------------------------------------------------------------------
# Cross-section grid building
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
) -> MSLEigenmodeData:
    """Solve the quasi-TEM eigenmode for an MSL cross-section.

    Returns the transverse (Ey, Ez) field shape from the static
    Laplace solver, the transverse (Hy, Hz) derived via the wave-
    impedance identity for a +x̂-propagating quasi-TEM mode, β(ω) from
    Hammerstad ε_eff, and Z0 (V/I and energy paths) for cross-checking.

    Parameters
    ----------
    grid : Grid (uniform)
    port : MSLPort
    eps_r_sub : float
        Substrate relative permittivity.
    freqs : (n_freqs,) array
        Measurement frequency grid.
    f_design : float, optional
        Frequency at which to define the design β = ω·√ε_eff/c. Default
        = midband of ``freqs``. v1 uses a frequency-independent ε_eff
        (weak-dispersion approx, accurate to ~1% on RO4350B for ≤10 GHz).
    pad_y_cells, pad_z_cells : int, optional
        Padding for the cross-section box. Defaults match the existing
        static-Laplace solver in rfx/sources/msl_port.py.
    refine : int
        Sub-mesh refinement factor for the static Laplace box (default 4).

    Notes
    -----
    Eigensolver runs in numpy/scipy outside JAX. Result cached on the
    port entry by the caller.
    """
    from rfx.sources.msl_port import compute_msl_mode_profile

    freqs = np.asarray(freqs, dtype=np.float64)
    if f_design is None:
        f_design = float(np.median(freqs))
    omega_design = 2.0 * np.pi * f_design

    H_sub = float(port.z_hi - port.z_lo)
    W = float(port.y_hi - port.y_lo)

    # Static-Laplace solve gives Ez(y, z), z0_static, eps_eff on the
    # FDTD-grid window cross-section. We reuse it verbatim for the
    # E-field shape; this is accurate to O((β·H)²) ≈ 0.16% per spec §2.2.
    static_profile = compute_msl_mode_profile(
        grid, port, eps_r_sub,
        pad_y_cells=pad_y_cells, pad_z_cells=pad_z_cells, refine=refine,
    )

    ez_static = np.asarray(static_profile["ez_profile"], dtype=np.float64)
    eps_eff_static = float(static_profile["eps_eff"])
    z0_static = float(static_profile["z0_static"])
    j_grid_lo = int(static_profile["j_grid_lo"])
    k_grid_lo = int(static_profile["k_grid_lo"])
    trace_j_lo = int(static_profile["trace_j_lo"])
    trace_j_hi = int(static_profile["trace_j_hi"])
    n_z_sub = int(static_profile["n_z_sub"])
    dy = float(static_profile["dy"])
    dz = float(static_profile["dz"])
    cell_indices = list(static_profile["cell_indices"])

    n_y_grid = ez_static.shape[0]
    n_z_grid = ez_static.shape[1]

    # Hammerstad-Jensen analytic ε_eff for cross-check.
    z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(W, H_sub, eps_r_sub)
    # Use the static-Laplace ε_eff (which captures the actual cross-section
    # discretisation); fall back to Hammerstad if static is degenerate.
    eps_eff = eps_eff_static if 1.0 < eps_eff_static < eps_r_sub else eps_eff_hj

    # β(ω) curve via weak-dispersion approximation
    beta_curve = (2.0 * np.pi * freqs / _C0) * np.sqrt(eps_eff)
    beta_design = float((omega_design / _C0) * np.sqrt(eps_eff))

    # Derive transverse Ey from -∂φ/∂y. The static solver returns Ez
    # already; we need Ey too. Compute Ey from the discrete profile via
    # the curl-free condition: Ey is the y-derivative of the electrostatic
    # potential, but ez_static is normalised, so we approximate Ey via
    # the y-derivative of ez integrated: simplest is to use centred
    # differences of the integrated potential.
    #
    # Reconstruct the potential via z-integration of -ez:
    #   φ(y, z) = ∫₀^z (-ez)(y, ζ) dζ           (with φ=0 at z=0 ground)
    phi = np.zeros((n_y_grid, n_z_sub + 1), dtype=np.float64)
    # phi[:, k] is the potential at z=k·dz (k=0 is ground, k=n_z_sub is trace)
    for k in range(n_z_sub):
        phi[:, k + 1] = phi[:, k] + (-ez_static[:, k]) * dz

    # Ey at cell centres = -∂φ/∂y, evaluated at z=k+0.5 (between φ rows).
    # Use centred differences in y of (phi[:, k] + phi[:, k+1])/2.
    phi_cc_z = 0.5 * (phi[:, :-1] + phi[:, 1:])  # (n_y, n_z_sub)
    ey = np.zeros((n_y_grid, n_z_sub), dtype=np.float64)
    ey[1:-1, :] = -(phi_cc_z[2:, :] - phi_cc_z[:-2, :]) / (2.0 * dy)
    ey[0, :] = -(phi_cc_z[1, :] - phi_cc_z[0, :]) / dy
    ey[-1, :] = -(phi_cc_z[-1, :] - phi_cc_z[-2, :]) / dy

    # Ez collocated with cell centres.
    ez = ez_static.copy()

    # Transverse H from the wave-impedance identity for +x̂ propagation:
    #   Ht = (1/η_eff) · (x̂ × Et)
    #      = (1/η_eff) · (-Ez·ŷ + Ey·ẑ)
    # i.e. Hy = -Ez/η_eff,  Hz = +Ey/η_eff.
    #
    # NOTE on inhomogeneous-fill scaling: this point-wise relation is
    # exact only for a uniform-medium plane wave. For an inhomogeneous
    # quasi-TEM MSL mode the SHAPE is close (O((β·H)² · (εr−1)/εr) ≈
    # 1.6e-3 correction on RO4350B at 4 GHz per spec §2.2) but the
    # global scale needs a one-shot correction so V/I matches the
    # static-Laplace Z0. We scale H_t by the ratio (z0_pre / z0_static)
    # below.
    eta_eff = omega_design * MU_0 / beta_design
    hy = -ez / eta_eff
    hz = +ey / eta_eff

    # Z0 (V/I): V at trace centre = +∫Ez·dz from ground to trace.
    # ez_static was normalised so ∫Ez·dz at trace centre = +1 V.
    j_centre = (trace_j_lo + trace_j_hi) // 2 - j_grid_lo
    j_centre = max(0, min(j_centre, n_y_grid - 1))
    V = float(np.sum(ez[j_centre, :]) * dz)
    # I = ∮ H·dl ≈ -∫ Hy(y, z=top) dy across the trace width.
    # For +x propagation with Ez > 0, Hy < 0 (right-hand rule), so the
    # raw integral is negative; the -1 makes I > 0 to match V > 0.
    j_lo_local = max(0, trace_j_lo - j_grid_lo)
    j_hi_local = min(n_y_grid - 1, trace_j_hi - j_grid_lo)
    k_top = n_z_sub - 1
    I_pre = float(-np.sum(hy[j_lo_local:j_hi_local + 1, k_top]) * dy)
    z0_pre = V / I_pre if abs(I_pre) > 1e-30 else z0_static

    # Rescale H_t so V/I matches the static-Laplace Z0 (the more
    # accurate Z0 for the inhomogeneous quasi-TEM mode).
    target_z0 = z0_static
    if abs(z0_pre) > 1e-30 and abs(target_z0) > 1e-30:
        h_scale = z0_pre / target_z0
        hy *= h_scale
        hz *= h_scale
        I = I_pre * h_scale
    else:
        I = I_pre
    z0_vi = float(V / I) if abs(I) > 1e-30 else target_z0

    # Z0 (energy): Poynting power and stored-energy ratio. For TEM,
    # Z0 = √(L'/C') = √(<μH²>/<εE²>) · η₀ scaled by 1/√ε_eff.
    we = 0.5 * np.sum(eps_r_sub * EPS_0 * (ey * ey + ez * ez)) * dy * dz
    wm = 0.5 * np.sum(MU_0 * (hy * hy + hz * hz)) * dy * dz
    if we > 0 and wm > 0:
        z0_en = float(np.sqrt(wm / we) * V * V / (2.0 * we) if we > 0 else 0.0)
        # Cleaner identity: P_TEM = V·I/2 = V²/(2·Z0), so Z0 = V²/(2·P).
        P = 0.5 * np.sum(ey * hz - ez * hy) * dy * dz
        if abs(P) > 1e-30:
            z0_en = float(V * V / (2.0 * P))
        else:
            z0_en = z0_vi
    else:
        z0_en = z0_vi

    # σ-support mask: cells where the mode carries energy. Use the
    # |E|² density relative to the trace-centre maximum.
    e_density = ey * ey + ez * ez
    e_max = float(np.max(e_density)) if e_density.size > 0 else 0.0
    if e_max > 0:
        sigma_support_mask = e_density > 1e-3 * e_max
    else:
        sigma_support_mask = np.ones_like(e_density, dtype=bool)

    return MSLEigenmodeData(
        ey=ey,
        ez=ez,
        hy=hy,
        hz=hz,
        beta=beta_curve,
        z0=float(z0_vi),
        z0_energy=float(z0_en),
        eps_eff=float(eps_eff),
        f_design=float(f_design),
        j_grid_lo=j_grid_lo,
        k_grid_lo=k_grid_lo,
        n_y_grid=int(n_y_grid),
        n_z_grid=int(n_z_sub),
        dy=float(dy),
        dz=float(dz),
        cell_indices=cell_indices,
        sigma_support_mask=sigma_support_mask,
        trace_j_lo_local=int(trace_j_lo - j_grid_lo),
        trace_j_hi_local=int(trace_j_hi - j_grid_lo),
    )
