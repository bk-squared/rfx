"""Radar Cross Section (RCS) computation pipeline.

Combines TFSF plane-wave illumination with NTFF near-to-far-field
transform to compute monostatic and bistatic RCS of scatterers.

The standard FDTD RCS approach:
1. TFSF illuminates the target with a plane wave
2. Scattered field (outside TFSF box) is captured by NTFF box
3. NTFF computes far-field pattern
4. RCS(theta, phi) = 4*pi*r^2 * |E_scat|^2 / |E_inc|^2

Validation scope: the monostatic (backscatter) bin is cross-validated against
the exact Mie series. The default unsubtracted bistatic pattern is not
validated because residual TFSF incident-field leakage contaminates oblique
angles. The two-run ``subtract_incident_reference=True`` path is validated for
the documented ka~1 PEC-sphere comparison; other targets and configurations
require their own convergence and reference checks.

Reference: Taflove & Hagness, Ch. 8-9.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.farfield import (
    FarFieldResult, NTFFBox, compute_far_field, compute_far_field_jax,
)
from rfx.sources.tfsf import init_tfsf
from rfx.simulation import run


class RCSResult(NamedTuple):
    """Radar cross section computation result.

    freqs : (n_freqs,) Hz
    theta : (n_theta,) radians
    phi : (n_phi,) radians
    rcs_dbsm : (n_freqs, n_theta, n_phi) in dBsm
    rcs_linear : (n_freqs, n_theta, n_phi) in m^2
    monostatic_rcs : (n_freqs,) backscatter RCS in dBsm, evaluated
        exactly at the backscatter direction (opposite the incident
        propagation vector), independent of the theta/phi observation
        grid. For normal-incidence +x propagation this is
        (theta=pi/2, phi=pi), i.e. -x.

    VALIDATION SCOPE (read before trusting the full pattern)
    -------------------------------------------------------
    ``monostatic_rcs`` agrees with exact Mie to about 0.06 dB for the
    committed ka~1 PEC-sphere case. The default unsubtracted
    ``rcs_dbsm`` / ``rcs_linear`` bistatic pattern is NOT validated: residual
    TFSF incident-field leakage contaminates off-backscatter angles, and
    increasing ``ntff_offset`` alone does not remove it. Results produced by
    ``compute_rcs(..., subtract_incident_reference=True)`` use two-run complex
    far-field subtraction; that path is validated only for the documented
    ka~1 sphere comparison. ``RCSResult`` does not record which option produced
    it, so retain the call configuration with archived results. See
    ``compute_rcs`` ("Bistatic pattern caveat") for limits and convergence
    checks.
    """
    freqs: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    rcs_dbsm: np.ndarray
    rcs_linear: np.ndarray
    monostatic_rcs: np.ndarray


def _incident_spectrum_amplitude(
    f0: float,
    bandwidth: float,
    freqs: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Compute the frequency-domain amplitude of the TFSF incident pulse.

    The TFSF 1-D source uses a differentiated Gaussian:
        s(t) = -2*arg * exp(-arg^2),  arg = (t - t0) / tau
    where tau = 1/(f0 * bandwidth * pi), t0 = 3*tau.

    We compute the DFT of this waveform at the requested frequencies
    to get the incident field spectral amplitude for RCS normalization.
    """
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau
    times = np.arange(n_steps) * dt
    arg = (times - t0) / tau
    waveform = -2.0 * arg * np.exp(-(arg ** 2))

    # DFT at requested frequencies
    # S(f) = sum_n s(n*dt) * exp(-j*2*pi*f*n*dt) * dt
    amplitudes = np.zeros(len(freqs), dtype=np.complex128)
    for i, f in enumerate(freqs):
        phase = np.exp(-1j * 2 * np.pi * f * times)
        amplitudes[i] = np.sum(waveform * phase) * dt

    return amplitudes


def compute_rcs_jax(
    ntff_data,
    box: NTFFBox,
    grid: Grid,
    theta,
    phi,
    e_inc_amplitude,
):
    """JAX-differentiable RCS(θ,φ) from NTFF data — the AD counterpart of ``compute_rcs``.

    ``compute_rcs`` orchestrates (TFSF setup → ``run`` → far-field → RCS) with a numpy
    post-processor, so it is NOT on the AD tape. This function is the differentiable
    POST-PROCESSING half: given ``ntff_data`` accumulated by a differentiable
    ``run(..., tfsf=…, ntff=…)`` (whose accumulators flow through the ``lax.scan`` AD
    tape w.r.t. the scatterer materials), it returns σ as a ``jnp`` array so
    ``jax.grad`` flows scatterer-ε → RCS. It is the primitive behind differentiable
    RCS-reduction / -shaping inverse design.

    Same formula as ``compute_rcs`` (σ = 4π·|E_far|²/|E_inc|², ``compute_far_field``'s
    ``E_θ``/``E_φ`` already carry the ``jk/4π`` factor so ``|E_θ|²+|E_φ|²`` is the
    r-scaled scattered power), evaluated with ``compute_far_field_jax``. For the
    monostatic (backscatter) bin at normal +x incidence pass ``theta=[π/2], phi=[π]``.

    Parameters
    ----------
    ntff_data : NTFFData
        NTFF surface accumulators from ``run(..., tfsf=…, ntff=…)`` (JAX arrays, on tape).
    box : NTFFBox
    grid : Grid
    theta, phi : arrays in radians
        Observation directions. Backscatter for +x incidence is ``(π/2, π)``.
    e_inc_amplitude : (n_freqs,) complex array
        Incident plane-wave spectral amplitude for normalization — source-only, hence a
        CONSTANT for the gradient. Use :func:`_incident_spectrum_amplitude`
        (``f0, bandwidth, freqs, dt, n_steps``) with the same TFSF waveform.

    Returns
    -------
    jnp.ndarray
        ``rcs_linear`` (n_freqs, n_theta, n_phi) in m², differentiable in ``ntff_data``
        (hence in the scatterer permittivity upstream). Take ``10·log10`` for dBsm.

    See Also
    --------
    compute_rcs : the numpy orchestrator (Mie-validated monostatic bin).
    compute_far_field_jax : the differentiable far-field this builds on.
    """
    ff = compute_far_field_jax(ntff_data, box, grid, theta, phi)
    power_scat = jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2  # (nf, nθ, nφ)
    power_inc = jnp.abs(jnp.asarray(e_inc_amplitude)) ** 2          # (nf,)
    safe_power_inc = jnp.where(power_inc > 0, power_inc, 1e-30)
    return 4.0 * jnp.pi * power_scat / safe_power_inc[:, None, None]


def compute_rcs(
    grid: Grid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    f0: float,
    bandwidth: float = 0.5,
    theta_inc: float = 0.0,
    phi_inc: float = 0.0,
    polarization: str = "ez",
    theta_obs: jnp.ndarray | np.ndarray | None = None,
    phi_obs: jnp.ndarray | np.ndarray | None = None,
    freqs: jnp.ndarray | np.ndarray | None = None,
    boundary: str = "cpml",
    cpml_layers: int = 8,
    tfsf_margin: int = 3,
    ntff_offset: int = 1,
    subtract_incident_reference: bool = False,
) -> RCSResult:
    """Compute radar cross section of the scatterer defined in materials.

    Parameters
    ----------
    grid : Grid
        Simulation grid (must already include CPML padding).
    materials : MaterialArrays
        Material arrays with the scatterer defined (e.g., PEC regions
        with very high conductivity or high eps_r).
    n_steps : int
        Number of FDTD timesteps.
    f0 : float
        Center frequency of the Gaussian pulse (Hz).
    bandwidth : float
        Fractional bandwidth of the pulse.
    theta_inc : float
        Incident angle in degrees (0 = +x propagation). Only normal incidence
        (0) is supported; a non-zero value raises ``NotImplementedError`` (the
        oblique TFSF reflection path is unvalidated — issue #404).
    phi_inc : float
        Incident azimuth in degrees (reserved for future oblique support).
    polarization : str
        Electric field polarization: "ez" or "ey".
    theta_obs : array or None
        Observation elevation angles in radians. Default: 0 to pi, 37 points.
    phi_obs : array or None
        Observation azimuth angles in radians. Default: [0, pi/2].
    freqs : array or None
        Frequencies at which to compute RCS (Hz). Default: [f0].
    boundary : str
        Boundary condition type ("cpml" or "pec").
    cpml_layers : int
        Number of CPML layers (must match grid.cpml_layers).
    tfsf_margin : int
        Cells between CPML edge and TFSF boundary.
    ntff_offset : int
        Cells between TFSF boundary and NTFF box (NTFF must be in the
        scattered-field region, i.e., outside the TFSF box). The default
        (1) places the Huygens surface deep in the reactive near field.
        See "Bistatic pattern caveat" below: this default is validated
        for the backscatter/monostatic bin, but is NOT a validated
        bistatic setup, and increasing it does not close the oblique gap
        at test scale.
    subtract_incident_reference : bool
        If True (default False), run a second vacuum (no-scatterer) pass
        with the identical TFSF+NTFF setup and subtract its far-field from
        the target's at the COMPLEX level (E_scat = E_far[target] -
        E_far[vacuum]) before forming the RCS. This is the standard
        total-field/scattered-field normalization; it removes the residual
        TFSF-boundary incident-field leakage that the NTFF box otherwise
        integrates into a spurious forward-oblique lobe (issue #280).
        Doubles the solve cost. Default False keeps the validated
        monostatic path byte-identical; opt in for the bistatic pattern.
        Note: ``monostatic_rcs`` is always computed from the raw (unsubtracted)
        run regardless of this flag -- the leakage nulls at backscatter (~90 dB
        down), so subtraction would change it by <0.02 dB, and keeping the
        validated monostatic extraction untouched is intentional.

    Returns
    -------
    RCSResult
        freqs, theta, phi, rcs_dbsm (dBsm), rcs_linear (m^2),
        monostatic_rcs (dBsm, evaluated exactly at the true backscatter
        direction — for +x incidence that is (theta=pi/2, phi=pi) under
        the farfield r_hat convention; it does not depend on
        theta_obs/phi_obs).

    Bistatic pattern caveat
    -----------------------
    ``RCSResult.monostatic_rcs`` agrees with exact Mie to about 0.06 dB for the
    committed ka~1 PEC-sphere case. It is evaluated from the raw target run at
    the exact backscatter direction, independently of the requested observation
    grid.

    The default unsubtracted ``rcs_dbsm`` / ``rcs_linear`` bistatic pattern is
    NOT VALIDATED. With ``ntff_offset=1`` the oblique bins can be several dB to
    about 20 dB from the reference. Increasing ``ntff_offset`` alone does not
    remove the forward-oblique lobe. An empty-domain run produces the same lobe,
    demonstrating that its dominant cause is target-independent incident-field
    leakage from the discrete TFSF boundary rather than the scatterer staircase.

    Set ``subtract_incident_reference=True`` for the validated bistatic path.
    It doubles the solve cost, subtracts a matching vacuum run at the complex
    far-field level, and removes the target-independent leakage. In the
    committed ka~1 PEC-sphere H-plane comparison, the largest 15-90 degree
    difference from exact Mie falls from 10.5 to 1.2 dB, full-pattern dB
    correlation is 0.965, mean absolute difference is 0.42 dB, and the
    backscatter difference is about 0.06 dB. Those values validate only the
    stated sphere, frequency, polarization, angle cut, and discretization. An
    independent Bempp cube comparison confirms the raw discrepancy on another
    geometry; it is not validation of a corrected cube pattern.

    After subtraction, refine curved surfaces, repeat with a longer run, vary
    NTFF placement and CPML thickness, and enlarge the domain when deep pattern
    nulls matter. Compare each new target and configuration with an analytic or
    independent reference before treating the bistatic values as quantitative.
    """
    # Defaults
    if theta_obs is None:
        theta_obs = np.linspace(0.01, np.pi - 0.01, 37)
    else:
        theta_obs = np.asarray(theta_obs, dtype=np.float64)

    if phi_obs is None:
        phi_obs = np.array([0.0, np.pi / 2])
    else:
        phi_obs = np.asarray(phi_obs, dtype=np.float64)

    if freqs is None:
        freqs_arr = np.array([f0], dtype=np.float64)
    else:
        freqs_arr = np.asarray(freqs, dtype=np.float64)

    len(freqs_arr)
    dx = grid.dx
    dt = grid.dt

    # Fail-loud guard (#404): a non-zero theta_inc silently dispatches init_tfsf
    # to the 2D-aux oblique path, whose reflection accuracy is UNVALIDATED (the
    # extracted |R| does not track the injection angle — issue #404, R2-STOP
    # 2026-07-20). Only normal incidence is supported here; refuse rather than
    # return an unvalidated oblique RCS.
    if abs(float(theta_inc)) > 1e-6:
        raise NotImplementedError(
            f"compute_rcs(theta_inc={theta_inc}) — oblique incidence is not "
            "supported: the oblique TFSF path's reflection accuracy is "
            "unvalidated (issue #404). Use theta_inc=0.0 (normal incidence)."
        )

    # --- 1. Set up TFSF source ---
    tfsf_cfg, tfsf_st = init_tfsf(
        nx=grid.nx,
        dx=dx,
        dt=dt,
        cpml_layers=cpml_layers,
        tfsf_margin=tfsf_margin,
        f0=f0,
        bandwidth=bandwidth,
        amplitude=1.0,
        polarization=polarization,
        direction="+x",
        angle_deg=theta_inc,
    )

    # --- 2. Set up NTFF box just outside TFSF box ---
    # NTFF box must be in scattered-field region (outside TFSF box).
    # Place it `ntff_offset` cells outside the TFSF boundaries.
    # Per-face CPML thicknesses from grid.face_layers so asymmetric
    # cavity configurations (e.g., thin ground-plane z_lo) still keep
    # the NTFF surface outside active-CPML cells.
    fl = getattr(grid, "face_layers", None) or {
        k: grid.cpml_layers for k in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    }
    ntff_i_lo = tfsf_cfg.x_lo - ntff_offset
    ntff_i_hi = tfsf_cfg.x_hi + ntff_offset + 1
    ntff_j_lo = fl["y_lo"] + ntff_offset
    ntff_j_hi = grid.ny - fl["y_hi"] - ntff_offset
    ntff_k_lo = fl["z_lo"] + ntff_offset
    ntff_k_hi = grid.nz - fl["z_hi"] - ntff_offset

    # Clamp to valid range
    ntff_i_lo = max(ntff_i_lo, 1)
    ntff_i_hi = min(ntff_i_hi, grid.nx - 2)
    ntff_j_lo = max(ntff_j_lo, 1)
    ntff_j_hi = min(ntff_j_hi, grid.ny - 2)
    ntff_k_lo = max(ntff_k_lo, 1)
    ntff_k_hi = min(ntff_k_hi, grid.nz - 2)

    ntff_box = NTFFBox.from_grid(
        grid,
        i_lo=ntff_i_lo,
        i_hi=ntff_i_hi,
        j_lo=ntff_j_lo,
        j_hi=ntff_j_hi,
        k_lo=ntff_k_lo,
        k_hi=ntff_k_hi,
        freqs=jnp.array(freqs_arr, dtype=jnp.float32),
    )

    # --- 3. Run simulation with TFSF + NTFF ---
    result = run(
        grid,
        materials,
        n_steps,
        boundary=boundary,
        tfsf=(tfsf_cfg, tfsf_st),
        ntff=ntff_box,
    )

    # --- 4. Compute far-field from NTFF data ---
    ff = compute_far_field(
        result.ntff_data,
        ntff_box,
        grid,
        theta_obs,
        phi_obs,
    )

    # --- 4b. Optional two-run incident-reference subtraction (issue #280) ---
    # The discrete TFSF boundary does not perfectly cancel the incident field in
    # the scattered-field region, so the NTFF box integrates a residual incident
    # leakage into a spurious forward-oblique far-field lobe (an empty-domain run
    # produces the same lobe with NO scatterer; it nulls at backscatter, ~90 dB
    # below its forward-oblique peak). Because that leakage is target-INDEPENDENT
    # (the TFSF injects the same incident field with or without a scatterer) it
    # cancels under a two-run subtraction at the COMPLEX far-field level:
    # E_scat = E_far[target] - E_far[vacuum] (equivalent to subtracting the
    # near-fields then transforming, by linearity of the NTFF integral). This is
    # the standard total-field/scattered-field normalization (as in Meep's
    # scattered-field runs, done here at the far-field level). Validated vs the
    # EXACT Mie bistatic on a PEC sphere (tests/fixtures/rcs280_reference_
    # subtraction/): H-plane forward-oblique gap 10.5 -> 1.2 dB, shape
    # correlation -0.14 -> 0.965; backscatter (leakage ~0) essentially unchanged.
    # Default OFF keeps the validated monostatic path byte-identical.
    if subtract_incident_reference:
        vacuum = MaterialArrays(
            eps_r=jnp.ones(grid.shape, dtype=jnp.float32),
            sigma=jnp.zeros(grid.shape, dtype=jnp.float32),
            mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
        )
        ref_result = run(
            grid, vacuum, n_steps, boundary=boundary,
            tfsf=(tfsf_cfg, tfsf_st), ntff=ntff_box,
        )
        ff_ref = compute_far_field(
            ref_result.ntff_data, ntff_box, grid, theta_obs, phi_obs,
        )
        ff = FarFieldResult(
            E_theta=ff.E_theta - ff_ref.E_theta,
            E_phi=ff.E_phi - ff_ref.E_phi,
            theta=ff.theta, phi=ff.phi, freqs=ff.freqs,
        )

    # --- 5. Compute incident field spectrum for normalization ---
    E_inc_spectrum = _incident_spectrum_amplitude(
        f0, bandwidth, freqs_arr, dt, n_steps,
    )

    # --- 6. Compute RCS ---
    # RCS = 4*pi * |E_far|^2 / |E_inc|^2
    # where E_far already includes the jk/(4*pi) factor from compute_far_field,
    # so: |E_far|^2 = |E_theta|^2 + |E_phi|^2
    # The far-field result has E_theta, E_phi in V*m (omitting 1/r).
    # The NTFF formulation gives: E_far(r) = (jk/4*pi*r) * [N, L integrals]
    # So |E_far * r|^2 = |E_theta|^2 + |E_phi|^2 as returned.
    #
    # RCS = 4*pi * r^2 * |E_scat|^2 / |E_inc|^2
    #     = 4*pi * |E_far_r|^2 / |E_inc|^2
    # where E_far_r = r * E_far (the quantity returned by compute_far_field).

    E_theta = np.asarray(ff.E_theta, dtype=np.complex128)  # (nf, n_theta, n_phi)
    E_phi = np.asarray(ff.E_phi, dtype=np.complex128)

    power_scat = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2  # (nf, n_theta, n_phi)
    power_inc = np.abs(E_inc_spectrum) ** 2  # (nf,)

    # Avoid division by zero
    safe_power_inc = np.where(power_inc > 0, power_inc, 1e-30)

    rcs_linear = 4.0 * np.pi * power_scat / safe_power_inc[:, None, None]

    # Convert to dBsm
    rcs_dbsm = 10.0 * np.log10(np.maximum(rcs_linear, 1e-30))

    # --- 7. Extract monostatic (backscatter) RCS ---
    # Backscatter is the direction OPPOSITE the incident propagation
    # vector.  The TFSF source above propagates along +x
    # (``direction="+x"`` in the init_tfsf call); a non-zero theta_inc
    # tilts the propagation in the x-y plane for "ez" polarization and
    # in the x-z plane for "ey" (see the t_delay lines in
    # rfx/sources/tfsf_2d.py).  So:
    #     k_hat = (cos(theta_inc), sin(theta_inc), 0)   for "ez"
    #     k_hat = (cos(theta_inc), 0, sin(theta_inc))   for "ey"
    #     b_hat = -k_hat                                (backscatter)
    # Under the far-field convention (rfx/farfield.py:
    # r_hat = [sin(th)cos(ph), sin(th)sin(ph), cos(th)], theta = polar
    # angle from +z), the spherical angles of b_hat are
    #     theta_back = arccos(b_hat_z)
    #     phi_back   = atan2(b_hat_y, b_hat_x)
    # For the supported normal-incidence case (theta_inc = 0) this gives
    # (theta_back, phi_back) = (pi/2, pi), i.e. the -x direction.
    # NOTE (issue #276): the pre-fix code hardcoded (theta=pi, phi=0),
    # which under this convention is the -z BROADSIDE direction, not
    # backscatter.
    #
    # The far field is evaluated EXACTLY at the backscatter direction
    # (one extra direction on the already-accumulated NTFF data) instead
    # of argmin-snapping to the observation grid: the default phi grid
    # ([0, pi/2]) does not contain phi=pi, so grid snapping would
    # silently return a different cut.
    theta_inc_rad = float(np.radians(theta_inc))
    if polarization == "ey":
        k_hat = np.array([np.cos(theta_inc_rad), 0.0, np.sin(theta_inc_rad)])
    else:  # "ez"
        k_hat = np.array([np.cos(theta_inc_rad), np.sin(theta_inc_rad), 0.0])
    b_hat = -k_hat
    theta_back = float(np.arccos(np.clip(b_hat[2], -1.0, 1.0)))
    phi_back = float(np.mod(np.arctan2(b_hat[1], b_hat[0]), 2.0 * np.pi))

    ff_back = compute_far_field(
        result.ntff_data,
        ntff_box,
        grid,
        np.array([theta_back]),
        np.array([phi_back]),
    )
    Eb_theta = np.asarray(ff_back.E_theta, dtype=np.complex128)[:, 0, 0]
    Eb_phi = np.asarray(ff_back.E_phi, dtype=np.complex128)[:, 0, 0]
    power_back = np.abs(Eb_theta) ** 2 + np.abs(Eb_phi) ** 2  # (nf,)
    mono_linear = 4.0 * np.pi * power_back / safe_power_inc
    monostatic_rcs = 10.0 * np.log10(np.maximum(mono_linear, 1e-30))

    return RCSResult(
        freqs=freqs_arr,
        theta=theta_obs,
        phi=phi_obs,
        rcs_dbsm=rcs_dbsm,
        rcs_linear=rcs_linear,
        monostatic_rcs=monostatic_rcs,
    )
