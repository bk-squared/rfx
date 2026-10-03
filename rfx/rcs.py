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
validated because the NTFF side faces cross the total-field slab and
record the full incident field at oblique angles. The two-run ``subtract_incident_reference=True`` path is validated for
the documented ka~1 PEC-sphere comparison; other targets and configurations
require their own convergence and reference checks.

Reference: Taflove & Hagness, Ch. 8-9.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import C0, Grid
from rfx.core.yee import MaterialArrays
from rfx.farfield import (
    FarFieldResult, NTFFBox, compute_far_field, compute_far_field_jax,
    _ntff_face_pads,
)
from rfx.sources.tfsf import init_tfsf, measure_normal_incident_spectrum
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
    ``monostatic_rcs`` is compared with exact Mie in the committed
    ``tests/fixtures/rcs_sphere_mie/`` PEC-sphere case. The default unsubtracted
    ``rcs_dbsm`` / ``rcs_linear`` bistatic pattern is NOT validated: the NTFF side faces
    record the full incident field inside the total-field slab, and
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


class ScatteringResponse(NamedTuple):
    """One incident-polarization column of a phase-referenced scattering matrix.

    With the ``exp(+j omega t)`` convention, the outgoing field at
    ``reference_position + R * s_hat`` is ``exp(-jkR) / R`` times
    ``(F_theta, F_phi) * E_inc(reference_position)``. F has units of metres,
    shape ``(n_freqs, n_theta, n_phi)``, and uses the spherical output basis
    of ``rcs.theta``/``rcs.phi``. Run both ``ey`` and ``ez`` for two columns.
    ``incident_spectrum`` is the actual E-field DFT at that reference, not
    the source waveform. Grid/record convergence still bounds accuracy.
    """
    rcs: RCSResult
    F_theta: np.ndarray
    F_phi: np.ndarray
    incident_spectrum: np.ndarray
    reference_position: np.ndarray
    polarization: str


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

    We compute the DFT of this waveform at the requested frequencies.
    This is the WAVEFORM, not the incident field: the auxiliary line adds it
    as a soft source and launches 1/(2 S cos(k~dx/2)) of it (issue #820), so
    it is not an RCS normalization -- use
    ``rfx.sources.tfsf.measure_normal_incident_spectrum``.
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
        CONSTANT for the gradient. At normal incidence use
        :func:`rfx.sources.tfsf.measure_normal_incident_spectrum`
        (``cfg, state, n_steps, freqs, dt``) on the same ``init_tfsf`` config the run
        uses — the incident the grid carries, as ``compute_rcs`` does. The source
        waveform's own DFT (``_incident_spectrum_amplitude``) is about 1.1 dB larger
        than that incident and makes σ low by the same 1.1 dB (issue #820).

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
    phase_reference: tuple[float, float, float] | None = None,
) -> RCSResult | ScatteringResponse:
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
        Incident angle in degrees (0 = +x propagation). Non-zero values tilt the
        illumination in the x-y plane and route through the open-domain oblique
        Method-B TFSF (2.5-D: open transverse y, thin-periodic z). Only the
        far-field PATTERN / specular direction is validated at oblique incidence
        (specular peak phi = 180 - theta_inc at theta_obs = 90 deg). Measured
        envelope on the committed gate configuration (2.2-lambda plate, 700
        steps, public path, corner-inclusive kernels): peak error 0 deg at
        theta_inc=20 and 3 deg at theta_inc=40 (4 deg on the pre-fix exclusive
        kernels), gated at +/-6 deg, with a 3-7 deg peak-azimuth
        sensitivity to domain size at fixed dx/plate/CPML (PR #461 audit,
        measured on the PRE-fix exclusive kernels; not re-measured in-tree
        on the inclusive kernels) —
        i.e. the argmax azimuth of the broad specular lobe is NOT converged to
        ~1 deg. See ``tests/oracle/test_oblique_rcs_specular.py``.

        ABSOLUTE oblique sigma (issue #471 F7, all numbers measured): the
        normalization is the MEASURED 1D-aux incident spectrum (see
        ``tfsf_oblique_open.measure_incident_spectrum``), which removed a
        +6.58/+4.51 dB (theta 20/40 deg) inflation the analytic normal-path
        spectrum caused. Against a PO uniform-aperture oracle on the gate
        grid (dx=2 mm = lambda/30): specular-peak Delta +0.86/+0.85 dB
        (theta 20/40), lobe-flat over +/-12 deg. The returned sigma is that
        of an aperture of height ``h_eff = (k_hi - k_lo)*dx`` of the NTFF
        z-box (span-doubling ratio 4.000 measured — sigma scales as
        h_eff^2); the oblique path's hardcoded 2-cell midplane box means
        h_eff = 2*dx, so scale by ``(L_z/h_eff)^2`` for a physical strip
        height ``L_z``. CAVEAT (measured, pinned-geometry): halving dx to
        lambda/60 moves Delta to -1.55 dB — a 2.4 dB resolution
        sensitivity, so treat absolute oblique sigma as a +/-2 dB-class
        number at lambda/30-lambda/60, NOT sub-dB (the committed test gates
        |Delta| <= 1.5 dB at the fixed gate grid as a regression lock, not
        an accuracy claim). Requires ``polarization='ez'`` and a uniform
        grid; other combos raise ``NotImplementedError`` (issue #404 arc).
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
        total-field/scattered-field normalization; it removes the full incident
        field read by NTFF side faces inside the total-field slab, which
        otherwise produces a spurious forward-oblique lobe (issue #280).
        Doubles the solve cost. Default False keeps the validated
        monostatic path byte-identical; opt in for the bistatic pattern.
        Without ``phase_reference``, ``monostatic_rcs`` is computed from the raw
        (unsubtracted) run regardless of this flag, preserving the separately
        checked monostatic extraction.
    phase_reference : (x, y, z) in metres, optional
        Opt in to a ``ScatteringResponse`` with complex scattering amplitudes
        referenced to this physical point. Requires normal +x incidence, a
        uniform 3-D grid with symmetric CPML, and
        ``subtract_incident_reference=True``. The x coordinate must be on an
        E-node plane inside the total-field slab; it is never silently
        snapped. The normal incident wave is uniform in y/z. Only finite,
        positive frequencies with a nonzero incident spectrum are admitted;
        the caller must additionally check adequate source bandwidth and
        record/mesh convergence. Without this option the return is unchanged.

    Returns
    -------
    RCSResult
        freqs, theta, phi, rcs_dbsm (dBsm), rcs_linear (m^2),
        monostatic_rcs (dBsm, evaluated exactly at the true backscatter
        direction — for +x incidence that is (theta=pi/2, phi=pi) under
        the farfield r_hat convention; it does not depend on
        theta_obs/phi_obs).
        With ``phase_reference`` set, ``ScatteringResponse.rcs`` holds the
        RCS normalized by the incident measured at that reference, consistent
        with its complex F components. Frequencies are the realized NTFF bins.

    Bistatic pattern caveat
    -----------------------
    ``RCSResult.monostatic_rcs`` is evaluated from the raw target run at
    exact backscatter, independently of the observation grid. Its current
    Mie comparison is in ``tests/fixtures/rcs_sphere_mie/``.

    The default unsubtracted ``rcs_dbsm`` / ``rcs_linear`` bistatic pattern
    is NOT VALIDATED. The source's total-field slab is infinite transversely,
    so the side faces of a closed NTFF box record the full incident field.
    Increasing ``ntff_offset`` does not remove that contribution; off-axis
    bins cannot be interpreted as the scatterer's bistatic response.

    Set ``subtract_incident_reference=True`` for the reference-subtracted
    path. It doubles the solve cost and subtracts a matching vacuum run at
    the complex far-field level. The committed sphere comparison is in
    ``tests/fixtures/rcs280_reference_subtraction/``; its scope is limited
    to that sphere, frequency, polarization, angle cut and discretization.

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

    if phase_reference is not None:
        # NTFF accumulates at float32 bins. Normalize and de-embed at those
        # same realized frequencies, not at a subtly different requested bin.
        freqs_arr = freqs_arr.astype(np.float32).astype(np.float64)
    dx = grid.dx
    dt = grid.dt

    # ---- Oblique incidence routing (issue #404 arc — fork (a) S4/S5) ----
    # theta_inc != 0 routes the illumination through the OPEN-DOMAIN oblique
    # Method-B TFSF (real float32, dispersion-matched 1D-aux-along-k̂ + a 4-edge
    # box; rfx/sources/tfsf_oblique_open.py), NOT the periodic complex-Bloch 2D-aux
    # path (#414) which assumes lateral periodicity and is the wrong tool for a
    # compact scatterer. Method B is a 2.5-D path: k̂ lies in the x-y plane, the
    # transverse y-axis is OPEN (CPML) and z is thin/periodic (z-invariant far
    # field), so the fields stay real and the NTFF/DFT read physical fields.
    #
    # VALIDATED by the specular-peak gate (tests/oracle/test_oblique_rcs_specular.py):
    # the bistatic far-field of a finite PEC plate (normal +x) peaks at the
    # reflection-law specular direction phi = pi - theta_inc (theta_obs = pi/2)
    # across {0, 20, 40} deg, within the measured envelope in the theta_inc
    # parameter docstring above (peak error <= 4 deg, gated +/-6 deg, 3-7 deg
    # domain-size sensitivity — NOT ~1 deg); the theta->0 limit reduces to
    # backscatter (= specular at normal). The +/-6 deg tolerance is ~0.2 of the
    # 2.2-lambda plate's ~30 deg physical-optics 3 dB specular lobe at
    # theta_inc=40 (0.886*lambda/(W*cos(theta))) — argmax on a 1 deg grid
    # inside a broad lobe, an envelope pin, not a loose bound.
    # ABSOLUTE sigma (issue #471 F7): normalized by the MEASURED 1D-aux
    # incident spectrum (measure_incident_spectrum below), validated vs a PO
    # uniform-aperture oracle at the gate grid to +0.86/+0.85 dB (theta 20/40)
    # with a measured 2.4 dB resolution sensitivity lambda/30 -> lambda/60 —
    # a +/-2 dB-class calibration, envelope-pinned by
    # tests/unit/farfield/test_oblique_rcs_absolute_sigma.py. sigma corresponds to an
    # aperture height h_eff = (k_hi-k_lo)*dx = 2*dx (z-faces cancel for
    # x-y-plane observation; span-doubling ratio 4.000 measured). Kept fenced
    # for combos Method B does not support: non-ez polarization and
    # non-uniform / distributed grids.
    _oblique = abs(float(theta_inc)) > 1e-6
    reference_index = None
    reference_position = None
    if phase_reference is not None:
        if (_oblique or float(theta_inc) != 0.0 or float(phi_inc) != 0.0
                or not isinstance(grid, Grid) or grid.is_2d or boundary != "cpml"
                or cpml_layers <= 0
                or any(v != cpml_layers for v in grid.face_layers.values())
                or any(getattr(grid, f"pad_{axis}_{side}") != cpml_layers
                       for axis in "xyz" for side in ("lo", "hi"))):
            raise NotImplementedError(
                "phase_reference requires normal +x incidence on a uniform 3-D "
                "grid with symmetric CPML matching cpml_layers"
            )
        if not subtract_incident_reference:
            raise ValueError("phase_reference requires subtract_incident_reference=True")
        reference_position = np.asarray(phase_reference, dtype=np.float64)
        if (reference_position.shape != (3,) or not np.all(np.isfinite(reference_position))
                or np.any(reference_position < 0)
                or np.any(reference_position > np.asarray(grid.domain))):
            raise ValueError("phase_reference must be a finite physical point inside grid.domain")
        reference_index = grid.index_of("x", float(reference_position[0]))
        if not np.isclose(grid.node_of("x", reference_index), reference_position[0],
                          rtol=0.0, atol=8 * np.finfo(float).eps
                          * max(abs(reference_position[0]), float(dx))):
            raise ValueError("phase_reference.x must lie on an E-node plane; it is not snapped")
        if (n_steps <= 0 or freqs_arr.ndim != 1 or freqs_arr.size == 0
                or not np.all(np.isfinite(freqs_arr)) or np.any(freqs_arr <= 0)
                or np.any(freqs_arr >= 0.5 / float(dt))):
            raise ValueError("phase_reference needs positive n_steps and frequencies below Nyquist")
    if _oblique:
        if polarization != "ez":
            raise NotImplementedError(
                f"compute_rcs(theta_inc={theta_inc}, polarization={polarization!r}) "
                "— oblique RCS is supported for polarization='ez' only (the "
                "open-domain Method-B TFSF is ez / transverse-y); 'ey' is future work."
            )
        if getattr(grid, "dz", None) is not None:
            raise NotImplementedError(
                "compute_rcs oblique incidence is not supported on non-uniform "
                "grids (Method B requires a uniform single-device grid). "
                "Use a uniform Grid or theta_inc=0.0."
            )
        # Method B is 2.5-D: z is thin + PERIODIC and the NTFF z-box is a
        # 2-cell midplane span, so the returned sigma is for an infinite
        # z-periodic replication of the target sampled at the midplane. Refuse
        # a z-varying (genuinely 3-D) target rather than return that number
        # silently (review F4 — "refuse rather than return an unvalidated
        # number").
        for _mname in ("eps_r", "sigma", "mu_r"):
            _marr = np.asarray(getattr(materials, _mname))
            if _marr.ndim == 3 and _marr.shape[2] > 1 and not np.array_equal(
                _marr, np.broadcast_to(_marr[:, :, :1], _marr.shape)
            ):
                raise NotImplementedError(
                    f"compute_rcs(theta_inc={theta_inc}): materials.{_mname} varies "
                    "along z, but oblique incidence uses the 2.5-D Method-B path "
                    "(thin-periodic z, midplane NTFF) which is only meaningful for "
                    "z-invariant targets. Make the target z-invariant or use "
                    "theta_inc=0.0."
                )

    # --- 1. Set up TFSF source ---
    # angle_deg=0 ignores `method` (normal 1D-aux path, byte-identical); the
    # oblique branch selects Method B explicitly. ny/nz are only consumed by the
    # oblique path and ignored at normal incidence.
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
        ny=grid.ny,
        nz=grid.nz,
        method="methodB" if _oblique else "bloch",
    )
    if reference_index is not None and not tfsf_cfg.x_lo <= reference_index <= tfsf_cfg.x_hi:
        raise ValueError("phase_reference.x must lie inside the total-field slab")

    if _oblique:
        # #471 F5: compute_rcs never ran the preflight vacuum validator. Run
        # the identical all-four-planes check here so a target crossing a
        # TFSF boundary plane fails loud instead of corrupting the scattered
        # field (the runner-side validator covers the Simulation API lane).
        from rfx.sources.tfsf_oblique_open import validate_vacuum_boundary

        validate_vacuum_boundary(materials, tfsf_cfg)

    # --- 2. Set up NTFF box just outside TFSF box ---
    # NTFF box must be in scattered-field region (outside TFSF box).
    # Place it `ntff_offset` cells outside the TFSF boundaries.
    # Use realized pads for placement as well as the phase origin: declared
    # face_layers may still carry a nonzero budget on a PEC/PMC face.
    fl = _ntff_face_pads(grid)
    ntff_i_lo = tfsf_cfg.x_lo - ntff_offset
    ntff_i_hi = tfsf_cfg.x_hi + ntff_offset + 1
    ntff_j_lo = fl["y_lo"] + ntff_offset
    ntff_j_hi = grid.ny - fl["y_hi"] - ntff_offset
    if _oblique:
        # Method B is 2.5-D: z is THIN + PERIODIC (no real z-CPML), so the box
        # z-faces sit symmetric about the mid-plane with a 2-cell span rather than
        # inset from a (nonexistent) z-CPML. The two z-faces carry identical
        # z-invariant tangential fields with opposite outward normals, so they
        # cancel for x-y-plane (theta_obs=pi/2) observation — the validated cut.
        _kz = grid.nz // 2
        ntff_k_lo = _kz - 1
        ntff_k_hi = _kz + 1
    else:
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

    # The box must enclose the whole injected region, or the far-field
    # integral is not over the scattered field. run() checks this too, but it
    # only sees indices; the caller of compute_rcs never chose one. Check here
    # with the levers that actually produced them, before the run starts.
    from rfx.farfield import require_box_encloses_injected_region
    from rfx.sources.tfsf import tfsf_injection_planes

    _clamped = [
        name for name, wanted, got in (
            ("i_lo", tfsf_cfg.x_lo - ntff_offset, ntff_i_lo),
            ("i_hi", tfsf_cfg.x_hi + ntff_offset + 1, ntff_i_hi),
            ("j_lo", fl["y_lo"] + ntff_offset, ntff_j_lo),
            ("j_hi", grid.ny - fl["y_hi"] - ntff_offset, ntff_j_hi),
        ) if wanted != got
    ]
    _ctx = (
        f"compute_rcs placed this box itself from ntff_offset={ntff_offset}, "
        f"tfsf_margin={tfsf_margin}, cpml_layers={cpml_layers} on a "
        f"{grid.nx}x{grid.ny}x{grid.nz}-cell grid "
        f"(domain {tuple(float(v) for v in grid.domain)} m at dx={dx:.4g} m); "
        f"realized faces i=({ntff_i_lo}, {ntff_i_hi}), "
        f"j=({ntff_j_lo}, {ntff_j_hi}), k=({ntff_k_lo}, {ntff_k_hi}), "
        f"injection planes {tfsf_injection_planes(tfsf_cfg)}."
    )
    if _clamped:
        _ctx += (
            " The domain bounds pulled " + ", ".join(_clamped) + " back from "
            "the requested placement, so raising ntff_offset alone will not "
            "move it — enlarge the domain or lower cpml_layers/tfsf_margin."
        )
    else:
        _ctx += (
            " Raise or lower ntff_offset so that every face clears the planes "
            "above, or change tfsf_margin to move the planes."
        )
    require_box_encloses_injected_region(
        ntff_box, tfsf_injection_planes(tfsf_cfg), context=_ctx,
        shape=grid.shape)

    # --- 3. Run simulation with TFSF + NTFF ---
    # Open-domain Method B forces the transverse y-axis OPEN (CPML) with
    # thin-periodic z; normal incidence keeps the historical full-open defaults
    # (byte-identical: `_run_kw` collapses to the pre-relaxation call).
    _run_kw = dict(boundary=boundary, tfsf=(tfsf_cfg, tfsf_st), ntff=ntff_box)
    if _oblique:
        _run_kw.update(cpml_axes="xy", periodic=(False, False, True), pec_axes="")
    result = run(grid, materials, n_steps, **_run_kw)

    # --- 4. Compute far-field from NTFF data ---
    ff = compute_far_field(
        result.ntff_data,
        ntff_box,
        grid,
        theta_obs,
        phi_obs,
    )

    # --- 4b. Optional two-run incident-reference subtraction (issue #280) ---
    # The NTFF side faces cut through the transversely infinite total-field
    # slab and read the full incident field. It is target-independent, so
    # subtract the matching vacuum run at the complex far-field level before
    # forming power: E_scat = E_far[target] - E_far[vacuum]. See the committed
    # sphere comparison in tests/fixtures/rcs280_reference_subtraction/.
    # Default OFF preserves the separately checked monostatic path.
    if subtract_incident_reference:
        vacuum = MaterialArrays(
            eps_r=jnp.ones(grid.shape, dtype=jnp.float32),
            sigma=jnp.zeros(grid.shape, dtype=jnp.float32),
            mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
        )
        if reference_position is not None:
            # Complex subtraction requires the same numerical precision in
            # both solves, including when the caller supplies float64 media.
            vacuum = MaterialArrays(jnp.ones_like(materials.eps_r),
                                    jnp.zeros_like(materials.sigma),
                                    jnp.ones_like(materials.mu_r))
        ref_result = run(grid, vacuum, n_steps, **_run_kw)
        ff_ref = compute_far_field(
            ref_result.ntff_data, ntff_box, grid, theta_obs, phi_obs,
        )
        ff = FarFieldResult(
            E_theta=ff.E_theta - ff_ref.E_theta,
            E_phi=ff.E_phi - ff_ref.E_phi,
            theta=ff.theta, phi=ff.phi, freqs=ff.freqs,
        )

    # --- 5. Compute incident field spectrum for normalization ---
    if _oblique:
        # Method B hardcodes its own waveform (t0=40·dt, tau=12·dt modulated
        # Gaussian) and its source->aux launch response is theta-dependent, so
        # the analytic normal-path spectrum is the wrong denominator (issue
        # #471 F7: sigma inflated +6.58 dB at theta=20 deg, +4.51 dB at 40 deg
        # on the gate grid). Normalize by the MEASURED 1D-aux spectrum instead
        # — a standalone replay of the same aux the box injects from.
        from rfx.sources.tfsf_oblique_open import measure_incident_spectrum

        E_inc_spectrum = measure_incident_spectrum(
            tfsf_cfg, tfsf_st, n_steps, freqs_arr, dx,
        )
    else:
        # Normal incidence: the 1-D auxiliary line ADDS the waveform at one
        # node (a soft source), and the wave it launches across the TF/SF
        # plane is 1/(2 S cos(k~dx/2)) of it -- -1.14 dB at 40 cells per
        # wavelength and the uniform-grid Courant number. sigma goes as
        # |E_scat/E_inc|^2 with E_scat following the launched wave, so
        # dividing by the waveform's own DFT made every normal-incidence
        # sigma low by that same 1.14 dB (issue #820). Replay
        # the same auxiliary line and normalize by what it carries.
        E_inc_spectrum = measure_normal_incident_spectrum(
            tfsf_cfg, tfsf_st, n_steps, freqs_arr, dt,
            **({"reference_index": reference_index} if reference_index is not None else {}),
        )
    if reference_index is not None and (
            not np.all(np.isfinite(E_inc_spectrum)) or np.any(np.abs(E_inc_spectrum) == 0)):
        raise ValueError("phase_reference cannot normalize a zero or non-finite incident spectrum")

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
    if reference_index is not None:
        ff_back_ref = compute_far_field(
            ref_result.ntff_data, ntff_box, grid,
            np.array([theta_back]), np.array([phi_back]),
        )
        ff_back = ff_back._replace(
            E_theta=ff_back.E_theta - ff_back_ref.E_theta,
            E_phi=ff_back.E_phi - ff_back_ref.E_phi,
        )
    Eb_theta = np.asarray(ff_back.E_theta, dtype=np.complex128)[:, 0, 0]
    Eb_phi = np.asarray(ff_back.E_phi, dtype=np.complex128)[:, 0, 0]
    power_back = np.abs(Eb_theta) ** 2 + np.abs(Eb_phi) ** 2  # (nf,)
    mono_linear = 4.0 * np.pi * power_back / safe_power_inc
    monostatic_rcs = 10.0 * np.log10(np.maximum(mono_linear, 1e-30))

    rcs = RCSResult(
        freqs=freqs_arr,
        theta=theta_obs,
        phi=phi_obs,
        rcs_dbsm=rcs_dbsm,
        rcs_linear=rcs_linear,
        monostatic_rcs=monostatic_rcs,
    )
    if reference_index is None:
        return rcs

    incident = E_inc_spectrum
    th, ph = np.meshgrid(theta_obs, phi_obs, indexing="ij")
    s_hat = np.stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), axis=-1)
    # NTFF uses global physical positions: remove the outgoing origin phase.
    phase = np.exp(-2j * np.pi * freqs_arr[:, None, None] / C0
                   * (s_hat @ reference_position)[None, :, :])
    factor = phase / incident[:, None, None]
    return ScatteringResponse(
        rcs=rcs, F_theta=E_theta * factor, F_phi=E_phi * factor,
        incident_spectrum=incident, reference_position=reference_position,
        polarization=polarization,
    )
