"""Radar cross sections and incident-spectrum normalization."""

from __future__ import annotations

from rfx.core.yee import MaterialArrays
from rfx.farfield import FarFieldResult
from rfx.farfield import NTFFBox
from rfx.farfield import _ntff_face_pads
from rfx.measurement.farfield import compute_far_field
from rfx.grid import C0
from rfx.grid import Grid
from rfx.rcs import RCSResult
from rfx.rcs import ScatteringResponse
from rfx.stepping import run_uniform_loop as run
from rfx.sources.tfsf import init_tfsf, is_oblique
from rfx.measurement.incident import measure_normal_incident_spectrum
import jax.numpy as jnp
import numpy as np


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
        Method-B TFSF (2.5-D: open transverse y, thin-periodic z). The
        far-field PATTERN / specular direction is validated at oblique incidence
        (specular peak phi = 180 - theta_inc at theta_obs = 90 deg). Measured
        envelope on the committed gate configuration (2.2-lambda plate, 700
        steps, public path, corner-inclusive kernels): peak error 0 deg at
        theta_inc=20 and 3 deg at theta_inc=40 (4 deg on the pre-fix exclusive
        kernels), gated at +/-6 deg, with a 3-7 deg peak-azimuth
        sensitivity to domain size at fixed dx/plate/CPML (measured on the PRE-fix exclusive
        kernels; not re-measured in-tree
        on the inclusive kernels) —
        i.e. the argmax azimuth of the broad specular lobe is NOT converged to
        ~1 deg. See ``tests/oracle/test_oblique_rcs_specular.py``.

        ABSOLUTE oblique sigma (all numbers measured): the
        normalization is the MEASURED 1D-aux incident spectrum (see
        ``tfsf_oblique_open.measure_incident_spectrum``), which removed a
        +6.58/+4.51 dB (theta 20/40 deg) inflation the analytic normal-path
        spectrum caused. Against a PO uniform-aperture oracle for the
        2.2-lambda plate at 20 and 40 deg, four settled meshes (lambda/30 to
        lambda/120) give specular Delta +0.08 to +0.16 dB at 5 GHz,
        converged within 0.006 dB from lambda/60 to lambda/120.
        The returned sigma is that
        of an aperture of height ``h_eff = (k_hi - k_lo)*dx`` of the NTFF
        z-box (span-doubling ratio 4.000 measured — sigma scales as
        h_eff^2); the oblique path's hardcoded 2-cell midplane box means
        h_eff = 2*dx, so scale by ``(L_z/h_eff)^2`` for a physical strip
        height ``L_z``. The always-on gate-grid test locks Delta within
        0.03 dB of +0.0833/+0.1583 dB at 20/40 deg.
        Requires ``polarization='ez'`` and a uniform
        grid; other combos raise ``NotImplementedError``.
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
        otherwise produces a spurious forward-oblique lobe.
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

    This route is float32 only: it has no run-precision argument.
    """
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
        freqs_arr = freqs_arr.astype(np.float32).astype(np.float64)
    dx = grid.dx
    dt = grid.dt

    _oblique = is_oblique(float(theta_inc))
    reference_index = None
    reference_position = None
    if phase_reference is not None:
        if (_oblique or float(theta_inc) != 0.0 or float(phi_inc) != 0.0
                or not isinstance(grid, Grid) or grid.is_2d or boundary != "cpml"
                or cpml_layers <= 0
                or any(face.declared != cpml_layers for face in grid.boundary_depths)
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
        dtype=jnp.float32)
    from rfx.boundaries.depths import FACES, grid_face_depths
    from rfx.boundaries.features import refuse_ntff_reflecting_faces
    walls = (tuple(face for face in FACES if not (_oblique and face.startswith("z_")))
             if boundary not in ("cpml", "upml") else None)
    faces = grid_face_depths(grid, pec_faces=walls)
    if _oblique:
        faces = tuple(face for face in faces if not face.name.startswith("z_"))
    refuse_ntff_reflecting_faces(faces, lane="compute_rcs")
    if reference_index is not None and not tfsf_cfg.x_lo <= reference_index <= tfsf_cfg.x_hi:
        raise ValueError("phase_reference.x must lie inside the total-field slab")

    if _oblique:
        from rfx.sources.tfsf_oblique_open import validate_vacuum_boundary

        validate_vacuum_boundary(materials, tfsf_cfg)

    fl = _ntff_face_pads(grid)
    ntff_i_lo = tfsf_cfg.x_lo - ntff_offset
    ntff_i_hi = tfsf_cfg.x_hi + ntff_offset + 1
    ntff_j_lo = fl["y_lo"] + ntff_offset
    ntff_j_hi = grid.ny - fl["y_hi"] - ntff_offset
    if _oblique:
        _kz = grid.nz // 2
        ntff_k_lo = _kz - 1
        ntff_k_hi = _kz + 1
    else:
        ntff_k_lo = fl["z_lo"] + ntff_offset
        ntff_k_hi = grid.nz - fl["z_hi"] - ntff_offset

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

    _run_kw = dict(boundary=boundary, tfsf=(tfsf_cfg, tfsf_st), ntff=ntff_box)
    if _oblique:
        _run_kw.update(cpml_axes="xy", periodic=(False, False, True))
    result = run(grid, materials, n_steps, **_run_kw)

    ff = compute_far_field(
        result.ntff_data,
        ntff_box,
        grid,
        theta_obs,
        phi_obs,
    )

    if subtract_incident_reference:
        vacuum = MaterialArrays(
            eps_r=jnp.ones(grid.shape, dtype=jnp.float32),
            sigma=jnp.zeros(grid.shape, dtype=jnp.float32),
            mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
        )
        if reference_position is not None:
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

    if _oblique:
        from rfx.measurement.incident import measure_incident_spectrum

        E_inc_spectrum = measure_incident_spectrum(
            tfsf_cfg, tfsf_st, n_steps, freqs_arr, dx,
        )
    else:
        E_inc_spectrum = measure_normal_incident_spectrum(
            tfsf_cfg, tfsf_st, n_steps, freqs_arr, dt,
            **({"reference_index": reference_index} if reference_index is not None else {}),
        )
    if reference_index is not None and (
            not np.all(np.isfinite(E_inc_spectrum)) or np.any(np.abs(E_inc_spectrum) == 0)):
        raise ValueError("phase_reference cannot normalize a zero or non-finite incident spectrum")


    E_theta = np.asarray(ff.E_theta, dtype=np.complex128)
    E_phi = np.asarray(ff.E_phi, dtype=np.complex128)

    power_scat = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2
    power_inc = np.abs(E_inc_spectrum) ** 2

    safe_power_inc = np.where(power_inc > 0, power_inc, 1e-30)

    rcs_linear = 4.0 * np.pi * power_scat / safe_power_inc[:, None, None]

    rcs_dbsm = 10.0 * np.log10(np.maximum(rcs_linear, 1e-30))

    theta_inc_rad = float(np.radians(theta_inc))
    if polarization == "ey":
        k_hat = np.array([np.cos(theta_inc_rad), 0.0, np.sin(theta_inc_rad)])
    else:
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
    power_back = np.abs(Eb_theta) ** 2 + np.abs(Eb_phi) ** 2
    mono_linear = 4.0 * np.pi * power_back / safe_power_inc
    monostatic_rcs = 10.0 * np.log10(np.maximum(mono_linear, 1e-30))

    from rfx.measurement.record_end import injection_end
    from types import SimpleNamespace
    from rfx.measurement.coverage import accumulator_coverage, with_coverage, warn_uncovered
    coverage = accumulator_coverage(SimpleNamespace(
        time_series=np.empty((n_steps, 0)), grid=grid, ntff_box=ntff_box), tfsf=tfsf_cfg)
    source_end = coverage.max_end()
    settling_db = None
    settling_witness = {"status": "absent", "source_end_index": source_end, "reason":
                        "no time-domain record; add an identification probe"}

    settling_db, settling_witness = with_coverage(settling_db, settling_witness, coverage)
    warn_uncovered(settling_witness)

    rcs = RCSResult(
        freqs=freqs_arr,
        theta=theta_obs,
        phi=phi_obs,
        rcs_dbsm=rcs_dbsm,
        rcs_linear=rcs_linear,
        monostatic_rcs=monostatic_rcs,
        settling_db=settling_db, settling_witness=settling_witness,
    )
    if reference_index is None:
        return rcs

    incident = E_inc_spectrum
    th, ph = np.meshgrid(theta_obs, phi_obs, indexing="ij")
    s_hat = np.stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)), axis=-1)
    phase = np.exp(-2j * np.pi * freqs_arr[:, None, None] / C0
                   * (s_hat @ reference_position)[None, :, :])
    factor = phase / incident[:, None, None]
    return ScatteringResponse(
        rcs=rcs, F_theta=E_theta * factor, F_phi=E_phi * factor,
        incident_spectrum=incident, reference_position=reference_position,
        polarization=polarization,
    )


def _incident_spectrum_amplitude(
    f0: float,
    bandwidth: float,
    freqs: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Compute the frequency-domain amplitude of the TFSF incident pulse.

    The TFSF 1-D source uses GaussianPulse's charge-free differentiated
    Gaussian: s(t) = charge_free_dg(t, tau, cutoff=3),
    where tau = 1/(f0 * bandwidth * pi), t0 = 3*tau.

    We compute the DFT of this waveform at the requested frequencies.
    This is the WAVEFORM, not the incident field: the auxiliary line adds it
    as a soft source and launches 1/(2 S cos(k~dx/2)) of it, so
    it is not an RCS normalization -- use
    ``rfx.measurement.incident.measure_normal_incident_spectrum``.
    """
    tau = 1.0 / (f0 * bandwidth * np.pi)
    from rfx.sources.sources import charge_free_dg
    times = np.arange(n_steps) * dt
    waveform = np.asarray(charge_free_dg(times, tau, 3.0))

    amplitudes = np.zeros(len(freqs), dtype=np.complex128)
    for i, f in enumerate(freqs):
        phase = np.exp(-1j * 2 * np.pi * f * times)
        amplitudes[i] = np.sum(waveform * phase) * dt

    return amplitudes
