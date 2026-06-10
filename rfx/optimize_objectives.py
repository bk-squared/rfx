"""Pre-built objective functions for ``rfx.optimize.optimize()``.

Each factory returns a callable ``objective(result) -> scalar`` that
is compatible with :func:`rfx.optimize.optimize` and differentiable
through JAX.

**S-parameter vs time-domain objectives**

The frequency-domain objectives (``minimize_s11``, ``maximize_s21``,
``target_impedance``, ``maximize_bandwidth``) require ``result.s_params``
to be populated.  This happens automatically when running a simulation
via ``Simulation.run()`` with ports, but the lightweight forward pass
used inside ``optimize()`` and ``topology_optimize()`` does **not**
compute S-parameters (it would break JAX traceability).

For gradient-based optimization, use the **time-domain proxy**
objectives instead:

- ``minimize_reflected_energy`` -- proxy for minimizing S11
- ``maximize_transmitted_energy`` -- proxy for maximizing S21

These operate directly on ``result.time_series`` and are fully
JAX-differentiable through the FDTD scan loop.

Typical usage
-------------
>>> from rfx import Simulation, optimize, DesignRegion
>>> from rfx.optimize_objectives import minimize_reflected_energy
>>> sim = Simulation(freq_max=10e9, domain=(0.05, 0.05, 0.025))
>>> # ... add geometry, ports, probes ...
>>> obj = minimize_reflected_energy(port_probe_idx=0)
>>> result = optimize(sim, region, obj, n_iters=50)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_time_series(result, objective_name: str) -> None:
    ts = result.time_series
    if ts.ndim < 2 or ts.shape[-1] == 0:
        raise ValueError(
            f"{objective_name} requires time-series probe data, but the "
            "forward result has none. Re-run forward() with "
            "emit_time_series=True (the default)."
        )


def _db_to_linear(db: float) -> float:
    """Convert dB power to linear scale: 10^(dB/10)."""
    return 10.0 ** (db / 10.0)


def _find_freq_indices(result_freqs: jnp.ndarray, target_freqs: jnp.ndarray) -> jnp.ndarray:
    """Find nearest indices in *result_freqs* for each entry in *target_freqs*.

    Uses pure JAX operations so the index lookup is differentiable-friendly
    (the indices themselves are integer, but the downstream gather is
    stop-gradient safe because we only use them for indexing).
    """
    # (n_target, n_result) absolute differences
    diffs = jnp.abs(target_freqs[:, None] - result_freqs[None, :])
    return jnp.argmin(diffs, axis=1)


# ---------------------------------------------------------------------------
# Public objective factories
# ---------------------------------------------------------------------------

def minimize_s11(
    freqs: jnp.ndarray | np.ndarray,
    target_db: float = -10.0,
) -> Callable:
    """Minimize |S11|² over specified frequencies.

    Returns a callable ``objective(result) -> scalar`` that computes the
    mean |S11|² across the nearest frequency bins.  If the mean S11 is
    already below *target_db* the loss is clamped to zero.

    Parameters
    ----------
    freqs : array-like, shape (n,)
        Target frequencies (Hz) over which to minimize S11.
    target_db : float
        Threshold in dB.  When mean |S11|² is below this, loss = 0.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    target_freqs = jnp.asarray(freqs, dtype=jnp.float32)
    threshold_linear = _db_to_linear(target_db)  # e.g. -10 dB -> 0.1

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params  # (n_ports, n_ports, n_freqs) complex
        if s_params is None:
            raise ValueError(
                "minimize_s11 requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use minimize_reflected_energy() as a "
                "time-domain proxy objective for gradient-based optimization."
            )
        result_freqs = result.freqs  # (n_freqs,)

        s11 = s_params[0, 0, :]  # (n_freqs,) complex
        mag_sq = jnp.abs(s11) ** 2  # |S11|² linear

        indices = _find_freq_indices(
            jnp.asarray(result_freqs, dtype=jnp.float32), target_freqs,
        )
        selected = mag_sq[indices]  # (n_target,)
        mean_mag_sq = jnp.mean(selected)

        # Clamp: if already below target, loss = 0
        return jnp.maximum(mean_mag_sq - threshold_linear, 0.0)

    return objective


def maximize_s21(
    freqs: jnp.ndarray | np.ndarray,
) -> Callable:
    """Maximize |S21|² (transmission) over specified frequencies.

    Returns ``-mean|S21|²`` so that *minimizing* this objective
    maximizes transmission.

    Parameters
    ----------
    freqs : array-like, shape (n,)
        Target frequencies (Hz).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    target_freqs = jnp.asarray(freqs, dtype=jnp.float32)

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params  # (n_ports, n_ports, n_freqs) complex
        if s_params is None:
            raise ValueError(
                "maximize_s21 requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use maximize_transmitted_energy() as a "
                "time-domain proxy objective for gradient-based optimization."
            )
        result_freqs = result.freqs

        s21 = s_params[1, 0, :]  # (n_freqs,) complex
        mag_sq = jnp.abs(s21) ** 2

        indices = _find_freq_indices(
            jnp.asarray(result_freqs, dtype=jnp.float32), target_freqs,
        )
        selected = mag_sq[indices]
        return -jnp.mean(selected)

    return objective


def target_impedance(
    freq: float,
    z_target: float = 50.0,
) -> Callable:
    """Minimize |Z_in − z_target|² at a specific frequency.

    Computes input impedance from S11: ``Z_in = Z0 * (1 + S11) / (1 − S11)``
    where Z0 = 50 Ω (standard reference impedance).

    Parameters
    ----------
    freq : float
        Target frequency (Hz).
    z_target : float
        Desired input impedance (Ω).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    freq_arr = jnp.asarray([freq], dtype=jnp.float32)
    z0 = 50.0  # reference impedance for S-parameter normalization

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params
        if s_params is None:
            raise ValueError(
                "target_impedance requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use a time-domain proxy objective "
                "for gradient-based optimization."
            )
        result_freqs = result.freqs

        s11 = s_params[0, 0, :]  # (n_freqs,) complex

        idx = _find_freq_indices(
            jnp.asarray(result_freqs, dtype=jnp.float32), freq_arr,
        )
        s11_at_f = s11[idx[0]]  # scalar complex

        # Z_in = Z0 * (1 + S11) / (1 - S11), with safety clamp
        denom = 1.0 - s11_at_f
        # Avoid division by zero: add tiny imaginary part
        denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12 + 0j, denom)
        z_in = z0 * (1.0 + s11_at_f) / denom

        return jnp.abs(z_in - z_target) ** 2

    return objective


def maximize_bandwidth(
    f_center: float,
    f_bw: float,
    s11_threshold: float = -10.0,
) -> Callable:
    """Maximize the bandwidth where |S11| < threshold.

    Evaluates S11 over a frequency band ``[f_center − f_bw/2, f_center + f_bw/2]``
    and returns a soft loss that is lower when more frequency bins satisfy
    ``|S11|_dB < s11_threshold``.

    The loss is computed as the mean of ``max(|S11|²_dB − threshold, 0)``
    across the band, so it drives the optimizer to push all bins below
    threshold simultaneously, effectively maximizing matched bandwidth.

    Parameters
    ----------
    f_center : float
        Center frequency (Hz).
    f_bw : float
        Bandwidth span (Hz).
    s11_threshold : float
        Threshold in dB (default −10 dB).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    f_lo = f_center - f_bw / 2.0
    f_hi = f_center + f_bw / 2.0

    def objective(result) -> jnp.ndarray:
        s_params = result.s_params
        if s_params is None:
            raise ValueError(
                "maximize_bandwidth requires result.s_params but got None. "
                "The optimize() / topology_optimize() forward pass does not "
                "compute S-parameters. Use minimize_reflected_energy() as a "
                "time-domain proxy objective for gradient-based optimization."
            )
        result_freqs = jnp.asarray(result.freqs, dtype=jnp.float32)

        s11 = s_params[0, 0, :]  # (n_freqs,) complex
        mag_sq = jnp.abs(s11) ** 2

        # Select frequency bins within the target band
        mask = (result_freqs >= f_lo) & (result_freqs <= f_hi)
        # Convert to dB: 10*log10(|S11|²), with floor to avoid log(0)
        mag_sq_safe = jnp.maximum(mag_sq, 1e-30)
        s11_db = 10.0 * jnp.log10(mag_sq_safe)

        # Hinge loss: penalize bins above threshold
        excess = jnp.maximum(s11_db - s11_threshold, 0.0)

        # Weighted mean over in-band frequencies (out-of-band contribute 0)
        n_in_band = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(excess * mask) / n_in_band

    return objective


def maximize_directivity(
    theta_target: float,
    phi_target: float,
    *,
    n_theta: int = 37,
    n_phi: int = 73,
    log_ratio: bool = False,
    eps: float = 1e-37,
) -> Callable:
    """Maximize directivity in a target direction (ratio-based, scale-invariant).

    Computes the directivity ratio ``U(θ_target, φ_target) / P_rad`` where
    ``U = |E_θ|² + |E_φ|²`` is the radiation intensity and ``P_rad`` is
    the total radiated power integrated over the upper hemisphere. The
    ratio is scale-invariant, so the absolute magnitude of the NTFF
    spectral integral drops out — gradients reflect pattern shape only.

    Prior versions used absolute ``|E|²`` at the target direction, which
    is ~1e-27 in rfx's spectral NTFF convention and produces zero
    gradients in ``topology_optimize`` (GitHub issue #32).

    Parameters
    ----------
    theta_target : float
        Polar angle in radians [0, π].
    phi_target : float
        Azimuthal angle in radians [0, 2π].
    n_theta : int, optional
        Number of polar samples in [0, π/2] for the hemisphere
        integration (default 37 → 2.5° spacing).
    n_phi : int, optional
        Number of azimuthal samples in [0, 2π] (default 73 → 5° spacing).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)

    Parameters (continued)
    ----------------------
    log_ratio : bool, optional
        If True, optimize ``-(log U - log P)`` (full, sign-correct quotient
        gradient ``U'/U - P'/P``) instead of the legacy ``-U/stop_gradient(P)``.
        Default False (legacy, back-compat). USE ``log_ratio=True`` (or the
        ``maximize_directivity_logratio`` factory) for any DoF that changes
        total radiated power — see the warning below.
    eps : float, optional
        Floor for the ``log_ratio`` log arguments (default 1e-37). It only
        guards ``log(0)``; the log-ratio gradient ``U'/U`` is scale-invariant, so
        eps must be BELOW the working U/P magnitude (rfx spectral NTFF U,P are
        ~1e-27 in full antenna runs, ~1e-32 in small/early sims). A floor at or
        above U/P (e.g. the old 1e-30 on a 1e-32 sim) clamps the log argument and
        zeros the gradient — keep eps well below min(U, P). (The legacy path's
        denominator floor is hardcoded 1e-30 and is unaffected by this.)

    Notes
    -----
    In the default (``log_ratio=False``) mode, ``stop_gradient`` is applied to
    the P_rad denominator: the ratio is scale-invariant, so a shape-preserving
    scaling leaves the directivity unchanged, and letting the denominator carry
    gradient would add noise + NaN risk when P_rad is near zero.

    WARNING (GitHub #129): that legacy mode drops the ``-U*P'/P^2`` quotient-rule
    term, so it yields WRONG-SIGN gradients for any DoF that changes total
    radiated power — conductors / PEC topology (``topology_optimize(material_fg=
    "pec")``, Yagi director/reflector offsets+lengths, parasitics), lossy/sigma
    DoFs, and (magnitude-only) dielectric reshaping. It is correct ONLY for pure
    shape-preserving DoFs (the original #32 target, ``P_rad ~ const``). For
    power-changing DoFs pass ``log_ratio=True``: the full quotient
    ``grad = U'/U - P'/P`` is sign-correct, still scale-invariant (preserving the
    #32 property), monotone in the directivity (same optimum), and NaN-safe via
    independent ``eps`` floors (each log argument is O(1), avoiding the
    1e-27/1e-30 backward blow-up of a naive full quotient).
    """
    theta_arr = np.array([theta_target])
    phi_arr = np.array([phi_target])
    theta_hemi = np.linspace(0.0, np.pi / 2.0, n_theta)
    phi_hemi = np.linspace(0.0, 2.0 * np.pi, n_phi)

    def objective(result) -> jnp.ndarray:
        from rfx.farfield import compute_far_field

        ntff_data = result.ntff_data
        ntff_box = result.ntff_box

        if ntff_data is None or ntff_box is None:
            raise ValueError(
                "maximize_directivity requires a simulation with an NTFF box. "
                "Use sim.add_ntff_box(...) before running."
            )

        grid = getattr(result, "grid", None)
        if grid is None:
            grid = getattr(getattr(result, "state", None), "grid", None)
        if grid is None:
            raise ValueError(
                "maximize_directivity requires result.grid so the far-field "
                "can be evaluated from NTFF data."
            )

        # U at target — (n_freqs, 1, 1)
        ff_tgt = compute_far_field(ntff_data, ntff_box, grid, theta_arr, phi_arr)
        u_target = (jnp.abs(ff_tgt.E_theta[:, 0, 0]) ** 2
                    + jnp.abs(ff_tgt.E_phi[:, 0, 0]) ** 2)

        # Hemisphere P_rad: ∬ U(θ,φ) sin(θ) dθ dφ — (n_freqs, n_theta, n_phi)
        ff_hemi = compute_far_field(ntff_data, ntff_box, grid, theta_hemi, phi_hemi)
        u_hemi = (jnp.abs(ff_hemi.E_theta) ** 2 + jnp.abs(ff_hemi.E_phi) ** 2)
        sin_theta = jnp.asarray(np.sin(theta_hemi), dtype=u_hemi.dtype)
        # trapz in θ then φ
        integrand = u_hemi * sin_theta[None, :, None]
        p_rad_phi = jnp.trapezoid(integrand, theta_hemi, axis=1)
        p_rad = jnp.trapezoid(p_rad_phi, phi_hemi, axis=1)  # (n_freqs,)

        if log_ratio:
            # Full, NaN-safe quotient: grad = U'/U - P'/P (== true dD/dθ up to
            # the positive factor 1/D, so sign-correct + monotone). Floors are
            # applied INDEPENDENTLY so each log argument is O(1) — a single
            # shared 1e-30 floor on U/P (the ~1e-27 spectral NTFF scale) makes
            # the backward pass NaN. No stop_gradient -> correct sign for
            # power-changing DoFs (GitHub #129).
            log_dir = (jnp.log(jnp.maximum(u_target, eps))
                       - jnp.log(jnp.maximum(p_rad, eps)))
            return -jnp.mean(log_dir)
        # Legacy scale-invariant ratio. CORRECT ONLY for shape-preserving DoFs;
        # WRONG-SIGN for power-changing DoFs (see #129 warning above). The 1e-30
        # denominator floor is hardcoded here for exact back-compat (the `eps`
        # param tunes the log_ratio floor only).
        directivity = u_target / (jax.lax.stop_gradient(p_rad) + 1e-30)
        # Minimizing -D = maximizing directivity; average across freqs
        return -jnp.mean(directivity)

    return objective


# Alias for discoverability — the issue #32 fix aligned the default with
# the ratio-based formulation, so `maximize_directivity_ratio` is simply
# the new default under an explicit name.
maximize_directivity_ratio = maximize_directivity


def maximize_directivity_logratio(
    theta_target: float,
    phi_target: float,
    *,
    n_theta: int = 37,
    n_phi: int = 73,
    eps: float = 1e-37,
) -> Callable:
    """Directivity objective with the full, sign-correct quotient gradient.

    Equivalent to ``maximize_directivity(..., log_ratio=True)``: optimizes
    ``-(log U - log P)`` so the gradient ``U'/U - P'/P`` carries the
    ``-U*P'/P^2`` term the legacy ``stop_gradient`` mode drops. PREFER this for
    any power-changing DoF (PEC / topology / lossy / dielectric reshaping); the
    legacy default gives wrong-sign gradients for those (GitHub #129). See
    :func:`maximize_directivity` for the full warning and parameters.
    """
    return maximize_directivity(
        theta_target, phi_target,
        n_theta=n_theta, n_phi=n_phi, log_ratio=True, eps=eps,
    )


# ---------------------------------------------------------------------------
# Time-domain proxy objectives (for use with optimize / topology_optimize)
# ---------------------------------------------------------------------------

def minimize_reflected_energy(
    port_probe_idx: int = 0,
    *,
    late_fraction: float = 0.5,
) -> Callable:
    """Time-domain S11 proxy: minimize late-time reflected energy at port.

    Computes the ratio of energy in the second half of the probe time
    series (dominated by reflections) to energy in the first half
    (dominated by the incident pulse).  Minimizing this ratio drives
    the optimizer toward better impedance matching.

    This objective works with ``optimize()`` and ``topology_optimize()``
    because it uses only ``result.time_series`` (no S-parameters needed).

    Parameters
    ----------
    port_probe_idx : int
        Index into ``result.time_series`` columns identifying the probe
        co-located with the excitation port (default 0).
    late_fraction : float
        Fraction of the time series considered "late" (default 0.5).
        A value of 0.5 means the second half is treated as reflection.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        _require_time_series(result, "minimize_reflected_energy")
        ts = result.time_series[:, port_probe_idx]
        n = ts.shape[0]
        split = int(n * (1.0 - late_fraction))
        early_energy = jnp.sum(ts[:split] ** 2) + 1e-30
        late_energy = jnp.sum(ts[split:] ** 2)
        return late_energy / early_energy

    return objective


def minimize_s11_at_freq(
    target_freq: float,
    port_probe_idx: int = 0,
    *,
    incident_fraction: float = 0.25,
    dt: float | None = None,
) -> Callable:
    """Single-frequency |S11|² proxy for the differentiable ``forward()`` path.

    Unlike :func:`minimize_s11` (which requires S-parameters from ``run()``)
    this objective works on the time-series output of ``forward()``, so it
    composes with ``optimize()`` / ``topology_optimize()``. Issue #50.

    Method
    ------
    At the target frequency ω, split the DFT of the port probe into an
    incident component (``X_inc``) and a reflected component (``X_refl``)
    using a time-gating heuristic:

        X_inc  = DFT(ts[:q],  ω)        # source-only window (zero-padded)
        X_tot  = DFT(ts[:N], ω)         # full window
        X_refl = X_tot − X_inc          # by linearity of DFT

    When (a) the source pulse has decayed by sample ``q`` and (b) the
    first reflection does not arrive before sample ``q``, the split is
    clean and ``|X_refl / X_inc|² ≈ |S11(ω)|²``. Violating either
    assumption causes overlap contamination and biases the estimate.
    Callers controlling strongly resonant / long-round-trip structures
    should either enlarge the simulation window, narrow the source
    bandwidth to compress the pulse, or prefer a wave-decomposition
    (V/I) probe which is exact.

    Previously this objective returned ``|X_tot/X_inc|² = |1+S11|²``
    (fixed 2026-04, commit 1923db2)
    (see regression test ``tests/test_minimize_s11_at_freq_physical.py``),
    which minimises toward ``S11 = −1`` (perfect short) rather than
    ``S11 = 0`` (matched load). Fixed on branch
    ``fix/lumped-port-s11-at-freq``.

    Parameters
    ----------
    target_freq : float
        Frequency of interest (Hz).
    port_probe_idx : int
        Index into ``result.time_series`` for the port-co-located probe.
    incident_fraction : float
        Fraction of the leading time series treated as incident-only
        (default 0.25). Shorten for large DUTs with quick round-trip,
        lengthen for slow resonators.
    dt : float, optional
        Time step. If None, read from ``result.dt`` at call time.

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)

    .. deprecated::
        The time-gating heuristic fails for short-round-trip antennas
        (DRA, thin-substrate patches) where source pulse and reflection
        overlap. Prefer :func:`minimize_s11_at_freq_wave_decomp` together
        with ``Simulation.forward(port_s11_freqs=...)`` (issue #72).
    """
    import warnings as _w
    _w.warn(
        "minimize_s11_at_freq uses a time-gating heuristic that is biased "
        "for short-round-trip antennas (issue #72). Prefer "
        "minimize_s11_at_freq_wave_decomp + "
        "Simulation.forward(port_s11_freqs=...).",
        DeprecationWarning,
        stacklevel=2,
    )

    def objective(result) -> jnp.ndarray:
        _require_time_series(result, "minimize_s11_at_freq")
        ts = result.time_series[:, port_probe_idx]
        n = ts.shape[0]
        _dt = float(result.dt) if dt is None else float(dt)
        t = jnp.arange(n) * _dt
        omega = 2.0 * jnp.pi * float(target_freq)
        cos_t = jnp.cos(omega * t)
        sin_t = jnp.sin(omega * t)
        # X_tot: DFT over the full window (incident + reflection).
        X_tot_re = jnp.sum(ts * cos_t)
        X_tot_im = jnp.sum(ts * sin_t)
        # X_inc: DFT over the leading window (source-only, zero-padded
        # elsewhere). Because DFT is linear in the sample sequence, this
        # equals DFT(ts_windowed, full N) and is directly subtractable
        # from X_tot to recover the reflected-wave DFT.
        q = max(1, int(n * float(incident_fraction)))
        X_inc_re = jnp.sum(ts[:q] * cos_t[:q])
        X_inc_im = jnp.sum(ts[:q] * sin_t[:q])
        # X_refl = X_tot − X_inc. Divide by |X_inc|² to get |S11|².
        X_refl_re = X_tot_re - X_inc_re
        X_refl_im = X_tot_im - X_inc_im
        power_refl = X_refl_re ** 2 + X_refl_im ** 2
        power_inc = X_inc_re ** 2 + X_inc_im ** 2
        return power_refl / (power_inc + 1e-30)

    return objective


def minimize_s11_at_freq_wave_decomp(
    target_freq: float,
    port_idx: int = 0,
) -> Callable:
    """Single-frequency |S11|² objective via V/I wave decomposition (issue #72).

    Reads ``result.s_params`` populated by
    ``Simulation.forward(port_s11_freqs=...)`` (which uses the
    JIT-integrated lumped port DFT path) and returns ``|S11(f)|²`` at the
    nearest stored frequency.  The wave decomposition

        a = (-V + Z0·I) / (2·√Z0)        # incident
        b = (-V - Z0·I) / (2·√Z0)        # reflected
        S11 = b / a

    is exact: there is no source-pulse-vs-reflection separability
    assumption, so it works for short-round-trip antennas (DRA,
    thin-substrate patches) where the legacy
    :func:`minimize_s11_at_freq` is biased.

    Parameters
    ----------
    target_freq : float
        Frequency of interest (Hz).
    port_idx : int
        Index into the port list (default 0 = first lumped port).

    Returns
    -------
    callable(ForwardResult) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        s_params = getattr(result, "s_params", None)
        freqs = getattr(result, "freqs", None)
        if s_params is None or freqs is None:
            raise ValueError(
                "minimize_s11_at_freq_wave_decomp requires forward(...) "
                "to be called with port_s11_freqs=... so that "
                "result.s_params / result.freqs are populated. See "
                "issue #72."
            )
        if s_params.ndim == 1:
            s11 = s_params
        else:
            s11 = s_params[port_idx]
        idx = jnp.argmin(jnp.abs(freqs - float(target_freq)))
        s_at = s11[idx]
        return jnp.real(s_at) ** 2 + jnp.imag(s_at) ** 2

    return objective


def maximize_transmitted_energy(
    output_probe_idx: int = -1,
) -> Callable:
    """Time-domain S21 proxy: maximize energy at an output probe.

    Returns the negated total squared energy at the output probe, so
    that *minimizing* this objective maximizes transmission.

    This objective works with ``optimize()`` and ``topology_optimize()``
    because it uses only ``result.time_series`` (no S-parameters needed).

    Parameters
    ----------
    output_probe_idx : int
        Index into ``result.time_series`` columns for the output probe.
        Default -1 (last probe).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        _require_time_series(result, "maximize_transmitted_energy")
        ts = result.time_series[:, output_probe_idx]
        return -jnp.sum(ts ** 2)

    return objective


# ---------------------------------------------------------------------------
# Near-field probe-array beam steering (no NTFF required)
# ---------------------------------------------------------------------------

def steer_probe_array(
    target_probe_idx: int,
    suppress_probe_idx: int = 0,
    *,
    late_fraction: float = 0.5,
) -> Callable:
    """Steer radiation toward *target_probe_idx* and away from *suppress_probe_idx*.

    Uses a near-field probe array as a differentiable surrogate for
    far-field beam steering.  Place probes at different spatial positions
    around the antenna (e.g., above-left and above-right), then this
    objective maximizes the power ratio between target and suppressed
    probe.

    This avoids NTFF DFT entirely and works reliably in float32.

    Setup example::

        # Probes at different angles in the near field
        sim.add_probe((x_center - 0.02, y_center, z_above), "ez")  # probe 0: left
        sim.add_probe((x_center + 0.02, y_center, z_above), "ez")  # probe 1: right

        # Steer toward probe 1 (right), suppress probe 0 (left)
        obj = steer_probe_array(target_probe_idx=1, suppress_probe_idx=0)

    Parameters
    ----------
    target_probe_idx : int
        Column index in ``result.time_series`` to maximize.
    suppress_probe_idx : int
        Column index to suppress (default 0).
    late_fraction : float
        Fraction of time series to use (late portion, after source decays).

    Returns
    -------
    callable(Result) -> scalar (JAX-differentiable)
    """
    def objective(result) -> jnp.ndarray:
        _require_time_series(result, "steer_probe_array")
        ts = result.time_series
        n = ts.shape[0]
        start = int(n * (1.0 - late_fraction))
        target_energy = jnp.sum(ts[start:, target_probe_idx] ** 2)
        suppress_energy = jnp.sum(ts[start:, suppress_probe_idx] ** 2)
        # Maximize ratio: minimize -(target / (suppress + eps))
        return -(target_energy / (suppress_energy + 1e-12))

    return objective
