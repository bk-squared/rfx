"""Identifiability & uncertainty quantification for differentiable full-wave S11.

This module is the analysis layer on top of the physically-scaled, AD-traceable
S11 forward used by :func:`rfx.differentiable_material_fit.calibrate_material_s11`.
Given the same ``sim_factory`` and frequency grid, it builds a **real-valued**
residual (stacked ``[Re(S11), Im(S11)]``), takes its AD Jacobian w.r.t. the
log-space material parameters, and turns that Jacobian into a local
identifiability / uncertainty report:

    log_params --(exp)--> poles --> Simulation.forward(port_s11_freqs=...).s_params
        --> [Re(S11), Im(S11)]              (residual, real-valued)
        --> J = jacfwd(residual)            (2*n_freqs x n_params, real)
        --> F = Jᵀ J / noise_std²           (Fisher information)
        --> eigen / SVD / covariance        (identifiability report)

Noise model & approximation (READ THIS)
---------------------------------------
The Fisher information assumes **i.i.d. zero-mean Gaussian noise on the real and
imaginary parts of the measured S11**, each with standard deviation
``noise_std`` (dimensionless, in linear S11 units).  Under that model the
maximum-likelihood covariance is the inverse Fisher information (the
Cramér-Rao lower bound), and this module reports a **local Laplace / Cramér-Rao
approximation of the posterior around the evaluation point** — NOT a full
Bayesian posterior.  It is only valid in the regime where the residual is
approximately linear in the parameters over the scale of the uncertainty
(the standard local-sensitivity assumption).  The condition number of ``F`` is
independent of ``noise_std`` (which scales ``F`` uniformly); ``noise_std`` only
sets the absolute Cramér-Rao standard deviations.

Numerical precision
-------------------
The rfx core is single-precision (complex64), so the Jacobian ``J`` is computed
in float32.  The **linear-algebra analysis** (eigendecomposition, SVD, matrix
inversion, correlation matrix) is performed in float64 by promoting ``J``/``F``
before the decomposition.  This is an offline analysis of an already-computed
float32 Jacobian — it does not (and cannot) enable x64 inside the forward FDTD
scan, whose accumulators are hardcoded complex64.  Promoting to float64 for the
decomposition keeps eigenvalues and condition numbers from being dominated by
float32 round-off.

Fail-closed policy
------------------
:func:`identifiability_report` never blindly inverts a singular or
near-singular Fisher matrix.  If the condition number exceeds
``cond_threshold``, or any eigenvalue is non-finite or negative (from numerical
error), it sets ``is_identifiable=False``, falls back to a pseudo-inverse with a
reported numerical ``rank``, and flags the sloppy (unidentified) singular
directions and the parameters that dominate them.  It never silently returns a
garbage standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rfx.differentiable_material_fit import (
    _apply_nuisance,
    _nuisance_tau_scale,
    _params_to_debye_poles,
    _params_to_lorentz_poles,
    _poles_to_params,
)


# ---------------------------------------------------------------------------
# Real-valued S11 residual (AD-traceable)
# ---------------------------------------------------------------------------

def s11_residual_fn(
    sim_factory: Callable,
    freqs: np.ndarray,
    n_debye: int,
    n_lorentz: int,
    *,
    num_periods: float = 14.0,
    checkpoint: bool = True,
    include_nuisance: bool = False,
) -> Callable:
    """Build a real-valued S11 residual over log-space material parameters.

    The returned function ``residual(log_params)`` maps a log-space parameter
    vector (same layout as
    :func:`rfx.differentiable_material_fit._poles_to_params`:
    ``[log(eps_inf), log(de_1), log(tau_1), ..., log(de_L1), log(w0_L1),
    log(delta_L1), ...]``) to a **real** vector of length ``2 * n_freqs``,
    namely ``concatenate([Re(S11), Im(S11)])`` evaluated by
    ``Simulation.forward(port_s11_freqs=...).s_params``.

    With ``include_nuisance=True`` the residual consumes the JOINT parameter
    vector of ``calibrate_material_s11(fit_nuisance=True)`` (issue #273
    Stage 2): the same material layout plus a trailing
    ``[log_alpha, phi, tau_hat]`` at offset ``1 + 2*n_debye + 3*n_lorentz``,
    and the simulated S11 is passed through the SAME
    :func:`~rfx.differentiable_material_fit._apply_nuisance` transform the
    fitter uses — one nuisance model, two consumers, no drift.  ``tau_hat``
    is the one-way delay divided by ``tau_scale = 1/(4*pi*max(freqs))``.

    The output is real-valued so that ``jax.jacfwd`` yields a real Jacobian
    directly (no complex-holomorphicity assumptions).

    Parameters
    ----------
    sim_factory : callable(eps_inf, debye_poles, lorentz_poles) -> Simulation
        Same contract as :func:`calibrate_material_s11`.  Called inside the
        AD tape with traced pole scalars.
    freqs : (n_freqs,) array in Hz
        Frequencies at which S11 is evaluated.
    n_debye, n_lorentz : int
        Number of Debye / Lorentz poles in the parameter vector.
    num_periods : float
        Periods at ``freq_max`` for the forward step count.
    checkpoint : bool
        Use gradient checkpointing in the FDTD scan.
    include_nuisance : bool
        Consume the joint vector with the ``[log_alpha, phi, tau_hat]`` tail
        (see above).  Default False — layout unchanged for existing callers.
    """
    freqs_jnp = jnp.asarray(np.asarray(freqs, dtype=np.float64), dtype=jnp.float32)
    lorentz_offset = 1 + 2 * n_debye
    nuisance_offset = 1 + 2 * n_debye + 3 * n_lorentz
    tau_scale = _nuisance_tau_scale(freqs)

    def residual(log_params):
        eps_inf, debye_poles = _params_to_debye_poles(log_params, n_debye)
        lorentz_poles = _params_to_lorentz_poles(log_params, n_lorentz, lorentz_offset)
        sim = sim_factory(eps_inf, debye_poles, lorentz_poles)
        s11 = sim.forward(
            port_s11_freqs=freqs_jnp,
            num_periods=num_periods,
            checkpoint=checkpoint,
            skip_preflight=True,
        ).s_params
        s11 = jnp.reshape(s11, (-1,))
        if include_nuisance:
            s11 = _apply_nuisance(
                s11, freqs_jnp,
                log_params[nuisance_offset],
                log_params[nuisance_offset + 1],
                log_params[nuisance_offset + 2],
                tau_scale,
            )
        return jnp.concatenate([jnp.real(s11), jnp.imag(s11)])

    return residual


def calibration_param_names(
    n_debye: int,
    n_lorentz: int,
    *,
    include_nuisance: bool = False,
) -> list:
    """Column names for the log-space parameter vector, in Jacobian order.

    Matches the layout of
    :func:`rfx.differentiable_material_fit._poles_to_params` plus, when
    ``include_nuisance=True``, the ``[log_alpha, phi, tau_hat]`` tail of
    :func:`s11_residual_fn`.
    """
    names = ["log_eps_inf"]
    for i in range(1, n_debye + 1):
        names += [f"log_de_{i}", f"log_tau_{i}"]
    for k in range(1, n_lorentz + 1):
        names += [f"log_de_L{k}", f"log_w0_L{k}", f"log_delta_L{k}"]
    if include_nuisance:
        names += ["log_alpha", "phi", "tau_hat"]
    return names


def s11_jacobian(residual_fn: Callable, params: np.ndarray) -> jnp.ndarray:
    """Forward-mode Jacobian ``J = d(residual)/d(params)``.

    Uses :func:`jax.jacfwd` (forward mode) because ``n_params`` is small and
    the residual is a tall vector (``2 * n_freqs >> n_params``), for which
    forward mode is efficient.

    Returns an array of shape ``(2 * n_freqs, n_params)``.
    """
    params = jnp.asarray(params, dtype=jnp.float32)
    return jax.jacfwd(residual_fn)(params)


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------

def fisher_information(J: np.ndarray, noise_std: float) -> np.ndarray:
    """Fisher information matrix ``F = (Jᵀ J) / noise_std²``.

    Assumes i.i.d. zero-mean Gaussian noise with standard deviation
    ``noise_std`` on each real residual component (Re/Im of S11).  Computed in
    float64 for a numerically stable decomposition downstream (see module
    docstring).

    Returns a ``(n_params, n_params)`` array.
    """
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError(f"noise_std must be finite and positive, got {noise_std!r}")
    J64 = np.asarray(J, dtype=np.float64)
    return (J64.T @ J64) / float(noise_std) ** 2


# ---------------------------------------------------------------------------
# Identifiability report
# ---------------------------------------------------------------------------

@dataclass
class IdentifiabilityReport:
    """Local (Laplace / Cramér-Rao) identifiability & uncertainty report.

    Attributes
    ----------
    param_names : list[str]
        Names of the parameters, in Jacobian-column order.
    noise_std : float
        Assumed i.i.d. Gaussian noise std on Re/Im S11 used to scale ``F``.
    eigenvalues : np.ndarray
        Eigenvalues of ``F``, sorted ascending.
    condition_number : float
        ``lambda_max / lambda_min`` of ``F`` (``inf`` if ``lambda_min <= 0``).
    singular_values : np.ndarray
        Singular values of ``J``, sorted descending.
    rank : int
        Numerical rank of ``J`` (singular values above a relative tolerance).
    is_identifiable : bool
        ``False`` (fail-closed) if ``F`` is singular / ill-conditioned /
        non-finite / has a negative eigenvalue.
    covariance : np.ndarray
        Parameter covariance ``C``.  ``inv(F)`` when identifiable, otherwise
        the pseudo-inverse ``pinv(F)`` (values in unidentified directions are
        not meaningful — see ``sloppy_directions``).
    correlation_matrix : np.ndarray
        ``corr_ij = C_ij / sqrt(C_ii C_jj)``.
    crlb_std : np.ndarray
        Cramér-Rao lower-bound std per parameter, ``sqrt(diag(C))``.  Reliable
        only when ``is_identifiable`` is ``True``.
    identifiable_directions : list[dict]
        For each singular value above tolerance: its index, singular value, and
        the parameters that dominate that (well-constrained) singular vector.
    sloppy_directions : list[dict]
        For each singular value below tolerance: its index, singular value, and
        the parameters that dominate that (unidentified) singular vector.
    cond_threshold : float
        Condition-number threshold used for the fail-closed decision.
    reason : str
        Human-readable explanation of the identifiability verdict.
    """

    param_names: list = field(default_factory=list)
    noise_std: float = 1.0
    eigenvalues: np.ndarray | None = None
    condition_number: float = float("inf")
    singular_values: np.ndarray | None = None
    rank: int = 0
    is_identifiable: bool = False
    covariance: np.ndarray | None = None
    correlation_matrix: np.ndarray | None = None
    crlb_std: np.ndarray | None = None
    identifiable_directions: list = field(default_factory=list)
    sloppy_directions: list = field(default_factory=list)
    cond_threshold: float = 1e10
    reason: str = ""


def _dominant_params(vec: np.ndarray, param_names: list, top: int = 3) -> list:
    """Return the parameters (name, weight) that dominate a singular vector."""
    order = np.argsort(np.abs(vec))[::-1]
    out = []
    for i in order[:top]:
        w = float(vec[i])
        if abs(w) < 1e-6:
            continue
        out.append({"param": param_names[i], "weight": w})
    return out


def identifiability_report(
    F: np.ndarray,
    param_names: list,
    *,
    J: np.ndarray | None = None,
    noise_std: float = 1.0,
    cond_threshold: float = 1e10,
    svd_rtol: float = 1e-6,
) -> IdentifiabilityReport:
    """Turn a Fisher matrix (and optional Jacobian) into an identifiability report.

    Performs the eigendecomposition, SVD, covariance, correlation, and
    Cramér-Rao bounds in float64, and applies the fail-closed policy documented
    at the module level.

    Parameters
    ----------
    F : (n_params, n_params) array
        Fisher information matrix from :func:`fisher_information`.
    param_names : list[str]
        Parameter names in column order.
    J : (2*n_freqs, n_params) array, optional
        Jacobian, used for the SVD / singular-direction analysis.  If omitted,
        the SVD is derived from ``F`` (singular values of ``J`` are recovered
        as ``sqrt(noise_std**2 * eig(F))``) and the singular vectors from the
        eigenvectors of ``F``.
    noise_std : float
        Noise std that was used to build ``F`` (recorded in the report; used to
        recover singular values from ``F`` when ``J`` is not given).
    cond_threshold : float
        Fail-closed if ``condition_number`` exceeds this.
    svd_rtol : float
        Relative tolerance (vs. the largest singular value) for the numerical
        rank of ``J``.
    """
    F64 = np.asarray(F, dtype=np.float64)
    n = F64.shape[0]

    report = IdentifiabilityReport(
        param_names=list(param_names),
        noise_std=float(noise_std),
        cond_threshold=float(cond_threshold),
    )

    # --- Eigendecomposition of F (symmetric PSD in exact arithmetic) --------
    # Symmetrize to kill float round-off asymmetry before eigh.
    F_sym = 0.5 * (F64 + F64.T)
    eigvals, eigvecs = np.linalg.eigh(F_sym)  # ascending
    report.eigenvalues = eigvals

    finite = bool(np.all(np.isfinite(eigvals)))
    lam_min = float(eigvals[0])
    lam_max = float(eigvals[-1])

    # --- SVD / singular directions of J -------------------------------------
    if J is not None:
        J64 = np.asarray(J, dtype=np.float64)
        U, svals, Vt = np.linalg.svd(J64, full_matrices=False)  # descending
        right_vecs = Vt  # rows are right singular vectors
    else:
        # Recover from F: sigma_i = noise_std * sqrt(lambda_i), vectors = eigvecs
        order = np.argsort(eigvals)[::-1]
        svals = float(noise_std) * np.sqrt(np.clip(eigvals[order], 0.0, None))
        right_vecs = eigvecs[:, order].T
    report.singular_values = svals

    smax = float(svals[0]) if svals.size else 0.0
    rank_tol = svd_rtol * smax
    rank = int(np.sum(svals > rank_tol))
    report.rank = rank

    # Map each singular direction to the parameters that dominate it.
    for k in range(len(svals)):
        entry = {
            "index": k,
            "singular_value": float(svals[k]),
            "dominant_params": _dominant_params(right_vecs[k], report.param_names),
        }
        if svals[k] > rank_tol:
            report.identifiable_directions.append(entry)
        else:
            report.sloppy_directions.append(entry)

    # --- Fail-closed verdict -------------------------------------------------
    negative_eig = lam_min < -abs(1e-9 * lam_max)
    if not finite:
        report.condition_number = float("inf")
        report.is_identifiable = False
        report.reason = "Fisher matrix has non-finite eigenvalues."
    elif lam_min <= 0.0 or negative_eig:
        report.condition_number = float("inf")
        report.is_identifiable = False
        report.reason = (
            f"Fisher matrix singular / not positive-definite "
            f"(lambda_min={lam_min:.3e}, lambda_max={lam_max:.3e}); "
            f"rank {rank}/{n}."
        )
    else:
        cond = lam_max / lam_min
        report.condition_number = float(cond)
        if cond > cond_threshold:
            report.is_identifiable = False
            report.reason = (
                f"Condition number {cond:.3e} exceeds threshold "
                f"{cond_threshold:.3e}; rank {rank}/{n}."
            )
        else:
            report.is_identifiable = True
            report.reason = (
                f"Well-conditioned: condition number {cond:.3e} "
                f"below threshold {cond_threshold:.3e}; full rank {rank}/{n}."
            )

    # --- Covariance / correlation / Cramér-Rao bound ------------------------
    if report.is_identifiable:
        C = np.linalg.inv(F_sym)
    else:
        # Do NOT invert blindly: use pinv with a reported rank.  Values along
        # unidentified directions (see sloppy_directions) are not meaningful.
        C = np.linalg.pinv(F_sym, rcond=svd_rtol)
    report.covariance = C

    diag = np.diag(C)
    with np.errstate(invalid="ignore"):
        crlb_std = np.sqrt(np.where(diag > 0, diag, np.nan))
    report.crlb_std = crlb_std

    d = np.sqrt(np.where(diag > 0, diag, np.nan))
    denom = np.outer(d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, C / denom, np.nan)
    report.correlation_matrix = corr

    return report


# ---------------------------------------------------------------------------
# Post-fit UQ entry point (issue #273 Stage 2)
# ---------------------------------------------------------------------------

def calibration_uq(
    sim_factory: Callable,
    freqs: np.ndarray,
    fit_result,
    *,
    n_debye_poles: int,
    n_lorentz_poles: int,
    fit_nuisance: bool = False,
    noise_std: float | None = None,
    s11_measured: np.ndarray | None = None,
    num_periods: float = 14.0,
    checkpoint: bool = True,
    cond_threshold: float = 1e10,
    svd_rtol: float = 1e-6,
) -> IdentifiabilityReport:
    """Fisher / Cramér-Rao identifiability report AT the fitted optimum.

    Packs the FITTED parameters from ``fit_result`` (a
    :class:`~rfx.differentiable_material_fit.MaterialFitResult`, plus the
    ``[log_alpha, phi, tau_hat]`` nuisance tail when ``fit_nuisance=True``),
    builds the joint residual via :func:`s11_residual_fn`, takes its AD
    Jacobian, and returns :func:`identifiability_report` of the resulting
    Fisher information.  This is the ``compute_uq=True`` backend of
    :func:`~rfx.differentiable_material_fit.calibrate_material_s11` and can
    also be called standalone on any fitted result.

    Unit conventions for ``report.crlb_std`` (column order =
    ``report.param_names``, from :func:`calibration_param_names`):

    - material parameters and ``log_alpha`` are log-space, so their
      ``crlb_std`` is a RELATIVE (fractional) standard deviation;
    - ``phi`` is linear radians;
    - ``tau_hat`` is the scaled delay — the report does NOT de-scale it.
      Convert to seconds yourself:
      ``tau_std_seconds = crlb_std[tau_hat] * tau_scale`` with
      ``tau_scale = 1 / (4*pi*max(freqs))``
      (:func:`rfx.differentiable_material_fit._nuisance_tau_scale`).

    Parameters
    ----------
    sim_factory : callable(eps_inf, debye_poles, lorentz_poles) -> Simulation
        Same factory the fit used (the Jacobian must be taken through the
        same forward model).
    freqs : (n_freqs,) array in Hz
        Same frequency grid the fit used.
    fit_result : MaterialFitResult
        Fitted optimum; supplies the material poles and (when
        ``fit_nuisance=True``) the ``nuisance_alpha/phi/tau`` estimates.
    n_debye_poles, n_lorentz_poles : int
        Model order of the fit (defines the parameter layout).
    fit_nuisance : bool
        Analyze the JOINT material+nuisance vector (must match how the fit
        was run; fails closed if ``fit_result`` carries no nuisance values).
    noise_std : float or None
        i.i.d. Gaussian noise std on Re/Im S11.  ``None`` estimates
        ``sigma_hat = sqrt(sum(r^2) / max(2*n_freqs - n_params, 1))`` from
        the residual at the optimum, which requires ``s11_measured``.
    s11_measured : (n_freqs,) complex array, optional
        Measured S11, only needed for the ``noise_std=None`` estimate.
    num_periods : float
        Periods at ``freq_max`` for the forward step count.
    checkpoint : bool
        Use gradient checkpointing in the FDTD scan.
    cond_threshold, svd_rtol : float
        Forwarded to :func:`identifiability_report`.
    """
    freqs = np.asarray(freqs, dtype=np.float64)

    # Fail closed on argument problems BEFORE the expensive AD Jacobian
    # (tens of seconds to minutes on real fixtures).
    if noise_std is None and s11_measured is None:
        raise ValueError(
            "noise_std=None estimates sigma_hat from the residual at the "
            "optimum, which requires s11_measured; pass one of the two."
        )
    # The [:n] slices below would silently truncate (and JAX indexing clamps
    # silently), so a model-order mismatch must be an explicit error.
    if len(fit_result.debye_poles) != n_debye_poles:
        raise ValueError(
            f"fit_result carries {len(fit_result.debye_poles)} Debye pole(s) "
            f"but n_debye_poles={n_debye_poles}; the declared model order "
            "must match the fitted result."
        )
    if len(fit_result.lorentz_poles) != n_lorentz_poles:
        raise ValueError(
            f"fit_result carries {len(fit_result.lorentz_poles)} Lorentz "
            f"pole(s) but n_lorentz_poles={n_lorentz_poles}; the declared "
            "model order must match the fitted result."
        )

    params = _poles_to_params(
        fit_result.eps_inf,
        list(fit_result.debye_poles)[:n_debye_poles],
        list(fit_result.lorentz_poles)[:n_lorentz_poles],
    )
    param_names = calibration_param_names(
        n_debye_poles, n_lorentz_poles, include_nuisance=fit_nuisance
    )

    if fit_nuisance:
        alpha = getattr(fit_result, "nuisance_alpha", None)
        phi = getattr(fit_result, "nuisance_phi", None)
        tau = getattr(fit_result, "nuisance_tau", None)
        if alpha is None or phi is None or tau is None:
            raise ValueError(
                "fit_nuisance=True but fit_result carries no nuisance estimates "
                "(nuisance_alpha/phi/tau are None); run calibrate_material_s11 "
                "with fit_nuisance=True, or pass fit_nuisance=False here."
            )
        tau_scale = _nuisance_tau_scale(freqs)
        nuisance_tail = jnp.array(
            [np.log(float(alpha)), float(phi), float(tau) / tau_scale],
            dtype=params.dtype,
        )
        params = jnp.concatenate([params, nuisance_tail])

    residual = s11_residual_fn(
        sim_factory, freqs, n_debye_poles, n_lorentz_poles,
        num_periods=num_periods,
        checkpoint=checkpoint,
        include_nuisance=fit_nuisance,
    )
    J = s11_jacobian(residual, params)

    n_params = int(params.shape[0])
    if noise_std is None:
        # (s11_measured presence already validated up front.)
        model = np.asarray(
            residual(jnp.asarray(params, dtype=jnp.float32)), dtype=np.float64
        )
        meas = np.asarray(s11_measured).reshape(-1)
        data = np.concatenate([meas.real, meas.imag]).astype(np.float64)
        r = model - data
        # Unbiased-ish variance estimate with n_params fit DOF removed.
        dof = max(r.size - n_params, 1)
        sigma = float(np.sqrt(np.sum(r**2) / dof))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"estimated sigma_hat is not usable ({sigma!r}); the fit "
                "residual is degenerate — pass noise_std explicitly."
            )
    else:
        sigma = float(noise_std)

    F = fisher_information(J, sigma)
    return identifiability_report(
        F, param_names, J=J, noise_std=sigma,
        cond_threshold=cond_threshold, svd_rtol=svd_rtol,
    )
