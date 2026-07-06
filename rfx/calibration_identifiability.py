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
    _params_to_debye_poles,
    _params_to_lorentz_poles,
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
) -> Callable:
    """Build a real-valued S11 residual over log-space material parameters.

    The returned function ``residual(log_params)`` maps a log-space parameter
    vector (same layout as
    :func:`rfx.differentiable_material_fit._poles_to_params`:
    ``[log(eps_inf), log(de_1), log(tau_1), ..., log(de_L1), log(w0_L1),
    log(delta_L1), ...]``) to a **real** vector of length ``2 * n_freqs``,
    namely ``concatenate([Re(S11), Im(S11)])`` evaluated by
    ``Simulation.forward(port_s11_freqs=...).s_params``.

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
    """
    freqs_jnp = jnp.asarray(np.asarray(freqs, dtype=np.float64), dtype=jnp.float32)
    lorentz_offset = 1 + 2 * n_debye

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
        return jnp.concatenate([jnp.real(s11), jnp.imag(s11)])

    return residual


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
