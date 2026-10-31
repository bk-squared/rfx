"""Warnings for the spectral passivity of returned port measurements."""

import warnings

import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.errors import AccuracyWarning


TOL = 1e-2


def warn_if_singular_value_exceeds_one(s_params, freqs, *, extractor: str) -> None:
    """Warn once when a finite returned S bin exceeds the spectral tolerance.

    A diagonal-only array gives a lower bound, not the full matrix norm.
    Observed columns likewise give a lower bound when drives are missing.
    Concrete measurement only: tracing and unavailable host values are skipped.
    """
    if is_tracer(s_params) or is_tracer(freqs):
        return
    s = np.asarray(s_params)
    f = np.asarray(freqs)
    if s.ndim not in (1, 2, 3) or not s.size or f.ndim != 1 or s.shape[-1] != f.size:
        return
    lower_bound = ""
    if s.ndim == 3:
        from rfx.sparams._common import _sigma_max_excess
        excess = _sigma_max_excess(s)
        if s.shape[0] != s.shape[1]:
            lower_bound = " (observed-column lower bound of sigma_max)"
    elif s.ndim in (1, 2):
        magnitude = np.abs(s)
        if s.ndim == 2:
            magnitude = np.max(magnitude, axis=0)
            lower_bound = " (diagonal-only lower bound of sigma_max)"
        excess = magnitude - 1.0
    else:
        return
    finite = np.isfinite(excess)
    over = finite & (excess > TOL)
    if not np.any(over):
        return
    k = int(np.argmax(np.where(finite, excess, -np.inf)))
    warnings.warn(
        f"{extractor}: the returned S is not passive at {int(over.sum())} of "
        f"{excess.size} frequency bins, worst sigma_max = {1.0 + excess[k]:.6g}"
        f"{lower_bound} at {f[k] / 1e9:.6g} GHz. S is returned as extracted. "
        "A passive structure cannot scatter more power than it receives, "
        "so these bins are an extraction or record artifact.",
        AccuracyWarning,
        stacklevel=2,
    )


def warn_if_nonpassive_lumped_s11(s_params, freqs, *, extractor: str,
                                  passivity_tol: float = 0.10) -> None:
    """Tracer-safe passivity self-check for lumped/wire port S11.

    ``run(compute_s_params=True)`` and ``forward(port_s11_freqs=...)`` for
    lumped/wire ports assemble S11 from the eager single-cell extractor, but —
    unlike ``compute_*_s_matrix`` (which calls
    :func:`rfx.api._sparams._warn_if_nonpassive_smatrix`) — they previously
    surfaced no passivity check. A passive one-port cannot have ``|S11| > 1``;
    a gross violation (measured up to ~1.98 on a high-eps closed cavity) means
    the extractor is unreliable at that frequency (single-cell curl-of-H
    current is ill-conditioned where the incident wave is weak, i.e. spectral
    band edges), NOT that the device is exotic. Surface it so an eager
    analysis / optimizer setup does not trust or chase those bins.

    ``s_params``: ``(n_freqs,)`` single port, or ``(n_ports, n_freqs)`` per-port
    diagonal S_ii. ``passivity_tol`` matches the repo standard (0.10) so mild
    float32 noise just above 1.0 is not flagged.

    Tracer-safe: under ``jax.grad`` / ``jax.jit`` ``s_params`` is an abstract
    tracer with no concrete value, so the check is skipped entirely — it never
    perturbs an AD/optimization run, only the eager research call.
    """
    import numpy as np
    from rfx.core.jax_utils import is_tracer
    if is_tracer(s_params) or is_tracer(freqs):
        return
    mag = np.abs(np.asarray(s_params))
    f = np.asarray(freqs)
    if mag.size == 0:
        return
    nonfinite = ~np.isfinite(mag)
    over = mag > (1.0 + passivity_tol)
    if not (over.any() or nonfinite.any()):
        return
    worst = float(np.nanmax(mag)) if np.isfinite(mag).any() else float("nan")
    bad_cols = np.unique(np.where(over | nonfinite)[-1])
    bad_f = [f"{float(f[c]) / 1e9:.3f}" for c in bad_cols[:6] if c < f.size]
    # Describe the actual trigger accurately: a non-finite bin is its own
    # failure mode (don't report a misleadingly-small finite max for it).
    if nonfinite.any():
        kind = f"non-finite (and max finite|S11|={worst:.3f})"
    else:
        kind = f"max|S11|={worst:.3f} > 1+{passivity_tol:g}"
    import warnings as _w
    _w.warn(
        f"{extractor}: lumped/wire port S11 is non-passive ({kind}) " f"at freq(GHz)={bad_f}"
        f"{' …' if len(bad_f) == 6 else ''}. A passive one-port cannot have "
        f"|S11|>1; the single-cell extractor is unreliable where the incident "
        f"wave is weak (spectral band edges). Do not trust or optimize against "
        f"these bins — restrict the band to where the source has energy, or use "
        f"run() for a passive analysis curve.", AccuracyWarning,
        stacklevel=3,
    )
