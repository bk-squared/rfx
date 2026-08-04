"""tests/_msl_ad_objective.py

Single shared definition of the MSL AD-gate objective (issue #530) so the
tight gate (``test_msl_ad_fd_converged.py``) and the AD smoke
(``test_msl_sparam_ad.py``) differentiate the EXACT same reduction instead of
two hand-written copies that can drift -- the same defect class
``tests/_gate_policy.py`` exists to prevent for the envelope->gate multiplier
(issue #528).

WHY BAND-MEAN |S21|^2, NOT sum_ij|S_ij|^2 (issue #530)
--------------------------------------------------------
``test_msl_ad_fd_converged_tight`` used to differentiate
``sum(|S|**2)`` over the full 2x2 matrix and all 8 frequency bins. For a
passive network ``S^dagger S <= I``, so ``sum_i|S_ij|^2 <= 1`` per driven
port and ``sum_ij|S_ij|^2 <= 2`` per frequency -- 16 over 8 bins. The
measured loss was 16.00599: 99.96% structural constant (the passivity
ceiling) and 0.037% residue. PR #516 (#511/#507) then moved |S11| from
0.2233 toward ~0.05, shrinking that residue's magnitude 50x
(g_ad -2.1143e-01 -> -4.2425e-03) while the loss's float32 ULP did not move,
so the FD signal ``2h|g|`` fell from ~222 ULP to 4.4 ULP and the comparator
stopped resolving it (issue #527). The gate was differentiating the
derivative of a near-cancelling residue riding on a constant, and it will go
blind the same way again the next time an extractor fix moves |S| closer to
unitary.

Band-mean ``|S21|**2`` (mean transmitted power over the gate's frequency
bins) is not passivity-pinned the way the full-matrix sum is. What moves the
dynamic range is MEASURED, not a mechanism narrative: the level dropped
16x (16.006 -> 0.998 on the gate's fixture), which cuts the loss's float32
ULP 32x (1.9073e-06 -> 5.9605e-08) and lifts f32 resolving power from 4.45
to 53.8 ULP at the gate's h. The residue from unity (~2.5e-3, on the order
of |S11|**2 with |S11| ~ 0.05 on this fixture) is now a physical observable
(reflected power) rather than a unitarity-violation artifact -- that
composition change, not a claim about which physical channel (beta,
reference-plane mismatch, or something else) drives d(loss)/d(alpha), is
the #530 cure. (An earlier draft of this docstring claimed the objective
"moves directly with... guided wavelength via beta" -- that mechanism is
UNMEASURED; a sign witness in the gate's own fixture points elsewhere, at
the wave-split's FROZEN Hammerstad-Jensen Z0 reference rather than a
beta-driven standing-wave shift, which would have no particular sign
preference. Whether the gradient is dominantly a reference-plane artifact
or genuine beta/reflection physics is an OPEN question, tracked in
**issue #560** (not run there either -- #560 is the tracker, not the
answer) -- see ``tests/test_msl_ad_fd_converged.py``'s docstring,
"GATE REBUILT", for the decisive z0_fit-vs-z0_hj probe that would settle
it.) It is computed from the SAME ``compute_msl_s_matrix``
call the old objective used; only the post-call reduction changes.

USAGE
-----
Both consumers call ``compute_msl_s_matrix(...)`` themselves (the fixture,
``num_periods``, ``n_freqs``, ``eps_override``/synthetic-fixture DOF, and
dtype discipline are each caller's own responsibility -- this module only
owns the reduction), then pass the resulting ``.S`` array through
``msl_band_mean_s21_sq``.
"""

from __future__ import annotations

import jax.numpy as jnp


def msl_band_mean_s21_sq(S: "jnp.ndarray") -> "jnp.ndarray":
    """Band-mean ``|S21|**2`` from an MSL S-matrix.

    Parameters
    ----------
    S : (n_ports, n_ports, n_freqs) complex
        As returned by ``MSLSMatrixResult.S`` / ``compute_msl_s_matrix(...).S``.
        Indexing is 0-based ``S[dst - 1, src - 1, :]`` (see
        ``MSLSMatrixResult.s(m, n)``, 1-indexed ports); S21 (port1 -> port2)
        is therefore ``S[1, 0, :]``.

    Returns
    -------
    Real scalar: ``mean_f |S21(f)|**2`` over whatever frequency bins are in
    ``S`` -- 8 bins on the tight gate's fixture, 1 bin on the #515 smoke's
    single-frequency fixture, so both callers share the identical reduction
    down to the single-bin case (mean of one element is that element).
    """
    s21 = S[1, 0, :]
    return jnp.real(jnp.mean(jnp.abs(s21) ** 2))
