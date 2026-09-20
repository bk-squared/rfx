"""How accurately the N-probe beta fit recovers a beta it was GIVEN.

``tests/unit/sparams/test_msl_nprobe_extractor.py`` proves the extractor
recovers alpha/gamma/beta/Z0 at its own tolerances.  This module asks the
narrower question those tolerances are too loose to answer: on data that
is EXACTLY the model the extractor fits, how far off is the beta it
returns, and which way?

Why it exists.  ``_estimate_beta`` scans ``_BETA_SCAN_NODES`` trial
values across ``beta0 * (1 +- _BETA_SCAN_FRAC)`` -- one node is 1.75 % of
``beta0`` -- and refines the best node with a 3-point parabola.  The
residual it refines is an L2 NORM, so near the true beta it is V-shaped
(it reaches zero and turns), not parabolic; fitting a parabola across
that kink leaves a bias that repeats with the node period.  Measured with
``scripts/diagnostics/msl_beta_fit_synthetic_accuracy.py``, which dumps
the whole curve: the bias oscillates with a period of exactly one scan
node, +0.143 % / -0.165 %, vanishing at the nodes and at the node
midpoints.  It is the same size at complex128 as at complex64, so it is
the scan geometry and not precision, and it does not move with the
backward-wave amplitude or phase, because the fit is exact at the true
beta for any gamma.

The same script tests that account the decisive way: refining the SQUARE
of the residual instead (monotone, so the argmin node is unchanged; the
square is locally parabolic where the norm is not) drops the sweep
maximum from 0.1649 % to 0.0161 % and removes every sign change.  What
survives is a remainder of about -0.011 % that keeps one sign, which
this account does not explain.

The envelope pinned below is MEASURED, not desired.  It is a regression
lock -- it bounds the error from above, so a change that shrinks the bias
passes -- and the accuracy the fit ought to have is named separately, as
a strict xfail, so that fixing the estimator turns this module red and
forces the bound to be tightened rather than quietly left loose.

Attribution of the cv20 signed-beta residual (#830) is a separate
question and is not decided here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, MU_0
from rfx.probes.msl_wave_decomp import extract_msl_nprobe
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

# The MSL phase-referee board (validation/crossval/20_msl_phase_referee.py):
# RO4350B, 600 um trace, 254 um declared substrate.  Only the ANCHOR matters
# here -- the scan is centred on it -- so this is the anchor production would
# compute for that port, not a claim about the board.
EPS_R = 3.66
W_TRACE_M = 600e-6
H_DECLARED_M = 254e-6
F_MID_HZ = 3.7586206896551726e9        # centre of the referee's 3.0-4.5 GHz gate band

# Probe arrays the envelope lock runs on. The first is the thru fixture's own
# (5 planes, 11 cells of 50 um); the second is deliberately unlike it.
LAYOUTS = {
    "thru_5_planes_550um": np.array([4.5e-3 + n * 550e-6 for n in range(5)]),
    "unlike_9_planes_250um": np.array([1.0e-3 + n * 250e-6 for n in range(9)]),
}

# A wider set for the array-independence witness: 3 to 9 planes, 150 to 550 um
# spacing, two origins -- a factor of ~15 in the electrical length beta*L the
# array subtends. The bias under test is a property of the beta SCAN, so it
# must not move across these.
SPREAD_ARRAYS = dict(LAYOUTS, **{
    "3_planes_150um_x0_0": np.array([n * 150e-6 for n in range(3)]),
    "3_planes_150um_x0_4p5mm": np.array([4.5e-3 + n * 150e-6 for n in range(3)]),
    "3_planes_300um_x0_0": np.array([n * 300e-6 for n in range(3)]),
    "5_planes_150um_x0_0": np.array([n * 150e-6 for n in range(5)]),
    "5_planes_250um_x0_0": np.array([n * 250e-6 for n in range(5)]),
    "9_planes_550um_x0_0": np.array([n * 550e-6 for n in range(9)]),
})

# Swept true beta, as a fraction of the anchor: +-3 % brackets every residual
# anyone has proposed for this board, at a step fine enough to resolve the
# 1.75 % node period.
RATIOS = np.round(np.arange(0.97, 1.03 + 1e-9, 0.001), 6)
GAMMA_OVER_ALPHA = (0.0, 0.05, 0.2)

# MEASURED (scripts/diagnostics/msl_beta_fit_synthetic_accuracy.py, 17424
# cases, complex64 and complex128 within 5e-6 of each other):
#   max |fitted/true - 1| = 0.1655 %,  signed mean over the window = -0.0090 %.
MEASURED_MAX_ABS_BIAS = 1.655e-3
# The lock: the measurement plus half again, so ordinary platform-to-platform
# float32 drift cannot red it but a real loss of accuracy can. Measured on the
# other side too: this bound reds at a beta scan 1.4x coarser than the current
# one (41 nodes -> 25) and stays green at 1.33x.
ACCURACY_ENVELOPE = 1.5 * MEASURED_MAX_ABS_BIAS
# MEASURED across SPREAD_ARRAYS: the worst-case bias spans 0.1625 % to 0.1691 %,
# a spread of 6.6e-5 on the script's 0.0005 ratio grid and 6.3e-5 on this test's
# 0.001 grid. The bound has to stay green when the beta scan alone gets coarser,
# because the failure message below tells the reader the scan is NOT the cause:
# with the scan degraded the spread was measured at 1.36e-4 (25 nodes), 7.0e-5
# (21) and 1.09e-4 (11) -- non-monotone, all under 2e-4. So the bound is 2e-4:
# about 3x the measurement, and still 8x under the 0.165 % effect it separates
# array dependence from.
MEASURED_ARRAY_SPREAD = 6.3e-5
ARRAY_SPREAD_BOUND = 2e-4
# What a least-squares fit of an exactly-representable model should deliver.
TARGET_ACCURACY = 5e-4


def _anchor(freq_hz: float) -> float:
    """``beta0`` the way ``rfx/sparams/msl.py`` builds it for this port."""
    c0 = 1.0 / float(np.sqrt(MU_0 * EPS_0))
    _z0, eps_eff = hammerstad_jensen_z0_eps_eff(W_TRACE_M, H_DECLARED_M, EPS_R)
    return 2.0 * np.pi * float(freq_hz) * float(np.sqrt(eps_eff)) / c0


def _sweep_bias(x_m: np.ndarray, gamma_over_alpha: float) -> np.ndarray:
    """``fitted/true - 1`` over ``RATIOS``, through the production extractor.

    The voltages are built from the extractor's OWN model with alpha = 1, so
    a perfect fit returns zero here by construction; everything nonzero below
    is the estimator's.
    """
    beta0 = _anchor(F_MID_HZ)
    beta_true = RATIOS * beta0
    bx = beta_true[:, None] * x_m[None, :]
    v = np.exp(-1j * bx) + gamma_over_alpha * np.exp(+1j * bx)
    out = extract_msl_nprobe(
        jnp.asarray(v), jnp.asarray(x_m),
        jnp.ones(v.shape[0], dtype=jnp.complex64),
        jnp.full(v.shape[0], beta0),
    )
    assert not bool(np.any(np.asarray(jax.device_get(out["beta_railed"])))), (
        "the beta scan railed inside a +-3 % sweep -- the window is +-35 %, so "
        "a rail here means the scan is not doing what this test assumes")
    fitted = np.asarray(jax.device_get(jnp.real(out["beta"])))
    return fitted / beta_true - 1.0


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("gamma_over_alpha", GAMMA_OVER_ALPHA)
def test_beta_fit_stays_inside_its_measured_accuracy_envelope(
        layout, gamma_over_alpha):
    """Upper bound only: the fit may get better, never worse.

    Data that IS the fitted model, so the whole excursion belongs to the
    estimator.  See the module docstring for where the number came from.
    """
    bias = _sweep_bias(LAYOUTS[layout], gamma_over_alpha)
    worst = float(np.max(np.abs(bias)))
    assert worst <= ACCURACY_ENVELOPE, (
        f"{layout}, |gamma/alpha|={gamma_over_alpha}: the beta fit is off by "
        f"{100 * worst:.4f} % on data that is exactly its own model, past the "
        f"envelope {100 * ACCURACY_ENVELOPE:.4f} % (the measured "
        f"{100 * MEASURED_MAX_ABS_BIAS:.4f} % plus half again). Re-measure "
        f"with scripts/diagnostics/msl_beta_fit_synthetic_accuracy.py before "
        f"moving this bound.")


def test_beta_fit_accuracy_is_a_property_of_the_scan_not_of_the_probe_array():
    """Eight unlike arrays must see the same error.

    This is what identifies the error as the beta scan's: the arrays differ
    in probe count, spacing and origin -- a factor of ~15 in beta*L -- so
    anything the probe geometry caused would separate them.  Stated as an
    absolute difference, so it still holds if the estimator is fixed and all
    of them go to zero.
    """
    worst = {
        name: float(np.max(np.abs(_sweep_bias(x, 0.0))))
        for name, x in SPREAD_ARRAYS.items()
    }
    spread = max(worst.values()) - min(worst.values())
    assert spread <= ARRAY_SPREAD_BOUND, (
        f"probe arrays disagree on the fit's accuracy by {100 * spread:.4f} %, "
        f"past the {100 * ARRAY_SPREAD_BOUND:.4f} % allowed on a measured "
        f"{100 * MEASURED_ARRAY_SPREAD:.4f} % -- the error is then not (only) "
        f"the beta scan's, and the module docstring's account of it needs "
        f"redoing. Per array: {worst}")


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED 2026-09-20 (#830): the 3-point parabolic refinement in "
    "_estimate_beta is applied to an L2-norm residual, which is V-shaped at "
    "its minimum rather than parabolic, leaving a bias that repeats with the "
    "scan-node period -- up to 0.166 %, about +0.12 % at the beta/beta0 the "
    "cv20 fixture sits at. Not fixed here: the fix is an estimator change and "
    "is a decision for the issue, not a side effect of measuring it. When it "
    "IS fixed this xfail turns into a failure, which is the point -- tighten "
    "ACCURACY_ENVELOPE to TARGET_ACCURACY and delete this marker."))
def test_beta_fit_reaches_the_accuracy_a_two_wave_fit_should_have():
    """0.05 % on data that is exactly the fitted model, for any array."""
    for name, x_m in LAYOUTS.items():
        worst = float(np.max(np.abs(_sweep_bias(x_m, 0.0))))
        assert worst <= TARGET_ACCURACY, f"{name}: {100 * worst:.4f} %"
