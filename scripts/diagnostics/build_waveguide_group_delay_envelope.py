"""Issue #490 Lane 3 -- aggregate the 3 near-cutoff group-delay runs into a
committed envelope + falsifier + settling/invariance witnesses.

Reads the 3 npz cases produced by ``build_waveguide_near_cutoff_group_delay.py``
(baseline_np60, settling_np120, grown_domain_np60) and:

1. Computes tau_g_measured(f) = -d(unwrap(angle(S21)))/d(omega) from the
   baseline run via a central-difference stencil (interior points) / one-
   sided stencil (2 endpoints, reported not gated).
2. LIKE-FOR-LIKE COMPARATOR (adversarial review of PR #536, finding M3): the
   measured tau_g is a finite difference, but the original oracle
   (``analytic_tau_g``, the exact closed-form derivative) is not -- the
   stencil's own truncation error (0.0072 ns at the worst interior point, 29%
   of the naive 0.0248 ns headline, OPPOSITE sign) partially cancelled the
   true residual and flattered it. The fix applies the SAME central-
   difference stencil to the analytic PHASE (``_analytic_phase`` ->
   ``_tau_g_analytic_via_stencil``) before differencing, so both sides of the
   comparison carry the identical discretization error and the residual
   isolates the physical S21-extraction error. The exact closed-form
   ``analytic_tau_g`` is retained ONLY for descriptive/contextual reporting
   (``tau_g_analytic_exact_ns``) -- it is NOT used for gating.
3. Gates the interior points against the stencil-consistent oracle with
   ``MAX_GROUP_DELAY_TOL_NS`` (H1: a pinned module constant, NOT computed
   on-the-fly from the residual it gates -- see the tolerance-envelope test
   for the two-sided bound).
4. Settling witness: baseline (num_periods=60) vs settling_np120
   (num_periods=120) tau_g, same like-for-like comparator -- must agree
   closely (record-length convergence substitutes for the MSL-style energy
   dB witness; see the producer's module docstring for why).
5. Domain invariance witness: baseline vs grown_domain_np60 tau_g and the
   gate verdict -- must not flip.
6. Falsifiers (post-processing only, no extra FDTD; M2 -- adversarial review
   found ``wrong_sign`` and the original ``conjugate_flip`` are the SAME
   operation on this derivation chain, identical to 6.7e-16: negating
   angle(S21) before unwrap flips the unwrapped phase's sign exactly the same
   way dropping the leading minus sign does, so they are not independent
   defects. ``conjugate_flip`` is REPLACED with ``wrong_l_eff``, a genuinely
   orthogonal defect on the comparator side -- using the full domain length
   instead of the reference-plane separation as L_eff, a real bookkeeping
   mistake class): (a) drop the unwrap step (use wrapped phase directly)
   before differentiating; (b) drop the leading minus sign
   (tau_g = +dphi/domega instead of -dphi/domega); (c) use the WRONG L_eff
   (domain_x_m instead of the reference-plane separation) in the analytic
   comparator. All three must red the gate.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_near_cutoff_group_delay import (  # noqa: E402
    OUT_DIR, analytic_tau_g, L_EFF_M, FC_TE10_HZ, C0,
)

# H1 -- pinned module constant (NOT derived on-the-fly from the residual it
# gates -- that was the tautology the adversarial review flagged). Provenance:
# the worst LIKE-FOR-LIKE interior residual across all 3 committed cases
# (baseline/settling/grown) is 0.03201 ns (baseline, f=2.06 GHz, the first
# interior point) -- see build_waveguide_group_delay_tolerance_envelope test
# for the re-derivation from the committed fixture. 0.03201 ns x ~1.3 margin
# = 0.0416 ns, rounded up to 0.042 ns. Bounded on both sides by
# tests/test_waveguide_group_delay_tolerance_envelope.py (>= worst measured
# like-for-like case so it never fails a validated case; <= worst x 1.5 so it
# is not slack), the same discipline as MAX_TOL (magnitude) and
# MAX_PHASE_TOL_DEG (Lane 1 phase).
MAX_GROUP_DELAY_TOL_NS = 0.042


def _central_diff_group_delay(phase: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """tau_g = -d(phase)/d(omega): central difference (interior, O(domega^2))
    / one-sided (2 endpoints, O(domega)). Shared by the measured leg and the
    analytic-through-the-same-stencil leg so the comparator is like-for-like."""
    n = len(phase)
    tau = np.full(n, np.nan)
    tau[1:-1] = -(phase[2:] - phase[:-2]) / (omega[2:] - omega[:-2])
    tau[0] = -(phase[1] - phase[0]) / (omega[1] - omega[0])
    tau[-1] = -(phase[-1] - phase[-2]) / (omega[-1] - omega[-2])
    return tau


def _tau_g_measured(freqs_hz: np.ndarray, s21: np.ndarray):
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * freqs_hz
    return _central_diff_group_delay(phase, omega), phase


def _analytic_phase(freqs_hz: np.ndarray, l_eff: float) -> np.ndarray:
    """Exact (continuous, unwrapped-by-construction) TE10 phase: -beta(f)*L_eff."""
    beta = (2.0 * np.pi * freqs_hz / C0) * np.sqrt(1.0 - (FC_TE10_HZ / freqs_hz) ** 2)
    return -beta * l_eff


def _tau_g_analytic_via_stencil(freqs_hz: np.ndarray, l_eff: float) -> np.ndarray:
    """M3: the like-for-like oracle -- SAME central-difference stencil applied
    to the analytic phase, so its own O(domega^2)/O(domega) truncation error
    is present on both sides of the comparison."""
    phase = _analytic_phase(freqs_hz, l_eff)
    omega = 2.0 * np.pi * freqs_hz
    return _central_diff_group_delay(phase, omega)


def _falsifier_skip_unwrap(freqs_hz, s21):
    phase = np.angle(s21)  # NO unwrap -- the planted defect
    omega = 2.0 * np.pi * freqs_hz
    return _central_diff_group_delay(phase, omega)


def _falsifier_wrong_sign(freqs_hz, s21):
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * freqs_hz
    n = len(freqs_hz)
    tau = np.full(n, np.nan)
    tau[1:-1] = +(phase[2:] - phase[:-2]) / (omega[2:] - omega[:-2])  # dropped minus
    tau[0] = +(phase[1] - phase[0]) / (omega[1] - omega[0])
    tau[-1] = +(phase[-1] - phase[-2]) / (omega[-1] - omega[-2])
    return tau


def _falsifier_wrong_l_eff(freqs_hz, wrong_l_eff):
    """M2 replacement for conjugate_flip (which was IDENTICAL to wrong_sign on
    this derivation chain, 6.7e-16, not an independent defect). Genuinely
    orthogonal: corrupts the ANALYTIC comparator (a real L_eff bookkeeping
    mistake -- using the domain length instead of the reference-plane
    separation), not the measured extraction, and is compared against the
    CORRECTLY measured tau_g."""
    return _tau_g_analytic_via_stencil(freqs_hz, wrong_l_eff)


def main():
    baseline = np.load(OUT_DIR / "baseline_np60.npz")
    settling = np.load(OUT_DIR / "settling_np120.npz")
    grown = np.load(OUT_DIR / "grown_domain_np60.npz")

    freqs = baseline["freqs_hz"]
    l_eff = float(baseline["l_eff_m"])
    assert abs(l_eff - L_EFF_M) < 1e-9

    tau_measured, phase_unwrapped = _tau_g_measured(freqs, baseline["s21"])
    tau_analytic_stencil = _tau_g_analytic_via_stencil(freqs, l_eff)
    tau_analytic_exact = analytic_tau_g(freqs, l_eff)  # informational only, NOT gated
    interior = slice(1, -1)

    # GATED metric: like-for-like (measured stencil vs analytic-through-same-
    # stencil).
    abs_diff_ns = np.abs(tau_measured[interior] - tau_analytic_stencil[interior]) * 1e9
    max_diff_ns = float(abs_diff_ns.max())
    mean_diff_ns = float(abs_diff_ns.mean())

    # Informational only (NOT gated): the original (pre-M3) comparison against
    # the exact closed-form derivative, and the stencil's own truncation error
    # relative to that exact derivative. Kept for transparency, not deleted.
    abs_diff_ns_vs_exact_oracle = np.abs(
        tau_measured[interior] - tau_analytic_exact[interior]
    ) * 1e9
    stencil_truncation_error_ns = (
        tau_analytic_stencil[interior] - tau_analytic_exact[interior]
    ) * 1e9

    tol_ns = MAX_GROUP_DELAY_TOL_NS

    # Qualitative expectations (pre-declared): monotonic decrease, smooth.
    # LOW(d): the ratio check uses INTERIOR bins (indices 1 and 9), not the
    # band-edge endpoints (0 and 10) -- a sibling test asserts the endpoints
    # are reported but not gated, so the qualitative check should not lean on
    # them either. The analytic side of this ratio uses the EXACT closed-form
    # oracle (not the stencil) -- it is a loose order-of-magnitude/sign
    # sanity check against the true physics, not the tight quantitative gate.
    monotonic = bool(np.all(np.diff(tau_measured) <= 1e-12))
    d_ratio = float(tau_measured[1] / tau_measured[-2])
    d_ratio_analytic = float(tau_analytic_exact[1] / tau_analytic_exact[-2])

    # Settling witness: baseline (np=60) vs settling (np=120), same
    # like-for-like comparator.
    tau_settling, _ = _tau_g_measured(settling["freqs_hz"], settling["s21"])
    settling_diff_ns = float(np.abs(tau_measured[interior] - tau_settling[interior]).max()) * 1e9

    # Domain-invariance witness: baseline vs grown domain, same like-for-like
    # comparator.
    tau_grown, _ = _tau_g_measured(grown["freqs_hz"], grown["s21"])
    tau_analytic_stencil_grown = _tau_g_analytic_via_stencil(grown["freqs_hz"], float(grown["l_eff_m"]))
    grown_abs_diff_ns = np.abs(tau_grown[interior] - tau_analytic_stencil_grown[interior]) * 1e9
    grown_max_diff_ns = float(grown_abs_diff_ns.max())
    baseline_verdict = "passed" if max_diff_ns <= tol_ns else "failed"
    grown_verdict = "passed" if grown_max_diff_ns <= tol_ns else "failed"

    # Falsifiers -- all compared against the like-for-like (stencil-
    # consistent) oracle for methodological consistency with the gate itself.
    tau_skip_unwrap = _falsifier_skip_unwrap(freqs, baseline["s21"])
    tau_wrong_sign = _falsifier_wrong_sign(freqs, baseline["s21"])
    tau_wrong_l_eff_oracle = _falsifier_wrong_l_eff(freqs, float(baseline["domain_x_m"]))
    fals = {}
    for name, tau_f, oracle in (
        ("skip_unwrap", tau_skip_unwrap, tau_analytic_stencil),
        ("wrong_sign", tau_wrong_sign, tau_analytic_stencil),
        # wrong_l_eff corrupts the ORACLE, not the measured leg: compare the
        # correctly-measured tau_g against the wrong-L_eff oracle.
        ("wrong_l_eff", tau_measured, tau_wrong_l_eff_oracle),
    ):
        d = np.abs(tau_f[interior] - oracle[interior]) * 1e9
        # guard against NaN from unwrap-skip producing huge spurious jumps
        worst = float(np.nanmax(d))
        fals[name] = {"worst_abs_diff_ns": worst, "gate_reds": bool(worst > tol_ns)}

    status = baseline_verdict
    envelope = {
        "schema": "rfx.waveguide_wr340_near_cutoff_group_delay_envelope",
        "schema_version": 2,
        "status": status,
        "evidence_level": "E2-analytic-oracle-group-delay-near-cutoff-wr340",
        "claim": (
            f"rfx compute_waveguide_s_matrix(normalize='flux') S21 group delay "
            f"(-d(angle(S21))/d(omega), central difference) on an empty WR-340 "
            f"guide over f/fc in [1.152, 1.498] {'matches' if status=='passed' else 'DOES NOT match'} "
            f"the analytic L_eff/v_g(f) oracle (same finite-difference stencil "
            f"applied to both sides, M3) to {tol_ns:.4f} ns "
            f"(max interior diff {max_diff_ns:.4f} ns)."
        ),
        "fc_te10_hz": FC_TE10_HZ,
        "l_eff_m": l_eff,
        "freqs_hz": freqs.tolist(),
        "fc_ratio": (freqs / FC_TE10_HZ).tolist(),
        "tau_g_measured_ns": (tau_measured * 1e9).tolist(),
        "tau_g_analytic_via_stencil_ns": (tau_analytic_stencil * 1e9).tolist(),
        "tau_g_analytic_exact_ns": (tau_analytic_exact * 1e9).tolist(),
        "endpoints_gated": False,
        "interior_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "max_abs_diff_ns": max_diff_ns,
        "mean_abs_diff_ns": mean_diff_ns,
        "max_abs_diff_ns_vs_exact_oracle_not_gated": float(abs_diff_ns_vs_exact_oracle.max()),
        "stencil_truncation_error_ns_max_not_gated": float(np.abs(stencil_truncation_error_ns).max()),
        "comparator_note": (
            "max_abs_diff_ns is the GATED, like-for-like metric (measured "
            "finite-difference tau_g vs the analytic phase run through the "
            "IDENTICAL finite-difference stencil). "
            "max_abs_diff_ns_vs_exact_oracle_not_gated compares against the "
            "exact closed-form derivative instead and is NOT gated -- it is "
            "smaller here only because the stencil's own truncation error "
            "partially cancels the true residual with opposite sign "
            "(adversarial review of PR #536, finding M3); do not use it as "
            "the headline number."
        ),
        "max_group_delay_tol_ns": tol_ns,
        "tol_provenance": (
            f"pinned module constant MAX_GROUP_DELAY_TOL_NS={MAX_GROUP_DELAY_TOL_NS} "
            f"(NOT derived from this run's own residual -- see the module "
            f"docstring / tolerance-envelope test for the measured-envelope "
            f"derivation across all 3 committed cases)"
        ),
        "qualitative_expectations": {
            "monotonic_decrease": monotonic,
            "measured_ratio_tau1_over_tau9_interior": d_ratio,
            "analytic_exact_ratio_tau1_over_tau9_interior": d_ratio_analytic,
        },
        "settling_witness": {
            "method": "record-length convergence (num_periods 60 vs 120); "
                      "compute_waveguide_s_matrix has no built-in energy-based "
                      "settling_db for the waveguide port family",
            "max_abs_diff_ns_np60_vs_np120": settling_diff_ns,
        },
        "domain_invariance_witness": {
            "baseline_domain_m": float(baseline["domain_x_m"]),
            "grown_domain_m": float(grown["domain_x_m"]),
            "baseline_max_abs_diff_ns": max_diff_ns,
            "grown_max_abs_diff_ns": grown_max_diff_ns,
            "baseline_verdict": baseline_verdict,
            "grown_verdict": grown_verdict,
            "verdict_flipped": baseline_verdict != grown_verdict,
        },
        "falsifiers": fals,
        "wallclock_s": {
            "baseline": float(baseline["wallclock_s"]),
            "settling": float(settling["wallclock_s"]),
            "grown": float(grown["wallclock_s"]),
        },
    }

    out_path = REPO / "tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2))

    print(f"status={status}  max_diff(like-for-like)={max_diff_ns:.4f}ns  tol={tol_ns:.4f}ns")
    print(f"  [context, not gated] vs exact oracle={abs_diff_ns_vs_exact_oracle.max():.4f}ns  "
          f"stencil truncation error max={np.abs(stencil_truncation_error_ns).max():.4f}ns")
    print(f"monotonic={monotonic}  interior ratio meas={d_ratio:.4f} analytic(exact)={d_ratio_analytic:.4f}")
    print(f"settling (np60 vs np120) max diff: {settling_diff_ns:.4f} ns")
    print(f"domain invariance: baseline={baseline_verdict} grown={grown_verdict} "
          f"grown_max_diff={grown_max_diff_ns:.4f}ns")
    for name, v in fals.items():
        print(f"falsifier[{name}]: worst_diff={v['worst_abs_diff_ns']:.4f}ns gate_reds={v['gate_reds']}")
    print(f"wrote {out_path.relative_to(REPO)}")
    for i, f in enumerate(freqs):
        print(f"  f={f/1e9:.4f}GHz f/fc={f/FC_TE10_HZ:.4f} tau_meas={tau_measured[i]*1e9:.4f}ns "
              f"tau_analytic_stencil={tau_analytic_stencil[i]*1e9:.4f}ns "
              f"tau_analytic_exact={tau_analytic_exact[i]*1e9:.4f}ns")


if __name__ == "__main__":
    main()
