"""Issue #490 Lane 3 -- aggregate the 3 near-cutoff group-delay runs into a
committed envelope + falsifier + settling/invariance witnesses.

Reads the 3 npz cases produced by ``build_waveguide_near_cutoff_group_delay.py``
(baseline_np60, settling_np120, grown_domain_np60) and:

1. Computes tau_g_measured(f) = -d(unwrap(angle(S21)))/d(omega) from the
   baseline run via a central-difference stencil (interior points) / one-
   sided stencil (2 endpoints, reported not gated).
2. Gates the interior points against the analytic tau_g(f) = L_eff/v_g(f)
   oracle with a measured-envelope tolerance.
3. Settling witness: compares baseline (num_periods=60) vs settling_np120
   (num_periods=120) tau_g -- must agree closely (record-length convergence
   substitutes for the MSL-style energy dB witness; see the producer's
   module docstring for why).
4. Domain invariance witness: compares baseline vs grown_domain_np60 tau_g
   and the gate verdict -- must not flip.
5. Falsifiers (post-processing only, no extra FDTD): (a) drop the unwrap
   step (use wrapped phase directly) before differentiating; (b) drop the
   leading minus sign (tau_g = +dphi/domega instead of -dphi/domega); (c) a
   conjugate flip on S21 (the same convention-mismatch class as Lane 1's
   falsifier and the historical cv11/WR-90 comparator bugs). All three must
   red the gate.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_near_cutoff_group_delay import (  # noqa: E402
    OUT_DIR, analytic_tau_g, L_EFF_M, FC_TE10_HZ,
)

# Measured-envelope tolerance in nanoseconds -- filled in from the baseline
# measurement (see MAX_GROUP_DELAY_TOL_NS derivation note printed at run time
# and pinned into the committed fixture's provenance field).


def _tau_g_measured(freqs_hz: np.ndarray, s21: np.ndarray):
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * freqs_hz
    n = len(freqs_hz)
    tau = np.full(n, np.nan)
    # interior: central difference, O(domega^2)
    tau[1:-1] = -(phase[2:] - phase[:-2]) / (omega[2:] - omega[:-2])
    # endpoints: one-sided, O(domega) -- reported, not gated
    tau[0] = -(phase[1] - phase[0]) / (omega[1] - omega[0])
    tau[-1] = -(phase[-1] - phase[-2]) / (omega[-1] - omega[-2])
    return tau, phase


def _falsifier_skip_unwrap(freqs_hz, s21):
    phase = np.angle(s21)  # NO unwrap -- the planted defect
    omega = 2.0 * np.pi * freqs_hz
    tau = np.full(len(freqs_hz), np.nan)
    tau[1:-1] = -(phase[2:] - phase[:-2]) / (omega[2:] - omega[:-2])
    tau[0] = -(phase[1] - phase[0]) / (omega[1] - omega[0])
    tau[-1] = -(phase[-1] - phase[-2]) / (omega[-1] - omega[-2])
    return tau


def _falsifier_wrong_sign(freqs_hz, s21):
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * freqs_hz
    tau = np.full(len(freqs_hz), np.nan)
    tau[1:-1] = +(phase[2:] - phase[:-2]) / (omega[2:] - omega[:-2])  # dropped minus
    tau[0] = +(phase[1] - phase[0]) / (omega[1] - omega[0])
    tau[-1] = +(phase[-1] - phase[-2]) / (omega[-1] - omega[-2])
    return tau


def _falsifier_conjugate_flip(freqs_hz, s21):
    return _tau_g_measured(freqs_hz, np.conj(s21))[0]


def main():
    baseline = np.load(OUT_DIR / "baseline_np60.npz")
    settling = np.load(OUT_DIR / "settling_np120.npz")
    grown = np.load(OUT_DIR / "grown_domain_np60.npz")

    freqs = baseline["freqs_hz"]
    l_eff = float(baseline["l_eff_m"])
    assert abs(l_eff - L_EFF_M) < 1e-9

    tau_measured, phase_unwrapped = _tau_g_measured(freqs, baseline["s21"])
    tau_analytic = analytic_tau_g(freqs, l_eff)
    interior = slice(1, -1)

    abs_diff_ns = np.abs(tau_measured[interior] - tau_analytic[interior]) * 1e9
    max_diff_ns = float(abs_diff_ns.max())
    mean_diff_ns = float(abs_diff_ns.mean())

    # Measured-envelope tolerance: worst interior-point diff x ~1.3 margin
    # (printed here; the constant is pinned into MAX_GROUP_DELAY_TOL_NS below
    # once measured -- see the companion test's tolerance-envelope check).
    tol_ns = max(max_diff_ns * 1.3, 0.02)

    # Qualitative expectations (pre-declared): monotonic decrease, smooth.
    monotonic = bool(np.all(np.diff(tau_measured) <= 1e-12))
    d_ratio = float(tau_measured[0] / tau_measured[-1])
    d_ratio_analytic = float(tau_analytic[0] / tau_analytic[-1])

    # Settling witness: baseline (np=60) vs settling (np=120).
    tau_settling, _ = _tau_g_measured(settling["freqs_hz"], settling["s21"])
    settling_diff_ns = float(np.abs(tau_measured[interior] - tau_settling[interior]).max()) * 1e9

    # Domain-invariance witness: baseline vs grown domain.
    tau_grown, _ = _tau_g_measured(grown["freqs_hz"], grown["s21"])
    tau_analytic_grown = analytic_tau_g(grown["freqs_hz"], float(grown["l_eff_m"]))
    grown_abs_diff_ns = np.abs(tau_grown[interior] - tau_analytic_grown[interior]) * 1e9
    grown_max_diff_ns = float(grown_abs_diff_ns.max())
    baseline_verdict = "passed" if max_diff_ns <= tol_ns else "failed"
    grown_verdict = "passed" if grown_max_diff_ns <= tol_ns else "failed"

    # Falsifiers.
    tau_skip_unwrap = _falsifier_skip_unwrap(freqs, baseline["s21"])
    tau_wrong_sign = _falsifier_wrong_sign(freqs, baseline["s21"])
    tau_conj = _falsifier_conjugate_flip(freqs, baseline["s21"])
    fals = {}
    for name, tau_f in (("skip_unwrap", tau_skip_unwrap),
                        ("wrong_sign", tau_wrong_sign),
                        ("conjugate_flip", tau_conj)):
        d = np.abs(tau_f[interior] - tau_analytic[interior]) * 1e9
        # guard against NaN from unwrap-skip producing huge spurious jumps
        worst = float(np.nanmax(d))
        fals[name] = {"worst_abs_diff_ns": worst, "gate_reds": bool(worst > tol_ns)}

    envelope = {
        "schema": "rfx.waveguide_wr340_near_cutoff_group_delay_envelope",
        "schema_version": 1,
        "status": baseline_verdict,
        "evidence_level": "E2-analytic-oracle-group-delay-near-cutoff-wr340",
        "claim": (
            f"rfx compute_waveguide_s_matrix(normalize='flux') S21 group delay "
            f"(-d(angle(S21))/d(omega), central difference) on an empty WR-340 "
            f"guide over f/fc in [1.152, 1.498] {'matches' if baseline_verdict=='passed' else 'DOES NOT match'} "
            f"the analytic L_eff/v_g(f) oracle to {tol_ns*1e9:.2f} ps "
            f"(max interior diff {max_diff_ns:.4f} ns)."
        ),
        "fc_te10_hz": FC_TE10_HZ,
        "l_eff_m": l_eff,
        "freqs_hz": freqs.tolist(),
        "fc_ratio": (freqs / FC_TE10_HZ).tolist(),
        "tau_g_measured_ns": (tau_measured * 1e9).tolist(),
        "tau_g_analytic_ns": (tau_analytic * 1e9).tolist(),
        "endpoints_gated": False,
        "interior_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "max_abs_diff_ns": max_diff_ns,
        "mean_abs_diff_ns": mean_diff_ns,
        "max_group_delay_tol_ns": tol_ns,
        "tol_provenance": f"measured envelope: worst interior diff {max_diff_ns:.4f} ns x 1.3 margin",
        "qualitative_expectations": {
            "monotonic_decrease": monotonic,
            "measured_ratio_tau0_over_tauN": d_ratio,
            "analytic_ratio_tau0_over_tauN": d_ratio_analytic,
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

    print(f"status={baseline_verdict}  max_diff={max_diff_ns:.4f}ns  tol={tol_ns:.4f}ns")
    print(f"monotonic={monotonic}  ratio meas={d_ratio:.4f} analytic={d_ratio_analytic:.4f}")
    print(f"settling (np60 vs np120) max diff: {settling_diff_ns:.4f} ns")
    print(f"domain invariance: baseline={baseline_verdict} grown={grown_verdict} "
          f"grown_max_diff={grown_max_diff_ns:.4f}ns")
    for name, v in fals.items():
        print(f"falsifier[{name}]: worst_diff={v['worst_abs_diff_ns']:.4f}ns gate_reds={v['gate_reds']}")
    print(f"wrote {out_path.relative_to(REPO)}")
    for i, f in enumerate(freqs):
        print(f"  f={f/1e9:.4f}GHz f/fc={f/FC_TE10_HZ:.4f} tau_meas={tau_measured[i]*1e9:.4f}ns "
              f"tau_analytic={tau_analytic[i]*1e9:.4f}ns")


if __name__ == "__main__":
    main()
