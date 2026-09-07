#!/usr/bin/env python3
"""Exact (semi-analytic) annulus modes for cv02 — a REPORTED-ONLY oracle.

**This script gates nothing.** It exists because cv02's Q comparison had no
third party: rfx and Meep were compared to each other, so the only thing a
disagreement could be attributed to was "one of them". The 2-D TM(Ez) annulus
the case builds — air inside ``r < R_in``, index ``n`` between ``R_in`` and
``R_out``, outgoing air outside — has an exact solution, so the two solvers'
errors can each be measured separately against it.

The mode condition is the 4x4 Bessel/Hankel matching determinant for
``Ez = f(r) e^{i m phi}`` (``Ez`` and ``dEz/dr`` continuous at both interfaces,
``J_m`` regular at the origin, ``H^(1)_m`` outgoing at infinity), solved for a
complex frequency by secant iteration. ``Q = Re(f) / (-2 Im(f))``, Meep's own
definition.

Why it is trustworthy without a second solver: **Maxwell scale invariance**.
Scaling both radii by the same fraction must leave ``Q`` exactly invariant and
scale ``f`` by exactly ``-1``. The printed oracle block recomputes
``R_out d lnQ/d R_out + R_in d lnQ/d R_in`` (must be 0) and the same
combination for ``f`` (must be -1) from independently taken numerical
derivatives. Both come back to 5-6 digits. That is a check on the solve, not a
solver cross-check, and this script says so rather than claiming more.

What it is used for (issue #907, pre-declaration Correction 4):

* the **decomposition** of the rfx-vs-Meep Q gap into the two codes' own errors
  against the exact annulus — the signed logs differ by the measured gap to
  four decimals on all three modes;
* the honest statement of how loose the Q gate's resolution envelope is: the
  reference's real accuracy here is ~17x better than the envelope, and rfx's
  own error against the exact annulus is LARGER than the reference's on two of
  three modes, so a pass on that gate is not evidence that rfx's radiation Q is
  accurate;
* the narrowed falsification: an effective radius or index shift of the ideal
  annulus large enough to produce the observed Q gap would move the frequency
  far more than the two solvers' 0.028% frequency agreement permits. That is
  NOT a falsification of discretization as a class — the same model
  under-predicts Meep's OWN Q error by 11x/7x/1x — and the script prints that
  self-test so the claim cannot be over-read.

Numbers are quoted, not measured, on both solver sides: the Meep column is the
Meep "Modes of a Ring Resonator" tutorial's published harminv output and the
rfx column is the committed frozen board in
``tests/crossval/test_cv02_ring_mode_judge.py``.

Usage::

    python scripts/diagnostics/cv02_annulus_exact_oracle.py [--json out.json]
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path

from scipy.special import h1vp, hankel1, jv, jvp, yv, yvp

# --- the cv02 geometry, verbatim from validation/crossval/02_ring_resonator.py
N_WG = 3.4
R_IN = 1.0
R_OUT = 2.0
DX = 0.1                      # a / resolution, resolution = 10

#: Meep "Modes of a Ring Resonator" tutorial, published harminv output
#: (meep.readthedocs.io, Python_Tutorials/Basics). Quoted, not measured here.
MEEP_REFERENCE = {
    3: (0.118101575043663, 80.683059081382),
    4: (0.147162555528154, 316.29272471914),
    5: (0.175246750722663, 1677.48461212767),
}

#: One unmodified run of 02_ring_resonator.py, 2026-08-31, commit 649b2cf —
#: the frozen board in tests/crossval/test_cv02_ring_mode_judge.py.
RFX_TODAY = {
    3: (0.118068, 86.5),
    4: (0.147213, 357.6),
    5: (0.175298, 1864.1),
}


def _det(f: complex, m: int, n: float = N_WG, r_in: float = R_IN,
         r_out: float = R_OUT) -> complex:
    """4x4 matching determinant of the annulus at complex frequency ``f``.

    Units: ``c = 1``, ``a = 1``, so ``k = 2 pi f`` and ``f`` is in ``c/a`` —
    the same normalisation Meep prints. Expanded by hand (rather than through
    a linear-algebra determinant) so the function stays a plain complex scalar
    of ``f`` for the secant iteration.
    """
    k = 2.0 * math.pi * f
    a1, a2 = k * r_in, n * k * r_in
    b1, b2 = n * k * r_out, k * r_out
    # Inner interface: A J_m(k r) = B J_m(nk r) + C Y_m(nk r) and the same for
    # the radial derivative. Eliminating A gives one relation between B and C,
    # written as the 2x2 minor below; likewise the outer interface eliminates
    # D. The mode condition is that the two relations are proportional.
    ji, jpi = jv(m, a1), jvp(m, a1)
    p1, dp1 = jv(m, a2), n * jvp(m, a2)
    q1, dq1 = yv(m, a2), n * yvp(m, a2)
    p2, dp2 = jv(m, b1), n * jvp(m, b1)
    q2, dq2 = yv(m, b1), n * yvp(m, b1)
    ho, hpo = hankel1(m, b2), h1vp(m, b2)
    # inner: (ji * dp1 - jpi * p1) B + (ji * dq1 - jpi * q1) C = 0  ... (I)
    ai = ji * dp1 - jpi * p1
    bi = ji * dq1 - jpi * q1
    # outer: (ho * dp2 - hpo * p2) B + (ho * dq2 - hpo * q2) C = 0  ... (II)
    ao = ho * dp2 - hpo * p2
    bo = ho * dq2 - hpo * q2
    return ai * bo - bi * ao


def solve_mode(m: int, f_guess: complex, *, n: float = N_WG,
               r_in: float = R_IN, r_out: float = R_OUT,
               tol: float = 1e-14, max_iter: int = 200) -> complex:
    """Complex resonant frequency of azimuthal order ``m`` (secant iteration).

    Returns ``f`` with ``Im(f) < 0`` (decaying). Raises ``RuntimeError`` rather
    than returning a non-converged value — a silently unconverged root would be
    an invented oracle.
    """
    x0 = complex(f_guess)
    x1 = x0 * (1.0 + 1e-6)
    f0 = _det(x0, m, n, r_in, r_out)
    f1 = _det(x1, m, n, r_in, r_out)
    for _ in range(max_iter):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not cmath.isfinite(x2):
            break
        x0, f0, x1 = x1, f1, x2
        f1 = _det(x1, m, n, r_in, r_out)
        if abs(x1 - x0) <= tol * abs(x1):
            return x1
    raise RuntimeError(f"annulus mode m={m} did not converge from {f_guess!r}")


def exact_mode(m: int, f_guess: complex, **kw) -> tuple[float, float]:
    """``(f, Q)`` of one exact annulus mode. ``Q = Re f / (-2 Im f)``."""
    f = solve_mode(m, f_guess, **kw)
    return f.real, f.real / (-2.0 * f.imag)


def _sensitivities(m: int, f_seed: complex, h: float = 1e-4) -> dict:
    """d ln Q / d(param) and d ln f / d(param) ALONG the mode branch.

    Central differences in which the complex root is re-solved at each
    perturbed geometry, so ``k`` follows the branch instead of being held
    fixed. (Holding ``k`` fixed — the WKB shortcut — gives a sensitivity 10-20x
    larger and of the opposite sign, and is wrong for comparing two solvers,
    which sit on the same ``m`` branch at different effective radii.)
    """
    out: dict[str, float] = {}
    for name, kwargs_p, kwargs_m in (
        ("R_out", dict(r_out=R_OUT + h), dict(r_out=R_OUT - h)),
        ("R_in", dict(r_in=R_IN + h), dict(r_in=R_IN - h)),
        ("n", dict(n=N_WG + h), dict(n=N_WG - h)),
    ):
        fp, qp = exact_mode(m, f_seed, **kwargs_p)
        fm, qm = exact_mode(m, f_seed, **kwargs_m)
        out[f"dlnQ_d{name}"] = (math.log(qp) - math.log(qm)) / (2.0 * h)
        out[f"dlnf_d{name}"] = (math.log(fp) - math.log(fm)) / (2.0 * h)
    return out


def report() -> dict:
    """Everything this oracle knows, as a plain dict (also what --json dumps)."""
    modes: dict[int, dict] = {}
    for m, (f_meep, q_meep) in MEEP_REFERENCE.items():
        seed = complex(f_meep, -f_meep / (2.0 * q_meep))
        f_ex, q_ex = exact_mode(m, seed)
        f_rfx, q_rfx = RFX_TODAY[m]
        sens = _sensitivities(m, complex(f_ex, -f_ex / (2.0 * q_ex)))
        ln_meep = math.log(q_meep / q_ex)
        ln_rfx = math.log(q_rfx / q_ex)
        df_solvers = abs(f_rfx - f_meep) / f_meep
        # What an effective-radius / effective-index shift consistent with the
        # SOLVERS' mutual frequency agreement could do to Q.
        d_r = df_solvers / abs(sens["dlnf_dR_out"])
        d_n = df_solvers / abs(sens["dlnf_dn"])
        # Self-test: feed the model Meep's own frequency error against the
        # exact annulus and ask it to predict Meep's own Q error.
        df_meep_exact = abs(f_meep - f_ex) / f_ex
        pred = max(df_meep_exact / abs(sens["dlnf_dR_out"])
                   * abs(sens["dlnQ_dR_out"]),
                   df_meep_exact / abs(sens["dlnf_dn"]) * abs(sens["dlnQ_dn"]))
        modes[m] = {
            "f_exact": f_ex, "Q_exact": q_ex,
            "f_meep": f_meep, "Q_meep": q_meep,
            "f_rfx": f_rfx, "Q_rfx": q_rfx,
            "ln_Q_meep_over_exact": ln_meep,
            "ln_Q_rfx_over_exact": ln_rfx,
            "decomposition_difference": ln_rfx - ln_meep,
            "measured_gap_ln_Q_rfx_over_meep": math.log(q_rfx / q_meep),
            "df_solvers_pct": df_solvers * 100.0,
            "half_cell_dlnQ": abs(sens["dlnQ_dR_out"]) * DX / 2.0,
            "half_cell_dlnf_pct": abs(sens["dlnf_dR_out"]) * DX / 2.0 * 100.0,
            "freq_bound_dlnQ_geometric": d_r * abs(sens["dlnQ_dR_out"]),
            "freq_bound_dlnQ_index": d_n * abs(sens["dlnQ_dn"]),
            "selftest_df_meep_vs_exact_pct": df_meep_exact * 100.0,
            "selftest_predicted_dlnQ": pred,
            "selftest_measured_dlnQ": abs(ln_meep),
            "selftest_underprediction_factor": abs(ln_meep) / pred,
            "scale_invariance_Q": (R_OUT * sens["dlnQ_dR_out"]
                                   + R_IN * sens["dlnQ_dR_in"]),
            "scale_invariance_f": (R_OUT * sens["dlnf_dR_out"]
                                   + R_IN * sens["dlnf_dR_in"]),
            **sens,
        }
    return {"geometry": {"n": N_WG, "R_in": R_IN, "R_out": R_OUT, "dx": DX},
            "modes": modes}


def format_report(data: dict) -> str:
    lines = ["cv02 exact-annulus oracle — REPORTED ONLY, gates nothing",
             f"  2-D TM(Ez) annulus n={N_WG}, R_in={R_IN}, R_out={R_OUT}; "
             f"cv02 mesh dx={DX}", ""]
    lines.append(f"  {'m':>2} {'f_exact':>10} {'Q_exact':>10} "
                 f"{'ln(Qmeep/Qex)':>14} {'ln(Qrfx/Qex)':>13} "
                 f"{'difference':>11} {'measured gap':>13}")
    for m, d in data["modes"].items():
        lines.append(
            f"  {m:>2} {d['f_exact']:>10.6f} {d['Q_exact']:>10.4f} "
            f"{d['ln_Q_meep_over_exact']:>+14.4f} "
            f"{d['ln_Q_rfx_over_exact']:>+13.4f} "
            f"{d['decomposition_difference']:>+11.4f} "
            f"{d['measured_gap_ln_Q_rfx_over_meep']:>+13.4f}")
    lines += ["",
              "  The last two columns agree to four decimals on every mode: "
              "the rfx-vs-Meep Q gap IS",
              "  the difference of the two codes' own discretization Q errors "
              "at resolution 10.",
              "",
              "  Maxwell scale-invariance oracle (exact values 0 and -1):"]
    for m, d in data["modes"].items():
        lines.append(f"    m={m}: Q combination {d['scale_invariance_Q']:+.6f}"
                     f"   f combination {d['scale_invariance_f']:+.6f}")
    lines += ["",
              "  Branch sensitivities (root re-solved at each perturbation):"]
    lines.append(f"    {'m':>2} {'dlnQ/dR_out':>12} {'dlnQ/dR_in':>11} "
                 f"{'dlnQ/dn':>9} {'dlnf/dR_out':>12} {'dlnf/dn':>9}")
    for m, d in data["modes"].items():
        lines.append(f"    {m:>2} {d['dlnQ_dR_out']:>+12.4f} "
                     f"{d['dlnQ_dR_in']:>+11.4f} {d['dlnQ_dn']:>+9.4f} "
                     f"{d['dlnf_dR_out']:>+12.4f} {d['dlnf_dn']:>+9.4f}")
    lines += ["",
              "  What the solvers' 0.03% frequency agreement permits, in Q:"]
    for m, d in data["modes"].items():
        lines.append(
            f"    m={m}: |df/f| = {d['df_solvers_pct']:.4f}%  ->  "
            f"|dlnQ| <= {d['freq_bound_dlnQ_geometric']:.2e} (radius) / "
            f"{d['freq_bound_dlnQ_index']:.2e} (index), against a measured gap "
            f"of {abs(d['measured_gap_ln_Q_rfx_over_meep']):.4f}")
    lines += ["",
              "  SELF-TEST of that model — predict Meep's OWN Q error from "
              "Meep's OWN f error:"]
    for m, d in data["modes"].items():
        lines.append(
            f"    m={m}: |df/f| = {d['selftest_df_meep_vs_exact_pct']:.4f}%  "
            f"predicts |dlnQ| <= {d['selftest_predicted_dlnQ']:.2e}, measured "
            f"{d['selftest_measured_dlnQ']:.4f}  "
            f"(under-predicts {d['selftest_underprediction_factor']:.1f}x)")
    lines += ["",
              "  So this model bounds only the EFFECTIVE-ANNULUS class of "
              "explanation (a radius or",
              "  index shift of the ideal annulus). It does not rule out "
              "discretization as a class:",
              "  radiation Q is out-coupled flux through a staircased curved "
              "boundary, which is not a",
              "  smooth function of the annulus the frequency sees.",
              "",
              "  Reference vs rfx accuracy against this oracle, as |ln(Q/Q_ex)| "
              "in % — the same",
              "  quantity the cv02 Q gate measures. This is why a pass on that "
              "gate is a statement",
              "  about the envelope and NOT about rfx's radiation Q: rfx's own "
              "error exceeds the",
              "  reference's on two of the three modes, so no reference-side "
              "accuracy bound could",
              "  admit this case."]
    for m, d in data["modes"].items():
        lines.append(
            f"    m={m}: reference {abs(d['ln_Q_meep_over_exact']) * 100:>5.1f}%"
            f"   rfx {abs(d['ln_Q_rfx_over_exact']) * 100:>5.1f}%")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    data = report()
    print(format_report(data))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
