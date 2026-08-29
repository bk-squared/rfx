"""Issue #683 follow-up: acceptance run for the UNIFORM-lane sampling flip.

Pre-declaration (binding, committed BEFORE the flip and before this is run):
    docs/design_notes/issue683_uniform_flip_predeclaration.md  (gate A1)
Protocol whose rule this re-applies unchanged:
    docs/design_notes/issue683_sampling_order_decision_protocol.md  (sec. 5-6)

This re-runs the decision harness's UNIFORM-lane arm (the arm that was
"PRE" and FAILED the circuit law with slope -0.62 / intercept -81 Ohm)
against the flipped uniform lane. Acceptance, same numbers as protocol
section 6, one run, no tuning:

    G1 coupling gates pass, and
    n_live*a in [0.90, 1.10] and n_live*|b| <= 10 Ohm at BOTH
    f1 = 0.05 GHz and f2 = 0.1 GHz.

It also runs the NU lane once at R_L = 50 and reports the max lane
difference of the raw accumulators (expected ~0 after the flip: gate G2 of
the decision run measured the lane difference to be EXACTLY the same-step
injection increment, which the flip removes). That report is corroborating
output, not a pass/fail gate here — the binding lane-parity gate is the
witness test in tests/test_nu_wire_port_lane_parity.py (A2).

Run from the worktree:

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu \
      /Users/byungkwankim/Documents/rfx/.venv/bin/python \
      validation/research/issue683_flip_acceptance.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from issue683_sampling_order_decision import (  # noqa: E402
    FREQS, F1, F2, RL_SWEEP, N_LIVE, MID_DRV, MID_LOAD,
    G1A_RATIO_MIN, G1B_RATIO_MIN, SLOPE_LO, SLOPE_HI, INTERCEPT_MAX_OHM,
    build, run_pre, run_post, pick, fit,
)


def main() -> int:
    results = {}
    print("[A1] uniform-lane R_L sweep on the FLIPPED code")
    for r_l in RL_SWEEP:
        accs, dt, _w = run_pre(build(nu=False, r_load=r_l))
        v_drv = pick(accs, MID_DRV)[0].astype(np.complex128)
        i_drv = pick(accs, MID_DRV)[1].astype(np.complex128)
        v_load = pick(accs, MID_LOAD)[0].astype(np.complex128)
        results[r_l] = dict(v=v_drv, i=i_drv, v_load=v_load)
        rho = v_drv / i_drv
        print(f"  R_L={r_l:6.1f}  n*Re(rho)(f1)={N_LIVE*rho[F1].real:+9.3f}  "
              f"(f2)={N_LIVE*rho[F2].real:+9.3f}")

    ok = True

    # G1 (fixture validity, ordering-independent observables)
    i_mag = np.array([abs(results[r]["i"][F1]) for r in RL_SWEEP])
    vl_mag = np.array([abs(results[r]["v_load"][F1]) for r in RL_SWEEP])
    mono = bool(np.all(np.diff(i_mag) < 0))
    ratio_i = float(i_mag[0] / i_mag[-1])
    ratio_v = float(vl_mag.max() / vl_mag.min())
    g1 = mono and ratio_i >= G1A_RATIO_MIN and ratio_v >= G1B_RATIO_MIN
    print(f"[G1] monotone={mono} ratio_I={ratio_i:.2f} "
          f"(gate >= {G1A_RATIO_MIN}) ratio_Vload={ratio_v:.2f} "
          f"(gate >= {G1B_RATIO_MIN}) -> {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # A1 decision fits (same rule as protocol section 6)
    rl = np.array(RL_SWEEP)
    for fb, name in ((F1, "f1=0.05GHz"), (F2, "f2=0.10GHz")):
        rho = np.array([(results[r]["v"][fb] / results[r]["i"][fb]).real
                        for r in RL_SWEEP])
        a, b = fit(rl, N_LIVE * rho)
        passes = (SLOPE_LO <= a <= SLOPE_HI) and abs(b) <= INTERCEPT_MAX_OHM
        print(f"[FIT] {name}  n*a={a:+.4f} (gate [{SLOPE_LO},{SLOPE_HI}])  "
              f"n*b={b:+.3f} Ohm (gate |b|<={INTERCEPT_MAX_OHM}) -> "
              f"{'PASS' if passes else 'FAIL'}")
        ok &= passes

    # Corroboration: lane difference at R_L = 50 (not a gate here)
    accs_u, dt_u, _ = run_pre(build(nu=False, r_load=50.0))
    accs_n, dt_n, _ = run_post(build(nu=True, r_load=50.0))
    dv = np.max(np.abs(pick(accs_u, MID_DRV)[0].astype(np.complex128)
                       - pick(accs_n, MID_DRV)[0].astype(np.complex128)))
    di = np.max(np.abs(pick(accs_u, MID_DRV)[1].astype(np.complex128)
                       - pick(accs_n, MID_DRV)[1].astype(np.complex128)))
    print(f"[LANE] R_L=50 raw-acc max|dV|={dv:.3e}  max|dI|={di:.3e}  "
          f"dt_uni={dt_u:.6e} dt_nu={dt_n:.6e}")

    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
