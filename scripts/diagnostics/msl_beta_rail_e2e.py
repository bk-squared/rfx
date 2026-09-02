"""Issue #681 end-to-end confirmation on real FDTD phasors (CPU-scale).

Three single-port MSL runs on the same physical board (eps_r 2.2 open
thru, dx = 200 um, f_max = 20 GHz — the tests/unit/sparams/test_msl_probe_offset_interval
open-thru geometry):

  A  control: port declared eps_r_sub = 2.2 (true). Expect NO beta-rail
     warning and result.beta_railed all False.
  B  mis-declared: port told eps_r_sub = 9.8 while the physical substrate
     stays 2.2. The HJ anchor is then wrong by ~sqrt(7.38/1.87): the true
     beta sits at ~0.50*beta0, outside the [0.65, 1.35]*beta0 scan window.
     Expect the rail warning and beta_railed True in >= 50% of bins.
     (On b29f9de this run returned Z0/beta silently pinned at the scan
     rail — the #681 silent-wrong class.)
  A' span control: as A but with the conservative pre-#681 spacing pinned
     explicitly (2 cells). |S11| must match run A within 0.02 everywhere:
     the #681 span widening must not move S (S rides on the analytic HJ
     anchor and the V*I split, never on the fitted beta).

Pre-declared falsifier (declared before the first run; one attempt):
  - B emits the "pinned at its own window limit" warning AND
    mean(beta_railed) >= 0.5;
  - A emits no such warning AND beta_railed is all False.
Either failure is reported as-is, no post-hoc tuning.

RESULT (2026-08-29, CPU, worktree agent/issue-681-nprobe-fit): the
falsifier FIRED on the control half. Run B railed loudly at 7/8 bins
(expected) and the A-vs-A' |S11| invariance held to 4 decimals, but
run A ALSO railed at its three lowest bins (4/6/8 GHz), beta_railed
[1,1,1,0,0,0,0,0]. The criterion was NOT retuned. Two readings, not
adjudicated here: false positives, or true detections of a genuinely
unbracketable low-bin fit on this deliberately marginal board (its own
preflight flags the port 900 um from the x-CPML vs 1588 um
recommended, and the low bins have the shortest span in lambda_g).
The follow-up adjudication is pre-registered in
scripts/vessl_msl_beta_rail_e2e_adjudication.yaml (clean-clearance
geometry + settling/residual dump); it is a NEW experiment, not a
rerun of this one.

ADJUDICATION RESULT (2026-08-29, VESSL run 369367256766, RTX 4090,
clean-clearance geometry, settling -46.2 dB, all bins reliable,
settling-conditioned exit criterion): FALSIFIED again — run A railed at
2/8 SETTLED bins (8 and 10 GHz), run B railed loudly at 6/8, A-vs-A'
max d|S11| = 0.0000. Adjudicated reading: these are TRUE detections,
not detector false positives — run A's fitted |Z0| is 16-34 Ohm across
ALL bins (railed or not) against the ~47 Ohm physical line, i.e. the
N-probe fitted Z0/beta lane is the known DIAGNOSTIC-ONLY quantity
(ledger 2026-08-20: "MSL extractor reports non-physical Z0/eps_eff on
a real board ... does NOT reach S11"), and the rail flag marks the
bins where that already-unreliable fit additionally failed to bracket.
The "correctly-declared run must not rail at all" expectation was the
wrong half of the falsifier — withdrawn on this evidence; the flag's
shipped semantics ("scan failed to bracket; do not quote Z0/beta at
flagged bins") stand. Committed evidence:
docs/design_notes/issue681_rail_adjudication_report.json.

Run:  python scripts/diagnostics/msl_beta_rail_e2e.py [--out out.json]
Exit code 0 = all assertions hold.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np

# Path bootstrap: `python scripts/diagnostics/...` puts the script dir, not the
# repo root, on sys.path — without this the venv's installed (main) rfx wins
# and the run crashes on the missing beta_railed attribute.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rfx import Box, Simulation

DX = 2e-4
Y_C = 0.01316
W_TRACE = 0.002413
H_SUB = 0.000794
EPS_TRUE = 2.2
FREQS = np.linspace(4e9, 18e9, 8)

RAIL_MSG = "pinned at its own window limit"


def _build(args, eps_r_sub_declared: float, n_probe_spacing=None) -> Simulation:
    domain = (args.domain_x, 0.02632, 0.0038)
    sim = Simulation(freq_max=20e9, domain=domain, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=EPS_TRUE)
    sim.add(Box((0, 0, 0), (domain[0], domain[1], H_SUB)), material="sub")
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (domain[0] - 0.001, Y_C + W_TRACE / 2, H_SUB + DX)),
            material="pec")
    kw = {} if n_probe_spacing is None else {"n_probe_spacing": n_probe_spacing}
    sim.add_msl_port(position=(args.feed_x, Y_C, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     eps_r_sub=eps_r_sub_declared, name="p1", **kw)
    return sim


def _run(args, sim: Simulation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(
            freqs=FREQS, num_periods=args.num_periods,
        )
    rail_warned = any(RAIL_MSG in str(w.message) for w in caught)
    return res, rail_warned, [str(w.message) for w in caught]


def _diag(res) -> dict:
    return dict(
        settling_db=(None if res.settling_db is None
                     else np.asarray(res.settling_db).tolist()),
        reliable=(None if res.reliable is None
                  else np.asarray(res.reliable).astype(int).tolist()),
        z0_fit_abs=np.abs(np.asarray(res.Z0[0])).tolist(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    # Defaults reproduce the 2026-08-29 (marginal-board) attempt; the
    # adjudication yaml passes a clean-clearance geometry instead:
    # --domain-x 0.040 --feed-x 0.005 --num-periods 40.
    ap.add_argument("--domain-x", type=float, default=0.020)
    ap.add_argument("--feed-x", type=float, default=0.0025)
    ap.add_argument("--num-periods", type=int, default=20)
    args = ap.parse_args()

    report: dict = {"freqs_ghz": (FREQS / 1e9).tolist()}
    failures: list[str] = []

    print("=== run A: correct eps_r_sub declaration (control) ===")
    res_a, warned_a, _ = _run(args, _build(args, EPS_TRUE))
    railed_a = np.asarray(res_a.beta_railed, dtype=bool)
    print(f"  beta_railed: {railed_a.astype(int).ravel().tolist()}  "
          f"rail-warned: {warned_a}")
    report["A"] = {"railed_frac": float(railed_a.mean()),
                   "beta_railed": railed_a.astype(int).ravel().tolist(),
                   "rail_warned": bool(warned_a), **_diag(res_a)}
    # Exit criterion matches the pre-registered adjudication declaration:
    # run A gates only on bins whose record is SETTLED (settling_db < -40 dB).
    # Rails on under-settled bins are recorded but not gating.
    sett_a = report["A"].get("settling_db")
    if sett_a is None:
        settled_a = np.ones_like(railed_a, dtype=bool)
    else:
        try:
            settled_a = np.broadcast_to(
                np.asarray(sett_a, dtype=float) < -40.0, railed_a.shape
            )
        except ValueError:  # shape-incompatible dump: gate on the worst record
            settled_a = np.full(
                railed_a.shape, bool(float(np.min(np.asarray(sett_a, dtype=float))) < -40.0)
            )
    railed_settled_a = railed_a & settled_a
    report["A"]["railed_settled_frac"] = float(railed_settled_a.mean())
    if railed_settled_a.any():
        failures.append("FALSIFIED: control run A railed on settled (< -40 dB) bins")
    elif warned_a or railed_a.any():
        print("note: run A railed/warned only on non-settled bins — "
              "recorded, not gating (settling-conditioned falsifier)")

    print("=== run B: port told eps_r_sub=9.8 on a physical 2.2 board ===")
    res_b, warned_b, _ = _run(args, _build(args, 9.8))
    railed_b = np.asarray(res_b.beta_railed, dtype=bool)
    print(f"  beta_railed: {railed_b.astype(int).ravel().tolist()}  "
          f"rail-warned: {warned_b}")
    report["B"] = {"railed_frac": float(railed_b.mean()),
                   "beta_railed": railed_b.astype(int).ravel().tolist(),
                   "rail_warned": bool(warned_b), **_diag(res_b)}
    if not warned_b or railed_b.mean() < 0.5:
        failures.append(
            "FALSIFIED: mis-declared run B did not rail loudly "
            f"(warned={warned_b}, railed_frac={railed_b.mean():.2f})"
        )

    print("=== run A': pre-#681 pinned spacing (S invariance control) ===")
    res_ap, _, _ = _run(args, _build(args, EPS_TRUE, n_probe_spacing=2))
    s_a = np.abs(np.asarray(res_a.S[0, 0, :]))
    s_ap = np.abs(np.asarray(res_ap.S[0, 0, :]))
    ds = float(np.max(np.abs(s_a - s_ap)))
    print(f"  max ||S11|_widened - |S11|_pinned| = {ds:.4f}")
    report["span_s_invariance_max_dS"] = ds
    if ds > 0.02:
        failures.append(f"S moved with the span widening: max dS = {ds:.4f}")

    report["failures"] = failures
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    print("RESULT:", "PASS" if not failures else "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
