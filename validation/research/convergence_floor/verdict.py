"""Issue #786 — synthesis: apply the frozen windows, apportion, verdict.

Reads only the committed discriminator outputs; computes nothing new
except the apportionment arithmetic the pre-declaration fixed.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.verdict
"""

from __future__ import annotations

import json
import os

import numpy as np

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "verdict.json")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    win = load("predeclared_windows_786.json")
    d0 = load("d0_reproduction.json")
    d1 = load("d1_geometry.json")
    d2 = load("d2_edge.json")
    d3 = load("d3_port.json")
    d4a = load("d4_reference_a.json")
    d4b = load("d4_reference_b.json")
    d4c = load("d4_reference_c.json")

    out = {"issue": 786, "verdicts": {}}
    out["verdicts"]["D0"] = d0["verdict"] if d0 else "NOT RUN"
    out["verdicts"]["D1"] = d1["verdict"] if d1 else "NOT RUN"
    out["verdicts"]["D2"] = d2["verdict"] if d2 else "NOT RUN"
    out["verdicts"]["D3"] = d3["verdict"] if d3 else "NOT RUN"
    out["verdicts"]["D4a"] = d4a["D4a"]["verdict"] if d4a else "NOT RUN"
    if d4b:
        out["verdicts"]["D4b"] = {k: v["verdict"]
                                  for k, v in d4b["D4b"].items()}
    if d4c:
        out["verdicts"]["D4c"] = {("%s_%s" % (r["arm"], r["scale"])):
                                  r["verdict"] for r in d4c["D4c"]["rows"]}

    # --- apportionment (rule frozen in the pre-declaration) -----------
    app = {}
    if d4b and d4c:
        f_inf = d4b["D4b"]["uniform"]["f_inf_hz"]
        ref_rows = [r for r in d4c["D4c"]["rows"]
                    if r["arm"] == "UC" and r["scale"] == fx.REF_SCALE]
        if ref_rows:
            r = ref_rows[0]
            e1 = r["E1_hz"]
            cons = r["mean_hz"]
            d_total = abs(e1 - f_inf)
            d_instr = abs(e1 - cons)
            d_phys = abs(cons - f_inf)
            app = {
                "f_inf_hz": f_inf,
                "f_E1_at_ref_hz": e1,
                "f_consensus_at_ref_hz": cons,
                "consensus_spread_hz": r["spread_hz"],
                "Delta_total_hz": d_total,
                "Delta_instr_hz": d_instr,
                "Delta_phys_hz": d_phys,
                "instr_pct": 100.0 * d_instr / d_total if d_total else None,
                "phys_pct": 100.0 * d_phys / d_total if d_total else None,
                "remedy_licensed_4a": bool(d_total and
                                           d_instr >= 0.5 * d_total),
            }
    out["apportionment"] = app

    # --- the ladder, re-judged against the independent reference ------
    if d4b:
        u = d4b["D4b"]["uniform"]
        errs = [u["err_vs_f_inf_mhz"][str(s)] for s in u["scales"]]
        out["ladder_against_independent_reference"] = {
            "scales": u["scales"],
            "err_mhz": u["err_vs_f_inf_mhz"],
            "monotone_decreasing": bool(all(errs[i] > errs[i + 1]
                                            for i in range(len(errs) - 1))),
            "p_loglog": u["p_from_loglog_vs_f_inf"],
            "p_nls": u["p"],
        }
        if "multiband" in d4b["D4b"]:
            m = d4b["D4b"]["multiband"]
            out["ladder_against_independent_reference"]["multiband"] = {
                "err_mhz": m["err_vs_f_inf_mhz"],
                "p_loglog": m["p_from_loglog_vs_f_inf"], "p_nls": m["p"]}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
