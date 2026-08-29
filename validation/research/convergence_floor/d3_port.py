"""D3 (issue #786) — port / probe loading.

First principles (pre-declared): rfx soft sources are ADDITIVE
(``rfx/simulation.py::make_soft_source``: ``E += Cb*w(t)``). An additive
forcing term in a linear time-invariant system leaves the system operator
-- hence every eigenfrequency -- exactly unchanged, so the predicted
coupling-induced df is EXACTLY ZERO. This arm measures it.

Variants, at two scales, everything else held at the PR #785 values:
  a  baseline (src_amp = 1.0)
  b  src_amp = 0.01     (100x weaker drive)
  c  src_amp = 100.0    (100x stronger drive)
  d  source pair moved inward to (9.0/18.0, 11.25, 0.75) mm -- same
     x-odd / y-even symmetry class, different physical coupling
  e  probe moved to (15.75, 11.25, 0.75) mm

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d3_port
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d3_port.json")
WINDOWS = os.path.join(RES, "predeclared_windows_786.json")

VARIANTS = {
    "a_baseline": dict(),
    "b_amp_0.01": dict(src_amp=0.01),
    "c_amp_100": dict(src_amp=100.0),
    "d_ports_inward": dict(src_p=(9.0e-3, 11.25e-3, 0.75e-3),
                           src_m=(18.0e-3, 11.25e-3, 0.75e-3)),
    "e_probe_moved": dict(prb=(15.75e-3, 11.25e-3, 0.75e-3)),
}
SCALES = (0.75, 0.5)


def main():
    warnings.filterwarnings("ignore")
    win = json.load(open(WINDOWS))["D3_port_loading"]
    exo = float(win["exonerate_hz"])
    att = float(win["attribute_hz"])

    out = {"issue": 786, "discriminator": "D3", "scales": list(SCALES),
           "rows": [], "per_scale": {}}
    for s in SCALES:
        fs = {}
        for name, kw in VARIANTS.items():
            r = fx.measure(s, multiband=False, **kw)
            fs[name] = r["f_target"]
            out["rows"].append({"scale": s, "variant": name,
                                "f_target": r["f_target"],
                                "dominance": r["dominance"],
                                "wallclock_s": r["wallclock_s"]})
            print("s=%-5s %-14s f=%.6f GHz  d(base)=%+9.3f kHz  wall=%.0fs"
                  % (s, name, r["f_target"] / 1e9,
                     (r["f_target"] - fs["a_baseline"]) / 1e3,
                     r["wallclock_s"]), flush=True)
        v = np.array(list(fs.values()))
        span = float(v.max() - v.min())
        # Coupling-strength arm only (a,b,c,d): monotone dependence test.
        coup = [fs["b_amp_0.01"], fs["a_baseline"], fs["c_amp_100"]]
        mono = bool(all(coup[i] < coup[i + 1] for i in range(2))
                    or all(coup[i] > coup[i + 1] for i in range(2)))
        coup_span = float(max(coup + [fs["d_ports_inward"]])
                          - min(coup + [fs["d_ports_inward"]]))
        out["per_scale"][str(s)] = {
            "f_by_variant_hz": fs, "max_pairwise_span_hz": span,
            "coupling_arm_span_hz": coup_span,
            "coupling_monotone": mono,
            "verdict": ("EXONERATED (span %.3f kHz <= 1 MHz)" % (span / 1e3)
                        if span <= exo else
                        ("ATTRIBUTED (monotone in coupling, span %.3f MHz "
                         ">= 10 MHz)" % (coup_span / 1e6)
                         if mono and coup_span >= att else
                         "INCONCLUSIVE (span %.3f MHz)" % (span / 1e6))),
        }
        print("  s=%s -> %s" % (s, out["per_scale"][str(s)]["verdict"]))

    verdicts = [v["verdict"] for v in out["per_scale"].values()]
    out["verdict"] = ("EXONERATED at every scale"
                      if all(v.startswith("EXONERATED") for v in verdicts)
                      else " | ".join(verdicts))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nD3:", out["verdict"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
