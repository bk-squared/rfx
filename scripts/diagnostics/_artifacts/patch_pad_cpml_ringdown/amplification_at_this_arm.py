#!/usr/bin/env python3
"""Does the #1043 amplification inequality reach the #801 arm?  (no FDTD)

Runs rfx's own one-step CPML amplification model -- the ONE copy, owned by
``tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py::amplification_rho``
-- at THIS arm's cell size, timestep and layer counts, for the epsilon pair the arm
actually assembles.

The #801 rig calls ``sim.run(num_periods=...)`` with no ``subpixel_smoothing``, and that
parameter defaults to ``False`` (``rfx/api/_execute.py``), so the E update and the psi
coefficient both read ``materials.eps_r``: eps_a == eps_b by construction.  The model's
own ``eps 12 pad, consistent (= subpixel OFF row)`` case already reads rho = 1; this
re-runs it at the arm's numbers rather than borrowing that row.

The comparator is checked first (a lossless slice with no absorber must sit exactly on
the unit circle), because a model that cannot reproduce the trivial case cannot be read
on the interesting one.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.pardir, os.pardir, os.pardir, os.pardir))
assert os.path.isdir(os.path.join(REPO, "rfx")), REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "unit", "boundaries"))

from test_cpml_subpixel_coefficient_consistency import amplification_rho  # noqa: E402

H = 0.787e-3
EPS_R = 3.38
DT = 3.751186055802254e-13   # the arm's own dt, from its recorded run
DX = H / 4                   # n = 4

out = {
    "arm": "isolated patch, n = 4, dx = h/4, dt from the recorded run",
    "dx_m": DX, "dt_s": DT, "eps_substrate": EPS_R,
    "model_owner": ("tests/unit/boundaries/test_cpml_subpixel_coefficient_consistency.py"
                    "::amplification_rho"),
    "comparator": {},
    "cases": [],
}
out["comparator"]["rho_lossless_no_absorber"] = amplification_rho(8, DX, DT, 1.0, 1.0,
                                                                 absorber=False)
out["comparator"]["rho_vacuum_consistent_cpml_8"] = amplification_rho(8, DX, DT, 1.0, 1.0)

for layers in (6, 8, 16, 32):
    for name, ea, eb in (
        ("as the arm assembles it: subpixel OFF, pad eps consistent", EPS_R, EPS_R),
        ("counterfactual #1043 pair: Yee half sees the substrate, psi sees vacuum",
         EPS_R, 1.0),
        ("counterfactual reversed: Yee half vacuum, psi sees the substrate", 1.0, EPS_R),
        ("vacuum pad, consistent (the +x pad this arm actually has at pad = 10h)",
         1.0, 1.0),
    ):
        rho = amplification_rho(layers, DX, DT, ea, eb, n=max(24, 3 * layers))
        out["cases"].append(dict(cpml_layers=layers, case=name, eps_a=ea, eps_b=eb,
                                 rho=rho, unstable=bool(rho > 1 + 1e-6)))

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "amplification_at_this_arm.json"), "w") as fh:
    json.dump(out, fh, indent=1)

print(f"comparator: no-absorber rho = {out['comparator']['rho_lossless_no_absorber']!r}, "
      f"vacuum CPML rho = {out['comparator']['rho_vacuum_consistent_cpml_8']!r}")
for c in out["cases"]:
    print(f"  layers {c['cpml_layers']:2d}  eps_a {c['eps_a']:5.2f} eps_b {c['eps_b']:5.2f}  "
          f"rho {c['rho']:.12f}  {'UNSTABLE' if c['unstable'] else 'bounded'}   {c['case']}")
