"""Small CPU ADI PEC falsifier. Run each case under timeout 60.

Example: JAX_PLATFORMS=cpu PYTHONPATH=. timeout 60 python
scripts/diagnostics/adi_pec_stability.py --mode 3d --kind sheet --factor 5
--steps 200 --output docs/design_notes/adi_pec_stability/3d_sheet_5_200.json

Uses the public run path; after the guard lands, --unguarded-measurement
explicitly bypasses only the guard for reproducing the numerical evidence.
"""
import argparse
import json
import time
import os
import threading
from unittest.mock import patch
import warnings
from pathlib import Path

import numpy as np
from rfx import Box, Simulation


def build(mode, kind, factor):
    sim = Simulation(freq_max=15e9, domain=(.02, .02, .02), dx=.001,
                     boundary="pec", solver="adi", mode=mode,
                     adi_cfl_factor=factor)
    thickness = {"sheet": 0, "volume1": .001, "volume3": .003}.get(kind)
    if mode == "3d":
        lo = (.004, .004, .010)
        hi = (.016, .016, .010 + (thickness or 0))
        src, probe = (.010, .010, .005), (.010, .010, .015)
    else:
        # In-plane sheet, tangential to the live Ez; collapse only z.
        lo = (.004, .010, 0)
        hi = (.016, .010 + (thickness or 0), .020)
        src, probe = (.010, .005, 0), (.010, .015, 0)
    if thickness is not None:
        sim.add(Box(lo, hi), material="pec")
    sim.add_source(src, "ez")
    sim.add_probe(probe, "ez")
    sim.add_probe(src, "ez")
    return sim


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("3d", "2d_tmz"))
    p.add_argument("--kind", choices=("none", "sheet", "volume1", "volume3"))
    p.add_argument("--factor", type=float)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unguarded-measurement", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="Run the 104-case serial battery; output is a directory")
    args = p.parse_args(argv)
    if args.sweep:
        args.output.mkdir(parents=True, exist_ok=True)
        for steps, factors in ((200, (.5, 1, 1.5, 2, 3, 5)),
                               (800, (1, 1.25, 1.5, 1.75, 2)),
                               (4000, (.5, 1))):
            for mode in ("3d", "2d_tmz"):
                for factor in factors:
                    for kind in ("none", "sheet", "volume1", "volume3"):
                        dest = args.output / f"{mode}_{kind}_{factor:g}_{steps}.json"
                        if dest.exists():
                            continue
                        argv_case = ["--mode", mode, "--kind", kind,
                                     "--factor", str(factor), "--steps", str(steps),
                                     "--output", str(dest)]
                        if args.unguarded_measurement:
                            argv_case.append("--unguarded-measurement")
                        watchdog = threading.Timer(60, lambda: os._exit(124))
                        watchdog.start()
                        try:
                            main(argv_case)
                        finally:
                            watchdog.cancel()
        return
    if args.mode is None or args.kind is None or args.factor is None:
        p.error("single cases require --mode, --kind, and --factor")
    start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = build(args.mode, args.kind, args.factor)
        if args.unguarded_measurement:
            with patch("rfx.adi._validate_interior_pec", lambda *a, **kw: None):
                result = sim.run(n_steps=args.steps, skip_preflight=True)
        else:
            result = sim.run(n_steps=args.steps, skip_preflight=True)
        traces = np.asarray(result.time_series)
        fields = [np.asarray(getattr(result.state, c)) for c in
                  (("ex", "ey", "ez", "hx", "hy", "hz") if args.mode == "3d"
                   else ("ez", "hx", "hy"))]
    probe = traces[:, 0]
    finite = np.isfinite(probe)
    # JSON null marks each non-finite sample; counts retain the distinction.
    def peak(a):
        a = np.asarray(a)
        return float(np.max(np.abs(a[np.isfinite(a)]))) if np.isfinite(a).any() else None
    out = dict(mode=args.mode, kind=args.kind, factor=args.factor,
               steps=args.steps, dt=float(result.dt), seconds=time.monotonic()-start,
               peak_finite=peak(probe), nonfinite=int((~finite).sum()),
               quarter_peaks=[peak(a) for a in np.array_split(probe, 4)],
               final_field_peak_finite=max((peak(a) or 0) for a in fields),
               final_field_nonfinite=sum(int((~np.isfinite(a)).sum()) for a in fields),
               traces=[[float(x) if np.isfinite(x) else None for x in row] for row in traces])
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k:v for k,v in out.items() if k != "traces"}), flush=True)

if __name__ == "__main__":
    main()
