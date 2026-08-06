#!/usr/bin/env python3
"""Audit any committed broad-E5 envelope's absorber depth and passivity witness.

Two numbers per case, both derived FROM the artifact — no constant copied out of
a producer, so this runs on a clean checkout against any of the committed
envelopes:

  * absorber depth as a fraction of lambda_g at the LOWEST measured frequency
    (where lambda_g is longest and the absorber weakest). The repo's far-port
    discipline is >= 0.5 (#496).
  * the per-case unitarity excess, |max(|S11|^2+|S21|^2) - 1|, on lossless slabs
    where column power must be 1. This is the passivity contamination the
    absorber leaks in, and the band envelopes already record it.

Written because the absorber depth was RECORDED in six committed envelope
fixtures and asserted by nothing, which is how five band lanes at 0.060-0.162
lambda_g and the WR-90 NU lane at 0.099 all shipped (#574, #576). It reports;
it gates nothing and promotes nothing.

    python scripts/diagnostics/i496_unitarity_witness_audit.py --all
    python scripts/diagnostics/i496_unitarity_witness_audit.py --envelope X.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C0 = 299_792_458.0
DISCIPLINE_FLOOR = 0.5          # #496


def audit(path: Path, label: str = "") -> dict:
    env = json.loads(path.read_text())
    es = env.get("envelope_summary") or {}
    recipe = es.get("setup_recipe") or {}
    cpml = recipe.get("cpml_layers")
    fc = es.get("cutoff_te10_hz")
    f_lo = (es.get("freq_range_hz") or [None])[0]
    tol = env.get("max_mag_abs_tol")
    name = label or path.name

    if cpml is None or fc is None or f_lo is None:
        print(f"{name}: cannot audit — missing "
              f"{'setup_recipe.cpml_layers ' if cpml is None else ''}"
              f"{'cutoff_te10_hz ' if fc is None else ''}"
              f"{'freq_range_hz' if f_lo is None else ''}")
        return {"path": str(path), "auditable": False}

    lam_g_low = (C0 / f_lo) / math.sqrt(1.0 - (fc / f_lo) ** 2)
    print(f"\n{name}")
    print(f"  cutoff {fc / 1e9:.3f} GHz, f_lo {f_lo / 1e9:.2f} GHz, "
          f"lambda_g(f_lo) {lam_g_low * 1e3:.2f} mm, absorber {cpml} cells, "
          f"gate {tol}")
    print(f"  {'case':18} {'dx_um':>7} {'depth_mm':>9} {'frac_lam_g':>10} "
          f"{'unit_excess':>12} {'env_err':>9} {'gate_used':>9}")
    worst_frac, worst_err = 9.9, 0.0
    worst_unit = None            # None = this lane records no witness
    for c in env.get("cases", []):
        dx = c.get("dx_m") or recipe.get("base_dx_m")
        if dx is None:
            continue
        depth = cpml * float(dx)
        frac = depth / lam_g_low
        um, un = c.get("unitarity_max"), c.get("unitarity_min")
        # BOTH directions: an absorber that leaks makes column power exceed 1,
        # while a truncated record makes it fall short. Reporting only the
        # over-unity side understates it — on WR-28 the max side reads 5.7e-4
        # while the min side is 0.996848, a 3.2e-3 SHORTFALL, 5.6x larger.
        #
        # And MISSING is not zero. `(None or 1.0) - 1.0` is 0.0, so an earlier
        # revision printed "0.0e+00" — perfect unitarity — for the WR-90 NU lane,
        # whose builder never writes these keys at all. A skipped check rendered
        # as a passing one is the #303 failure; absent stays absent here.
        unit = None
        if um is not None or un is not None:
            unit = max(abs((um if um is not None else 1.0) - 1.0),
                       abs((un if un is not None else 1.0) - 1.0))
        err = float(c["max_mag_abs_diff"])
        used = err / tol if tol else float("nan")
        unit_txt = "  NOT MEASURED" if unit is None else f"{unit:>12.2e}"
        print(f"  {c['tag']:18} {float(dx) * 1e6:>7.0f} {depth * 1e3:>9.3f} "
              f"{frac:>10.3f} {unit_txt} {err:>9.5f} {used:>8.0%}")
        worst_frac = min(worst_frac, frac)
        if unit is not None:
            worst_unit = max(worst_unit or 0.0, unit)
        worst_err = max(worst_err, err)
    verdict = "OK" if worst_frac >= DISCIPLINE_FLOOR else (
        f"BELOW the {DISCIPLINE_FLOOR} lambda_g far-port discipline (#496)")
    wu = ("NOT MEASURED (this builder writes no unitarity keys — the lane has "
          "no passivity witness at all)" if worst_unit is None
          else f"{worst_unit:.2e}")
    print(f"  worst: {worst_frac:.3f} lambda_g, unitarity excess "
          f"{wu}, envelope error {worst_err:.5f} "
          f"({worst_err / tol if tol else float('nan'):.0%} of gate)  -> {verdict}")
    return {"path": str(path), "auditable": True, "worst_fraction": worst_frac,
            "worst_unitarity_excess": worst_unit, "worst_env_error": worst_err,
            "cpml_layers": int(cpml), "gate": tol}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envelope", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--all", action="store_true",
                    help="every committed broad-E5 envelope fixture")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.envelope]
    if args.all or not paths:
        paths = sorted(REPO.glob("tests/fixtures/waveguide_*broad_e5/"
                                 "*_broad_e5_envelope.json"))
    results = [audit(p, args.label if len(paths) == 1 else "") for p in paths]
    bad = [r for r in results
           if r.get("auditable") and r["worst_fraction"] < DISCIPLINE_FLOOR]
    print(f"\n{len(bad)} of {len(results)} audited lanes are below "
          f"{DISCIPLINE_FLOOR} lambda_g")
    for r in sorted(bad, key=lambda r: r["worst_fraction"]):
        print(f"  {Path(r['path']).name:52} {r['worst_fraction']:.3f} lambda_g, "
              f"unitarity {'NOT MEASURED' if r['worst_unitarity_excess'] is None else format(r['worst_unitarity_excess'], '.1e')}, "
              f"envelope {r['worst_env_error']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
