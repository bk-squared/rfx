"""Print cv15's before/after in the four fields the manifest clause needs.

Issue #931 (lattice ownership contract), case X-C. cv15's rfx leg is re-solved
with both conductors declared as SHEETS and the feed spanning the full
substrate; this reads the pre-#931 leg and the regenerated one side by side and
computes exactly the quantities ``compare()`` gates on, so the manifest
claim_scope text is filled from a measurement rather than translated from the
old digits.

    python scripts/diagnostics/cv15_before_after_931.py [--before PATH] [--after PATH]

Defaults: before = ``_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json``
(the #768 leg, preserved verbatim), after = ``_15_patch_results/rfx.json``.
The openEMS leg is unchanged and is read from ``_15_patch_results/openems.json``.

No solve. Reads committed JSON only.
"""
from __future__ import annotations

import argparse
import json
import os

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "validation", "crossval", "_15_patch_results")


def _row(name: str, leg: dict, f_oe: float) -> str:
    f = leg["f_primary_hz"]
    fa = leg["f_analytic_hz"]
    d_oe = (f - f_oe) / f_oe * 100.0
    d_an = (f - fa) / fa * 100.0
    sc = leg.get("stack_check") or {}
    return (f"{name}\n"
            f"  f_primary        {f/1e9:.6f} GHz\n"
            f"  vs openEMS       {abs(d_oe):.2f}% {'LOW' if d_oe < 0 else 'HIGH'}\n"
            f"  vs analytic      {d_an:+.2f}%\n"
            f"  Q (harminv)      {leg.get('q_harminv')}\n"
            f"  gain_dbi         {leg.get('gain_dbi')}\n"
            f"  settle_db        {leg.get('settle_db')}\n"
            f"  runtime_s        {leg.get('runtime_s')}\n"
            f"  ground/patch     {sc.get('ground_realization')} / "
            f"{sc.get('patch_realization')}\n"
            f"  n_distinct_eps   {sc.get('n_distinct_eps')}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", default=os.path.join(
        RES, "rfx_pre931_two_plane_ground_1f005d0d.json"))
    ap.add_argument("--after", default=os.path.join(RES, "rfx.json"))
    ap.add_argument("--openems", default=os.path.join(RES, "openems.json"))
    a = ap.parse_args()

    oe = json.load(open(a.openems, encoding="utf-8"))
    f_oe = oe["f_dip_hz"]
    before = json.load(open(a.before, encoding="utf-8"))
    after = json.load(open(a.after, encoding="utf-8"))
    print(_row("BEFORE  (#768: two_plane ground, feed one cell short of the "
               "patch)", before, f_oe))
    print(_row("AFTER   (#931: both conductors declared sheets, full-span "
               "feed)", after, f_oe))
    print(f"openEMS f_dip      {f_oe/1e9:.6f} GHz   (leg unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
