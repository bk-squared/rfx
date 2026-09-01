"""cv17 material blind window — the numbers behind the #812 material re-gate.

WHY THIS FILE EXISTS (issue #812, round 2).  cv17's ``claim_scope`` states how
wide a permittivity error the 6.3 dB RCS gate tolerates, and round 1 wrote
that window as four FDTD-measured digits that no committed artifact carried.
This builder re-derives the same window from the committed record plus the
Mie oracle, with NO FDTD, so the prose can reference an artifact key.

Model (the same first-order class cv18 uses for its aperture defect): the
solver's own discretization error is held fixed and only the material moves,

    delta_db(eps) = committed delta_db  +  [ mie_db(eps) - mie_db(2.56) ]

with the ORACLE left at the declared 2.56, which is what the gate compares
against.  A permittivity is "inside the blind window" when max over the four
gated coarse ka bins of |delta_db(eps)| stays within the 6.3 dB gate.

The model is corroborated, not replaced, by the live defect runs recorded in
docs/design_notes/issue812_cv17_cv18_geometry_sensitivity_predeclaration.md
section 5.1: it reproduces the live PASS/FAIL verdict at all four probed
permittivities.  Those live magnitudes remain prose-only and are deliberately
NOT copied into this artifact.

Usage:
  python scripts/diagnostics/build_cv17_material_blind_window.py [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CV17 = _REPO_ROOT / "validation/crossval/17_dielectric_sphere_mie.py"
_SOURCE = _REPO_ROOT / "validation/crossval/_17_dielectric_results/rfx.json"
_OUT = _REPO_ROOT / "validation/crossval/_17_dielectric_results/material_blind_window.json"

# Declared BEFORE the measurement: a fixed permittivity grid, not a solved-for
# edge, so the emitted bracket is reproducible bit-for-bit and cannot be tuned.
EPS_GRID = (1.6, 1.8, 2.0, 2.2, 2.56, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5)
DECLARED_EPS = 2.56


def _load_cv17():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_cv17_module", _CV17)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build() -> dict:
    m = _load_cv17()
    rec = json.loads(_SOURCE.read_text(encoding="utf-8"))
    rows = rec["gated_coarse"]
    gate = rec["gates"]["coarse_gate_db"]

    def mie_db(eps, ka):
        return 10 * np.log10(
            m.mie_backscatter_over_pi_a2(float(np.sqrt(eps)), ka))

    scan = []
    for eps in EPS_GRID:
        per_bin = []
        for r in rows:
            shift = mie_db(eps, r["ka"]) - mie_db(DECLARED_EPS, r["ka"])
            per_bin.append(round(float(abs(r["delta_db"] + shift)), 3))
        worst = max(per_bin)
        scan.append({
            "eps_r": eps,
            "eps_rel_dev": round(abs(eps / DECLARED_EPS - 1.0), 6),
            "max_abs_delta_db": worst,
            "inside_db_gate": bool(worst <= gate),
            "per_bin_abs_delta_db": per_bin,
        })

    inside = [s["eps_r"] for s in scan if s["inside_db_gate"]]
    # widest CONTIGUOUS run on the declared grid that contains the declared eps
    i0 = EPS_GRID.index(DECLARED_EPS)
    lo = hi = i0
    while lo - 1 >= 0 and scan[lo - 1]["inside_db_gate"]:
        lo -= 1
    while hi + 1 < len(EPS_GRID) and scan[hi + 1]["inside_db_gate"]:
        hi += 1

    return {
        "schema": "rfx.rcs_mie_material_blind_window",
        "schema_version": 1,
        "issue": 812,
        "generated_by": "scripts/diagnostics/build_cv17_material_blind_window.py",
        "reads": ["validation/crossval/_17_dielectric_results/rfx.json"],
        "runs_fdtd": False,
        "method": (
            "delta_db(eps) = committed gated_coarse delta_db + [mie_db(eps) - "
            "mie_db(2.56)] at the same ka, oracle held at the DECLARED 2.56; "
            "a permittivity is inside the blind window when the worst of the "
            "four gated coarse bins stays within coarse_gate_db. First-order "
            "model: the solver's discretization error is held fixed and only "
            "the material moves. It reproduces the live defect runs' PASS/FAIL "
            "verdict at every probed permittivity (design note 5.1); those "
            "live magnitudes are prose-only and are not copied here."),
        "config": {
            "declared_eps_r": DECLARED_EPS,
            "coarse_ka": [r["ka"] for r in rows],
            "coarse_gate_db": gate,
            "eps_realized_tol": rec["gates"]["eps_realized_tol"],
        },
        "eps_grid": list(EPS_GRID),
        "scan": scan,
        "summary": {
            "blind_window_eps_grid_values": inside,
            "blind_window_bracket_eps": [EPS_GRID[lo], EPS_GRID[hi]],
            "first_failing_eps_below": (EPS_GRID[lo - 1] if lo > 0 else None),
            "first_failing_eps_above": (EPS_GRID[hi + 1]
                                        if hi + 1 < len(EPS_GRID) else None),
            "blind_window_max_rel_dev": round(
                max(abs(e / DECLARED_EPS - 1.0) for e in inside), 4),
            "material_gate_rel_tol": rec["gates"]["eps_realized_tol"],
            "blind_window_over_material_gate_x": round(
                max(abs(e / DECLARED_EPS - 1.0) for e in inside)
                / rec["gates"]["eps_realized_tol"], 1),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fresh = build()
    if args.check:
        if json.loads(_OUT.read_text(encoding="utf-8")) != fresh:
            print("MISMATCH: committed material_blind_window.json != rebuild")
            return 1
        print("OK: committed material_blind_window.json reproduces")
        return 0
    _OUT.write_text(json.dumps(fresh, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_REPO_ROOT)}")
    print(json.dumps(fresh["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
