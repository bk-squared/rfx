"""Live cv17 defect probe across the blind-window FAIL island (#812 round 2).

The committed material-blind-window model
(``validation/crossval/_17_dielectric_results/material_blind_window.json``)
says the 6.3 dB gate PASSES a rasterized permittivity anywhere in
[2.0, 4.5] and [5.0, 5.6] but FAILS in an island at 4.6-4.9, where the
ka = 1.25 bin sits on a Mie resonance. Round 1's live defect runs sampled
1.8 / 2.0 / 5.5 / 6.0 only, so the island's edges are model-only. This
probe runs the real solver with the rasterizer delivering ``eps_r`` while
the oracle stays at the DECLARED 2.56 (the audit's defect, exactly as the
round-1 runs built it), on the four gated coarse bins, and records per-bin
``delta_db`` next to the model's prediction.

Report-only: no gate, window or fixture is touched. The realized-material
gate G17-A fires on every one of these builds by construction; what is
measured here is what the dB channel alone would have said.

Usage::

    python scripts/diagnostics/probe_cv17_permittivity_island.py \
        --eps 4.5 4.6 4.7 4.8 4.9 5.0 --output OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CV17 = _REPO_ROOT / "validation/crossval/17_dielectric_sphere_mie.py"
_MODEL = _REPO_ROOT / "validation/crossval/_17_dielectric_results/material_blind_window.json"


def _load_cv17():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_cv17_probe_module", _CV17)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eps", type=float, nargs="+", default=[4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    m = _load_cv17()
    model = json.loads(_MODEL.read_text(encoding="utf-8"))
    model_rows = {row["eps_r"]: row for row in model["scan"]}
    declared = float(m.EPS_R)
    gate = float(m.GATE_COARSE_DB)
    assert abs(m.M_IDX - np.sqrt(declared)) < 1e-12

    out = {
        "schema": "rfx.cv17_permittivity_island_probe",
        "issue": 812,
        "produced_by": "scripts/diagnostics/probe_cv17_permittivity_island.py",
        "declared_eps_r": declared,
        "oracle_eps_r": declared,
        "coarse_gate_db": gate,
        "ka_gated": list(m.KA_GATED_COARSE),
        "model_artifact": str(_MODEL.relative_to(_REPO_ROOT)),
        "runs": [],
    }
    for eps in args.eps:
        m.EPS_R = float(eps)          # what the rasterizer delivers
        # m.M_IDX untouched: the oracle stays at the declared 2.56
        rows, t0 = [], time.time()
        for ka in m.KA_GATED_COARSE:
            r = m.run_point(ka, m.COARSE_CPR, m.CLEAR_CELLS_DEFAULT)
            rows.append({"ka": ka, "delta_db": r["delta_db"],
                         "eps_realized": r["eps_realized"],
                         "n_distinct_eps": r["n_distinct_eps"]})
        worst = max(abs(x["delta_db"]) for x in rows)
        mrow = model_rows.get(round(float(eps), 2))
        rec = {
            "eps_r": float(eps),
            "per_bin_delta_db": [x["delta_db"] for x in rows],
            "eps_realized_per_bin": [x["eps_realized"] for x in rows],
            "max_abs_delta_db": round(float(worst), 3),
            "inside_db_gate": bool(worst <= gate),
            "model_max_abs_delta_db": (mrow["max_abs_delta_db"] if mrow else None),
            "model_inside_db_gate": (mrow["inside_db_gate"] if mrow else None),
            "model_per_bin_abs_delta_db": (mrow["per_bin_abs_delta_db"] if mrow else None),
            "verdict_agrees_with_model": (bool(worst <= gate) == mrow["inside_db_gate"]) if mrow else None,
            "wall_s": round(time.time() - t0, 1),
        }
        out["runs"].append(rec)
        print(f"eps {eps:5.2f}: live max|delta| {worst:6.3f} dB -> "
              f"{'PASS' if rec['inside_db_gate'] else 'FAIL'}; model "
              f"{rec['model_max_abs_delta_db']} -> "
              f"{'PASS' if rec['model_inside_db_gate'] else 'FAIL'}"
              f"{'' if mrow is None else ('  agree' if rec['verdict_agrees_with_model'] else '  DISAGREE')}",
              flush=True)
    m.EPS_R = declared
    out["summary"] = {
        "n_runs": len(out["runs"]),
        "n_verdicts_agree_with_model": sum(1 for r in out["runs"] if r["verdict_agrees_with_model"]),
        "live_fail_eps": [r["eps_r"] for r in out["runs"] if not r["inside_db_gate"]],
        "model_fail_eps": [r["eps_r"] for r in out["runs"] if r["model_inside_db_gate"] is False],
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
