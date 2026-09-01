"""cv18 aperture-resolution witness — the numbers behind the #812 re-gate prose.

WHY THIS FILE EXISTS (issue #812, round 2).  The APERTURE RESOLUTION paragraph
of cv18's ``claim_scope`` asserts quantities that lived only in prose, and one
of them shipped SIGN-INVERTED and mis-sourced: it said the committed fine trace
sits closer to the oracle one fine cell NARROW than to the oracle at the
declared d, quoting the one-cell UNDER-aperture defect metric as if it were
that distance.  Both halves are re-derived here and emitted as JSON, so the
prose can reference an artifact key instead of restating a digit.

NO FDTD.  This builder reads the committed cv18 record
(``validation/crossval/_18_wr90_iris_results/rfx.json``) and evaluates the
same mode-matching oracle the case gates against.  It is deterministic and
runs in seconds; ``tests/test_wr90_iris_modematch_gates.py`` re-derives every
emitted number from an INDEPENDENT re-implementation of that oracle, so the
artifact is mechanically checked rather than trusted.

Two distinct quantities are emitted, and conflating them is the defect this
file closes:

  * ``oracle_distance_abs[g]`` = max_f | committed fine |S11| - oracle(d + g*dx_fine) |.
    How far the trace as measured sits from the oracle at a SHIFTED aperture.
    Its argmin over the declared offset grid is the fine rung's own effective
    aperture, i.e. a property of the rasterization, not a defect.
  * ``one_cell_defect.{over,under}.fine_gap_abs`` = max_f | (fine + [oracle(d +- dx)
    - oracle(d)]) - oracle(d) |.  The audit's DEFECT model: geometry moved by
    one cell at each rung, record and oracle still nominal.  This is what the
    gate would see.

The two differ by the sign with which the oracle shift enters, which is exactly
how the round-1 claim inverted.

Usage:
  python scripts/diagnostics/build_cv18_aperture_resolution.py [--check]

``--check`` regenerates in memory and diffs against the committed artifact
(exit 1 on any difference) without writing.
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
_CV18 = _REPO_ROOT / "validation/crossval/18_wr90_iris_modematch.py"
_SOURCE = _REPO_ROOT / "validation/crossval/_18_wr90_iris_results/rfx.json"
_OUT = _REPO_ROOT / "validation/crossval/_18_wr90_iris_results/aperture_resolution.json"

# Declared BEFORE the measurement (issue #812 pre-declaration note section 2.4):
# the effective-aperture probe is a fixed grid of aperture offsets in fine
# cells, not a free optimisation, so the emitted argmin is reproducible
# bit-for-bit and cannot be tuned after the fact.
OFFSET_GRID_FINE_CELLS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _load_cv18():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_cv18_module", _CV18)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build() -> dict:
    m = _load_cv18()
    rec = json.loads(_SOURCE.read_text(encoding="utf-8"))
    a = m.A_WR90
    dx_f = a / m.FINE_CELLS
    dx_c = a / m.COARSE_CELLS
    freqs = rec["config"]["freqs_hz"]
    rich_gate = rec["gates"]["richardson_gate_abs"]

    cache: dict[float, np.ndarray] = {}

    def oracle(d_phys: float) -> np.ndarray:
        key = float(d_phys)
        if key not in cache:
            cache[key] = np.array(
                [abs(m.iris_smatrix(a, key, m.T_IRIS, f)[0]) for f in freqs])
        return cache[key]

    pairs = []
    for fr in rec["gated_fine"]:
        # resolve the DECLARED aperture object (18.288 * 1e-3 is not the same
        # float as 18.288e-3), the same rule the crossval script now uses
        d = next(x for x in m.D_APERTURES if abs(x * 1e3 - fr["d_mm"]) < 1e-9)
        key = m.config_key(fr["d_mm"], fr["glen_m"], fr["iris_frac"])
        gate = m.GATE_FINE_ABS_PER_CONFIG[key]
        cr = next(c for c in rec["coarse_diagnostic"]
                  if (c["d_mm"], c["glen_m"], c["iris_frac"])
                  == (fr["d_mm"], fr["glen_m"], fr["iris_frac"]))
        s_f = np.asarray(fr["s11"], dtype=float)
        s_c = np.asarray(cr["s11"], dtype=float)
        base = oracle(d)

        dist = {f"{g:+.1f}": round(float(np.max(np.abs(s_f - oracle(d + g * dx_f)))), 4)
                for g in OFFSET_GRID_FINE_CELLS}
        nearest = min(OFFSET_GRID_FINE_CELLS,
                      key=lambda g: dist[f"{g:+.1f}"])

        defect = {}
        for sgn, name in ((+1, "over"), (-1, "under")):
            f_def = s_f + (oracle(d + sgn * dx_f) - base)
            c_def = s_c + (oracle(d + sgn * dx_c) - base)
            gap = float(np.max(np.abs(f_def - base)))
            rich = float(np.max(np.abs(2 * f_def - c_def - base)))
            defect[name] = {
                "fine_gap_abs": round(gap, 4),
                "detected_by_fine_gate": bool(gap > gate),
                "fine_margin_x": round(gap / gate, 3),
                "scores_better_than_undefected": bool(gap < fr["max_gap_abs"]),
                "richardson_dev_abs": round(rich, 4),
                "detected_by_richardson_gate": bool(rich > rich_gate),
            }

        pairs.append({
            "config": key,
            "d_mm": fr["d_mm"], "glen_m": fr["glen_m"], "iris_frac": fr["iris_frac"],
            "fine_gate_abs": gate,
            "committed_fine_gap_abs": fr["max_gap_abs"],
            "oracle_distance_abs": dist,
            "nearest_offset_fine_cells": nearest,
            "one_cell_defect": defect,
        })

    over = [p["one_cell_defect"]["over"] for p in pairs]
    under = [p["one_cell_defect"]["under"] for p in pairs]
    nearest_set = sorted({p["nearest_offset_fine_cells"] for p in pairs})
    summary = {
        "n_pairs": len(pairs),
        "nearest_offset_fine_cells_values": nearest_set,
        "nearest_offset_is_positive_at_all_pairs": bool(
            all(p["nearest_offset_fine_cells"] > 0 for p in pairs)),
        "over_aperture_detected": sum(d["detected_by_fine_gate"] for d in over),
        "over_aperture_min_margin_x": round(
            min(d["fine_margin_x"] for d in over), 3),
        "under_aperture_detected": sum(d["detected_by_fine_gate"] for d in under),
        "under_aperture_max_margin_x": round(
            max(d["fine_margin_x"] for d in under), 3),
        "under_aperture_detected_configs": [
            p["config"] for p in pairs
            if p["one_cell_defect"]["under"]["detected_by_fine_gate"]],
        "under_aperture_scores_better_configs": [
            p["config"] for p in pairs
            if p["one_cell_defect"]["under"]["scores_better_than_undefected"]],
        "richardson_detected_either_sign": sum(
            d["detected_by_richardson_gate"] for d in over + under),
    }

    return {
        "schema": "rfx.wr90_iris_aperture_resolution",
        "schema_version": 1,
        "issue": 812,
        "generated_by": "scripts/diagnostics/build_cv18_aperture_resolution.py",
        "reads": ["validation/crossval/_18_wr90_iris_results/rfx.json"],
        "runs_fdtd": False,
        "method": (
            "oracle_distance_abs[g] = max_f |committed fine |S11| - "
            "oracle(d + g*dx_fine)| over the DECLARED offset grid "
            "(a property of the fine rung's rasterization); "
            "one_cell_defect.{over,under}.fine_gap_abs = max_f |(fine + "
            "[oracle(d +- dx_rung) - oracle(d)]) - oracle(d)|, the audit's "
            "one-cell-at-each-rung geometry defect against the NOMINAL "
            "oracle (what the gate would see). The two are NOT the same "
            "quantity; conflating them inverted the round-1 prose claim."),
        "config": {
            "a_m": m.A_WR90, "t_m": m.T_IRIS,
            "dx_fine_mm": round(dx_f * 1e3, 4),
            "dx_coarse_mm": round(dx_c * 1e3, 4),
            "n_freqs": len(freqs),
            "richardson_gate_abs": rich_gate,
            "pooled_fine_gate_abs": rec["gates"]["fine_gate_abs"],
        },
        "offset_grid_fine_cells": list(OFFSET_GRID_FINE_CELLS),
        "pairs": pairs,
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diff against the committed artifact, write nothing")
    args = ap.parse_args()
    fresh = build()
    if args.check:
        committed = json.loads(_OUT.read_text(encoding="utf-8"))
        if committed != fresh:
            print("MISMATCH: committed aperture_resolution.json != rebuild")
            return 1
        print("OK: committed aperture_resolution.json reproduces")
        return 0
    _OUT.write_text(json.dumps(fresh, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_REPO_ROOT)}")
    s = fresh["summary"]
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
