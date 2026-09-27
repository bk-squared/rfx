"""The rfx curves behind three cross-validation comparisons, saved as arrays at the judged rung.

For the RT/Duroid 5880 probe-fed patch (|S11| and its input impedance), the
MSL open-stub notch filter and the Sheen stepped-impedance low-pass filter
(|S21|, |S11|), this script imports the case's own test module by path
(``tests/crossval/<case>/test_<case>.py``, read-only), calls its ``run_rung``
at the finest rung of its ``LADDER_M`` (the rung each test compares with the
openEMS record's ``stage_b_fine``), and saves the curves together with the
path and sha256 of every reference record the test reads.  It judges nothing:
the verdict at the same commit comes from the case's own pytest ladder, run in
its own job (``scripts/vessl_showcase_ladder_<case>.yaml``).

    python scripts/showcase/forward_curves.py --case rt5880_patch --out DIR
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))       # this checkout's rfx, not an installed one
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _record  # noqa: E402

CASES = {
    "rt5880_patch": {
        "module": "tests/crossval/rt5880_patch/test_rt5880_patch.py",
        "references": {"openems": "tests/crossval/rt5880_patch/reference/openems_patch.json"},
        "structure": "RT/Duroid 5880 probe-fed patch antenna, one 50 ohm wire port",
    },
    "msl_notch_filter": {
        "module": "tests/crossval/msl_notch_filter/test_msl_notch_filter.py",
        "references": {"openems": "tests/crossval/msl_notch_filter/reference/openems_tutorial.json",
                       "palace": "tests/crossval/msl_notch_filter/reference/palace_fem.json"},
        "structure": "microstrip open-stub notch filter, two MSL ports",
    },
    "sheen_lpf": {
        "module": "tests/crossval/sheen_lpf/test_sheen_lpf.py",
        "references": {"openems": "tests/crossval/sheen_lpf/reference/openems_sheen.json",
                       "palace": "tests/crossval/sheen_lpf/reference/palace_fem.json"},
        "structure": "Sheen stepped-impedance microstrip low-pass filter, two MSL ports",
    },
}
# The reference arrays copied beside the rfx curves, per record kind: which
# stages or meshes, and which fields.
_OPENEMS_STAGES = ("stage_b_coarse", "stage_b_mid", "stage_b_fine")
_PALACE_MESHES = ("coarse", "mid")


def load_case_module(case: str):
    name = f"_showcase_{case}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / CASES[case]["module"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_record._jsonable) + "\n")
    tmp.replace(path)


def reference_arrays(case: str) -> tuple[dict, dict]:
    """The reference curves as arrays, and where they came from."""
    arrays, meta = {}, {}
    for kind, rel in CASES[case]["references"].items():
        path = REPO / rel
        rec = json.loads(path.read_text())
        meta[kind] = {"path": rel, "sha256": _record.sha256_file(path)}
        names = _OPENEMS_STAGES if kind == "openems" else _PALACE_MESHES
        for st in names:
            if st not in rec:
                continue
            d = rec[st]
            arrays[f"{kind}_{st}_freqs_hz"] = np.asarray(d["freqs_ghz"], dtype=float) * 1e9
            for key in ("s11_mag", "s21_mag", "s11_deg", "s21_deg", "zin_re_ohm", "zin_im_ohm"):
                if key in d:
                    arrays[f"{kind}_{st}_{key}"] = np.asarray(d[key], dtype=float)
        meta[kind]["stages"] = [st for st in names if st in rec]
    return arrays, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-dir", type=Path, default=REPO)
    ap.add_argument("--dx", type=float, default=None, help="smoke runs only")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    source = _record.source_block(a.repo_dir, precision="float32")
    mod = load_case_module(a.case)
    dx = a.dx if a.dx is not None else float(mod.LADDER_M[-1])
    print(f"{a.case}: run_rung at dx = {dx*1e6:.3f} um (LADDER_M[-1] = "
          f"{mod.LADDER_M[-1]*1e6:.3f} um)", flush=True)

    ref_arrays, ref_meta = reference_arrays(a.case)
    np.savez(out / "reference_curves.npz", **ref_arrays)
    _save_json(out / "reference_records.json", ref_meta)

    t0 = time.perf_counter()
    r = mod.run_rung(dx)
    wall = time.perf_counter() - t0
    freqs = np.asarray(r["freqs_hz"], dtype=float)
    arrays = {"freqs_hz": freqs, "s11": np.asarray(r["s11"], dtype=complex)}
    if "s21" in r:
        arrays["s21"] = np.asarray(r["s21"], dtype=complex)
    if a.case == "rt5880_patch":
        arrays["zin_ohm"] = mod.zin_from_s11(arrays["s11"])
    if r.get("z0") is not None:
        arrays["z0_ohm"] = np.asarray(r["z0"])
    if r.get("sigma_max_excess") is not None:
        arrays["sigma_max_excess"] = np.asarray(r["sigma_max_excess"], dtype=float)
    np.savez(out / "rfx_curves.npz", **arrays)
    settling = np.atleast_1d(np.asarray(r["settling_db"], dtype=float))
    rung = {"case": a.case, "dx_m": dx, "ladder_m": [float(v) for v in mod.LADDER_M],
            "judged_rung_is_finest": dx == float(mod.LADDER_M[-1]),
            "num_periods": float(mod.NUM_PERIODS), "n_freqs": int(freqs.size),
            "n_cells": int(r["n_cells"]), "grid_shape": list(r["grid_shape"]),
            "wall_s": wall, "settling_db": settling.tolist(),
            "realized": {k: v for k, v in r["realized"].items()
                         if isinstance(v, (int, float, str, list, tuple, bool)) or v is None},
            "max_abs_s11": float(np.max(np.abs(arrays["s11"]))),
            "max_abs_s21": float(np.max(np.abs(arrays["s21"]))) if "s21" in arrays else None}
    _save_json(out / "rung.json", rung)

    claims = [
        _record.claim("worst ring-down settling of the judged rung's record(s)",
                      float(settling.max()), "dB", "rung.json"),
        _record.claim("max |S11| over the sweep", rung["max_abs_s11"], "1", "rung.json"),
        _record.claim("cells in the judged rung's grid", rung["n_cells"], "cells", "rung.json"),
        _record.claim("wall time of run_rung at the judged rung", wall, "s", "rung.json"),
    ]
    if rung["max_abs_s21"] is not None:
        claims.append(_record.claim("max |S21| over the sweep", rung["max_abs_s21"], "1",
                                    "rung.json"))
    files = _record.data_files(out)
    rec = {
        "schema": _record.SCHEMA, "id": f"forward-curves-{a.case}",
        "question": (f"The rfx curves of the {CASES[a.case]['structure']} at the rung its "
                     "cross-validation test judges, beside the reference records it reads."),
        "source": source,
        "run": {"platform": "VESSL", "preset": os.environ.get("RFX_SHOWCASE_PRESET"),
                "run_id": None, "wall_s": {"run_rung_s": wall}},
        "model": {"structure": CASES[a.case]["structure"], "module": CASES[a.case]["module"],
                  "builder": "the module's build(dx) through run_rung(dx)", "rung": rung,
                  "reference_records": ref_meta},
        "claims": claims, "derived": [],
        "out_of_scope": ["the verdict against the references: the case's pytest ladder job at "
                         "the same commit holds it",
                         "the coarser rungs of the ladder (in that job's log)"],
    }
    path = _record.write_result(out, rec, files)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
