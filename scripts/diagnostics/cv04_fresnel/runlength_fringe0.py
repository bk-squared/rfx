#!/usr/bin/env python3
"""Section 14 falsifier: is fringe 0's displacement a truncation artefact?

Section 12 of ``docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md``
hypothesises that cv04's fringe *maxima* are displaced by truncating the run at
719 steps while the tail witness still reads 3.6 % of incident peak, and
predicts the residual shrinks with run length.  Section 14 makes that
load-bearing: at fringe 0 correct code consumes 101.6 % of the window's
*derived* budget (``fringe_gate_geometry.json::windows.fringes[0].position_derived_budget_hz``)
and passes only on ``windows.safety``.

``n_steps`` is not an independent knob -- the case derives it from the CPML
round-trip safe time, so it follows ``nx_interior``.  Raising 600 -> 1500 gives
the committed rung-C4 length (~1940 steps).

Declared BEFORE the run (section 14):
  PASS  |dev(fringe 0)| <= 29.52 MHz -> the truncation-artefact hypothesis is
        confirmed and criterion (A) is physically underwritten.
  FAIL  |dev| stays ~30 MHz or grows -> the window's derivation is missing a
        term.  That term must then be derived.  The window is NOT widened to
        admit the observation.

Runs the unmodified case source with one exact substitution, so it cannot
drift from what the crossval script actually does.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CASE = os.path.join(REPO, "validation", "crossval", "04_multilayer_fresnel.py")

_ANCHOR = "# Exit code (rfx crossval convention)"

_DUMP = '''
# --- appended by scripts/diagnostics/cv04_fresnel/runlength_fringe0.py ---
import json as _json                                            # noqa: E402
with open(os.environ["CV04_RUNLEN_JSON"], "w") as _fh:
    _json.dump({
        "nx_interior": int(nx_interior),
        "n_steps": int(n_steps),
        "fringe_ok": bool(fringe_ok),
        "rows": [
            {"kind": _r.kind,
             "f_ref_hz": float(_r.f_ref_hz),
             "f_meas_hz": float(_r.f_meas_hz),
             "df_hz": float(_r.df_hz),
             "f_window_hz": float(_r.f_window_hz),
             "value_ref": float(_r.value_ref),
             "value_meas": float(_r.value_meas),
             "dvalue": float(_r.dvalue)}
            for _r in fringe_verdict.rows
        ],
        "reasons": list(fringe_verdict.reasons),
    }, _fh, indent=2)
# --- end appended block ---
'''

# key, why it exists, [(exact old text, exact new text), ...]
CONFIGS = [
    ("nx600_committed_719_steps",
     "the committed recipe, reproduced here so both arms come from one driver",
     []),
    ("nx1500_rungC4_1940_steps",
     "rung C4: n_steps follows nx_interior through the CPML safe time",
     [("nx_interior = 600", "nx_interior = 1500")]),
]


def run_one(edits, workdir, json_path):
    src = open(CASE).read()
    for old, new in edits:
        assert src.count(old) == 1, f"pattern is not unique: {old!r}"
        src = src.replace(old, new)
    assert src.count(_ANCHOR) == 1, "exit-code anchor not found in the case"
    src = src.replace(_ANCHOR, _DUMP + "\n" + _ANCHOR)
    path = os.path.join(workdir, "case.py")
    with open(path, "w") as fh:
        fh.write(src)
    env = dict(os.environ, PYTHONPATH=REPO, MPLBACKEND="Agg",
               CV04_RUNLEN_JSON=json_path)
    proc = subprocess.run([sys.executable, path], capture_output=True,
                          text=True, env=env, cwd=workdir)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    geom = json.load(open(os.path.join(
        REPO, "validation", "crossval", "_04_fresnel_results",
        "fringe_gate_geometry.json")))
    budget = geom["windows"]["fringes"][0]["position_derived_budget_hz"]

    out = {"schema": "issue812-cv04-runlength-fringe0-v1", "issue": 812,
           "runs_fdtd": True,
           "declared_pass_if": "abs(dev(fringe 0)) <= "
                               "windows.fringes[0].position_derived_budget_hz",
           "derived_budget_hz": budget,
           "configs": {}}

    for key, why, edits in CONFIGS:
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(
                os.path.join(REPO, "validation", "crossval", "comparators"),
                os.path.join(td, "comparators"))
            jp = os.path.join(td, "dump.json")
            rc, log = run_one(edits, td, jp)
            payload = json.load(open(jp)) if os.path.exists(jp) else None
            rec = {"why": why, "edits": edits, "exit_code": rc,
                   "payload": payload, "log_tail": log[-4000:]}
            if payload is not None and payload["rows"]:
                dev = payload["rows"][0]["df_hz"]
                rec["fringe0_dev_hz"] = dev
                rec["fringe0_abs_dev_over_derived_budget"] = abs(dev) / budget
                rec["fringe0_verdict"] = (
                    "PASS (artefact hypothesis confirmed)"
                    if abs(dev) <= budget else
                    "FAIL (derivation is missing a term)")
            out["configs"][key] = rec
            print(f"[{key}] exit={rc} "
                  f"dev0={rec.get('fringe0_dev_hz')} "
                  f"-> {rec.get('fringe0_verdict')}", flush=True)

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
