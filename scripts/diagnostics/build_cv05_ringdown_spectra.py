"""Regenerate cv05's ring-down spectra fixture with provenance (#812 round 2).

``tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`` holds
the rfx harminv spectrum of the declared 29.5 mm patch and of four builds with
the resonant length mis-realized (22.5 / 22.0 / 21.0 / 38.0 mm) -- the
falsifier record for the mode-resolved selector. Round 1 produced it with an
uncommitted runpy harness; this script is that harness, committed, so the
fixture can be rebuilt from the repo.

For each length it executes ``validation/crossval/05_patch_antenna.py`` in a
fresh globals dict with ``RFX_CV05_PATCH_L_MM`` set, catches the ``SystemExit``
the script raises when openEMS is absent (PART 2), and reads the PART 1
ring-down results (``modes_good``) out of the script's globals. Two solves per
length are what the script itself does (probe + harminv); nothing is changed
in the physics.

Usage::

    python scripts/diagnostics/build_cv05_ringdown_spectra.py --output OUT.json
    python scripts/diagnostics/build_cv05_ringdown_spectra.py --check   # rebuild, compare at 1e-6 rel
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "validation/crossval/05_patch_antenna.py"
FIXTURE = REPO / "tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json"
LENGTHS_MM = {"baseline": 29.5, "patch_len_22p5mm": 22.5, "patch_len_22p0mm": 22.0,
              "patch_len_21p0mm": 21.0, "patch_len_38p0mm": 38.0}

# Declared patch lengths for the geometry-only census (no FDTD). Same list the
# committed fixture carried; it spans the design length and the neighbourhood of
# the 21-23 mm rows where one lattice cell decides which cavity mode the
# selector is allowed to find.
CENSUS_LENGTHS_MM = [29.5, 23.0, 22.5, 22.0, 21.65, 21.5, 21.0, 20.5, 38.0]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _exec_script(env_extra: dict) -> dict:
    """Run 05_patch_antenna.py in a fresh globals dict and hand them back.

    ``env_extra`` is applied on top of the environment and removed again, and
    the case's other hooks are cleared first: a stray ``RFX_CV05_BUILD_ONLY``
    or ``RFX_CV05_SHEET_PLANE_DELTA`` in the shell would silently produce a
    fixture of a different board.
    """
    for key in ("RFX_CROSSVAL05_JSON", "RFX_CV05_BUILD_ONLY",
                "RFX_CV05_REALIZED_JSON", "RFX_CV05_SHEET_PLANE_DELTA",
                "RFX_CV05_PATCH_L_MM"):
        os.environ.pop(key, None)
    os.environ.update(env_extra)
    src = SCRIPT.read_text(encoding="utf-8")
    code = compile(src, str(SCRIPT), "exec")
    g = {"__name__": "__main__", "__file__": str(SCRIPT), "__builtins__": __builtins__}
    sys.argv = [str(SCRIPT)]
    g["__exit_code__"] = 0
    try:
        exec(code, g)  # noqa: S102 -- the case script, run as itself
    except SystemExit as e:
        g["__exit_code__"] = int(e.code or 0)
    finally:
        for key in env_extra:
            os.environ.pop(key, None)
    return g


def census(lengths_mm=None) -> dict:
    """Declared patch length -> realized conductor extent, geometry only.

    Measured from the REALIZED PEC edge set (the script's own
    ``assert_realized_sheets``, which reads ``realized_pec_edge_masks``), not
    from a rasterization rule re-derived here. Since #931 the patch is a sheet
    and its realized length is the number of tangential E edges its footprint
    stands on: 29 masked nodes carry 28 edges, and 28 mm of metal is what sets
    the resonance. The pre-#931 census counted the NODES (29), which is why
    its rows are one larger throughout. Re-measured, never carried forward.
    """
    lengths = list(CENSUS_LENGTHS_MM if lengths_mm is None else lengths_mm)
    out = {
        "_what": "patch x-extent measured from the REALIZED PEC edge set on "
                 "cv05's own grid (dx = 1 mm), no FDTD: declared length (mm) -> "
                 "realized conductor length in edges. Since #931 the patch is a "
                 "sheet and this counts tangential E edges, not masked nodes; "
                 "the pre-#931 census counted nodes and every row read one "
                 "higher. Establishes which design-mode errors are reachable.",
        "_unit": "edges (= mm at dx = 1 mm)",
    }
    for lm in lengths:
        print(f"== census: L = {lm} mm ==", flush=True)
        g = _exec_script({"RFX_CV05_PATCH_L_MM": f"{lm:.4f}",
                          "RFX_CV05_BUILD_ONLY": "1"})
        st = g["REALIZED"]
        out[f"{lm}"] = int(st["patch"]["x_edge_cells"])
    return out


def run_one(length_mm: float) -> dict:
    g = _exec_script({"RFX_CV05_PATCH_L_MM": f"{length_mm:.4f}"})
    exit_code = g["__exit_code__"]
    modes = g["modes_good"]
    st = g["REALIZED"]
    return {
        "realized_patch_len_mm": float(g["L"] * 1e3),
        "declared_patch_len_mm": 29.5,
        "script_exit": exit_code,
        # The realized stack of THIS build, from realized_pec_edge_masks. A
        # fixture whose provenance is a resonance and nothing else cannot say
        # which cavity produced it; that is how the +21.1 % error survived.
        "realized_stack": st,
        "harminv_band_hz": [float(g["HARMINV_F_LO"]), float(g["HARMINV_F_HI"])]
        if "HARMINV_F_LO" in g else None,
        "modes": [{"freq": float(m.freq), "Q": float(m.Q), "amplitude": float(abs(m.amplitude))}
                  for m in sorted(modes, key=lambda m: m.freq)],
    }


def build(names: list[str]) -> dict:
    committed = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else {}
    out = {k: v for k, v in committed.items() if k.startswith("_") and k != "_provenance"}
    out["_provenance"] = {
        "repo_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain")),
        "recorded_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harness": "scripts/diagnostics/build_cv05_ringdown_spectra.py: exec of "
                   "validation/crossval/05_patch_antenna.py per length with "
                   "RFX_CV05_PATCH_L_MM set, num_periods=60 (the script's own value), CPU",
    }
    out["runs"] = {}
    for name in names:
        print(f"== {name}: L = {LENGTHS_MM[name]} mm ==", flush=True)
        out["runs"][name] = run_one(LENGTHS_MM[name])
    # The census is RE-MEASURED, never copied from the committed fixture. It
    # is a record of what the lattice realizes, and #931 changed that; carrying
    # the old rows forward would put a pre-contract measurement inside a
    # post-contract artifact.
    out["_realized_x_cell_census"] = census()
    return out


def _compare(fresh: dict, committed: dict, rel: float) -> list[str]:
    bad = []
    for name, run in fresh["runs"].items():
        c = committed.get("runs", {}).get(name)
        if c is None:
            bad.append(f"runs.{name}: missing in committed"); continue
        if len(run["modes"]) != len(c["modes"]):
            bad.append(f"runs.{name}: {len(run['modes'])} modes vs committed {len(c['modes'])}"); continue
        for i, (a, b) in enumerate(zip(run["modes"], c["modes"])):
            for key in ("freq", "Q"):
                if not math.isclose(a[key], b[key], rel_tol=rel):
                    bad.append(f"runs.{name}.modes[{i}].{key}: {a[key]} vs {b[key]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(FIXTURE))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rel", type=float, default=1e-6)
    ap.add_argument("--only", nargs="*", default=None, help="subset of run names")
    ap.add_argument("--census-only", action="store_true",
                    help="geometry-only census (no FDTD); prints and writes just "
                         "the realized-extent table")
    args = ap.parse_args(argv)
    if args.census_only:
        table = census()
        print(json.dumps(table, indent=1))
        if args.output != str(FIXTURE):
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(table, indent=1) + "\n",
                                         encoding="utf-8")
            print(f"wrote {args.output}")
        return 0
    names = args.only or list(LENGTHS_MM)
    fresh = build(names)
    if args.check:
        bad = _compare(fresh, json.loads(FIXTURE.read_text(encoding="utf-8")), args.rel)
        print("OK: committed cv05_ringdown_spectra.json reproduces" if not bad
              else "MISMATCH: " + "; ".join(bad[:10]))
        return 1 if bad else 0
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fresh, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
