#!/usr/bin/env python3
"""Dielectric-only control witness for the lattice ownership contract (#931).

The contract changes CONDUCTOR realization and nothing else: "Dielectric
sampling is untouched (node, half-open), so every dielectric-only fixture
stays bit-identical" (design note §1.1, §5). cv01, cv02 and cv03 are the
crossval cases that can falsify that sentence — a dielectric Box on a
2-D TMz lattice (cv01, cv03) and a pair of Cylinders (cv02), no conductor
anywhere. cv04 never touches the geometry layer at all and is checked by
re-running it and diffing its two committed JSON artifacts instead.

Those three cases need Meep for their reference leg, which is not
installed on every machine, and their rfx leg is minutes of FDTD. Neither
is necessary to answer the question: what the contract could change is
what the DECLARATION rasterizes to, and that is decided before the first
time step. So this driver runs each script with ``Simulation.run`` and
``Simulation.forward`` replaced by a stop hook that assembles the
materials, records a digest of every array plus the realized PEC edge
set, and raises to end the script.

Run it twice — once against the checkout under test, once against a
pre-#931 checkout — and compare the JSON. Equal digests are the
bit-identity evidence; a non-empty conductor census on a dielectric-only
case is a defect on its own, whichever side it appears on.

    RFX=<checkout> python scripts/diagnostics/cv0104_dielectric_control_witness.py \
        --out control_<label>.json

The digests cover the assembled material arrays as raw bytes, so they
compare bit patterns, not printed values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

CASES = ["01_waveguide_bend", "02_ring_resonator", "03_straight_waveguide_flux"]


def _digest(arr) -> dict:
    a = np.asarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "min": float(a.min()) if a.size else None,
        "max": float(a.max()) if a.size else None,
    }


class _Stop(Exception):
    """Raised after the first assembly to end the script before any solve."""


def _capture(sim) -> dict:
    """Assemble ``sim`` (no solve) and describe what it realizes."""
    sheets: list = []
    wires: list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nonuniform = any(getattr(sim, n, None) is not None for n in
                         ("_dz_profile", "_dx_profile", "_dy_profile"))
        # A pre-#931 checkout has no sheet/wire collectors at all. Fall back
        # to the bare call there rather than reporting an error: the point of
        # the comparison is the MATERIAL arrays, which both sides return in
        # the same positions.
        if nonuniform:
            grid = sim._build_nonuniform_grid()
            try:
                mats, _d, _l, pec = sim._assemble_materials_nu(
                    grid, pec_sheets=sheets, pec_wires=wires)
            except TypeError:
                mats, _d, _l, pec = sim._assemble_materials_nu(grid)
        else:
            grid = sim._build_grid()
            try:
                out = sim._assemble_materials(grid, pec_sheets=sheets,
                                              pec_wires=wires)
            except TypeError:
                out = sim._assemble_materials(grid)
            mats, pec = out[0], out[3]
    rec = {
        "grid_shape": [int(v) for v in grid.shape],
        "eps_r": _digest(mats.eps_r),
        "sigma": _digest(mats.sigma),
        "n_pec_sheets": len(sheets),
        "n_pec_wires": len(wires),
        "pec_cells": int(np.asarray(pec).sum()) if pec is not None else 0,
    }
    if pec is None and not sheets and not wires:
        # A conductor-free case: there is no edge set to realize, and
        # asking for one is an error by construction. Say so explicitly
        # rather than fabricating an all-False triple.
        rec["pec_edges"] = 0
        rec["conductor_free"] = True
    else:
        rec["conductor_free"] = False
        try:
            from rfx.boundaries.pec import realized_pec_edge_masks
        except ImportError:                    # pre-#931 checkout
            rec["pec_edges"] = None
        else:
            edges = realized_pec_edge_masks(
                pec, sheets=tuple(sheets), wires=tuple(wires),
                periodic=sim._periodic_flags())
            rec["pec_edges"] = int(sum(int(np.asarray(m).sum()) for m in edges))
    return rec


def run_case(case: str, repo: Path) -> dict:
    from rfx import Simulation

    captured: list[dict] = []
    originals = {}
    for name in ("run", "forward"):
        fn = getattr(Simulation, name, None)
        if fn is None:
            continue
        originals[name] = fn

        def _stub(self, *a, _n=name, **kw):
            captured.append({"call": _n, **_capture(self)})
            raise _Stop(_n)

        setattr(Simulation, name, _stub)

    script = repo / "validation" / "crossval" / f"{case}.py"
    argv, cwd = list(sys.argv), os.getcwd()
    err = None
    try:
        sys.argv = [str(script)]
        os.chdir(script.parent)
        runpy.run_path(str(script), run_name="__main__")
    except _Stop:
        pass
    except SystemExit as exc:
        err = f"SystemExit({exc.code})"
    except BaseException:                     # noqa: BLE001 - reported, not raised
        err = traceback.format_exc(limit=6)
    finally:
        sys.argv, _ = argv, os.chdir(cwd)
        for name, fn in originals.items():
            setattr(Simulation, name, fn)
    return {"case": case, "captured": captured, "error": err}


def _describe(repo: Path, *args: str) -> str:
    """git output for ``repo``, or ``"unknown"`` — never fatal to the witness."""
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.environ.get("RFX", ""))
    ap.add_argument("--cases", nargs="*", default=CASES)
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else \
        Path(__file__).resolve().parents[2]
    import rfx
    payload = {
        "repo": str(repo),
        # The comparison is only readable if each side says WHICH checkout it
        # is. Without this the filename is the only claim about provenance,
        # and a filename is not evidence.
        "repo_commit": _describe(repo, "rev-parse", "HEAD"),
        "repo_dirty": bool(_describe(repo, "status", "--porcelain")),
        "rfx_module": str(Path(rfx.__file__).resolve()),
        "rfx_version": getattr(rfx, "__version__", "?"),
        "cases": [run_case(c, repo) for c in args.cases],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"checkout: {repo} @ {payload['repo_commit'][:8]}"
          + ("  (DIRTY)" if payload["repo_dirty"] else ""))
    for c in payload["cases"]:
        n = len(c["captured"])
        print(f"{c['case']}: {n} assembled simulation(s)"
              + (f"  [after-stop: {c['error'].splitlines()[-1][:90]}]"
                 if c["error"] else ""))
        for rec in c["captured"]:
            print(f"    {rec['call']}: shape={rec['grid_shape']} "
                  f"eps={rec['eps_r']['sha256'][:16]} "
                  f"sigma={rec['sigma']['sha256'][:16]} "
                  f"pec_cells={rec['pec_cells']} sheets={rec['n_pec_sheets']} "
                  f"edges={rec['pec_edges']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
