#!/usr/bin/env python3
"""Re-solve the #752 floor sweep and score it on the board each point realizes.

Issue #752 re-verification. This is the runner that produced
``msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep_realized_anchor_2026-09-19.json``.
It is committed so that artifact's citation can be followed; the first version
of that measurement was made by a script that lived only in a scratch
directory, which is the same defect the guide-citation gate in
``tests/contracts/test_guide_citations_resolve.py`` exists to stop.

WHAT IT DOES, and why it is not the sibling scripts
----------------------------------------------------
``msl_z0_bias_floor_sweep.py`` is inspection-only: its ``main()`` refuses to
regenerate, because its JSON is a PRE-DECLARED experiment record and
overwriting it would destroy the thing that makes it auditable.
``msl_z0_bias_floor_sweep_realized_anchor.py`` re-scores those frozen rows
without solving. Neither can answer "what does the extractor read TODAY", so
this one imports ``run_one`` from the first, solves the six points fresh, and
writes a NEW artifact with its own provenance. It never touches either frozen
file, and it verifies their sha256 before and after.

The anchor is Hammerstad-Jensen on the board THAT SAME BUILD realized:
``sim.fidelity_report(print_report=False)`` on the identical geometry, the
dielectric's realized z-extent as ``h`` and the trace's realized y-extent as
``W``. Both numbers come from one build, which is what the frozen pair could
no longer offer once #802/#834 and then #931 moved the realized board.

READ THE ARTIFACT'S CAVEATS BEFORE QUOTING A DEVIATION. Two known contributors
are NOT modelled here: the conductor is a one-cell PEC volume (#931 realizes it
with walls on both faces; the same trace declared zero-thickness reads 4.27
percentage points differently at aligned h_sub/3), and
``hammerstad_jensen_z0_eps_eff`` is a simplified quasi-static formula rather
than the complete Hammerstad-Jensen 1980 model
(``docs/guides/msl_geometry_diagnostics.md``). Neither has an error budget
here.

COST, and where to run it
-------------------------
Six FDTD points, measured 2026-09-19 on a CPU pod: 170 / 366 / 600 / 1181 /
182 / 413 s, about 48 min in total. Under the 2026-09-20 rule anything over
~10 minutes of CPU belongs on VESSL (remilab-c0) rather than a shared pod:
package this file with a CPU-preset yaml in the pattern of
``scripts/vessl_gpu_suite.yaml``, persist the provider log and the artifact
before ``vessl run delete``, never delete a non-terminal run, and quote the run
id as the witness. The 2026-09-19 artifact predates that rule and was produced
on the pod; its provenance block says so.

Run
---
    python scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor_rerun.py
        # inspection only: prints the committed artifact's provenance and
        # acceptance block, solves nothing.

    python scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor_rerun.py \\
        --run --out <path>
        # six FDTD solves; refuses to overwrite an existing artifact.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
FROZEN = OUT_DIR / "msl_z0_bias_floor_sweep.json"
FROZEN_ANCHOR = OUT_DIR / "msl_z0_bias_floor_sweep_realized_anchor.json"
DEFAULT_OUT = OUT_DIR / "msl_z0_bias_floor_sweep_realized_anchor_2026-09-19.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sweep_module():
    spec = importlib.util.spec_from_file_location(
        "_msl_floor_sweep", OUT_DIR.parent / "msl_z0_bias_floor_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(sw, dx: float):
    """The sweep's own ``run_one`` geometry, rebuilt for a metadata read."""
    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    lx = sw.L_LINE + 2 * sw.PORT_MARGIN
    ly = sw.W_TRACE + 2 * (2 * sw.H_SUB + 8 * dx)
    lz = sw.H_SUB + 1.5e-3
    sim = Simulation(freq_max=sw.F_MAX, domain=(lx, ly, lz), dx=dx,
                     cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro4350b", eps_r=sw.EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, sw.H_SUB)), material="ro4350b")
    yc = ly / 2.0
    sim.add(Box((0.0, yc - sw.W_TRACE / 2.0, sw.H_SUB),
                (lx, yc + sw.W_TRACE / 2.0, sw.H_SUB + dx)), material="pec")
    for x0, direction in ((sw.PORT_MARGIN, "+x"),
                          (sw.PORT_MARGIN + sw.L_LINE, "-x")):
        sim.add_msl_port(position=(x0, yc, 0.0), width=sw.W_TRACE,
                         height=sw.H_SUB, direction=direction, impedance=50.0)
    return sim


def _realized(sim) -> tuple[float, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        report = sim.fidelity_report(print_report=False)
    sub = [g for g in report if "ro4350b" in str(g.get("entity"))][0]
    pec = [g for g in report if "'pec'" in str(g.get("entity"))][0]

    def axis(record, name):
        return [a for a in record["axes"] if a["axis"] == name][0]

    return (axis(sub, "z")["realized_extent_um"],
            axis(pec, "y")["realized_extent_um"])


def resolve(out_path: Path) -> dict:
    """Six FDTD solves. Returns the artifact dict; does not write it."""
    import jax
    import rfx
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    sw = _sweep_module()
    before = {p.name: _sha256(p) for p in (FROZEN, FROZEN_ANCHOR)}
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"rfx.__file__ = {rfx.__file__}", flush=True)
    print(f"commit = {commit}  jax = {jax.__version__}  "
          f"x64 = {os.environ.get('JAX_ENABLE_X64', '0')}", flush=True)

    rows, t_all = [], time.time()
    for label, dx in sw.DX_GRID:
        h_um, w_um = _realized(_build(sw, dx))
        print(f"[{label}] solving (realized {w_um:.3f} x {h_um:.3f} um) ...",
              flush=True)
        record = sw.run_one(label, dx)
        hj, eps_eff = hammerstad_jensen_z0_eps_eff(
            w_um * 1e-6, h_um * 1e-6, sw.EPS_R)
        z0 = record["z0_measured_ohm"]
        record.update(
            h_sub_realized_um=round(h_um, 3),
            w_trace_realized_um=round(w_um, 3),
            z0_hj_realized_board_ohm=round(hj, 3),
            eps_eff_hj_realized_board=round(eps_eff, 4),
            dev_vs_realized_board_pct=round(100.0 * (z0 - hj) / hj, 4),
            dev_vs_declared_board_pct=round(
                100.0 * (z0 - record["z0_hj_ohm"]) / record["z0_hj_ohm"], 4),
        )
        rows.append(record)
        print(f"[{label}] Z0={z0:.3f} HJ_real={hj:.3f} "
              f"dev={record['dev_vs_realized_board_pct']:+.4f}%", flush=True)

    after = {p.name: _sha256(p) for p in (FROZEN, FROZEN_ANCHOR)}
    if after != before:
        raise SystemExit("a frozen artifact changed during the run -- ABORT")

    devs = [abs(r["dev_vs_realized_board_pct"]) for r in rows]
    aligned = [abs(r["dev_vs_realized_board_pct"]) for r in rows
               if r["label"].startswith("aligned")]
    misaligned = [abs(r["dev_vs_realized_board_pct"]) for r in rows
                  if r["label"].startswith("misaligned")]
    return {
        "note": (
            "Issue #752 six-point RE-SOLVE. NEW artifact: the pre-declared "
            "msl_z0_bias_floor_sweep.json, its as-run verdict block and "
            "msl_z0_bias_floor_sweep_realized_anchor.json are untouched and "
            "their sha256 was re-checked after this run. Every row is freshly "
            "solved AND freshly measured on the board that same build "
            "realized. Known contributors NOT modelled: the conductor is a "
            "one-cell PEC volume (#931), and hammerstad_jensen_z0_eps_eff is "
            "a simplified quasi-static formula, not full HJ1980."),
        "provenance": {
            "commit": commit, "rfx_file": rfx.__file__,
            "jax_version": jax.__version__,
            "jax_enable_x64": os.environ.get("JAX_ENABLE_X64", "0"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
            "python": platform.python_version(), "host": platform.platform(),
            "wallclock_total_s": round(time.time() - t_all, 1),
            "frozen_sha256_unchanged": before,
            "settings": ("run_one at its committed n_freqs=30, "
                         "num_periods=12, enforce_passivity=False, "
                         "gate 3.0-4.5 GHz"),
            "produced_by": Path(__file__).relative_to(REPO).as_posix(),
        },
        "predeclared_acceptance": {
            "c1_all_six_within_0p4_pct": bool(max(devs) <= 0.4),
            "c2_misaligned_within_2x_aligned":
                bool(max(misaligned) <= 2.0 * max(aligned)),
            "c3_all_settled_below_minus_40_db": bool(all(
                r["settling_db"] is not None and max(r["settling_db"]) < -40.0
                for r in rows)),
            "max_abs_dev_vs_realized_board_pct_all_six": round(max(devs), 4),
            "max_abs_dev_aligned_pct": round(max(aligned), 4),
            "max_abs_dev_misaligned_pct": round(max(misaligned), 4),
        },
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="solve the six points (about 48 min of CPU; "
                             "prefer VESSL, see this file's docstring)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="artifact to write with --run")
    args = parser.parse_args(argv)

    if not args.run:
        artifact = json.loads(DEFAULT_OUT.read_text())
        print(json.dumps({
            "record_kind": "msl_floor_sweep_realized_anchor_resolve",
            "new_field_solves": 0,
            "artifact": DEFAULT_OUT.relative_to(REPO).as_posix(),
            "artifact_sha256": _sha256(DEFAULT_OUT),
            "provenance": artifact["provenance"],
            "predeclared_acceptance": artifact["predeclared_acceptance"],
        }, indent=2))
        return 0

    if args.out.exists():
        parser.error(f"{args.out} exists; a run record is never overwritten. "
                     "Pass --out with a new dated name.")
    args.out.write_text(json.dumps(resolve(args.out), indent=1) + "\n")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
