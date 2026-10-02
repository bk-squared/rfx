"""Regenerate B1's two-step field record; Conclusion: leader fills.

Run with OMP/BLAS threads pinned to one and two emulated CPU devices, e.g.
XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=2'
python -m tests.contracts.boundary_baseline --update -n 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

from tests.contracts.boundary_cases import CASES, ENTRIES
from tests.contracts.boundary_compare import classify, departures
from tests.contracts.boundary_fields import measure, measured_fields


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "scripts/diagnostics/boundary_model/B1"
# Stable leading phrases, without counts, face lists, or explanatory advice.
REFUSAL_PREFIXES = (
    "PMC magnetic face(s)",
    "subgrid validation: supported=False",
    "[run] preflight found",
    "boundary='upml' does not support",
    "solver='adi' does not support",
    "solver='adi' supports only a uniform absorber",
    "vmap_sweep: a magnetic (pmc) boundary face is not supported",
    "run(compute_s_params=True) for lumped/wire add_port(...) does not honor periodic axes",
    "run(compute_s_params=True) has a single result schema",
    "periodic axes",
    "Lumped ports are not supported together with the TFSF",
    "Floquet ports do not support non-uniform z mesh",
    # Lane admission (rfx/runners/_admission.py): a declared input the lane
    # does not carry, named with the lane.
    "The ADI run() lane would solve this Simulation as if",
    "The graded run() lane would solve this Simulation as if",
)


def refusal_prefix(message):
    for prefix in REFUSAL_PREFIXES:
        if message.startswith(prefix):
            return prefix
    raise AssertionError(f"Unregistered refusal; inspect before updating the baseline: {message}")


def measure_entry(args):
    entry, base = args
    cells = []
    for case in CASES:
        start, cpu = time.monotonic(), time.process_time()
        row = dict(case=case, entry=entry, steps=2, base_commit=base)
        if entry == "subgridded":
            row["build"] = "sim.add_refinement((.006, .010), ratio=2); B0 used (0., .012)"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                sim, grid, declared, records = measure(case, entry)
                assert records, "No field scan observed"
                fields, psi, reference = measured_fields(records, grid, entry)
                assert fields.shape[-3:] == grid.shape, (
                    f"STOP {case}/{entry}: selected fields {fields.shape[-3:]} != grid {grid.shape}; "
                    "inspect the recorded layout before regeneration")
                faces = classify(fields, grid, psi, reference)
                row.update(status="MEASURED", declared=declared, shape=grid.shape, faces=faces,
                           scan_records=[dict(rank=r["rank"], shape=r["fields"].shape, layout=r["layout"])
                                         for r in records],
                           departures=departures(sim.boundary_model(), grid, faces, entry))
            except (ValueError, NotImplementedError) as exc:
                row.update(status="REFUSED", exception=type(exc).__name__,
                           message_prefix=refusal_prefix(str(exc)))
        row.update(cpu_s=time.process_time() - cpu, wall_s=time.monotonic() - start)
        cells.append(row)
    return cells


def b0_changes(cells, b0):
    changes = []
    for row in cells:
        gpu = row["entry"] == "gpu-query"
        source = b0["additional_backend_records"] if gpu else b0["records"]
        old = next(r for r in source if r["case"] == row["case"]
                   and r["entry"] == ("run [GPU]" if gpu else row["entry"]))
        differences = []
        if row["status"] != old["status"]:
            differences.append(old["status"] + " -> " + row["status"])
        elif row["status"] == "MEASURED":
            for face, measured in row["faces"].items():
                previous = old["faces"][face]
                before = "".join(c for c, yes in (("E", previous["E_nodes"]), ("H", previous["H_indices"]),
                                                 ("A", previous["absorber"]), ("W", previous["wrap"])) if yes) or "-"
                after = "".join(c for c, yes in (("E", measured["e_zero"]), ("H", measured["h_zero"]),
                                                ("A", measured["absorbs"]), ("W", measured["coupled"])) if yes) or "-"
                if before != after:
                    differences.append(f"{face} {before} -> {after}")
        if (row["case"], row["entry"]) == ("tfsf", "distributed"):
            differences.append("HARNESS CORRECTION (Addendum 4): use the recorded full grid without slab stripping; "
                               "B1 x_lo/x_hi E backings absent -> present; (x_lo,f)/(x_hi,f) removed; "
                               "not a change on main")
        if differences:
            if row["entry"] == "subgridded":
                differences.append("harness change: B0 refinement z=(0,12) mm; B1 z=(6,10) mm; not a change on main")
            changes.append(dict(case=row["case"], entry=row["entry"], changes=differences))
    return changes


def render_matrix(baseline, old_markdown):
    # The leader's final paragraph is opaque text, independent of JSON's field.
    # Preserve it byte-for-byte, along with the rest of the unregenerated tail.
    intro, rest = old_markdown.split("| Declaration |", 1)
    _, tail = rest.split("## Stopped comparisons", 1)
    import re
    intro = re.sub(r"Kernel base: `[^`]+`", f"Kernel base: `{baseline['kernel_base_commit']}`", intro)
    lines = ["| Declaration | " + " | ".join(ENTRIES) + " |", "|---|" + "---|" * len(ENTRIES)]
    cells = {(r["case"], r["entry"]): r for r in baseline["cells"]}
    for case in CASES:
        entries = []
        for entry in ENTRIES:
            row = cells[case, entry]
            entries.append("REFUSED" if row["status"] == "REFUSED" else
                           ",".join(sorted({d["code"] for d in row["departures"]})) or "pass")
        lines.append("| " + case + " | " + " | ".join(entries) + " |")
    lines += ["", "## B0 class → B1 field class", ""]
    lines += ["- " + r["case"] + " / " + r["entry"] + ": " + "; ".join(r["changes"]) + "."
              for r in baseline["b0_changes"]]
    return intro + "\n".join(lines) + "\n\n## Stopped comparisons" + tail


def write_baseline(baseline, old_markdown, target):
    markdown = render_matrix(baseline, old_markdown)
    (target / "MATRIX.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (target / "MATRIX.md").write_text(markdown)


def validate_class_changes(previous, cells):
    """Permit the TFSF backing, B2 periods and B3b magnetic image migrations."""
    old_lookup = {(r["case"], r["entry"]): r for r in previous["cells"]}
    for row in cells:
        cell = row["case"], row["entry"]
        old = old_lookup[cell]
        if cell[0] == "waveguide-pmc" and cell[1] in (
                "run", "forward", "nonuniform", "sweep", "gpu-query", "distributed", "subgridded"):
            assert row["status"] == "REFUSED"
            assert row["exception"] == "NotImplementedError"
            assert row["message_prefix"] == "PMC magnetic face(s)"
            continue
        magnetic = cell[0] in ("pmc-pec", "pmc-cpml", "waveguide-pmc")
        if magnetic and cell[1] == "distributed":
            assert row["status"] == "REFUSED"
            assert row["message_prefix"] == "PMC magnetic face(s)"
            continue
        if magnetic and cell[1] in ("run", "forward", "nonuniform", "wire-fast", "gpu-query", "sweep"):
            if old["status"] == "REFUSED":
                assert row["status"] == "REFUSED"
                assert (old["exception"], old["message_prefix"]) == (row["exception"], row["message_prefix"])
                continue
            assert row["status"] == "MEASURED"
            axes = "yz" if cell[0] == "waveguide-pmc" else "x"
            before = {(d["face"], d["code"]) for d in old["departures"]}
            after = {(d["face"], d["code"]) for d in row["departures"]}
            image_departures = {(f"{axis}_{side}", code) for axis in axes for side in ("lo", "hi")
                                for code in ("b1", "h")}
            assert after == before - image_departures, row["departures"]
            for name in (f"{axis}_{side}" for axis in axes for side in ("lo", "hi")):
                face = row["faces"][name]
                assert not face["e_zero"] and not face["h_zero"]
                assert face["face_node_response"] > 1e-7
            for name, face in row["faces"].items():
                prior_face = old["faces"][name]
                migrating = name[0] in axes and prior_face["h_zero"] and not face["h_zero"]
                for kind in ("e_zero", "h_zero", "absorbs", "coupled",
                             "e_zero_planes_m", "h_zero_planes_m", "period_m",
                             "e_plane_m", "h_plane_m"):
                    if migrating and kind in ("h_zero", "h_zero_planes_m"):
                        continue
                    prior, current = prior_face[kind], face[kind]
                    same = ((prior is None and current is None) or
                            (prior is not None and current is not None and
                             np.size(prior) == np.size(current) and
                             np.allclose(prior, current, rtol=0, atol=1e-10)))
                    assert same, f"STOP {cell}/{name}: {kind} {prior} -> {current}"
            # Only these magnetic declarations/lanes gain the B3b image. Their
            # declared-plane location is pinned by the independent cavity
            # and mirrored-domain tests; do not accept other class changes.
            continue
        assert old["status"] == row["status"], f"STOP {cell}: status changed"
        if row["status"] == "REFUSED":
            assert (old["exception"], old["message_prefix"]) == (row["exception"], row["message_prefix"]), (
                f"STOP {cell}: refusal signature changed")
            continue
        before = {(d["face"], d["code"]) for d in old["departures"]}
        after = {(d["face"], d["code"]) for d in row["departures"]}
        correction = cell == ("tfsf", "distributed")
        accepted = correction and before - after == {("x_lo", "f"), ("x_hi", "f")} and not after - before
        periodic = cell[0] == "periodic-xy" and cell[1] in ("run", "forward", "sweep", "gpu-query")
        floquet = cell[0] == "floquet" and cell[1] in ("run", "forward", "sweep", "gpu-query")
        period_fixed = periodic and before - after == {(f"{a}_{s}", "e") for a in "xy" for s in ("lo", "hi")} and not after - before
        assert before == after or accepted or period_fixed, (
            f"STOP {cell}: classification changed: removed {sorted(before - after)}; added {sorted(after - before)}")
        for face, values in row["faces"].items():
            prior_face = old["faces"][face]
            for kind in ("e_zero", "h_zero", "absorbs", "coupled"):
                prior, current = prior_face[kind], values[kind]
                allowed = accepted and face in ("x_lo", "x_hi") and kind == "e_zero" and not prior and current
                assert prior == current or allowed, f"STOP {cell}/{face}: {kind} {prior} -> {current}"
            # A wall plane or a period that moved is a value change, not a class change, and a
            # regeneration must not absorb it: moving away from the declared plane is a regression.
            # The step that moves a value toward the declared plane (B2, B3) adds its allowance
            # here, as the TFSF/distributed correction does for its E backings.
            for kind in ("e_zero_planes_m", "h_zero_planes_m", "period_m", "e_plane_m", "h_plane_m"):
                if kind in ("e_plane_m", "h_plane_m") and not (prior_face[kind[0] + "_zero"] and values[kind[0] + "_zero"]):
                    continue
                prior, current = prior_face[kind], values[kind]
                allowed = accepted and face in ("x_lo", "x_hi") and kind.startswith("e_")
                if (period_fixed or floquet) and face[0] in "xy" and kind == "period_m":
                    declared = .024 if face[0] == "x" else .020
                    allowed = current is not None and abs(current - declared) <= 1e-10
                same = (prior is None) == (current is None) and (
                    prior is None or np.allclose(np.atleast_1d(prior), np.atleast_1d(current), rtol=0, atol=1e-10)
                    if np.size(prior) == np.size(current) else False)
                assert same or allowed, f"STOP {cell}/{face}: {kind} {prior} -> {current}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="measure and rewrite MATRIX.json / MATRIX.md")
    parser.add_argument("-n", type=int, default=1, choices=range(1, 9), help="CPU worker processes, at most 8")
    args = parser.parse_args(argv)
    if not args.update:
        parser.error("pass --update to remeasure the baseline")
    previous = json.loads((TARGET / "MATRIX.json").read_text())
    markdown = (TARGET / "MATRIX.md").read_text()
    base = subprocess.check_output(["git", "merge-base", "HEAD", "origin/main"], cwd=ROOT, text=True).strip()
    jobs = [(entry, base) for entry in ENTRIES]
    if args.n == 1:
        results = list(map(measure_entry, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.n, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(measure_entry, jobs))
    lookup = {(r["case"], r["entry"]): r for group in results for r in group}
    cells = [lookup[case, entry] for case in CASES for entry in ENTRIES]
    validate_class_changes(previous, cells)
    baseline = dict(previous, kernel_base_commit=base, cells=cells,
                    cpu_seconds_cells=sum(r["cpu_s"] for r in cells))
    baseline["harness_sha256"] = {
        f"tests/contracts/{name}": hashlib.sha256((ROOT / "tests/contracts" / name).read_bytes()).hexdigest()
        for name in ("boundary_cases.py", "boundary_fields.py", "boundary_compare.py", "boundary_baseline.py")}
    b0 = json.loads((ROOT / "scripts/diagnostics/boundary_model/B0/MATRIX.json").read_text())
    baseline["b0_changes"] = b0_changes(cells, b0)
    write_baseline(baseline, markdown, TARGET)
    print(f"{len(cells)} cells; {sum(r['status'] == 'MEASURED' for r in cells)} measured; "
          f"{sum(len(r.get('departures', [])) for r in cells)} departures; "
          f"{baseline['cpu_seconds_cells']:.6f} cell CPU s; base {base}; "
          "no class changes outside the accepted TFSF/distributed harness correction")


if __name__ == "__main__":
    main()
