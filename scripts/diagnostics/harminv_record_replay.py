"""Compare estimator revisions on identical captured cv02/cv24 inputs.

No FDTD runs and no overwrite of existing reports. Both auto-decimation
paths are evaluated; when the case explicitly opts out, its undecimated
baseline is also evaluated. Baseline output is a comparison, not an oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def load_estimator(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse-baseline", type=Path,
                    help="reuse baseline results after checking source and input hashes")
    ap.add_argument("--skip-undecimated", action="store_true",
                    help="compare only the automatic paths (e.g. a new physical falsifier)")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists; choose a new report path")
    capture = json.loads((args.records / "capture.json").read_text())
    old = load_estimator(args.baseline, "_old_harminv")
    new = load_estimator(args.candidate, "_new_harminv")
    result = dict(case=capture["case"], source_commit=capture["source_commit"],
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  records=[])
    cached = {}
    if args.reuse_baseline:
        prior = json.loads(args.reuse_baseline.read_text())
        assert prior["baseline_sha256"] == result["baseline_sha256"]
        assert prior["case"] == result["case"]
        cached = {r["input"]["file"]: r for r in prior["records"]}
        result["baseline_reused_report_sha256"] = hashlib.sha256(args.reuse_baseline.read_bytes()).hexdigest()
    for record in capture["records"]:
        path = args.records / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with np.load(path) as data:
            signal = data["signal"]
        dt, fmin, fmax = (record[k] for k in ("dt", "f_min", "f_max"))
        row = dict(input=record, variants={})
        variants = [("baseline_auto", old, "auto"), ("candidate_auto", new, "auto")]
        if record["kwargs"].get("decimate") is False and not args.skip_undecimated:
            variants.extend([("baseline_undecimated", old, False),
                             ("candidate_undecimated", new, False)])
        for name, module, decimate in variants:
            if name.startswith("baseline_") and cached:
                previous = cached[record["file"]]
                assert previous["input"] == record, "cached input/parameters differ"
                row["variants"][name] = dict(previous["variants"][name], reused=True)
                continue
            kwargs = dict(record["kwargs"], decimate=decimate)
            start = time.monotonic()
            modes = module.harminv(signal, dt, fmin, fmax, **kwargs)
            duration = (new.harminv_record_duration(len(signal), dt, fmax, decimate=decimate)
                        if name.startswith("candidate_") else (len(signal) - 1) * dt)
            row["variants"][name] = dict(modes=[m._asdict() for m in modes],
                                          reported_input_or_analysis_duration_s=duration,
                                          wall_s=time.monotonic() - start)
            print(record["file"], name, len(modes), "modes", flush=True)
        result["records"].append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
