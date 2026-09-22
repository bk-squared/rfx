#!/usr/bin/env python3
"""Print JSON measurements for matching oracle/two-worker probe directories.

Usage: compare_multinode_probe.py --oracle ORACLE_ROOT --distributed TWO_ROOT
Directories are searched recursively. Missing traces/timings remain null with a
reason; a scan-output trace after a gather failure never marks gather successful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def load(root, count):
    groups = {}
    for path in sorted(root.rglob("*.rank*.json")):
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or data.get("process_count") != count:
            continue
        shape = tuple(data.get("realized_shape", data["requested_shape"]))
        rank = data["process_id"]
        group = groups.setdefault(shape, {})
        if rank in group:
            raise ValueError(f"Duplicate shape/rank {shape}/{rank}: {path}")
        group[rank] = (path, data)
    return groups


def timing(group, count, field):
    if set(group) != set(range(count)):
        return {"median_seconds": None, "median_per_step_seconds": None,
                "reason": "Missing rank JSON", "ranks_present": sorted(group)}
    records = [group[rank][1] for rank in range(count)]
    repeats = records[0]["arguments"]["repeats"]
    steps = records[0]["arguments"]["steps"]
    assert all(d["arguments"]["repeats"] == repeats and d["arguments"]["steps"] == steps
               for d in records), "Rank repeat/step mismatch"
    values = []
    for index in range(1, repeats + 1):
        per_rank = [next((r.get(field) for r in d["runs"] if r["index"] == index), None)
                    for d in records]
        if all(v is not None for v in per_rank):
            values.append(max(per_rank))
    median = statistics.median(values) if len(values) == repeats else None
    return {"max_rank_seconds_by_repeat": values, "median_seconds": median,
            "median_per_step_seconds": median / steps if median is not None else None,
            "reason": None if median is not None else "Incomplete timed repeats"}


def compare_traces(oracle, distributed):
    if 0 not in oracle or 0 not in distributed:
        return {"probes": None, "reason": "Missing rank-0 JSON"}
    arrays = []
    origins = []
    for group in (oracle, distributed):
        path, data = group[0]
        if not data.get("trace_file"):
            return {"probes": None, "reason": f"No trace recorded in {path}"}
        arrays.append(np.load(path.parent / data["trace_file"], allow_pickle=False))
        origins.append(data["trace_origin"])
    a, b = arrays
    assert a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3, "Trace shape mismatch"
    assert a.dtype == b.dtype == np.float32, "Expected float32 traces"
    assert np.isfinite(a).all() and np.isfinite(b).all(), "Non-finite trace"
    diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=0)
    peak = np.max(np.abs(a), axis=0)
    other_peak = np.max(np.abs(b), axis=0)
    return {"trace_origins": origins, "max_abs_diff": float(diff.max()),
            "probes": [{"probe": i, "max_abs_diff": float(diff[i]),
                        "oracle_peak": float(peak[i]), "distributed_peak": float(other_peak[i]),
                        "max_abs_diff_over_oracle_peak": float(diff[i] / peak[i]) if peak[i] else None,
                        "zero_oracle_peak": bool(peak[i] == 0)} for i in range(3)]}


def compare(oracle_root, distributed_root):
    oracles, distributed = load(oracle_root, 1), load(distributed_root, 2)
    if not oracles and not distributed:
        raise ValueError("No probe JSONs found")
    rows = []
    for shape in sorted(oracles.keys() | distributed.keys()):
        one, two = oracles.get(shape, {}), distributed.get(shape, {})
        present = [data for group in (one, two) for _, data in group.values()]
        for key in ("boundary", "dtype", "dx_m", "dt_seconds", "source_position_m",
                    "probe_positions_m", "waveform", "rfx_tree", "jax_version", "jaxlib_version"):
            values = [d[key] for d in present if key in d]
            if values and any(v != values[0] for v in values):
                raise ValueError(f"Mismatched {key} for shape {shape}: {values}")
        row = {"shape": shape, "nx_per_rank_two": shape[0] / 2,
               "trace": compare_traces(one, two), "timing": {}, "ranks": {}}
        for field in ("run_seconds", "scan_seconds"):
            a, b = timing(one, 1, field), timing(two, 2, field)
            ratio = (b["median_seconds"] / a["median_seconds"]
                     if a["median_seconds"] and b["median_seconds"] is not None else None)
            row["timing"][field] = {"oracle": a, "two_workers": b,
                                    "two_over_one_ratio": ratio}
        for name, group in (("oracle", one), ("two_workers", two)):
            row["ranks"][name] = [
                {"json": str(path), "process_id": rank, "hostname": data["hostname"],
                 "status": data.get("status"), "exception": data.get("exception"),
                 "memory": [m for m in data.get("memory", data.get("devices", [])) if m["local"]],
                 "runs": [{k: r.get(k) for k in
                           ("index", "status", "scan_completed", "gather_succeeded", "exception")}
                          for r in data["runs"]]}
                for rank, (path, data) in sorted(group.items())]
        rows.append(row)
    return {"timing_aggregation": "median of max rank wall-clock for each timed repeat; "
                                  "all ranks and repeats required; warm-up excluded",
            "measurements": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--distributed", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.oracle, args.distributed), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
