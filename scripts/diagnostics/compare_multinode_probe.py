#!/usr/bin/env python3
"""Compare distributed probe runs with one-rank runs of the same solver, and A with B.

Usage: compare_multinode_probe.py --oracle ROOT [ROOT ...] --distributed ROOT [ROOT ...]

Directories are searched recursively; the two options may name the same directory.
A run is one probe tag in one directory with all its rank JSONs. The oracle is a
run with n_ranks == 1 (the whole domain on one device); a one-process run over two
devices has n_ranks 2 and is not one (older probes wrote `oracle: true` for it, so
this script never reads that field).

Each distributed run (n_ranks > 1, under --distributed) is paired with every
one-rank run under --oracle of the same rfx tree whose configuration matches:
realized shape, model, lane, grad, steps, steps_short, checkpoint_every, the JAX
and jaxlib versions, and the grid, source, probe and waveform fields. A one-rank
run of the same rfx tree that differs, a distributed run with no one-rank run of
its rfx tree, and a run with uncommitted rfx/ changes are refused with a message
naming the runs and the fields; nothing is printed on stdout then. A directory
holding both <job> and <job>-b (the job script's A/B mode: --ref, then --ref-b on
the same nodes) also gets an A-versus-B comparison, whatever the rank count.

Differences are taken against the second run of each pair (the oracle, or B):
max |a - b|; that over the reference peak; that in float32 ULP of the reference
peak, max |a - b| / spacing(float32(peak)); and bitwise equality. Each trace probe
is compared against its own peak. Gradients are assembled from the shards every
rank saved, by their recorded x ranges, and must cover every row exactly once.
Timings take the slowest rank of each timed repeat and apply the probe's own
estimators (distributed_multinode_probe.timing_summary). Probe JSONs of schema 1
(before the estimator fields) are read from their per-repeat records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distributed_multinode_probe import (  # noqa: E402
    ESTIMATOR, FLAG_RULE, TIMED_FIELDS, timing_summary)

PHYSICS_FIELDS = ("boundary", "dtype", "dx_m", "dt_seconds", "source_position_m",
                  "probe_positions_m", "waveform")
REFERENCE_NOTE = ("Differences are taken against the second run of each pair (the one-rank oracle, or "
                  "B). ulp_of_peak = max|a-b| / spacing(float32(reference peak)); relative_to_peak = "
                  "max|a-b| / reference peak. Each trace probe against its own peak; the gradient "
                  "over the realized grid rows.")


def config_of(data):
    """What must match for two runs to be comparable (schema 1 defaults filled in)."""
    arguments = data.get("arguments") or {}
    config = {
        "realized_shape": data.get("realized_shape"),
        "model": data.get("model", arguments.get("model", "vacuum")),
        "lane": data.get("lane", arguments.get("lane", "run")),
        "grad": data.get("grad", arguments.get("grad", False)),
        "steps": data.get("steps", arguments.get("steps")),
        "steps_short": data.get("steps_short", arguments.get("steps_short")) or 0,
        "checkpoint_every": data.get("checkpoint_every", arguments.get("checkpoint_every")) or 0,
        "jax_version": data.get("jax_version"),
        "jaxlib_version": data.get("jaxlib_version"),
        # Fusion flags move rounding and time; the memory fraction moves time and capacity.
        "xla_flags": (data.get("environment") or {}).get("XLA_FLAGS"),
        "mem_fraction": (data.get("environment") or {}).get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    }
    config.update({key: data.get(key) for key in PHYSICS_FIELDS})
    return config


def rank_count(data):
    """Devices the domain was split over; one process can drive several."""
    if data.get("n_ranks") is not None:
        return data["n_ranks"]
    if data.get("devices"):
        return len(data["devices"])
    if not (data.get("arguments") or {}).get("local_devices"):
        return data.get("process_count")
    return None


class Run:
    """One probe invocation: a tag in one directory and the JSON of each rank."""

    def __init__(self, directory, tag, ranks):
        self.directory, self.tag = directory, tag
        self.ranks = dict(sorted(ranks.items()))
        self.head = self.ranks[min(self.ranks)][1]
        self.process_count = self.head.get("process_count")
        self.n_ranks = rank_count(self.head)
        self.commit = self.head.get("commit")
        self.rfx_tree = self.head.get("rfx_tree")
        self.config = config_of(self.head)
        self.label = f"{tag} [{directory}]"

    def disagreement(self):
        """Rank JSONs of one run must describe the same code and configuration.

        A field a rank never reached (it stopped early) is not a disagreement.
        """
        first = min(self.ranks)
        for rank, (_, data) in self.ranks.items():
            recorded = {"commit": data.get("commit"), "rfx_tree": data.get("rfx_tree"), **config_of(data)}
            expected = {"commit": self.commit, "rfx_tree": self.rfx_tree, **self.config}
            for key, value in recorded.items():
                if value is not None and expected[key] is not None and value != expected[key]:
                    return f"{self.label}: rank {rank} records {key} {value!r}, rank {first} {expected[key]!r}"
        return None

    def describe(self):
        arguments = self.head.get("arguments") or {}
        return {"directory": str(self.directory), "tag": self.tag, "commit": self.commit,
                "rfx_tree": self.rfx_tree, "rfx_dirty": self.head.get("rfx_dirty"),
                "probe_sha256": self.head.get("probe_sha256"),
                "schema_version": self.head.get("schema_version"),
                "n_ranks": self.n_ranks, "process_count": self.process_count,
                "local_devices": bool(arguments.get("local_devices")),
                "ranks_present": sorted(self.ranks),
                "status": {str(rank): data.get("status") for rank, (_, data) in self.ranks.items()},
                "hostnames": {str(rank): data.get("hostname") for rank, (_, data) in self.ranks.items()}}


def load_runs(roots):
    grouped = {}
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"compare_multinode_probe: {root} is not a directory")
        for path in sorted(root.rglob("*.rank*.json")):
            if path.name.startswith(("summary.", "._")):
                continue
            data = json.loads(path.read_text())
            if data.get("schema_version") not in (1, 2) or "tag" not in data or "process_id" not in data:
                continue
            ranks = grouped.setdefault((path.parent.resolve(), data["tag"]), {})
            rank = data["process_id"]
            if rank in ranks and ranks[rank][0].resolve() != path.resolve():
                raise SystemExit(f"compare_multinode_probe: two JSONs for rank {rank} of {data['tag']} "
                                 f"in {path.parent}")
            ranks[rank] = (path, data)
    return {key: Run(key[0], key[1], ranks) for key, ranks in grouped.items()}


def difference(a, b):
    """a against the reference b (same shape)."""
    bitwise = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"reason": "non-finite values", "bitwise_equal": bitwise,
                "non_finite_elements": [int(np.count_nonzero(~np.isfinite(a))),
                                        int(np.count_nonzero(~np.isfinite(b)))]}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    max_diff = float(np.max(np.abs(a64 - b64))) if a.size else 0.0
    peak = float(np.max(np.abs(b64))) if b.size else 0.0
    return {"max_abs_diff": max_diff, "reference_peak": peak,
            "relative_to_peak": max_diff / peak if peak else None,
            "ulp_of_peak": max_diff / float(np.spacing(np.float32(peak))) if peak else None,
            "elements_differing": int(np.count_nonzero(a != b)), "bitwise_equal": bitwise}


def load_trace(run):
    if 0 not in run.ranks:
        return None, "missing rank-0 JSON"
    path, data = run.ranks[0]
    if not data.get("trace_file"):
        return None, f"no trace recorded in {path.name}"
    trace = np.load(path.parent / data["trace_file"], allow_pickle=False)
    if trace.ndim != 2 or trace.dtype != np.float32 or not np.isfinite(trace).all():
        return None, f"{data['trace_file']}: expected a finite float32 (steps, probes) array"
    return trace, None


def compare_traces(run, reference):
    (a, why_a), (b, why_b) = load_trace(run), load_trace(reference)
    if why_a or why_b:
        return {"reason": why_a or why_b}
    if a.shape != b.shape:
        return {"reason": f"trace shapes differ: {a.shape} and {b.shape}"}
    origins = [(r.ranks[0][1].get("trace_origin"), r.ranks[0][1].get("trace_run_index"))
               for r in (run, reference)]
    return {"trace_origin_and_repeat": origins, "bitwise_equal": a.tobytes() == b.tobytes(),
            "probes": [{"probe": i, **difference(a[:, i], b[:, i])} for i in range(a.shape[1])]}


def assemble_gradient(run):
    """The global gradient from every rank's saved shards, placed by recorded x range."""
    shards = []
    for path, data in run.ranks.values():
        records = [r for r in data.get("runs", []) if r.get("grad_shards")]
        if records:
            last = max(records, key=lambda r: r["index"])
            shards += [(path.parent / s["file"], s["x"][0], s["x"][1]) for s in last["grad_shards"]]
    if not shards:
        return None, "no gradient shards recorded"
    shape = run.head.get("design_shape")
    if not shape:
        return None, "no design_shape recorded"
    rows = np.zeros(shape[0], dtype=np.int64)
    parts = []
    for file, lo, hi in sorted(shards, key=lambda s: s[1]):
        if not 0 <= lo < hi <= shape[0]:
            return None, f"{file.name}: x range [{lo}, {hi}) outside the design's {shape[0]} rows"
        array = np.load(file, allow_pickle=False)
        if array.shape != (hi - lo, *shape[1:]):
            return None, f"{file.name}: shape {array.shape} does not hold x [{lo}, {hi}) of {shape}"
        rows[lo:hi] += 1
        parts.append((lo, hi, array))
    if not (rows == 1).all():
        return None, (f"shards leave {int(np.count_nonzero(rows == 0))} of {shape[0]} x rows uncovered "
                      f"and cover {int(np.count_nonzero(rows > 1))} more than once")
    dtypes = {array.dtype for _, _, array in parts}
    if len(dtypes) != 1:
        return None, f"shards have different dtypes {sorted(map(str, dtypes))}"
    gradient = np.empty(shape, dtype=dtypes.pop())
    for lo, hi, array in parts:
        gradient[lo:hi] = array
    return gradient, {"shards": [[lo, hi] for lo, hi, _ in parts], "design_shape": shape}


def compare_gradients(run, reference):
    if not (run.config["grad"] and reference.config["grad"]):
        return {"reason": "not a --grad run"}
    (a, info_a), (b, info_b) = assemble_gradient(run), assemble_gradient(reference)
    if a is None or b is None:
        return {"reason": info_a if a is None else info_b}
    nx = run.config["realized_shape"][0]
    if a.shape[1:] != b.shape[1:] or min(a.shape[0], b.shape[0]) < nx:
        return {"reason": f"gradient shapes {a.shape} and {b.shape} do not both hold {nx} x rows"}
    result = {"shards": [info_a["shards"], info_b["shards"]], "compared_x_rows": nx,
              **difference(a[:nx], b[:nx])}
    for name, array in (("run", a), ("reference", b)):
        if array.shape[0] > nx:
            result[f"{name}_alignment_pad_rows_zero"] = bool(np.all(array[nx:] == 0))
    return result


def timing_of(run, field):
    """The probe's estimators over the slowest rank of each timed repeat."""
    missing = sorted(set(range(run.process_count or 0)) - set(run.ranks))
    if missing:
        return {"values": [], "reason": f"missing rank JSONs {missing}"}
    repeats = (run.head.get("arguments") or {}).get("repeats") or 0
    records, incomplete = [], []
    for index in range(1, repeats + 1):
        per_rank = [next((r for r in data.get("runs", []) if r.get("index") == index), None)
                    for _, data in run.ranks.values()]
        if any(r is None for r in per_rank):
            incomplete.append(index)
            continue
        record = {"index": index, "warmup": False, "compile_seconds": {}}
        for name in (field, field + "_short"):
            values = [r.get(name) for r in per_rank]
            record[name] = max(values) if all(v is not None for v in values) else None
            compiles = [((r.get("compile_seconds") or {}).get(name) or {}).get("backend_compile")
                        for r in per_rank]
            if all(c is not None for c in compiles):
                record["compile_seconds"][name] = {"backend_compile": max(compiles)}
        if record[field] is None:
            incomplete.append(index)
        records.append(record)
    summary = timing_summary(records, field, run.config["steps"], run.config["steps_short"])
    summary["aggregation"] = "slowest rank per timed repeat"
    if incomplete:
        summary["reason"] = f"timed repeats {incomplete} are missing, or have no {field}, on some rank"
    return summary


def ratio(numerator, denominator, key):
    """top / bottom, or the reason there is none (a marginal can come out <= 0)."""
    if numerator.get("reason") or denominator.get("reason"):
        return None, "a timing has missing repeats"
    top, bottom = numerator.get(key), denominator.get(key)
    if top is None or bottom is None:
        return None, f"no {key} on one side"
    if top <= 0 or bottom <= 0:
        return None, f"not computed: {key} is {top:.3g} and {bottom:.3g}; a ratio needs both > 0"
    return top / bottom, None


def compare_runs(run, reference, run_name, reference_name):
    row = {run_name: run.describe(), reference_name: reference.describe(), "config": run.config,
           "trace": compare_traces(run, reference), "gradient": compare_gradients(run, reference),
           "timing": {}}
    for field in TIMED_FIELDS:
        mine, theirs = timing_of(run, field), timing_of(reference, field)
        if not mine.get("values") and not theirs.get("values"):
            continue
        row["timing"][field] = {run_name: mine, reference_name: theirs}
        for key, name in (("median_marginal_per_step", "median_marginal_per_step"),
                          ("median", "median_incl_compile")):
            value, why = ratio(mine, theirs, key)
            row["timing"][field][f"{run_name}_over_{reference_name}_{name}"] = value
            if why:
                row["timing"][field][f"{run_name}_over_{reference_name}_{name}_note"] = why
    row["ranks"] = {}
    for name, item in ((run_name, run), (reference_name, reference)):
        row["ranks"][name] = [
            {"json": str(path), "process_id": rank, "hostname": data.get("hostname"),
             "status": data.get("status"), "exception": data.get("exception"),
             "memory": [m for m in data.get("memory", data.get("devices", [])) if m.get("local")],
             "runs": [{k: r.get(k) for k in
                       ("index", "status", "scan_completed", "gather_succeeded", "failure_phase",
                        "exception")} for r in data.get("runs", [])]}
            for rank, (path, data) in item.ranks.items()]
    return row


def mismatch(run, other):
    return "; ".join(f"{key} {run.config[key]!r} vs {other.config[key]!r}"
                     for key in run.config if run.config[key] != other.config[key])


def compare(oracle_roots, distributed_roots):
    oracle_runs = load_runs(oracle_roots)
    distributed_runs = load_runs(distributed_roots)
    every = {**oracle_runs, **distributed_runs}
    problems = [p for p in (run.disagreement() for run in every.values()) if p]
    oracles = [run for run in oracle_runs.values() if run.n_ranks == 1]
    distributed = [run for run in distributed_runs.values() if (run.n_ranks or 0) > 1]
    rows, unpaired = [], []
    for run in sorted(distributed, key=lambda r: r.label):
        if run.rfx_tree is None or run.config["realized_shape"] is None:
            unpaired.append({"run": run.describe(), "exception": run.head.get("exception"),
                             "reason": "the run stopped before it recorded its rfx tree and grid"})
            continue
        same_tree = [o for o in oracles if o.rfx_tree == run.rfx_tree]
        matched = [o for o in same_tree if o.config == run.config]
        rows += [compare_runs(run, o, "distributed", "oracle") for o in sorted(matched, key=lambda r: r.label)]
        if matched:
            continue
        if same_tree:
            problems += [f"{run.label} and the one-rank run {o.label} are rfx tree {run.rfx_tree} but "
                         f"differ: {mismatch(run, o)}" for o in same_tree]
        else:
            # No oracle of this tree is not an error: an A/B job needs none. Report it.
            found = ", ".join(f"{o.label} (tree {o.rfx_tree})" for o in oracles) or "none"
            others = [r for r in oracle_runs.values() if r.n_ranks != 1]
            note = ("; runs there with more than one rank are not oracles: "
                    + ", ".join(f"{r.label} (n_ranks {r.n_ranks})" for r in others)) if others else ""
            unpaired.append({"run": run.describe(),
                             "reason": f"no one-rank run of rfx tree {run.rfx_tree} (commit {run.commit}) "
                                       f"under --oracle; one-rank runs found: {found}{note}"})
    ab_rows = []
    for (directory, tag), a in sorted(every.items()):
        b = every.get((directory, tag + "-b"))
        if b is None:
            continue
        if a.config != b.config:
            problems.append(f"A/B {a.label} and {b.label} differ: {mismatch(a, b)}")
            continue
        row = compare_runs(a, b, "a", "b")
        row.update(same_rfx_tree=a.rfx_tree == b.rfx_tree,
                   same_probe_sha256=a.head.get("probe_sha256") == b.head.get("probe_sha256"))
        ab_rows.append(row)
    if not rows and not ab_rows and not problems:
        problems.append("nothing to compare: no distributed run under --distributed and no <job>/<job>-b pair")
    compared = {(row[name]["directory"], row[name]["tag"]) for row in rows + ab_rows
                for name in ("distributed", "oracle", "a", "b") if name in row}
    problems += [f"{run.label}: {run.head['rfx_dirty']} uncommitted path(s) under rfx/, so its rfx tree "
                 "does not identify the solver that ran" for (directory, tag), run in sorted(every.items())
                 if (str(directory), tag) in compared and run.head.get("rfx_dirty")]
    if problems:
        raise SystemExit("compare_multinode_probe: refused\n" + "\n".join(f"  - {p}" for p in problems))
    not_compared = [{"run": run.label, "n_ranks": run.n_ranks, "status": run.head.get("status")}
                    for (directory, tag), run in sorted(every.items())
                    if (str(directory), tag) not in compared]
    return {"reference": REFERENCE_NOTE,
            "timing_aggregation": "slowest rank per timed repeat, warm-up excluded",
            "estimator": ESTIMATOR, "flag_rule": FLAG_RULE,
            "distributed_vs_oracle": rows, "a_vs_b": ab_rows,
            "unpaired": unpaired, "not_compared": not_compared}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracle", type=Path, nargs="+", required=True)
    parser.add_argument("--distributed", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.oracle, args.distributed), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
