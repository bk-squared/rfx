#!/usr/bin/env python3
"""Measure the unmodified public runner, including failures before/after scan.

One process owns one device. --nx-per-rank is the *realized array* x length
per process, excluding ghost cells. For an oracle of N ranks of S cells, pass
--process-count 1 --nx-per-rank N*S: the whole domain is on ONE device.

A scoped Python line observer times the runner's existing scan call and blocks
its outputs before the existing gather. It neither replaces the scan/gather
nor constructs global arrays. Timings include compilation on every invocation
where main recompiles; these are public sim.run calls, not cached kernel replays.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback


def exception_record(exc):
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ScanObserver:
    """Observe only the outer runner frame, at its actual scan assignment."""

    def __init__(self, function, jax, distributed):
        self.jax = jax
        self.code = function.__code__
        source, first = inspect.getsourcelines(function)
        tree = ast.parse("".join(source))
        self.lines = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            targets = {n.id for t in node.targets for n in ast.walk(t)
                       if isinstance(n, ast.Name)}
            if {"final_carry", "probe_ts" if distributed else "outputs"} <= targets:
                called = ast.unparse(node.value.func)
                if called == ("run_fn" if distributed else "jax.lax.scan"):
                    self.lines.add(first + node.lineno - 1)
        # Distributed: one assignment per boundary type on main; since #1202
        # each boundary type has a single-process and a multi-process
        # assignment (four). Only the executed one fires the timer.
        if len(self.lines) not in ({2, 4} if distributed else {1}):
            raise RuntimeError("Runner scan instrumentation anchor changed")
        self.output_name = "probe_ts" if distributed else "outputs"
        self.scan_seconds = None
        self.started = None
        self.trace = None
        self.phase = "setup"

    def global_trace(self, frame, event, arg):
        if event == "call" and frame.f_code is self.code:
            return self.local_trace
        return None

    def local_trace(self, frame, event, arg):
        if event == "line":
            if self.started is not None and self.scan_seconds is None:
                if "final_carry" in frame.f_locals:
                    carry = frame.f_locals["final_carry"]
                    self.trace = frame.f_locals[self.output_name]
                    self.jax.block_until_ready((carry, self.trace))
                    self.scan_seconds = time.perf_counter() - self.started
                    self.phase = "gather"
            if frame.f_lineno in self.lines:
                # Drain setup arrays so the scan interval starts at the call.
                self.jax.block_until_ready((frame.f_locals["carry_init"],
                                            frame.f_locals["xs"]))
                self.started = time.perf_counter()
                self.phase = "scan"
        return self.local_trace


def memory_stats(devices, process_id):
    records = []
    for device in devices:
        item = {"id": device.id, "process_index": device.process_index,
                "platform": device.platform, "device_kind": device.device_kind,
                "local": device.process_index == process_id,
                "peak_bytes_in_use": None, "bytes_limit": None}
        if item["local"]:
            try:
                stats = device.memory_stats()
                item["available"] = stats is not None
                if stats:
                    for key in ("peak_bytes_in_use", "bytes_limit"):
                        item[key] = int(stats[key]) if key in stats else None
            except Exception as exc:
                item["exception"] = exception_record(exc)
        records.append(item)
    return records


def measure(sim, args, jax, observer):
    record = {"scan_seconds": None, "run_seconds": None,
              "scan_completed": False, "gather_succeeded": False}
    result = None
    previous = sys.gettrace()
    start = time.perf_counter()
    try:
        sys.settrace(observer.global_trace)
        result = sim.run(n_steps=args.steps, devices=jax.devices())
        sys.settrace(previous)
        observer.phase = "gather"
        jax.block_until_ready((result.state, result.time_series))
        record["run_seconds"] = time.perf_counter() - start
        record["gather_succeeded"] = True
        record["status"] = "ok"
        record["final_state_fully_addressable"] = bool(result.state.ez.is_fully_addressable)
        record["trace_fully_addressable"] = bool(result.time_series.is_fully_addressable)
    except Exception as exc:
        record["failure_phase"] = observer.phase
        record["exception"] = exception_record(exc)
        record["status"] = "gather_failed" if observer.phase == "gather" else "error"
    finally:
        sys.settrace(previous)
        record["attempt_seconds"] = time.perf_counter() - start
        record["scan_seconds"] = observer.scan_seconds
        record["scan_completed"] = observer.scan_seconds is not None
        for field in ("run_seconds", "scan_seconds"):
            record[field + "_per_step"] = (record[field] / args.steps
                                             if record[field] is not None else None)
    trace = result.time_series if result is not None else observer.trace
    record["trace_origin"] = "result.time_series" if result is not None else "scan_output"
    record["memory"] = memory_stats(jax.devices(), args.process_id)
    return record, trace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-address")
    parser.add_argument("--process-count", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--local-device-id", type=int, default=0)
    parser.add_argument("--nx-per-rank", type=int, required=True)
    parser.add_argument("--ny", type=int, default=116)
    parser.add_argument("--nz", type=int, default=116)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="probe")
    args = parser.parse_args()
    if min(args.process_count, args.steps, args.repeats) < 1:
        parser.error("process-count, steps and repeats must be positive")
    if min(args.nx_per_rank, args.ny, args.nz) < 4:
        parser.error("each slab/grid dimension must be at least 4")
    if not 0 <= args.process_id < args.process_count:
        parser.error("process-id must be in [0, process-count)")
    if args.process_count > 1 and not args.coordinator_address:
        parser.error("multiple processes require --coordinator-address")
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                           for c in args.tag):
        parser.error("tag must contain only letters, digits, hyphens, underscores, dots")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.tag}.rank{args.process_id}.json"
    nx = args.nx_per_rank * args.process_count
    data = {"schema_version": 1, "tag": args.tag, "hostname": socket.gethostname(),
            "process_id": args.process_id, "process_count": args.process_count,
            "arguments": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()},
            "requested_shape": [nx, args.ny, args.nz], "runs": [], "status": "initializing",
            "oracle": args.process_count == 1,
            "oracle_domain_note": "One process: nx-per-rank is the entire x array on ONE device; "
                                  "to match N ranks of S cells, supply nx-per-rank=N*S.",
            "timing_note": "Public sim.run plus block_until_ready; includes setup, any compilation, "
                           "scan and native gather. scan_seconds excludes setup and gather but "
                           "includes any scan compilation. Warm-up excluded from medians. "
                           "Scoped Python line observer adds overhead. No kernel cache replacement.",
            "environment": {k: os.environ.get(k) for k in
                            ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                             "CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                             "RFX_NODE_NAME", "NODE_NAME", "RFX_TOOLING_SHA")}}
    write_json(path, data)
    initialized = False
    exit_code = 1
    try:
        import jax
        import jaxlib
        import numpy as np

        data.update(jax_version=jax.__version__, jaxlib_version=jaxlib.__version__)
        write_json(path, data)
        if args.process_count > 1:
            jax.distributed.initialize(coordinator_address=args.coordinator_address,
                                       num_processes=args.process_count,
                                       process_id=args.process_id,
                                       local_device_ids=[args.local_device_id],
                                       initialization_timeout=900)
            initialized = True
        assert len(jax.local_devices()) == 1, str(jax.local_devices())
        assert len(jax.devices()) == args.process_count, str(jax.devices())
        assert jax.process_index() == args.process_id
        data["devices"] = memory_stats(jax.devices(), args.process_id)
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from rfx import Simulation
        from rfx.runners.distributed_v2 import run_distributed
        from rfx.simulation import run as run_single

        data["commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        data["rfx_tree"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD:rfx"], text=True).strip()
        data["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        sim = Simulation(freq_max=15e9, domain=tuple((n - 1) * 1e-3 for n in (nx, args.ny, args.nz)),
                         dx=1e-3, boundary="pec", precision="float32")
        source = (nx * 0.5e-3, args.ny * 0.5e-3, args.nz * 0.5e-3)
        sim.add_source(source, "ez", amplitude_kind="field")
        probes = [(nx * fraction * 1e-3, source[1], source[2]) for fraction in (0.25, 0.5, 0.75)]
        for position in probes:
            sim.add_probe(position, "ez")
        grid = sim._build_grid()
        data.update(realized_shape=list(grid.shape), boundary="pec", dtype="float32",
                    dx_m=1e-3, dt_seconds=float(grid.dt), freq_max_hz=15e9,
                    source_position_m=source, probe_positions_m=probes,
                    waveform="default GaussianPulse(f0=7.5e9, bandwidth=0.8)",
                    domain_m=list(sim._domain))
        assert tuple(grid.shape) == (nx, args.ny, args.nz), str(grid.shape)
        function = run_distributed if args.process_count > 1 else run_single
        data["status"] = "running"
        for index in range(args.repeats + 1):
            observer = ScanObserver(function, jax, args.process_count > 1)
            record, trace = measure(sim, args, jax, observer)
            record.update(index=index, warmup=index == 0)
            if trace is not None:
                try:
                    # Native replicated trace only; no process_allgather or recovery sharding.
                    host_trace = np.asarray(trace, dtype=np.float32)
                except Exception as exc:
                    record["trace_exception"] = exception_record(exc)
                    record["status"] = "gather_failed"
                    record["gather_succeeded"] = False
                else:
                    try:
                        assert host_trace.shape == (args.steps, 3), str(host_trace.shape)
                        if not np.isfinite(host_trace).all():
                            raise ValueError("Trace contains non-finite values")
                        if args.process_id == 0:
                            name = f"{args.tag}.trace.npy"
                            np.save(args.output / name, host_trace, allow_pickle=False)
                            data.update(trace_file=name, trace_run_index=index,
                                        trace_origin=record["trace_origin"])
                    except Exception as exc:
                        record["exception"] = exception_record(exc)
                        record["failure_phase"] = "trace_validation_or_save"
                        record["status"] = "error"
            data["runs"].append(record)
            write_json(path, data)
            print(json.dumps({"process_id": args.process_id, **record}), flush=True)
            if record["status"] == "error":
                break
        statuses = [r["status"] for r in data["runs"]]
        exit_code = 1 if "error" in statuses else 2 if "gather_failed" in statuses else 0
        for field in ("run_seconds", "scan_seconds"):
            values = [r[field] for r in data["runs"] if not r["warmup"] and r[field] is not None]
            median = statistics.median(values) if values else None
            data[field] = {"values": values, "median": median,
                           "median_per_step": median / args.steps if median is not None else None}
        data["memory"] = memory_stats(jax.devices(), args.process_id)
    except Exception as exc:
        data["exception"] = exception_record(exc)
        traceback.print_exc()
    finally:
        data.update(exit_code=exit_code, status={0: "ok", 1: "error", 2: "gather_failed"}[exit_code])
        write_json(path, data)
        if initialized:
            jax.distributed.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
