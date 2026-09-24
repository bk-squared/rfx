#!/usr/bin/env python3
"""Measure the unmodified public runner, including failures before/after scan.

One process owns one device. --nx-per-rank is the *realized array* x length
per process, excluding ghost cells. For an oracle of N ranks of S cells, pass
--process-count 1 --nx-per-rank N*S: the whole domain is on ONE device.

A scoped Python line observer times the runner's existing scan call and blocks
its outputs before the existing gather. It neither replaces the scan/gather
nor constructs global arrays. These are public sim.run / forward calls, not
cached kernel replays: every call traces, lowers and compiles again, so one
call's time includes compilation. With --steps-short each repeat times the same
call at steps-short and then at --steps; the paired difference over the step
difference is the loop's cost per step with the per-call compilation cancelled,
provided both calls compile the same program (checked at argument parsing). A
jax.monitoring listener records the compile seconds that fall inside every
timed window.
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

TIMED_FIELDS = ("run_seconds", "scan_seconds", "grad_seconds")
# The compile-stage durations JAX reports through jax.monitoring. The three names
# are defined identically in jax/_src/dispatch.py of JAX 0.4.33, 0.6.2 and 0.10.2;
# backend_compile_duration is the XLA compilation itself.
COMPILE_EVENTS = {
    "/jax/core/compile/backend_compile_duration": "backend_compile",
    "/jax/core/compile/jaxpr_trace_duration": "jaxpr_trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "jaxpr_to_mlir_module",
}
FLAG_FACTOR = 1.5
ESTIMATOR = ("median_marginal_per_step is the median, over the timed repeats (warm-up excluded), of "
             "(t_long - t_short) / (steps - steps_short) with both times taken in the same repeat. "
             "difference_of_medians_per_step is (median t_long - median t_short) / (steps - steps_short).")
FLAG_RULE = (f"a timed repeat is flagged when its short or its long time exceeds {FLAG_FACTOR}x the median "
             "of the timed repeats' times of the same kind; flagged repeats stay in every statistic")


def exception_record(exc):
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def segmentation(steps, checkpoint_every):
    """The program forward(distributed=True) compiles for one call of `steps` steps.

    rfx/runners/distributed_nu.py (run_nonuniform_distributed_pec), with the
    n_warmup=0 this probe uses, segments the loop only when
    0 < checkpoint_every < steps, and pads the last segment when checkpoint_every
    does not divide steps.
    """
    if not checkpoint_every or checkpoint_every >= steps:
        return "one unsegmented scan"
    segments = -(-steps // checkpoint_every)
    pad = segments * checkpoint_every - steps
    return (f"{segments} segments of {checkpoint_every}"
            + (f", the last padded with {pad} computed steps" if pad else ""))


def timing_plan_error(lane, steps, steps_short, checkpoint_every):
    """Why these step counts cannot be timed as asked, or None.

    prepare_multinode_launch.py calls this too, so a launch is refused before
    it is submitted.
    """
    if steps_short and not 0 < steps_short < steps:
        return f"--steps-short must satisfy 0 < steps-short < steps; got {steps_short} with steps {steps}"
    if checkpoint_every < 0:
        return "--checkpoint-every must be >= 0"
    if checkpoint_every and lane != "forward":
        return "--checkpoint-every needs --lane forward (the run lane never passes it to the runner)"
    if checkpoint_every:
        counts = [n for n in (steps_short, steps) if n]
        if any(n <= checkpoint_every or n % checkpoint_every for n in counts):
            plans = "; ".join(f"{n} steps -> {segmentation(n, checkpoint_every)}" for n in counts)

            def kind(n):
                return (n > checkpoint_every, n % checkpoint_every == 0)

            if steps_short and kind(steps_short) != kind(steps):
                why = ("The short and long calls would compile different programs, so their difference "
                       "would not be the loop's cost per step.")
            elif any(n <= checkpoint_every for n in counts):
                why = "The remat length would be recorded but never applied."
            else:
                why = "The padded steps would be computed but not counted in the per-step cost."
            return (f"--checkpoint-every {checkpoint_every} needs every timed step count to be a multiple "
                    f"of it larger than it: forward(distributed=True) segments a call only when "
                    f"checkpoint_every < steps and pads the last segment when it does not divide steps "
                    f"(rfx/runners/distributed_nu.py, run_nonuniform_distributed_pec). Here: {plans}. {why}")
    return None


class CompileClock:
    """Compile-stage durations JAX reports, stamped when each one ends.

    JAX calls the listener as a stage finishes, in the thread that runs it,
    so a duration stamped inside a timed window was spent inside it.
    """

    def __init__(self, monitoring):
        self.events = []
        monitoring.register_event_duration_secs_listener(self._listen)

    def _listen(self, event, duration, **_):
        name = COMPILE_EVENTS.get(event)
        if name is not None:
            self.events.append((time.perf_counter(), name, float(duration)))

    def within(self, start, end):
        totals = dict.fromkeys(COMPILE_EVENTS.values(), 0.0)
        for stamp, name, duration in self.events:
            if start <= stamp <= end:
                totals[name] += duration
        return totals


def book(record, field, start, end, clock):
    """Record one timed window and the compile seconds inside it."""
    record[field] = end - start
    record.setdefault("compile_seconds", {})[field] = clock.within(start, end)


class ScanObserver:
    """Observe only the outer runner frame, at its actual scan assignment."""

    def __init__(self, function, jax, distributed):
        self.jax = jax
        self.code = function.__code__
        source, first = inspect.getsourcelines(function)
        tree = ast.parse("".join(source))
        self.lines = set()
        self.call_args = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            targets = {n.id for t in node.targets for n in ast.walk(t)
                       if isinstance(n, ast.Name)}
            if {"final_carry", "probe_ts" if distributed else "outputs"} <= targets:
                called = ast.unparse(node.value.func)
                if called == ("run_fn" if distributed else "jax.lax.scan"):
                    line = first + node.lineno - 1
                    self.lines.add(line)
                    self.call_args[line] = (list(node.value.args)
                                            + [k.value for k in node.value.keywords])
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
        self.drained_arrays = None
        self.args_inside_window = None

    def drain(self, frame):
        """Wait for every live JAX array the runner frame holds.

        The frame's locals hold every argument of the anchored call that is a
        bare name (all of them on main) and the arrays a scan body closes over,
        so setup work still running asynchronously finishes before the clock
        starts. Nothing is evaluated; a local that does not flatten as a pytree
        is skipped, and so is a deleted (donated) array.
        """
        arrays = []
        for value in list(frame.f_locals.values()):
            try:
                leaves = self.jax.tree_util.tree_leaves(value)
            except Exception:
                continue
            for leaf in leaves:
                if isinstance(leaf, self.jax.Array):
                    deleted = getattr(leaf, "is_deleted", None)
                    if deleted is None or not deleted():
                        arrays.append(leaf)
        self.jax.block_until_ready(arrays)
        return len(arrays)

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
                bound = frame.f_locals
                # Arguments that are not already-bound names are evaluated by the
                # call itself, inside the window; none are on main.
                self.args_inside_window = [
                    ast.unparse(a) for a in self.call_args[frame.f_lineno]
                    if not (isinstance(a, ast.Name) and a.id in bound)]
                self.drained_arrays = self.drain(frame)
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


def measure(sim, steps, jax, observer, process_id, clock):
    """One public sim.run call of `steps` steps."""
    record = {"scan_seconds": None, "run_seconds": None,
              "scan_completed": False, "gather_succeeded": False}
    result = None
    previous = sys.gettrace()
    start = time.perf_counter()
    try:
        sys.settrace(observer.global_trace)
        result = sim.run(n_steps=steps, devices=jax.devices())
        sys.settrace(previous)
        observer.phase = "gather"
        jax.block_until_ready((result.state, result.time_series))
        book(record, "run_seconds", start, time.perf_counter(), clock)
        record["gather_succeeded"] = True
        record["status"] = "ok"
        # Since #1202 the multi-process path returns host numpy fields (the
        # cross-process gather), so the JAX-only attribute is read when present.
        for key, array in (("final_state", result.state.ez),
                           ("trace", result.time_series)):
            record[key + "_type"] = type(array).__module__ + "." + type(array).__name__
            addressable = getattr(array, "is_fully_addressable", None)
            record[key + "_fully_addressable"] = (
                None if addressable is None else bool(addressable))
    except Exception as exc:
        record["failure_phase"] = observer.phase
        record["exception"] = exception_record(exc)
        record["status"] = "gather_failed" if observer.phase == "gather" else "error"
    finally:
        sys.settrace(previous)
        record["attempt_seconds"] = time.perf_counter() - start
        record["scan_seconds"] = observer.scan_seconds
        record["scan_completed"] = observer.scan_seconds is not None
        if observer.scan_seconds is not None:
            record.setdefault("compile_seconds", {})["scan_seconds"] = clock.within(
                observer.started, observer.started + observer.scan_seconds)
        record["pre_clock"] = {"drained_arrays": observer.drained_arrays,
                               "call_args_evaluated_inside_window": observer.args_inside_window}
    trace = result.time_series if result is not None else observer.trace
    record["trace_origin"] = "result.time_series" if result is not None else "scan_output"
    record["memory"] = memory_stats(jax.devices(), process_id)
    return record, trace


def measure_run(sim, args, jax, function, distributed, clock):
    """Run lane: with --steps-short, one sim.run at steps-short, then one at --steps.

    The long call goes last: its trace is the one saved. A failed short call
    fails the repeat, and the long call is not made.
    """
    record = {}
    if args.steps_short:
        short, _ = measure(sim, args.steps_short, jax, ScanObserver(function, jax, distributed),
                           args.process_id, clock)
        for key in ("run_seconds", "scan_seconds", "status", "attempt_seconds", "pre_clock"):
            record[key + "_short"] = short.get(key)
        record["compile_seconds"] = {k + "_short": v for k, v in short.get("compile_seconds", {}).items()}
        if short["status"] != "ok":
            record.update(status=short["status"], gather_succeeded=False,
                          failure_phase="short call: " + str(short.get("failure_phase")),
                          exception=short.get("exception"), scan_seconds=None, run_seconds=None,
                          trace_origin=None, memory=short["memory"])
            return record, None
    long, trace = measure(sim, args.steps, jax, ScanObserver(function, jax, distributed),
                          args.process_id, clock)
    compile_seconds = record.pop("compile_seconds", {})
    compile_seconds.update(long.pop("compile_seconds", {}))
    record.update(long, compile_seconds=compile_seconds)
    return record, trace


def forward_design(sim, jax, np):
    """An x-sharded eps design built slab by slab from global indices, so no
    process ever holds the whole domain: 1.0 plus 1.5 inside the middle third."""
    shape, sharding = sim.distributed_override_layout(jax.devices())
    nx = sim._build_nonuniform_grid().shape[0]

    def local(index):
        lo, hi, _ = index[0].indices(shape[0])
        rows = np.arange(lo, hi)
        values = np.where((rows >= nx // 3) & (rows < 2 * nx // 3), 2.5, 1.0)
        return np.broadcast_to(values[:, None, None], (hi - lo, *shape[1:])).astype(np.float32)

    return jax.make_array_from_callback(shape, sharding, local), shape, sharding


def measure_forward(sim, args, jax, eps, clock):
    """Wall time of one forward (and, with --grad, one jax.grad) call.

    forward() compiles on every call, so these times include compilation. With
    --steps-short each repeat first runs the same calls at that step count; the
    difference of the two times over the step difference is the loop's cost per
    step without compilation.
    """
    import jax.numpy as jnp

    kwargs = dict(distributed=True, devices=jax.devices(), skip_preflight=True)
    if args.checkpoint_every:
        kwargs["checkpoint_every"] = args.checkpoint_every

    def trace_of(e, steps):
        return sim.forward(eps_override=e, n_steps=steps, **kwargs).time_series

    record = {"status": "ok", "gather_succeeded": True}
    trace, grad = None, None
    # The long run goes last. Only its trace and gradient are returned (to be
    # validated and saved): a short call's must never stand in for a failed long one.
    runs = ([("_short", args.steps_short)] if args.steps_short else []) + [("", args.steps)]
    try:
        for suffix, steps in runs:
            start = time.perf_counter()
            series = trace_of(eps, steps)
            jax.block_until_ready(series)
            book(record, "run_seconds" + suffix, start, time.perf_counter(), clock)
            if not suffix:
                trace = series
            if args.grad:
                start = time.perf_counter()
                gradient = jax.grad(lambda e: jnp.sum(trace_of(e, steps) ** 2))(eps)
                jax.block_until_ready(gradient)
                book(record, "grad_seconds" + suffix, start, time.perf_counter(), clock)
                record["grad_sharding"] = str(gradient.sharding)
                if not suffix:
                    grad = gradient
        record["trace_fully_replicated"] = bool(getattr(trace, "is_fully_replicated", False))
    except Exception as exc:
        record["exception"] = exception_record(exc)
        record["status"] = "error"
        record["failure_phase"] = "forward"
    record["scan_seconds"] = None
    record["trace_origin"] = "forward.time_series"
    record["memory"] = memory_stats(jax.devices(), args.process_id)
    return record, trace, grad


def per_step_fields(record, steps, steps_short):
    """Per-repeat derived values: the time over steps (compilation included) and
    the paired marginal of this repeat's short and long calls."""
    for field in TIMED_FIELDS:
        if record.get(field) is not None:
            record[field + "_per_step_incl_compile"] = record[field] / steps
        if steps_short and record.get(field) is not None and record.get(field + "_short") is not None:
            record[field + "_marginal_per_step"] = (
                (record[field] - record[field + "_short"]) / (steps - steps_short))


def timing_summary(runs, field, steps, steps_short):
    """Statistics of one timed quantity over the timed repeats (warm-up excluded).

    compare_multinode_probe.py applies the same function to the slowest rank of
    each repeat, so the probe and the collector report the same estimators.
    """
    timed = [r for r in runs if not r.get("warmup") and r.get(field) is not None]
    long = [r[field] for r in timed]
    summary = {"steps": steps, "values": long, "median": None, "median_per_step_incl_compile": None}
    if not long:
        return summary
    median_long = statistics.median(long)
    summary.update(median=median_long, median_per_step_incl_compile=median_long / steps)
    short = [r.get(field + "_short") for r in timed]
    paired = bool(steps_short) and all(v is not None for v in short)
    median_short = statistics.median(short) if paired else None
    rows = []
    for record, t_long, t_short in zip(timed, long, short):
        compile_seconds = record.get("compile_seconds") or {}
        row = {"index": record.get("index"), "long": t_long,
               "long_over_median": t_long / median_long if median_long else None,
               "backend_compile_long": (compile_seconds.get(field) or {}).get("backend_compile")}
        if paired:
            row.update(short=t_short,
                       short_over_median=t_short / median_short if median_short else None,
                       backend_compile_short=(compile_seconds.get(field + "_short") or {}).get(
                           "backend_compile"),
                       marginal_per_step=(t_long - t_short) / (steps - steps_short))
        row["flagged"] = any((row.get(key) or 0) > FLAG_FACTOR
                             for key in ("long_over_median", "short_over_median"))
        rows.append(row)
    summary.update(repeats=rows, flagged_repeats=[row["index"] for row in rows if row["flagged"]],
                   flag_rule=FLAG_RULE)
    if paired:
        marginal = [row["marginal_per_step"] for row in rows]
        summary.update(steps_short=steps_short, values_short=short, median_short=median_short,
                       marginal_per_step=marginal,
                       median_marginal_per_step=statistics.median(marginal),
                       min_marginal_per_step=min(marginal), max_marginal_per_step=max(marginal),
                       difference_of_medians_per_step=(median_long - median_short) / (steps - steps_short),
                       estimator=ESTIMATOR)
    elif steps_short:
        summary["reason"] = "a timed repeat has no short-call time; no marginal"
    return summary


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True)


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
    parser.add_argument("--model", choices=("vacuum", "loaded"), default="vacuum",
                        help="loaded: lossy substrate, a PEC block, a Debye and a Lorentz block, so the "
                             "material arrays are not uniform")
    parser.add_argument("--lane", choices=("run", "forward"), default="run",
                        help="run: sim.run(devices=...); forward: the differentiable "
                             "forward(distributed=True) with an x-sharded eps design built "
                             "slab by slab on each process")
    parser.add_argument("--grad", action="store_true",
                        help="forward lane: also time jax.grad of sum(trace**2) w.r.t. eps "
                             "and save each process's gradient shards")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="forward lane: segmented remat length (0 = none); every timed step "
                             "count must be a multiple of it larger than it")
    parser.add_argument("--steps-short", type=int, default=0,
                        help="each repeat first times the same call at this many steps; the paired "
                             "difference over the step difference is the per-step cost with the "
                             "per-call compilation cancelled (0 = off)")
    parser.add_argument("--local-devices", action="store_true",
                        help="one process drives every local device (e.g. a 2xA6000 node); "
                             "--nx-per-rank is per device")
    args = parser.parse_args()
    if min(args.process_count, args.steps, args.repeats) < 1:
        parser.error("process-count, steps and repeats must be positive")
    if min(args.nx_per_rank, args.ny, args.nz) < 4:
        parser.error("each slab/grid dimension must be at least 4")
    if not 0 <= args.process_id < args.process_count:
        parser.error("process-id must be in [0, process-count)")
    if args.grad and args.lane != "forward":
        parser.error("--grad needs --lane forward")
    problem = timing_plan_error(args.lane, args.steps, args.steps_short, args.checkpoint_every)
    if problem:
        parser.error(problem)
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
    environment = {k: os.environ.get(k) for k in
                   ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_COMPILATION_CACHE_DIR",
                    "CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "RFX_NODE_NAME", "NODE_NAME", "RFX_TOOLING_SHA", "RFX_PIP_JAX")}
    # Every NCCL_* and JAX_* variable: they change collectives and the compilation cache.
    environment.update({k: v for k, v in sorted(os.environ.items())
                        if k.startswith(("NCCL_", "JAX_"))})
    data = {"schema_version": 2, "tag": args.tag, "hostname": socket.gethostname(),
            "process_id": args.process_id, "process_count": args.process_count,
            "arguments": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()},
            "requested_shape": [nx, args.ny, args.nz], "runs": [], "status": "initializing",
            # Set once the device count is known: an oracle is one rank on one device.
            "oracle": None,
            "oracle_domain_note": "One process: nx-per-rank is the entire x array on ONE device; "
                                  "to match N ranks of S cells, supply nx-per-rank=N*S.",
            "timing_note": "Every call traces, lowers and compiles again. run_seconds: the public call "
                           "plus block_until_ready (setup, compilation, the loop and the native gather). "
                           "scan_seconds (run lane): from the anchored scan call, after every JAX array "
                           "in the runner frame is ready, to its outputs being ready; it includes that "
                           "call's own tracing, lowering and compilation. compile_seconds: the seconds "
                           "of each compile stage JAX reported inside each window. With steps_short each "
                           "repeat times steps_short, then steps; *_marginal_per_step pairs the two. "
                           "Warm-up excluded from every summary. The scoped Python line observer adds "
                           "overhead. No kernel cache replacement.",
            "environment": environment}
    write_json(path, data)
    initialized = False
    exit_code = 1
    try:
        import jax
        import jaxlib
        import numpy as np
        from jax import monitoring

        clock = CompileClock(monitoring)
        data.update(jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
                    compile_events=sorted(COMPILE_EVENTS))
        try:
            data["jax_compilation_cache_dir"] = getattr(jax.config, "jax_compilation_cache_dir", None)
        except Exception as exc:
            data["jax_compilation_cache_dir"] = f"unreadable: {exc}"
        write_json(path, data)
        if args.process_count > 1:
            jax.distributed.initialize(coordinator_address=args.coordinator_address,
                                       num_processes=args.process_count,
                                       process_id=args.process_id,
                                       local_device_ids=[args.local_device_id],
                                       initialization_timeout=900)
            initialized = True
        if args.local_devices:
            assert args.process_count == 1, "--local-devices is a one-process mode"
            n_ranks = len(jax.devices())
        else:
            assert len(jax.local_devices()) == 1, str(jax.local_devices())
            assert len(jax.devices()) == args.process_count, str(jax.devices())
            n_ranks = args.process_count
        assert jax.process_index() == args.process_id
        nx = args.nx_per_rank * n_ranks
        data["requested_shape"] = [nx, args.ny, args.nz]
        data["n_ranks"] = n_ranks
        data["oracle"] = n_ranks == 1
        data["devices"] = memory_stats(jax.devices(), args.process_id)
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        import rfx
        # The commit and rfx tree recorded below describe the checkout at root;
        # refuse to measure an rfx imported from anywhere else.
        rfx_module = Path(rfx.__file__).resolve()
        data["rfx_module"] = str(rfx_module)
        if root not in rfx_module.parents:
            raise RuntimeError(f"rfx was imported from {rfx_module}, not from the checkout {root}")
        from rfx import Simulation
        from rfx.runners.distributed_v2 import run_distributed
        from rfx.simulation import run as run_single

        data["commit"] = git(root, "rev-parse", "HEAD").strip()
        data["rfx_tree"] = git(root, "rev-parse", "HEAD:rfx").strip()
        # rfx_tree is the committed tree; rfx_dirty counts the paths under rfx/ that
        # differ from it (`git status --porcelain -- rfx`), so 0 means it ran as recorded.
        data["rfx_dirty"] = len(git(root, "--no-optional-locks", "status", "--porcelain",
                                    "--", "rfx").splitlines())
        data["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        mesh_kwargs = {}
        if args.lane == "forward":
            # forward(distributed=True) is the non-uniform lane; an explicit
            # (uniform-valued) z profile selects it without changing the cells.
            mesh_kwargs["dz_profile"] = np.full(args.nz - 1, 1e-3)
        sim = Simulation(freq_max=15e9, domain=tuple((n - 1) * 1e-3 for n in (nx, args.ny, args.nz)),
                         dx=1e-3, boundary="pec", precision="float32", **mesh_kwargs)
        source = (nx * 0.5e-3, args.ny * 0.5e-3, args.nz * 0.5e-3)
        if args.model == "loaded":
            from rfx import Box, DebyePole
            from rfx.materials.lorentz import lorentz_pole
            lx, ly, lz = ((n - 1) * 1e-3 for n in (nx, args.ny, args.nz))
            def box(x0, x1, y0, y1, z0, z1):
                return Box((x0 * lx, y0 * ly, z0 * lz), (x1 * lx, y1 * ly, z1 * lz))
            sim.add_material("substrate", eps_r=4.4, sigma=0.02)
            sim.add(box(0, 1, 0, 1, 0, 0.3), material="substrate")
            sim.add(box(0.1, 0.2, 0.2, 0.3, 0.4, 0.5), material="pec")
            sim.add_material("debye_block", eps_r=2.0,
                             debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
            sim.add(box(0.6, 0.7, 0.6, 0.7, 0.6, 0.7), material="debye_block")
            sim.add_material("lorentz_block", eps_r=2.0, lorentz_poles=[
                lorentz_pole(delta_eps=1.0, omega_0=2 * np.pi * 3e9, delta=1e9)])
            sim.add(box(0.3, 0.4, 0.6, 0.7, 0.6, 0.7), material="lorentz_block")
        data["model"] = args.model
        sim.add_source(source, "ez", amplitude_kind="field")
        probes = [(nx * fraction * 1e-3, source[1], source[2]) for fraction in (0.25, 0.5, 0.75)]
        for position in probes:
            sim.add_probe(position, "ez")
        grid = sim._build_nonuniform_grid() if args.lane == "forward" else sim._build_grid()
        data.update(lane=args.lane, grad=args.grad, checkpoint_every=args.checkpoint_every,
                    steps=args.steps, steps_short=args.steps_short)
        data.update(realized_shape=list(grid.shape), boundary="pec", dtype="float32",
                    dx_m=1e-3, dt_seconds=float(grid.dt), freq_max_hz=15e9,
                    source_position_m=source, probe_positions_m=probes,
                    waveform="default GaussianPulse(f0=7.5e9, bandwidth=0.8)",
                    domain_m=list(sim._domain))
        assert tuple(grid.shape) == (nx, args.ny, args.nz), str(grid.shape)
        distributed = n_ranks > 1
        function = run_distributed if distributed else run_single
        data["status"] = "running"
        design = None
        if args.lane == "forward":
            design, design_shape, design_sharding = forward_design(sim, jax, np)
            data.update(design_shape=list(design_shape), design_sharding=str(design_sharding))
        for index in range(args.repeats + 1):
            grad = None
            if args.lane == "forward":
                record, trace, grad = measure_forward(sim, args, jax, design, clock)
            else:
                record, trace = measure_run(sim, args, jax, function, distributed, clock)
            record.update(index=index, warmup=index == 0)
            per_step_fields(record, args.steps, args.steps_short)
            if grad is not None and index == args.repeats:
                # Each process saves its own gradient shards with their global
                # x ranges; compare_multinode_probe.py assembles and compares them.
                shards = []
                for shard in grad.addressable_shards:
                    lo, hi, _ = shard.index[0].indices(grad.shape[0])
                    name = f"{args.tag}.grad.x{lo:06d}-{hi:06d}.npy"
                    np.save(args.output / name, np.asarray(shard.data), allow_pickle=False)
                    shards.append({"file": name, "x": [lo, hi], "device": shard.device.id})
                record["grad_shards"] = shards
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
        for field in TIMED_FIELDS:
            data[field] = timing_summary(data["runs"], field, args.steps, args.steps_short)
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
