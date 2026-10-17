"""Per-feature CPU step-cost census, Design 20261017, Amendments 1-6.

Quantities (seconds unless explicitly ns; MAD is median absolute deviation):
S = 200; R = 7; t_step = median / S. Compile once, discard one warm run
of each executable, then interleave A, B, A, B, ... for 7+7 timed runs,
each ended by block_until_ready. Raw run seconds are retained; MAD and
medians are divided by S before any per-step comparison.
σ_row = 1.4826 × MAD of that row's 7 feature runs; σ_base likewise from
the 7 base runs interleaved with them.
σ_Δ = sqrt(σ_row² + σ_base²).
Added time Δt(N) = t_step(feature, N) − t_step(base, N).
Per-footprint cost: Δt(N) / F, in ns per step per footprint cell.
Δt_f(N) = max(Δt(N), 3σ_Δ(N)).
s = least-squares slope of log Δt_f against log C over the three sizes.
s_F is always the least-squares slope of log F against log C, using the
footprint cell counts the run actually realizes at the three sizes.
Excess e = s − s_F.
N = 56, 88, 144 interior cells; C = (N+16)³ = 373248, 1124864, 4096000.
For the CPML row, F = C − N³; its base is the same interior PEC box.
(a): e > 0.3, and Δt(144) − Δt(56)·(F(144)/F(56)) > 3σ_Δ(144).
Below noise: Δt is below 3σ_Δ at every size; timing (a) is ignored.
Trust: if σ_base(144) > 5 % of t_step(base, 144) on any row, report the
table on (b), without re-running with other settings; (c) is reported only.
(b): for any op kind, the domain-sized op count at 2F exceeds the count
at F (at N = 88). Count EVERY compiled-HLO instruction whose output has
at least C elements, by opcode. Excluded opcodes: parameter, tuple,
get-tuple-element, bitcast and constant. Inspect every computation,
including fusion bodies; count a tuple-producing instruction once if any
of its output arrays has at least C elements. Report absolute counts and
feature minus base counts. Only outputs are inspected, not input operands.
Amendment 6, multi-device only: replace C in the domain-sized threshold
with the per-device field shard's cell count, read from the compiled
program's field-shard shape. Review correction: single-device threshold
is the product of (compiled field dimension - 1), the realized cell count;
it equals C for full-grid scenes. C in the slopes remains unchanged.
A distributed result with devices_used=1 is labelled "falls back (one device)";
quote its diagnostic warning and record no timing samples.
The former four-op rule is retained as a diagnostic, not used for (b).
(c): XLA cost_analysis()["bytes accessed"] of the compiled step.
ΔB(N) = feature − base; s_c = least-squares slope of log ΔB against log C;
e_c = s_c − s_F. (c) is reported in every row, never flags.
Nonpositive ΔB has no logarithm: report s_c/e_c as null, without a floor
or an invented epsilon. (b) is PRIMARY; timing (a) is WITNESS.
A row where (b) and (a) disagree is listed as "disagree"; neither rule
resolves it alone. Otherwise (b) true is "flagged", (b) false is "none".
Ignored timing (below noise or untrusted) is null, not a negative witness.
An unavailable exact-2F comparison leaves (b) and the flag unavailable;
its message is retained under "unavailable 2F". No footprint is invented.
The scan counting convention is measured by compiling the same scan body
at 2 and 200 trips and also compiling its body alone. Raw cost dictionaries,
body counts and differences are retained. B is the compiler's whole-program
estimate, including setup/condition/output costs; it is not hardware traffic
and is not divided by S. Scan multiplicity is reported from the calibration,
not assumed. The control footprint is 8 cells (2F = 16).
Memory total = argument_size_in_bytes + output_size_in_bytes +
temp_size_in_bytes − alias_size_in_bytes, plus the same four host terms.
Compile seconds and memory_analysis total are records, not statistics.
Multi-device executable_call_seconds records each cached executable call
through block_until_ready separately (including compile-entry and warm calls);
interleaved executable_seconds identifies the seven timed samples per side.
Existing t_step still measures the whole public forward entry.

The injected control is test-only: per footprint cell and per E component,
an identity gather copies the full field, with optimization barriers on
its indices and result. Opaque indices prevent an identity rewrite or an
in-place update: a general gather cannot overwrite unread input elements.
This copies every E value once without changing any electric field values.
Lowered AND optimized counts are recorded: an optimizer removing a copy
must remain visible in the evidence, never be assumed away.

CLI defaults to STAGE 0 only. No timings are allowed on macOS. No gradient
or full-table run is implicit. Multi-device uses direct forward calls and
captures/reuses the runners' own compiled executables; no enclosing jit.
Optional start/end reference bases retain seven runs per size at each end.
They do not replace the interleaved bases or enter the Amendment 1 noise.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
from statistics import median
import sys
import time
from unittest.mock import patch

S = 200
R = 7
SIZES = (56, 88, 144)
KINDS = ("scatter", "dynamic-update-slice", "copy", "fusion")
EXCLUDED_OPCODES = frozenset(("parameter", "tuple", "get-tuple-element", "bitcast", "constant"))


def mad_sigma(values):
    centre = median(values)
    return 1.4826 * median(abs(x - centre) for x in values)


def pair_statistics(base_seconds, row_seconds, steps=S):
    if len(base_seconds) != R or len(row_seconds) != R:
        raise ValueError("each interleaved pair requires exactly 7+7 runs")
    if steps <= 0 or any(not math.isfinite(x) or x < 0 for x in [*base_seconds, *row_seconds]):
        raise ValueError("run times must be finite and nonnegative; steps positive")
    tb, tr = median(base_seconds) / steps, median(row_seconds) / steps
    sb, sr = mad_sigma(base_seconds) / steps, mad_sigma(row_seconds) / steps
    sd = math.hypot(sb, sr)
    delta = tr - tb
    return dict(base_runs_seconds=list(base_seconds), row_runs_seconds=list(row_seconds),
                base_median_seconds=median(base_seconds), row_median_seconds=median(row_seconds),
                t_base_seconds=tb, t_row_seconds=tr, sigma_base_seconds=sb,
                sigma_row_seconds=sr, sigma_delta_seconds=sd,
                delta_seconds=delta, floored_delta_seconds=max(delta, 3 * sd))


def log_slope(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("slope needs matching sequences of at least two values")
    if any(x <= 0 or not math.isfinite(x) for x in [*xs, *ys]):
        # No undeclared epsilon: a zero-MAD, nonpositive delta has no log.
        return None
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    denominator = sum((x - mx)**2 for x in lx)
    if not denominator:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / denominator


def classify(points, counts_f, counts_2f, *, timing_trusted=True):
    points = sorted(points, key=lambda p: p["N"])
    if [p["N"] for p in points] != list(SIZES):
        raise ValueError("classification requires the three Design sizes 56, 88, 144")
    s = log_slope([p["C"] for p in points], [p["floored_delta_seconds"] for p in points])
    sf = log_slope([p["C"] for p in points], [p["F"] for p in points])
    excess = None if s is None or sf is None else s - sf
    below = all(p["delta_seconds"] < 3 * p["sigma_delta_seconds"] for p in points)
    small, large = points[0], points[-1]
    growth = large["delta_seconds"] - small["delta_seconds"] * large["F"] / small["F"]
    a_raw = bool(not below and excess is not None and excess > .3 and
                 growth > 3 * large["sigma_delta_seconds"])
    growing = None if counts_f is None or counts_2f is None else sorted(
        k for k in counts_f.keys() | counts_2f.keys() if counts_2f.get(k, 0) > counts_f.get(k, 0))
    b = None if growing is None else bool(growing)
    byte_deltas = [p.get("delta_bytes_accessed") for p in points]
    sc = None if any(x is None for x in byte_deltas) else log_slope([p["C"] for p in points], byte_deltas)
    ec = None if sc is None or sf is None else sc - sf
    a = a_raw if timing_trusted and not below else None
    disagree = b is not None and a is not None and b != a
    flag = "unavailable 2F" if b is None else "disagree" if disagree else "flagged" if b else "none"
    return dict(s=s, s_F=sf, e=excess, below_noise=below,
                growth_seconds=growth, timing_trusted=timing_trusted,
                timing_clause_raw=a_raw, a=a, b=b, growing_opcodes=growing,
                s_c=sc, e_c=ec, c_report_only=True, disagree=disagree,
                flag=flag,
                conclusion="리더가 채움")


def _tensor_sizes(text):
    return [math.prod(map(int, dims.rstrip("x").split("x")))
            for dims in re.findall(r"tensor<((?:\d+x)+)(?:[a-z]\w*)", text)]


def structural_counts(lowered, compiled, cells):
    counts = dict.fromkeys(KINDS, 0)
    # The contract's scatter block regex, narrowed to the output signature.
    for block in re.findall(r'"stablehlo.scatter"[\s\S]*?->[^\n]*', lowered):
        if any(size >= cells for size in _tensor_sizes(block.rsplit("->", 1)[1])):
            counts["scatter"] += 1
    for line in lowered.splitlines():
        if re.search(r"stablehlo\.dynamic_update_slice\b", line):
            output = line.rsplit("->", 1)[-1] if "->" in line else line.rsplit(":", 1)[-1]
            if any(size >= cells for size in _tensor_sizes(output)):
                counts["dynamic-update-slice"] += 1
    for line in compiled.splitlines():
        match = re.search(r"=\s*(.*?)\b(copy|fusion)\(", line)
        if match:
            # Tuple results count if any output array has C elements, never
            # by adding many unrelated small outputs together.
            dims = re.findall(r"\b(?:[a-z]+\d*|pred)\[([\d,]*)\]", match[1])
            if any(math.prod(int(x) for x in d.split(",") if x) >= cells for d in dims):
                counts[match[2]] += 1
    return counts


def subtract_counts(feature, base):
    return {k: feature.get(k, 0) - base.get(k, 0) for k in sorted(feature.keys() | base.keys())}


def compiled_opcode_counts(compiled, cells):
    """Count instruction definitions, including nested computation bodies."""
    counts = {}
    for line in compiled.splitlines():
        match = re.match(r"\s*(?:ROOT\s+)?%?[\w.-]+\s*=\s*(.*?)\s+([a-z][a-z0-9-]*)\(", line)
        if not match or match[2] in EXCLUDED_OPCODES:
            continue
        dims = re.findall(r"\b(?:[a-z]+\d*|pred)\[([\d,]*)\]", match[1])
        if any(math.prod(int(x) for x in d.split(",") if x) >= cells for d in dims):
            counts[match[2]] = counts.get(match[2], 0) + 1
    return dict(sorted(counts.items()))


def field_shard_record(compiled_text):
    """Read all six field parameters in partitioned, optimized entry HLO."""
    entry = False
    fields = {}
    evidence = []
    for line in compiled_text.splitlines():
        if line.startswith("ENTRY "):
            entry = True
        if not entry:
            continue
        match = re.search(r"= (?:f|bf|c)\d+\[([0-9,]+)\].*?parameter\(\d+\).*?op_name=\"([^\"]+)\"", line)
        if not match:
            continue
        component = re.search(r"\.(ex|ey|ez|hx|hy|hz)$", match[2])
        if component:
            shape = tuple(map(int, match[1].split(",")))
            if len(shape) != 3:
                raise RuntimeError(f"field shard must have rank three: {line}")
            fields.setdefault(component[1], set()).add(shape)
            evidence.append(line.strip())
    if set(fields) != {"ex", "ey", "ez", "hx", "hy", "hz"}:
        raise RuntimeError("compiled entry does not identify all six field-shard parameters")
    shapes = set().union(*fields.values())
    if len(shapes) != 1:
        raise RuntimeError(f"compiled field-shard shapes disagree: {shapes}")
    shape = next(iter(shapes))
    return {"shape": list(shape), "cells": math.prod(shape), "parameter_instructions": evidence}


def single_device_field_record(compiled_text):
    """Read the full-field shape from the optimized scan carry.

    The six Yee fields share a rank-three shape dominating the CPML slabs.
    Require six occurrences and a unique componentwise maximal shape; never
    infer a field from an arbitrary largest temporary or from nominal N.
    """
    candidates, evidence = set(), []
    for line in compiled_text.splitlines():
        if " while(" not in line:
            continue
        signature = line.split(" while(", 1)[0].split("=", 1)[-1]
        shapes = [tuple(map(int, d.split(","))) for d in
                  re.findall(r"\b(?:f|bf|c)\d+\[(\d+,\d+,\d+)\]", signature)]
        for shape in set(shapes):
            if shapes.count(shape) >= 6 and all(
                    all(a >= b for a, b in zip(shape, other)) for other in shapes):
                candidates.add(shape)
                evidence.append(line.strip())
    if len(candidates) != 1:
        raise RuntimeError(f"compiled scan does not identify one six-field shape: {candidates}")
    shape = next(iter(candidates))
    if min(shape) <= 1:
        raise RuntimeError(f"field dimensions must exceed one: {shape}")
    return {"shape": list(shape), "cells": math.prod(d - 1 for d in shape),
            "carry_instructions": evidence}


def domain_size_threshold(compiled_text, cells, *, distributed=False):
    # cells remains the nominal C used by callers for slopes, not this count.
    record = field_shard_record if distributed else single_device_field_record
    return record(compiled_text)["cells"]


class OneDeviceFallback(Exception):
    """A returned product fallback, distinct from refusal or a timed row."""


def check_distributed_result(result):
    for diagnostic in getattr(result, "diagnostics", ()):
        if diagnostic.values.get("devices_used") == 1:
            raise OneDeviceFallback(diagnostic.message)


def fallback_record(exc):
    return {"status": "falls back (one device)", "devices_used": 1,
            "warning": str(exc), "message": str(exc), "conclusion": "리더가 채움"}


def cost_record(compiled):
    raw = compiled.cost_analysis()
    # Older JAX releases return a singleton list. Never sum per-device
    # dictionaries, which may describe replicated rather than distinct work.
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError("cost_analysis must expose one executable cost dictionary")
        raw = raw[0]
    raw = {str(k): float(v) for k, v in raw.items()}
    value = raw["bytes accessed"]
    if not math.isfinite(value) or value < 0:
        raise ValueError("bytes accessed must be finite and nonnegative")
    return {"bytes_accessed": value, "raw": raw}


def scan_cost_calibration():
    """Measure this compiler's scan accounting, without executing a timing run."""
    import jax
    import jax.numpy as jnp

    def body(x):
        return jax.lax.optimization_barrier(x) + jnp.float32(1)

    shape = jax.ShapeDtypeStruct((4096,), jnp.float32)
    body_executable = jax.jit(body).lower(shape).compile()
    body_record = cost_record(body_executable)
    scans = []
    for trips in (2, S):
        def scanned(x):
            return jax.lax.scan(lambda carry, _: (body(carry), None), x, None, length=trips)[0]
        executable = jax.jit(scanned).lower(shape).compile()
        counts = compiled_opcode_counts(executable.as_text(), 4096)
        record = cost_record(executable)
        scans.append(dict(trips=trips, **record, counts=counts,
                          bytes_minus_standalone_body=record["bytes_accessed"] - body_record["bytes_accessed"]))
    independent = scans[0]["bytes_accessed"] == scans[1]["bytes_accessed"]
    one_body = all(r["counts"].get("while") == 1 and r["counts"].get("add") == 1 for r in scans)
    return {"array_elements": 4096, "body": body_record, "scans": scans,
            "trip_count_independent": independent, "one_body_definition": one_body,
            "scan_body_counted_once_observed": independent and one_body,
            "interpretation": "Costs include one body plus loop overhead; no division by trip count."
            if independent and one_body else "Calibration did not establish once-per-body accounting.",
            "conclusion": "리더가 채움"}


def memory_record(compiled):
    memory = compiled.memory_analysis()
    if memory is None:
        return {"total_bytes": None, "reason": "memory_analysis returned None"}
    names = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
    fields = {prefix + name: int(getattr(memory, prefix + name, 0))
              for prefix in ("", "host_") for name in names}
    total = sum(fields[prefix + names[0]] + fields[prefix + names[1]] + fields[prefix + names[2]]
                - fields[prefix + names[3]] for prefix in ("", "host_"))
    return dict(total_bytes=total, components=fields)


@contextmanager
def injected_control(footprint, path):
    """Patch only in-process Python references; never mutate product files."""
    if not footprint:
        yield
        return
    import jax
    import jax.numpy as jnp
    module = importlib.import_module(
        "rfx.stepping.graded" if path == "multi-graded" else
        "rfx.runners._distributed_common" if path.startswith("multi") else
        "rfx.stepping.graded" if "graded" in path else "rfx.simulation")
    original = module.inject_drives

    def inject(state, drives, values):
        state = original(state, drives, values)
        for _ in range(footprint):
            updates = {}
            for name in ("ex", "ey", "ez"):
                field = getattr(state, name)
                indices = jax.lax.optimization_barrier(jnp.arange(field.shape[0], dtype=jnp.int32))
                # A runtime-index gather requires a distinct full-field result.
                # Its identity indices preserve the exact field bits. Barriers
                # prevent adjacent copies from being composed or eliminated.
                updates[name] = jax.lax.optimization_barrier(jnp.take(field, indices, axis=0, mode="clip"))
            state = state._replace(**updates)
        return state

    with patch.object(module, "inject_drives", inject):
        yield


class Program:
    def __init__(self, scene, path, cells, control_cells=0):
        import jax
        self.scene, self.path, self.cells = scene, path, cells
        self.control_cells = control_cells
        self.record = {}
        self.cache = {}
        self.compiled = None
        self.original_jit = jax.jit
        self.distributed = path.startswith("multi")

    def _record_executable(self, lowered, executable, seconds):
        lt, ct = lowered.as_text(), executable.as_text()
        threshold = domain_size_threshold(ct, self.cells, distributed=self.distributed)
        record = {"compile_seconds": seconds, "memory_analysis": memory_record(executable),
                "counts": compiled_opcode_counts(ct, threshold),
                "legacy_counts": structural_counts(lt, ct, threshold),
                "cost_analysis": cost_record(executable),
                "lowered_sha256": hashlib.sha256(lt.encode()).hexdigest(),
                "compiled_sha256": hashlib.sha256(ct.encode()).hexdigest()}
        if self.distributed:
            record["field_shard"] = field_shard_record(ct)
        else:
            record["realized_field"] = single_device_field_record(ct)
        record["domain_size_threshold"] = threshold
        return record

    def _capture_jit(self, fun=None, **kwargs):
        if fun is None:
            return lambda f: self._capture_jit(f, **kwargs)
        compiled_fn = self.original_jit(fun, **kwargs)
        legacy_runner = (fun.__module__ in ("rfx.runners.distributed_v2", "rfx.runners.distributed_nu")
                         and fun.__name__ in ("_run_cpml", "_run_pec", "run_fn"))
        wrapped_graded_runner = (fun.__module__, fun.__name__) == ("rfx.stepping.slab_wrap", "run_fn")
        if not (legacy_runner or wrapped_graded_runner):
            return compiled_fn

        def call(*args, **kw):
            key = fun.__code__
            if key not in self.cache:
                t0 = time.perf_counter()
                lowered = compiled_fn.lower(*args, **kw)
                executable = lowered.compile()
                elapsed = time.perf_counter() - t0
                self.cache[key] = executable
                self.record = self._record_executable(lowered, executable, elapsed)
                self.record["runner"] = {"module": fun.__module__, "function": fun.__name__}
            import jax
            started = time.perf_counter()
            result = self.cache[key](*args, **kw)
            jax.block_until_ready(result)
            seconds = time.perf_counter() - started
            self.last_executable_seconds = seconds
            self.record.setdefault("executable_call_seconds", []).append(seconds)
            return result
        return call

    def compile(self):
        import jax
        if self.distributed:
            # Direct public entry: compile/capture its own executable once.
            self.call()
            if not self.cache:
                raise RuntimeError("distributed entry did not expose a two-device runner executable")
        else:
            with injected_control(self.control_cells, self.path):
                t0 = time.perf_counter()
                def forward_arrays():
                    result = self.scene.sim.forward(
                        n_steps=S, skip_preflight=True, **self.scene.forward_kwargs)
                    # ForwardResult also contains static Grid/NTFF geometry
                    # objects and strings. Keep every array observable so
                    # feature accumulators stay live, but no host metadata
                    # crosses the JIT output boundary.
                    return tuple(x for x in jax.tree.leaves(result)
                                 if isinstance(x, (jax.Array, jax.core.Tracer)))
                lowered = jax.jit(forward_arrays).lower()
                self.compiled = lowered.compile()
                elapsed = time.perf_counter() - t0
            self.record = self._record_executable(lowered, self.compiled, elapsed)

    def call(self):
        import jax
        if self.distributed:
            with patch.object(jax, "jit", self._capture_jit), injected_control(self.control_cells, self.path):
                result = self.scene.sim.forward(n_steps=S, skip_preflight=True,
                                                distributed=True, devices=jax.devices("cpu")[:2],
                                                **self.scene.forward_kwargs)
            check_distributed_result(result)
        else:
            result = self.compiled()
        jax.block_until_ready(result)
        return result


def _scenes():
    if __package__:
        from . import _census_scenes
    else:
        import _census_scenes
    return _census_scenes


def _refusal(exc):
    return {"status": "refuses", "exception": type(exc).__name__, "message": str(exc),
            "conclusion": "리더가 채움"}


class Unavailable2F(Exception):
    """Harness builder/footprint limitation, never a product refusal."""


def unavailable_2f_record(exc):
    return {"status": "unavailable 2F", "exception": type(exc).__name__,
            "message": str(exc), "conclusion": "리더가 채움"}


def is_api_refusal(exc):
    if not (isinstance(exc, (ValueError, NotImplementedError)) or (
            isinstance(exc, TypeError) and "Simulation.forward() got unexpected keyword" in str(exc))):
        return False
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    return Path(tb.tb_frame.f_code.co_filename).resolve().is_relative_to(
        Path(__file__).resolve().parents[2] / "rfx")


def build_scene(builder, *args, **kwargs):
    try:
        return builder(*args, **kwargs)
    except NotImplementedError as exc:
        if is_api_refusal(exc):
            raise
        raise Unavailable2F(str(exc)) from exc


def run_pair(row, n, path):
    import jax
    scenes = _scenes()
    started = time.perf_counter()
    cells = (n + 16)**3
    print(f"BEGIN row={row} N={n} C={cells}", flush=True)
    base_scene = scenes.pec_base(n, path) if row == "cpml" else scenes.base(n, path)
    scene = build_scene(scenes.BUILDERS[row], n, path)
    geometry, base_geometry = scene.realized(), base_scene.realized()
    if geometry["interior_cells"] != [n] * 3:
        raise RuntimeError(f"unexpected interior realization: {geometry}")
    base_program, program = Program(base_scene, path, cells), Program(scene, path, cells, scene.footprint if row == "control" else 0)
    base_program.compile()
    program.compile()
    # Both are compiled before either warm-up. Warm-up is not one of the 7.
    warm_start = time.perf_counter()
    base_program.call()
    base_warm_seconds = time.perf_counter() - warm_start
    warm_start = time.perf_counter()
    program.call()
    row_warm_seconds = time.perf_counter() - warm_start
    base_runs, row_runs, interleaved = [], [], []
    for i in range(R):
        for label, runner, runs in (("A", base_program, base_runs), ("B", program, row_runs)):
            t0 = time.perf_counter()
            runner.call()
            elapsed = time.perf_counter() - t0
            runs.append(elapsed)
            sample = {"pair": i, "side": label, "seconds": elapsed}
            if runner.distributed:
                sample["executable_seconds"] = runner.last_executable_seconds
            interleaved.append(sample)
            print(f"RUN row={row} N={n} pair={i} side={label} seconds={elapsed:.9f}", flush=True)
    pair_wall = time.perf_counter() - started
    result = dict(N=n, C=cells, F=scene.footprint, status="ok", geometry=geometry,
                  base_geometry=base_geometry, **pair_statistics(base_runs, row_runs),
                  interleaved=interleaved, base_program=base_program.record, row_program=program.record,
                  pair_wall_seconds=pair_wall,
                  base_warm_seconds=base_warm_seconds, row_warm_seconds=row_warm_seconds,
                  row_compile_plus_warm_and_7_seconds=program.record["compile_seconds"] + row_warm_seconds + sum(row_runs))
    result["ns_per_step_per_footprint_cell"] = result["delta_seconds"] * 1e9 / scene.footprint
    result["delta_bytes_accessed"] = (program.record["cost_analysis"]["bytes_accessed"]
                                     - base_program.record["cost_analysis"]["bytes_accessed"])
    if n == 88:
        f_delta = subtract_counts(program.record["counts"], base_program.record["counts"])
        structural = {"F": {"footprint": scene.footprint, "absolute": program.record["counts"], "minus_base": f_delta},
                      "base": base_program.record["counts"]}
        try:
            doubled = build_scene(scenes.BUILDERS[row], n, path, scale=2)
            doubled_geometry = doubled.realized()
            if row == "base":
                # A reference has no feature to double; both comparisons are 0.
                doubled.footprint = 2 * scene.footprint
            elif doubled.footprint != 2 * scene.footprint:
                raise Unavailable2F(f"2F realization is {doubled.footprint}, expected {2 * scene.footprint}")
            p2 = Program(doubled, path, cells, doubled.footprint if row == "control" else 0)
            p2.compile()
            structural["2F"] = {"footprint": doubled.footprint, "geometry": doubled_geometry,
                                "absolute": p2.record["counts"],
                                "minus_base": subtract_counts(p2.record["counts"], base_program.record["counts"]),
                                "program": p2.record}
            del p2, doubled
        except Unavailable2F as exc:
            structural["2F"] = unavailable_2f_record(exc)
        except OneDeviceFallback as exc:
            structural["2F"] = fallback_record(exc)
        except (ValueError, NotImplementedError, TypeError) as exc:
            if not is_api_refusal(exc):
                raise
            structural["2F"] = _refusal(exc)
        result["structure"] = structural
    print(f"END row={row} N={n} pair_wall_seconds={pair_wall:.6f}", flush=True)
    del program, base_program, scene, base_scene
    jax.clear_caches()
    gc.collect()
    return result


def finish_report(report):
    largest = [p for row in report["rows"].values() for p in row["sizes"]
               if p["N"] == 144 and p["status"] == "ok"]
    trusted = not any(p["sigma_base_seconds"] > .05 * p["t_base_seconds"] for p in largest)
    report["timing_trusted"] = trusted if largest else None
    for row in report["rows"].values():
        points = [p for p in row["sizes"] if p["status"] == "ok"]
        if [p["N"] for p in points] != list(SIZES):
            continue
        structure = points[1].get("structure", {})
        f = structure.get("F", {}).get("minus_base")
        f2 = structure.get("2F", {}).get("minus_base")
        row["statistics"] = classify(points, f, f2, timing_trusted=trusted)
        for p in points:
            p["sigma_base_fraction_of_t_base"] = p["sigma_base_seconds"] / p["t_base_seconds"]
    control = report["rows"].get("control", {}).get("statistics")
    if control:
        report["control_flagged"] = control["flag"] == "flagged"
        report["control_b"] = control["b"]


def run_reference(n, path, phase):
    """Record the Design's start/end base; never pool it into row noise."""
    import jax
    scene = _scenes().base(n, path)
    geometry = scene.realized()
    program = Program(scene, path, (n + 16)**3)
    program.compile()
    program.call()
    runs = []
    for i in range(R):
        started = time.perf_counter()
        program.call()
        runs.append(time.perf_counter() - started)
        print(f"REFERENCE phase={phase} N={n} run={i} seconds={runs[-1]:.9f}", flush=True)
    t_step = median(runs) / S
    sigma = mad_sigma(runs) / S
    result = dict(N=n, C=(n + 16)**3, phase=phase, status="ok", geometry=geometry,
                  runs_seconds=runs, median_seconds=median(runs), t_step_seconds=t_step,
                  sigma_seconds=sigma, sigma_fraction_of_t_step=sigma / t_step,
                  program=program.record, conclusion="리더가 채움")
    del program, scene
    jax.clear_caches()
    gc.collect()
    return result


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("uniform", "graded", "multi-uniform", "multi-graded"), default="uniform")
    parser.add_argument("--rows", nargs="+", default=["base", "control", "wire"])
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true", help="compile, warm and time base at N=56 once")
    parser.add_argument("--reference-bases", action="store_true",
                        help="record seven start/end base runs per size; row noise remains interleaved")
    args = parser.parse_args(argv)
    if platform.system() == "Darwin":
        parser.error("timing is cluster-only; macOS may run the synthetic unit tests")
    if not hasattr(os, "sched_getaffinity"):
        parser.error("Linux CPU affinity is required")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 16:
        parser.error("the Design requires a fixed 16-CPU pin")
    os.sched_setaffinity(0, affinity[:16])
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "false"
    if args.path.startswith("multi"):
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    import jax
    import rfx
    scenes = _scenes()
    unknown = set(args.rows) - scenes.BUILDERS.keys()
    if unknown or not set(args.sizes) <= set(SIZES):
        parser.error(f"unknown rows {sorted(unknown)} or sizes outside {SIZES}")
    if args.path.startswith("multi") and len(jax.devices("cpu")) != 2:
        parser.error("multi-device path needs exactly two virtual CPU devices")
    report = {"schema": 3, "design_amendments": [1, 2, 3, 4, 5] + ([6] if args.path.startswith("multi") else []), "path": args.path, "steps": S, "repeats": R,
              "job": os.environ.get("REMI_JOB"), "jax_version": jax.__version__,
              "python": sys.version, "rfx_import": rfx.__file__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "hostname": platform.node(), "platform": platform.platform(),
              "source_base_commit": os.environ.get("CENSUS_BASE_COMMIT"),
              "source_tar_sha256": os.environ.get("CENSUS_TAR_SHA256"),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scenes_sha256": hashlib.sha256(Path(scenes.__file__).read_bytes()).hexdigest(),
              "rows": {}, "conclusion": "리더가 채움"}
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False), flush=True)
    t0 = time.perf_counter()
    if args.smoke:
        scene = scenes.base(56, args.path)
        program = Program(scene, args.path, 72**3)
        program.compile()
        program.call()
        t1 = time.perf_counter()
        program.call()
        seconds = time.perf_counter() - t1
        report["smoke"] = {"N": 56, "C": 72**3, "run_seconds": seconds,
                           "t_step_seconds": seconds / S, "program": program.record}
        print("SMOKE " + json.dumps(report["smoke"]), flush=True)
    else:
        report["scan_cost_calibration"] = scan_cost_calibration()
        print("SCAN_COST_CALIBRATION " + json.dumps(report["scan_cost_calibration"], ensure_ascii=False), flush=True)
        if args.reference_bases:
            report["reference_bases"] = {"start": [], "end": []}
            for n in sorted(set(args.sizes)):
                report["reference_bases"]["start"].append(run_reference(n, args.path, "start"))
                save_report(report, args.output)
        for row in args.rows:
            report["rows"][row] = {"sizes": []}
            for n in sorted(set(args.sizes)):
                try:
                    result = run_pair(row, n, args.path)
                except Unavailable2F as exc:
                    result = dict(N=n, C=(n+16)**3, **unavailable_2f_record(exc))
                except OneDeviceFallback as exc:
                    result = dict(N=n, C=(n+16)**3, **fallback_record(exc))
                    print("FALLBACK " + json.dumps(result, ensure_ascii=False), flush=True)
                    jax.clear_caches()
                    gc.collect()
                except (ValueError, NotImplementedError, TypeError) as exc:
                    if not is_api_refusal(exc):
                        raise
                    result = dict(N=n, C=(n+16)**3, **_refusal(exc))
                    print("REFUSES " + json.dumps(result, ensure_ascii=False), flush=True)
                    jax.clear_caches()
                    gc.collect()
                report["rows"][row]["sizes"].append(result)
                save_report(report, args.output)
        if args.reference_bases:
            for n in sorted(set(args.sizes)):
                report["reference_bases"]["end"].append(run_reference(n, args.path, "end"))
                save_report(report, args.output)
        finish_report(report)
    report["total_wall_seconds"] = time.perf_counter() - t0
    save_report(report, args.output)
    print("COMPLETE " + json.dumps({"output": args.output, "seconds": report["total_wall_seconds"],
                                    "control_flagged": report.get("control_flagged"),
                                    "control_b": report.get("control_b")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
