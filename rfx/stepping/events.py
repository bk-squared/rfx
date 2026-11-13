"""Host-only progress event transport and call state (schema v1)."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from dataclasses import fields, is_dataclass
import math
import os
import sys
import time
import uuid

_SINK = ContextVar("rfx_event_sink", default=None)
_SCOPE = ContextVar("rfx_event_scope", default=None)
_TRACED = set()
PHASES = ("assemble", "compile", "step", "post-process")


def _line(event):
    return "RFX_EVENT " + json.dumps(event, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


@contextmanager
def events(sink):
    """Send progress dictionaries to a callable, stderr, or an appended file."""
    stream = None
    if not callable(sink):
        stream = sys.stderr if sink == "stderr" else open(os.fspath(sink), "a", encoding="utf-8")
        def sink(event):
            stream.write(_line(event))
            stream.flush()
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)
        if stream is not None and stream is not sys.stderr:
            stream.close()


def enabled():
    return _SINK.get() is not None or "RFX_EVENTS" in os.environ


def in_scope():
    return _SCOPE.get() is not None


def current():
    scope = _SCOPE.get()
    return scope if scope is not None and not scope.refused else None


def traced(*trees):
    import jax
    from rfx.core.jax_utils import is_tracer
    return any(is_tracer(x) for x in jax.tree_util.tree_leaves(trees))


def ready(outputs):
    import jax
    # Results include strings and diagnostic objects as well as array leaves.
    for x in jax.tree_util.tree_leaves(outputs):
        if is_dataclass(x) and not isinstance(x, type):
            ready(tuple(getattr(x, f.name) for f in fields(x)))
        elif hasattr(x, "block_until_ready"):
            x.block_until_ready()


class Scope:
    def __init__(self, sink, entry, path, passes, label):
        self.sink, self.entry, self.path = sink, entry, path
        self.passes, self.label = passes, label
        self.run_id = uuid.uuid4().hex[:12]
        self.index, self.phase, self.outputs = 0, None, None
        self.refused = False
        self.last_tick = -math.inf
        self.done = 0

    def emit(self, kind, **fields):
        event = dict(v=1, t=time.time(), kind=kind, path=self.path, entry=self.entry, run_id=self.run_id)
        if self.passes and kind != "plan":
            event.update(pass_index=max(1, self.index), pass_count=len(self.passes))
        event.update(fields)
        for name in ("label", "detail"):
            if name in event:
                original = event[name]
                limit = min(48, len(original)) if name == "label" else len(original)
                while True:
                    event[name] = original if limit == len(original) else original[:max(0, limit - 1)] + "…"
                    if len(_line(event).encode()) <= 500 or limit == 0:
                        break
                    limit -= 1
        if len(_line(event).encode()) > 500:
            raise ValueError("RFX event exceeds 500 bytes")
        self.sink(event)

    def plan(self):
        stamp, parts = time.time(), 1
        while True:
            groups, group = [], []
            for label in self.passes:
                candidate = dict(v=1, t=stamp, kind="plan", path=self.path, entry=self.entry,
                                 run_id=self.run_id, phases=list(PHASES) if not groups else [],
                                 passes=group + [label], part=len(groups) + 1, parts=parts)
                if len(_line(candidate).encode()) > 500 and group:
                    groups.append(group)
                    group = []
                group.append(label)
            groups.append(group)
            if len(groups) == parts:
                break
            parts = len(groups)
        for i, group in enumerate(groups, 1):
            self.emit("plan", t=stamp, phases=list(PHASES) if i == 1 else [],
                      passes=group, part=i, parts=parts)

    def start(self, phase, **fields):
        self.phase = phase
        if not self.passes or 1 <= self.index <= len(self.passes):
            fields["label"] = self.passes[self.index - 1] if self.passes else self.label
        self.emit("phase_start", phase=phase, **fields)

    def end(self, outputs=None, **fields):
        ready(outputs)
        self.emit("phase_end", phase=self.phase, **fields)
        self.phase = None

    def refuse(self, code, detail):
        self.refused = True
        for index in range(1, max(1, len(self.passes)) + 1):
            self.index = index
            for phase in PHASES:
                reason = "compile_not_separated" if phase == "compile" else code
                self.emit("not_reported", phase=phase, reported=False, reason_code=reason, detail=detail)


@contextmanager
def scope(entry, path, passes=(), label="", refusal=None, probes=()):
    if _SCOPE.get() is not None:
        yield
        return
    env = os.environ.get("RFX_EVENTS")
    if env is not None and env != "stderr":
        raise ValueError("RFX_EVENTS must be 'stderr'")
    sink = _SINK.get()
    if sink is None and env is None:
        yield
        return
    if sink is None:
        def sink(event):
            sys.stderr.write(_line(event))
            sys.stderr.flush()
    state = Scope(sink, entry, path, list(passes), label)
    token = _SCOPE.set(state)
    try:
        is_traced = traced(*probes)
        if refusal and entry == "forward":
            state.pending, state.refused, state.probes = True, True, probes
        elif is_traced and (entry, path) in _TRACED:
            state.refused = True
        else:
            state.plan()
            if is_traced:
                _TRACED.add((entry, path))
                state.refuse("traced", "Host timing is unavailable under a transform")
            elif refusal:
                state.refuse(refusal, "This path is deferred to the next progress-events change")
            else:
                state.index = 1
                state.start("assemble")
        try:
            yield
        except BaseException:
            resolve_deferred(())
            if not state.refused and state.phase:
                state.end(state.outputs, **({"stop": "error", "steps_done": state.done} if state.phase == "step" else {}))
            raise
        else:
            resolve_deferred(())
            if not state.refused and state.phase:
                state.end(state.outputs)
    finally:
        _SCOPE.reset(token)


def next_solve():
    state = current()
    if state and state.phase == "post-process":
        state.end(state.outputs)
        state.index += 1
        state.start("assemble")


def begin_step(outputs, *, total, ticks, label="", cap=False, probes=()):
    resolve_deferred((outputs, probes))
    state = current()
    if state is None:
        return False
    if traced(outputs, *probes):
        # Probe the actual carry, xs and captured material/geometry operands.
        state.end(None)
        key = (state.entry, state.path)
        if key not in _TRACED:
            _TRACED.add(key)
            state.refuse("traced", "Traced step operands have no host timing")
        else:
            state.refused = True
        return False
    state.end((outputs, probes))
    state.emit("not_reported", phase="compile", reported=False,
               reason_code="compile_not_separated", detail="Compile is included in the first step chunk")
    if label and not state.passes:
        state.label = label
    state.start("step", **({} if ticks else {"ticks": False}))
    state.total, state.cap, state.ticks = int(total), bool(cap), ticks
    state.started, state.last_done, state.done = time.monotonic(), 0, 0
    return ticks


def tick(done, outputs=None, *, every=None, rule=None):
    state = current()
    if state is None or state.phase != "step" or not state.ticks:
        return
    state.done = int(done)
    if every and done - state.last_done < every and done < state.total:
        return
    state.last_done = done
    ready(outputs)
    now = time.monotonic()
    if now - state.last_tick < 1:
        return
    state.last_tick = now
    rate = done / max(now - state.started, 1e-12)
    fields = dict(steps_done=int(done), steps_total=state.total, total_is_cap=state.cap,
                  rate_steps_per_s=rate, eta_s=(state.total - done) / rate if rate > 0 else None)
    if rule is not None and rule.thr > 0:
        last = getattr(rule, "last", None)
        ratio = last[1] / last[2] if last is not None and last[2] > 0 else None
        if hasattr(rule, "history"):
            ratio = max(rule.history[-rule.env_checks:]) / rule.peak if rule.history and rule.peak > 0 else None
        fields.update(criterion="radiated_flux" if hasattr(rule, "history") else "conserved_energy",
                      criterion_value=ratio if ratio is None or math.isfinite(ratio) else None,
                      criterion_threshold=float(rule.thr))
    state.emit("progress", phase="step", **fields)


def end_step(outputs, done, diagnostic=None):
    state = current()
    if state is None:
        return
    state.done = int(done)
    stop = (diagnostic.values["criterion"] if diagnostic is not None else
            "threshold" if state.cap and done < state.total else "cap" if state.cap else "steps_done")
    state.end(outputs, stop=stop, steps_done=int(done))
    state.outputs = outputs
    state.start("post-process", extractor=getattr(state, "extractor", None) or (
        "s_matrix" if state.passes or state.entry.startswith("compute_") else "dft"))


def resolve_deferred(probes):
    state = _SCOPE.get()
    if state is None or not getattr(state, "pending", False):
        return
    state.pending = False
    is_traced = traced(probes, state.probes)
    key = (state.entry, state.path)
    if is_traced and key in _TRACED:
        return
    state.plan()
    if is_traced:
        _TRACED.add(key)
    state.refuse("traced" if is_traced else "path_not_wired",
                 "Host timing is unavailable under a transform" if is_traced else "This path is deferred")


def result_ready(result):
    resolve_deferred(result)
    state = current()
    if state:
        state.outputs = result
    return result


@contextmanager
def extractor_scope(entry, grid, count, probes):
    """Account for a diagnostic extractor after its own argument checks."""
    if in_scope() or not enabled():
        yield
        return
    passes = [f"p{i + 1}" for i in range(count)] if count > 1 else []
    path = "graded" if type(grid).__name__ == "NonUniformGrid" else "uniform"
    with scope(entry, path, passes, refusal="path_not_wired", probes=(probes,)):
        yield
