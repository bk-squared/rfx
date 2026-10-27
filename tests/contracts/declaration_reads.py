"""Declaration read-log twin, reused from the declaration inventory Instrument.

The proxy and scan/record hooks follow _scratch/inventory_measure.py supplied
with step A; declarations and nested record reads retain their root attribute.
"""

import collections
import contextlib
import copy
import dataclasses
import json
import os
import sys

import jax
import numpy as np
from jax._src import core
from unittest.mock import patch

from tests.contracts import path_disposition as T
from tests.unit.runners import test_path_disposition_cells as B


def traced():
    return type(core.trace_ctx.trace).__name__ in ("DynamicJaxprTrace", "JVPTrace")


def exception(exc):
    return dict(exception=type(exc).__name__, message=str(exc))


class TwinError(RuntimeError):
    pass


class Instrument:
    def __init__(self, sink, identity, enabled=True, records=True):
        self.sink, self.identity, self.enabled, self.records = (
            sink,
            identity,
            enabled,
            records,
        )
        self.phase = "before-kernel-scan"
        self.stage = "execution"
        self.count = 0
        self.attrs = set()
        self.twins = {}
        self.originals = []
        self.record_meta = {}
        self.schema = []
        self.phase_reads = collections.Counter()
        self.trace_reads = collections.Counter()
        self.issues = []
        self.active = False

    def log(self, attr, typ, field, operation="getattribute"):
        if not self.active or not self.enabled:
            return
        f = sys._getframe(1)
        reader = None
        nearest = {}
        while f:
            if f.f_globals.get("__name__", "").startswith("rfx."):
                module = f.f_globals.get("__name__", "")
                row = dict(module=module, function=f.f_code.co_name, line=f.f_lineno)
                if reader is None:
                    reader = row
                for group in ("model", "stepping", "measurement"):
                    if module.startswith("rfx." + group + ".") and group not in nearest:
                        nearest[group] = row
            f = f.f_back
        if reader is None:
            return
        self.count += 1
        self.phase_reads[self.phase] += 1
        self.trace_reads[str(traced())] += 1
        self.attrs.add(attr)
        self.sink.write(
            json.dumps(
                self.identity
                | dict(
                    attr=attr,
                    record_type=typ,
                    field=field,
                    operation=operation,
                    reader=reader,
                    nearest=nearest,
                    under_trace=traced(),
                    trace_type=type(core.trace_ctx.trace).__name__,
                    phase=self.phase,
                    stage=self.stage,
                )
            )
            + "\n"
        )

    def twin(self, x, attr, path=""):
        if (
            not self.records
            or x is None
            or isinstance(
                x,
                (
                    str,
                    bytes,
                    bool,
                    int,
                    float,
                    complex,
                    np.generic,
                    np.ndarray,
                    jax.Array,
                ),
            )
        ):
            return x
        if id(x) in self.twins:
            return self.twins[id(x)]
        typ = type(x)
        typename = typ.__module__ + "." + typ.__qualname__
        self.record_meta[id(x)] = (attr, typename, path)
        self.originals.append(x)
        inst = self

        def emit(field, op="getattribute"):
            inst.log(attr, typename, field, op)

        try:
            if typ in (list, dict, set):
                if typ is dict:

                    class Twin(dict):
                        def __getitem__(s, k):
                            emit("<all fields>", "getitem")
                            return super().__getitem__(k)

                        def __iter__(s):
                            emit("<all fields>", "iter")
                            return super().__iter__()

                        def items(s):
                            emit("<all fields>", "items")
                            return super().items()

                        def values(s):
                            emit("<all fields>", "values")
                            return super().values()

                        def get(s, k, *args):
                            emit(str(k), "get")
                            return super().get(k, *args)

                        def copy(s):
                            emit("<all fields>", "copy")
                            return super().copy()

                    y = Twin(
                        {
                            k: self.twin(v, attr, path + "." + str(k))
                            for k, v in x.items()
                        }
                    )
                elif typ is list:

                    class Twin(list):
                        def __iter__(s):
                            emit("<all fields>", "iter")
                            return super().__iter__()

                        def __getitem__(s, k):
                            emit("<all fields>", "getitem")
                            return super().__getitem__(k)

                        def copy(s):
                            emit("<all fields>", "copy")
                            return super().copy()

                    y = Twin(self.twin(v, attr, path) for v in x)
                else:

                    class Twin(set):
                        def __iter__(s):
                            emit("<all fields>", "iter")
                            return super().__iter__()

                        def copy(s):
                            emit("<all fields>", "copy")
                            return super().copy()

                    y = Twin(x)
            elif isinstance(x, tuple):
                self.schema.append(
                    dict(
                        attr=attr,
                        record_type=typename,
                        path=path,
                        fields=list(getattr(typ, "_fields", map(str, range(len(x))))),
                    )
                )

                def ga(s, name):
                    if name in getattr(typ, "_fields", ()):
                        emit(name)
                    return typ.__getattribute__(s, name)

                def it(s):
                    emit("<all fields>", "iter")
                    return tuple.__iter__(s)

                def gi(s, k):
                    emit("<all fields>", "getitem")
                    return tuple.__getitem__(s, k)

                attrs = dict(
                    __slots__=(), __getattribute__=ga, __iter__=it, __getitem__=gi
                )
                if hasattr(typ, "_asdict"):

                    def ad(s):
                        emit("<all fields>", "_asdict")
                        return typ._asdict(s)

                    attrs["_asdict"] = ad
                if hasattr(typ, "_replace"):

                    def rp(s, **kw):
                        emit("<all fields>", "_replace")
                        return typ._replace(s, **kw)

                    attrs["_replace"] = rp
                cls = type("ReadLog_" + typ.__name__, (typ,), attrs)
                vals = [self.twin(v, attr, path) for v in x]
                y = cls(*vals) if hasattr(typ, "_fields") else cls(vals)
            elif dataclasses.is_dataclass(x) or (
                typ.__module__.startswith("rfx") and hasattr(x, "__dict__")
            ):
                fields = (
                    [f.name for f in dataclasses.fields(x)]
                    if dataclasses.is_dataclass(x)
                    else list(vars(x))
                )

                def ga(s, name):
                    if name in fields:
                        emit(name)
                    return typ.__getattribute__(s, name)

                self.schema.append(
                    dict(attr=attr, record_type=typename, path=path, fields=fields)
                )
                cls = type(
                    "ReadLog_" + typ.__name__,
                    (typ,),
                    dict(__slots__=(), __getattribute__=ga),
                )
                y = copy.copy(x)
                object.__setattr__(y, "__class__", cls)
                self.twins[id(x)] = y
                for name in fields:
                    v = object.__getattribute__(x, name)
                    object.__setattr__(y, name, self.twin(v, attr, path + "." + name))
            else:
                return x
            self.twins[id(x)] = y
            self.record_meta[id(y)] = (attr, typename, path)
            return y
        except Exception as exc:
            self.issues.append(dict(record_type=typename, **exception(exc)))
            raise TwinError(typename + ": " + str(exc)) from exc

    def install(self, sim):
        if not self.enabled:
            return
        self.active = False
        for attr in T.TABLE:
            if hasattr(sim, attr):
                object.__setattr__(
                    sim, attr, self.twin(object.__getattribute__(sim, attr), attr)
                )
        typ, inst = type(sim), self

        def ga(s, name):
            if name in T.TABLE:
                inst.log(name, "Simulation", "<attribute>")
            return typ.__getattribute__(s, name)

        object.__setattr__(
            sim,
            "__class__",
            type("ReadLogSimulation", (typ,), dict(__getattribute__=ga)),
        )
        self.active = True

    @contextlib.contextmanager
    def watch(self):
        inst = self
        real_scan = jax.lax.scan

        def phase_scan(f, *a, **kw):
            # The imported watcher's entry is outside this wrapper, so it sees
            # the real source caller rather than this harness frame.
            prev = inst.phase
            kernel = bool(started)

            def body(*args):
                saved = inst.phase
                if kernel:
                    inst.phase = "inside-scan-body"
                try:
                    return f(*args)
                finally:
                    inst.phase = saved

            try:
                return real_scan(body, *a, **kw)
            finally:
                inst.phase = (
                    "after-scan" if kernel and prev != "inside-scan-body" else prev
                )

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(jax.lax, "scan", phase_scan))
            # recorded_scan is a production wrapper around lax.scan; the
            # existing helper's direct-caller list predates that wrapper.
            stack.enter_context(
                patch.object(
                    B,
                    "_KERNEL_MODULES",
                    B._KERNEL_MODULES
                    + (
                        "rfx/stepping/scan.py",
                        "rfx/stepping/slab_wrap.py",
                        "rfx/stepping/remat.py",
                    ),
                )
            )
            started = stack.enter_context(B._watch_kernel_scans())
            from rfx.stepping.slab_wrap import run_steps

            # The wrapped slab loop captured scan as a keyword default at
            # import time. Route that entry through the same imported hook.
            stack.enter_context(
                patch.dict(run_steps.__kwdefaults__, scan_loop=jax.lax.scan)
            )
            operations = []
            for module, names in (
                (dataclasses, ("asdict", "astuple", "replace")),
                (copy, ("copy", "deepcopy")),
            ):
                for name in names:
                    real = getattr(module, name)

                    def wrap(x, *a, __real=real, __name=name, **kw):
                        meta = inst.record_meta.get(id(x))
                        if meta:
                            inst.log(meta[0], meta[1], "<all fields>", __name)
                        return __real(x, *a, **kw)

                    operations.append((module, name, real, wrap))
            # Patch aliases imported by rfx as well as their defining modules.
            for module, name, real, wrap in operations:
                for loaded in list(sys.modules.values()):
                    if loaded is not None and getattr(
                        loaded, "__name__", ""
                    ).startswith("rfx"):
                        for alias, value in list(vars(loaded).items()):
                            if value is real:
                                stack.enter_context(patch.object(loaded, alias, wrap))
                stack.enter_context(patch.object(module, name, wrap))
            try:
                yield started
            finally:
                # A module imported during this scope may have bound a new
                # from-import alias; restore it as well as pre-existing aliases.
                for loaded in list(sys.modules.values()):
                    if loaded is not None and getattr(
                        loaded, "__name__", ""
                    ).startswith("rfx"):
                        for alias, value in list(vars(loaded).items()):
                            for module, name, real, wrap in operations:
                                if value is wrap:
                                    setattr(loaded, alias, real)
