"""Record base-branch profiles through pytest; the leader saves stdout's JSON.

Load with ``-p scripts.dev.freeze_edge_aware_today`` and explicitly name all
caller test files. This is an extraction session: the crossval ladder visits
each rung and arm witness independently, stopping each at its first solve.
The other tests execute normally, including the fin physics test.
Entries use float.hex strings. If compact JSON exceeds 1 MB, the entries
are losslessly stored in ``entries_gzip_base64``; the lock reads both forms.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import wraps
import gzip
import inspect
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
CROSSVAL = "tests/crossval/msl_notch_filter/test_msl_notch_filter.py"
CALLER_FILES = (
    "tests/unit/nonuniform/test_edge_aware_profile.py",
    "tests/contracts/test_auto_z_conductor_planes.py",
    "tests/contracts/test_sheet_end_junction.py",
    "tests/contracts/test_sheet_size_verdict.py",
    "tests/contracts/test_sheet_solved_edge_reports.py",
    "tests/contracts/test_sheet_cavity_spacing.py",
    "tests/unit/preflight/test_graded_axis_findings.py",
    CROSSVAL,
)
EXAMPLES = (
    "examples/tutorials/nonuniform_patch_demo.py",
    "examples/tutorials/patch_antenna_demo.py",
    "examples/tutorials/ports_and_sparams_101.py",
)
FIELDS = ("cells", "nodes", "metal_end_nodes", "edge_cells")
SCAN_C = (2.30, 2.40, 2.45, 2.50, 2.85, 2.90, 2.95, 3.00)


class _BeforeTimeStepping(BaseException):
    """Stop extraction before Simulation can enter a time stepper."""


def _hex(value):
    return float(value).hex()


class _Recorder:
    def __init__(self):
        import rfx.mesh_edges as mesh_edges

        self.mesh_edges = mesh_edges
        self.original = mesh_edges.edge_aware_profile
        self.original_build = mesh_edges._build
        self.signature = inspect.signature(self.original)
        self.entries = {}
        self.caller = None
        self.patched_sites = set()
        self.stops = []
        self.skipped_build_calls = 0

        @wraps(self.original)
        def wrapped(*args, **kwargs):
            pristine = mesh_edges._build is self.original_build
            if not pristine:
                self.skipped_build_calls += 1
                return self.original(*args, **kwargs)
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            frozen_args = {
                name: _hex(values[name])
                for name in ("lo", "hi", "dx", "max_ratio", "edge_offset")
            }
            frozen_args["boundary_cell"] = (
                None if values["boundary_cell"] is None
                else _hex(values["boundary_cell"])
            )
            frozen_args["faces"] = [_hex(v) for v in values["faces"]]
            frozen_args["sheet_edges"] = [
                [_hex(e.position), int(e.metal_side)] for e in values["sheet_edges"]
            ]
            result = self.original(*args, **kwargs)
            if mesh_edges._build is not self.original_build:
                self.skipped_build_calls += 1
                return result
            if self.caller is None:
                raise RuntimeError("a profile returned without a caller identity")
            outputs = {name: [_hex(v) for v in getattr(result, name)] for name in FIELDS}
            key = json.dumps(frozen_args, sort_keys=True, separators=(",", ":"))
            if key in self.entries:
                entry = self.entries[key]
                if entry["outputs"] != outputs:
                    raise RuntimeError(f"same arguments returned different bits: {self.caller}")
                if self.caller not in entry["callers"]:
                    entry["callers"].append(self.caller)
            else:
                self.entries[key] = {
                    "args": frozen_args, "outputs": outputs, "callers": [self.caller],
                }
            return result

        self.wrapped = wrapped

    def wrap_import_sites(self):
        """Replace existing aliases; later imports receive the patched export."""
        for module in tuple(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if namespace is None:
                continue
            for name, value in tuple(namespace.items()):
                if value is self.original:
                    namespace[name] = self.wrapped
                if value is self.original or value is self.wrapped:
                    self.patched_sites.add(f"{namespace.get('__name__', '?')}.{name}")

    @contextmanager
    def stop_simulation(self):
        from rfx import Simulation

        def stop_run(*args, **kwargs):
            self.stops.append({"caller": self.caller, "method": "Simulation.run"})
            raise _BeforeTimeStepping

        def stop_forward(*args, **kwargs):
            self.stops.append({"caller": self.caller, "method": "Simulation.forward"})
            raise _BeforeTimeStepping

        with patch.object(Simulation, "run", stop_run), patch.object(Simulation, "forward", stop_forward):
            yield

    def until_stopped(self, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except _BeforeTimeStepping:
            return
        raise RuntimeError(f"expected a pre-time-stepping stop: {self.caller}")

    def crossval_ladder(self, item):
        """A stopped rung cannot return S; visit the remaining builds explicitly."""
        module = item.module
        rungs = module._rungs()
        if rungs != module.LADDER_M:
            raise RuntimeError("unset RFX_MSL_NOTCH_RUNGS to freeze the complete ladder")
        for index, dx in enumerate(rungs):
            self.until_stopped(module.run_rung, dx)
            if index < module.ARM_WITNESS_RUNGS:
                self.until_stopped(module.run_rung, dx, module.WITNESS_ARM_LENGTH_M)

    def examples(self):
        for path in EXAMPLES:
            self.caller = path
            with self.stop_simulation():
                if path.endswith("ports_and_sparams_101.py"):
                    namespace = runpy.run_path(str(ROOT / path))
                    namespace["build_microstrip_ports"]()
                    self.until_stopped(namespace["main"])
                else:
                    self.until_stopped(runpy.run_path, str(ROOT / path), run_name="__main__")

    def scan(self):
        import numpy as np
        from rfx import Box

        h, w, lx = 254e-6, 600e-6, 12e-3
        dx, lz = h / 3, h + 1.5e-3
        for c in np.round(np.arange(1.5, 3.001, 0.05), 2):
            if float(c) not in SCAN_C:
                continue
            self.caller = f"scan:c={c:.2f}"
            ly = w + 2 * c * dx
            strip = Box((0, ly / 2 - w / 2, h), (lx, ly / 2 + w / 2, h))
            slab = Box((0, 0, 0), (lx, ly, h))
            self.mesh_edges.edge_aware_profiles(
                (lx, ly, lz), dx, sheets=[strip], solids=[slab], axes="y",
            )

    def emit(self, session):
        import jax
        import numpy as np

        provenance = {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "date": datetime.now(timezone.utc).date().isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "jax": jax.__version__,
            "platform": platform.platform(),
            "pytest_exitstatus": int(session.exitstatus),
        }
        payload = {
            "schema_version": 1,
            "provenance": provenance,
            "patched_import_sites": sorted(self.patched_sites),
            "skipped_build_calls": self.skipped_build_calls,
            "stops": self.stops,
            "entries": [self.entries[key] for key in sorted(self.entries)],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        plain_bytes = len(encoded.encode("utf-8"))
        if plain_bytes + 1 > 1_000_000:
            entries_json = json.dumps(payload.pop("entries"), sort_keys=True, separators=(",", ":"))
            payload["entries_gzip_base64"] = base64.b64encode(
                gzip.compress(entries_json.encode("utf-8"), mtime=0),
            ).decode("ascii")
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        size = len(encoded.encode("utf-8")) + 1
        if size > 1_000_000:
            raise RuntimeError(f"freeze exceeds the 1 MB bar even after compression: {size} bytes")
        stream = sys.__stdout__
        print("EDGE-FREEZE-PROVENANCE " + json.dumps(provenance, sort_keys=True), file=stream)
        print(f"EDGE-FREEZE-UNCOMPRESSED-JSON-BYTES {plain_bytes + 1}", file=stream)
        print(f"EDGE-FREEZE-JSON-BYTES {size}", file=stream)
        print("EDGE-FREEZE-BEGIN", file=stream)
        print(encoded, file=stream)
        print("EDGE-FREEZE-END", file=stream, flush=True)


_recorder = None


def pytest_sessionstart(session):
    global _recorder
    if hasattr(session.config, "workerinput") or session.config.getoption("numprocesses", default=0):
        raise pytest.UsageError("freeze_edge_aware_today requires serial pytest")
    _recorder = _Recorder()
    _recorder.wrap_import_sites()


def pytest_collection_finish(session):
    _recorder.wrap_import_sites()
    if not session.config.option.collectonly:
        selected = {item.path for item in session.items}
        missing = [path for path in CALLER_FILES if ROOT / path not in selected]
        if missing:
            raise pytest.UsageError(f"freeze requires every caller test file; missing: {missing}")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    _recorder.wrap_import_sites()
    _recorder.caller = item.nodeid
    guard = _recorder.stop_simulation() if item.path == ROOT / CROSSVAL else nullcontext()
    try:
        with guard:
            yield
    finally:
        _recorder.caller = None


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if (pyfuncitem.path == ROOT / CROSSVAL
            and pyfuncitem.name == "test_msl_notch_filter_matches_the_openems_tutorial"):
        _recorder.crossval_ladder(pyfuncitem)
        return True
    return None


def pytest_sessionfinish(session, exitstatus):
    if _recorder is None or session.config.option.collectonly:
        return
    _recorder.wrap_import_sites()
    _recorder.examples()
    _recorder.scan()
    _recorder.caller = None
    _recorder.emit(session)
