"""Asymmetric scenes and byte-level observations for progress path contracts."""
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from time import perf_counter

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.progress import events
from tests.contracts.test_progress_events import check_phases


def require_current_moment_determinism():
    if jax.default_backend() == "gpu" and "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("on GPUs the current-moment monitor is bit-reproducible only with XLA_FLAGS=--xla_gpu_deterministic_ops=true; see docs/guides/support_matrix.md")


def scene(*, graded=False, boundary="cpml", ports=1, source=True, drive_all_ports=False):
    sim = Simulation(freq_max=15e9, domain=(.012, .009, .007), dx=.001,
                     boundary=boundary, cpml_layers=2 if boundary == "cpml" else 0,
                     **({"dz_profile": [.0008, .0012, .001, .001, .001, .001, .001]} if graded else {}))
    if source:
        sim.add_source((.005, .003, .002), "ez", amplitude_kind="current",
                       waveform=GaussianPulse(5e9))
    for index in range(ports):
        excitation = {} if drive_all_ports else {"excite": index == 0}
        sim.add_port((.004 + .004 * index, .003, .003), "ez", impedance=50.,
                     waveform=GaussianPulse(5e9), **excitation)
    sim.add_probe((.008, .005, .004), "ez")
    sim.add_probe((.003, .002, .002), "hy")
    return sim


def add_records(sim):
    sim.add_dft_plane_probe(axis="y", coordinate=.004, component="ez",
                            freqs=[3e9, 7e9], name="plane")
    sim.add_flux_monitor(axis="y", coordinate=.004, freqs=[3e9, 7e9], name="flux")
    sim.add_ntff_box((.003, .002, .002), (.009, .007, .006), freqs=[3e9, 7e9])
    sim.add_current_moment_monitor((.001, .001, .001), (.010, .007, .006),
                                   block_size=.002, freqs=[3e9, 7e9])


def arrays(value, path="result"):
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from arrays(getattr(value, field.name), path + "." + field.name)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from arrays(value[key], path + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from arrays(item, path + "." + str(i))
    elif hasattr(value, "dtype") and hasattr(value, "shape"):
        yield path, np.asarray(value)
    else:
        leaves = jax.tree.leaves(value)
        if len(leaves) != 1 or leaves[0] is not value:
            for i, leaf in enumerate(leaves):
                yield from arrays(leaf, path + "." + str(i))


def compare(actual, reference, report, tmp_path):
    left, right = dict(arrays(actual)), dict(arrays(reference))
    assert left.keys() == right.keys() and left
    report["arrays"] = {}
    for name, a in left.items():
        b = right[name]
        ah, bh = (hashlib.sha256(v.tobytes()).hexdigest() for v in (a, b))
        same = a.dtype == b.dtype and a.shape == b.shape and ah == bh
        row = dict(shape=a.shape, dtype=str(a.dtype), on_sha256=ah, off_sha256=bh, equal=same)
        if not same:
            row["off_max_abs"] = float(np.max(np.abs(b)))
        if not same and a.shape == b.shape and a.dtype == b.dtype:
            row["max_abs"] = float(np.max(np.abs(a - b)))
            row["different_bytes"] = int(np.count_nonzero(
                np.frombuffer(a.tobytes(), dtype=np.uint8) != np.frombuffer(b.tobytes(), dtype=np.uint8)))
        report["arrays"][name] = row
    (tmp_path / "identity.json").write_text(json.dumps(report, indent=2))
    differences = {k: v for k, v in report["arrays"].items() if not v["equal"]}
    print(json.dumps({k: v for k, v in report.items() if k != "arrays"}))
    assert not differences, ("R1 raw-byte differences", differences)


def sequence(rows, path, *, passes=1, ticks=True):
    check_phases(rows, path, passes)
    expected = [("phase_start", "assemble"), ("phase_end", "assemble"),
                ("not_reported", "compile"), ("phase_start", "step"),
                ("phase_end", "step"), ("phase_start", "post-process"),
                ("phase_end", "post-process")]
    actual = [(r["kind"], r["phase"]) for r in rows if r["kind"] not in ("plan", "progress")]
    assert actual == expected * passes
    starts = [r for r in rows if r["kind"] == "phase_start" and r["phase"] == "step"]
    assert all(r.get("ticks", True) is ticks for r in starts)
    assert any(r["kind"] == "progress" for r in rows) is ticks


def measure(call, *, path, tmp_path, monkeypatch, passes=1, ticks=True, metadata=None):
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    from rfx.stepping import events as transport, uniform_loop, graded_loop
    completed = []
    original = transport.end_step
    def capture(outputs, *args, **kwargs):
        completed.append(dict(arrays(outputs)))
        return original(outputs, *args, **kwargs)
    for module in (transport, uniform_loop, graded_loop):
        monkeypatch.setattr(module, "end_step", capture)
    start = perf_counter()
    reference = call()
    # Converting every array synchronizes records as well as fields.
    reference = dict(arrays((reference, completed)))
    off_seconds = perf_counter() - start
    completed.clear()
    rows = []
    start = perf_counter()
    with events(rows.append):
        result = call()
        actual = dict(arrays((result, completed)))
    on_seconds = perf_counter() - start
    report = dict(path=path, off_seconds=off_seconds, on_seconds=on_seconds,
                  **(metadata or {}))
    compare(actual, reference, report, tmp_path)
    sequence(rows, path, passes=passes, ticks=ticks)
    return rows
