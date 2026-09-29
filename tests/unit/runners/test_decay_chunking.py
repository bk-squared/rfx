"""Uniform decay scans: exact stop steps and cross-trace output contracts.

The reference is the pre-change loop, frozen on 4725b748 in
_until_decay_reference.py. These are driver contracts, not accuracy claims.
Per-step arrays use the PI's 9-float32-ULP-at-peak rule; record sums use
1e-4 of their own peak. Neither rule changes the exact stop-step contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields, is_dataclass

import jax
import numpy as np
import pytest

from rfx import SnapshotSpec
from rfx import simulation
from rfx.probes.probes import flux_spectrum
from tests.unit.autodiff.test_forward_jit_compile_once import (
    MAX_REL_SUMMED,
    MAX_ULP_AT_PEAK,
    _rel_at_peak,
    _ulp_at_peak,
)
from tests.unit.runners._until_decay_reference import run_until_decay_reference
from tests.unit.runners.test_snapshot_interval_and_axes import _loaded_sim, _observables
from tests.unit.sparams.test_ringdown_run import _box


def _arrays(value, name=""):
    """Include every array leaf, including metadata and compensation."""
    if is_dataclass(value):
        for field in fields(value):
            yield from _arrays(getattr(value, field.name), f"{name}.{field.name}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _arrays(child, f"{name}[{key!r}]")
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        for key in value._fields:
            yield from _arrays(getattr(value, key), f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from _arrays(child, f"{name}[{i}]")
    elif hasattr(value, "dtype") and hasattr(value, "shape"):
        yield name, np.asarray(value)
    elif isinstance(value, (int, float, complex, bool)):
        yield name, np.asarray(value)


def _record_metrics(got, ref, record_property, *, comparison):
    """Measurements do not select or relax the assertion bar below.

    Complex differences retain both components. Empty arrays have peak and
    difference zero; the float32 spacing at zero is used for zero peaks.
    """
    left, right = dict(_arrays(got)), dict(_arrays(ref))
    assert left.keys() == right.keys()
    metrics = {}
    for name, b in right.items():
        a = left[name]
        assert a.shape == b.shape, name
        assert a.dtype == b.dtype, name
        aa, bb = a.astype(np.complex128), b.astype(np.complex128)
        peak = float(np.max(np.abs(bb), initial=0.0))
        delta = float(np.max(np.abs(aa - bb), initial=0.0))
        metrics[name] = dict(
            shape=list(b.shape), dtype=str(b.dtype), reference_peak=peak,
            max_abs_difference=delta,
            max_ulp_at_peak=delta / float(np.spacing(np.float32(peak))),
            array_equal=np.array_equal(a, b),
        )
    record_property(f"all_array_metrics_{comparison}", json.dumps(metrics, sort_keys=True))


def _run(
    branch, driver, monkeypatch, *, cap=241, interval=17, snapshot_interval=7, report_every=None
):
    sim = _loaded_sim("pec" if branch in ("point", "min-cap", "zero-cap") else "cpml")
    freqs = np.array([2e9, 4e9])
    sim.add_ntff_box((0.002, 0.002, 0.002), (0.010, 0.010, 0.010), freqs=freqs)
    if branch in ("energy", "flux"):
        sim.add_current_moment_monitor(
            (0.001, 0.001, 0.001), (0.011, 0.011, 0.011), block_size=0.005, freqs=freqs
        )
    spec = (
        None
        if snapshot_interval is None
        else SnapshotSpec(
            interval=snapshot_interval,
            components=("ez", "hy"),
            slice_axis=2,
            slice_index=sim._build_grid().position_to_index((0, 0, 0.006))[2],
        )
    )
    captured = {}

    def capture(*args, **kwargs):
        captured["low"] = driver(*args, **kwargs)
        return captured["low"]

    with monkeypatch.context() as m:
        m.setattr(simulation, "run_until_decay", capture)
        result = sim.run(
            until_decay=0.0 if branch == "zero-cap" else 0.8,
            decay_check_interval=interval,
            decay_min_steps=cap + 1 if branch == "min-cap" else 35,
            decay_max_steps=cap,
            decay_monitor_position=(0.009, 0.004, 0.006),
            radiated_flux_box=((0.002, 0.002, 0.002), (0.010, 0.010, 0.010))
            if branch == "flux"
            else None,
            flux_env_checks=2,
            snapshot=spec,
            report_every=report_every,
            compute_s_params=True,
            skip_preflight=True,
        )
    return result, captured["low"]


def _assert_contract(got, ref, record_property):
    got_steps, ref_steps = got[0].time_series.shape[0], ref[0].time_series.shape[0]
    record_property("stop_steps", got_steps)
    record_property("reference_stop_steps", ref_steps)
    assert got_steps == ref_steps, f"stop step: got {got_steps}, reference {ref_steps}"
    for result in (got[0], got[1], ref[0], ref[1]):
        assert int(result.state.step) == ref_steps
    _record_metrics(got[1], ref[1], record_property, comparison="old_loop")
    _record_metrics(_observables(got[0]), _observables(ref[0]), record_property,
                    comparison="old_loop_api")
    record_property("assertion_bar", "exact stop; per-step <=9 ULP at peak; sums <=1e-4 of peak")

    # The API S and dt plus every low-level accumulator (including Kahan
    # compensation) are observable separately from the final six fields.
    a, b = _observables(got[0]), _observables(ref[0])
    a["dt"], b["dt"] = np.asarray(got[0].dt), np.asarray(ref[0].dt)
    for name in ref[0].flux_monitors:
        a[f"flux_spectrum[{name}]"] = np.asarray(flux_spectrum(got[0].flux_monitors[name]))
        b[f"flux_spectrum[{name}]"] = np.asarray(flux_spectrum(ref[0].flux_monitors[name]))
    for name in (
        "ntff_data",
        "current_moment_data",
        "wire_port_sparams",
        "snapshots",
        "snapshot_axes",
    ):
        left, structure = jax.tree_util.tree_flatten_with_path(getattr(got[1], name))
        right = jax.tree_util.tree_leaves(getattr(ref[1], name))
        assert structure == jax.tree_util.tree_structure(getattr(ref[1], name)), name
        assert len(left) == len(right), name
        for (path, x), y in zip(left, right):
            if hasattr(x, "__dataclass_fields__"):
                for key in x.__dataclass_fields__:
                    xx, yy = getattr(x, key), getattr(y, key)
                    if isinstance(xx, dict):
                        assert xx.keys() == yy.keys()
                        for k in xx:
                            np.testing.assert_array_equal(xx[k], yy[k], err_msg=f"{name}.{key}.{k}")
                    else:
                        np.testing.assert_array_equal(xx, yy, err_msg=f"{name}.{key}")
            else:
                key = f"{name}{jax.tree_util.keystr(path)}"
                a[key], b[key] = np.asarray(x), np.asarray(y)
    assert got[0].time_series.shape == ref[0].time_series.shape
    ulps, summed = {}, {}
    for name in a:
        assert a[name].shape == b[name].shape, name
        assert a[name].dtype == b[name].dtype, name
        if name == "dt" or b[name].dtype.kind not in "fc":
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        elif name == "time_series" or name.startswith(("state.", "snapshots[")):
            # Equal/empty arrays need no reduction (some caps record no frames).
            ulps[name] = 0.0 if np.array_equal(a[name], b[name]) else _ulp_at_peak(b[name], a[name])
        else:
            summed[name] = 0.0 if np.array_equal(a[name], b[name]) else _rel_at_peak(b[name], a[name])
    record_property("ulp_at_peak_by_array", json.dumps(ulps, sort_keys=True))
    record_property("rel_summed_by_array", json.dumps(summed, sort_keys=True))
    record_property("max_ulp_at_peak", max(ulps.values()))
    record_property("max_rel_summed", max(summed.values()))
    failures = {name: f"{value} ULP > {MAX_ULP_AT_PEAK}"
                for name, value in ulps.items() if not value <= MAX_ULP_AT_PEAK}
    failures.update({name: f"{value} of peak > {MAX_REL_SUMMED}"
                     for name, value in summed.items() if not value <= MAX_REL_SUMMED})
    assert not failures, failures
    if ref[0].time_series.shape[0] > 1:
        for name in ("time_series", "s_params", "dft[pz]", "flux[fx].e1_dft", "state.ez"):
            assert np.max(np.abs(b[name])) > 0, name


@pytest.mark.parametrize("branch", ["point", "energy", "flux", "min-cap", "zero-cap"])
@pytest.mark.parametrize("snapshot_interval", [None, 7])
@pytest.mark.parametrize("report_every", [None, 13])
def test_decay_chunks_match_old_loop_per_stop_branch(
    branch, snapshot_interval, report_every, monkeypatch, record_property
):
    interval = 37 if branch == "point" else 17
    ref = _run(
        branch,
        run_until_decay_reference,
        monkeypatch,
        snapshot_interval=snapshot_interval,
        interval=interval,
        report_every=report_every,
    )
    got = _run(
        branch,
        simulation.run_until_decay,
        monkeypatch,
        snapshot_interval=snapshot_interval,
        interval=interval,
        report_every=report_every,
    )
    _assert_contract(got, ref, record_property)
    n = ref[0].time_series.shape[0]
    if branch.endswith("cap"):
        assert n == 241
    else:
        assert 35 <= n < 241
        assert (n - 1) % interval == 0
        if branch == "point":
            # Reading peaks only at check steps would stop at 186, not 112.
            assert n == 112


@pytest.mark.parametrize("branch,stop,interval", [("point", 112, 37), ("energy", 137, 17),
                                                ("flux", 188, 17)])
def test_decay_stops_before_absorbed_tail(branch, stop, interval, monkeypatch, record_property):
    kw = dict(cap=stop + 1, interval=interval, report_every=13)
    ref = _run(branch, run_until_decay_reference, monkeypatch, **kw)
    got = _run(branch, simulation.run_until_decay, monkeypatch, **kw)
    _assert_contract(got, ref, record_property)
    assert got[0].time_series.shape[0] == stop


@pytest.mark.parametrize(
    "cap,interval,snapshot_interval,report_every",
    [
        (87, 17, 1, None),  # one-step final chunk; every-step snapshots
        (83, 17, 7, 13),  # neither snapshot nor progress interval divides chunk
        (83, 100, 100, 19),  # check/snapshot interval exceeds cap (no frames)
        (19, 1, 3, None),  # check at every step
        (1, 17, 1, None),  # first step is also the cap
    ],
)
def test_decay_chunk_boundaries_and_snapshots_match_old_loop(
    cap, interval, snapshot_interval, report_every, monkeypatch, record_property
):
    kw = dict(
        cap=cap, interval=interval, snapshot_interval=snapshot_interval, report_every=report_every
    )
    ref = _run("zero-cap", run_until_decay_reference, monkeypatch, **kw)
    got = _run("zero-cap", simulation.run_until_decay, monkeypatch, **kw)
    _assert_contract(got, ref, record_property)
    assert got[0].time_series.shape[0] == cap
    assert got[0].snapshots["ez"].shape[0] == cap // snapshot_interval


def test_until_decay_with_probes_costs_less_than_ten_fixed_scans(record_property):
    """Include compilation and block on every returned value before timing.

    Warm setup with a short run; both measured drivers still compile their
    step/scan. A per-step loop plus flat stack fails on the same 4k record.
    """
    sim = _box("uniform")
    sim.run(n_steps=20, skip_preflight=True)
    elapsed = {}
    for name in ("fixed", "decay"):
        kw = (
            dict(n_steps=4000)
            if name == "fixed"
            else dict(
                until_decay=0.0, decay_max_steps=4000, decay_min_steps=4000, decay_check_interval=50
            )
        )
        start = time.perf_counter()
        r = sim.run(**kw, skip_preflight=True)
        jax.block_until_ready((r.time_series, r.state))
        elapsed[name] = time.perf_counter() - start
        assert r.time_series.shape[0] == 4000
    for name, seconds in elapsed.items():
        record_property(f"{name}_seconds", seconds)
    assert elapsed["decay"] < 10 * elapsed["fixed"], elapsed


def test_decay_without_stop_matches_run_scan(record_property):
    """The A/B lock's forced-N trajectory, under the 9-ULP-at-peak bar."""
    from tests.locks.test_run_until_decay_ab_identity import _build

    grid, materials, n, sources, probes = _build()
    kw = dict(sources=sources, probes=probes, return_state=True)
    ref = simulation.run(grid, materials, n, **kw)
    got = simulation.run_until_decay(
        grid, materials, decay_by=0.0, min_steps=n, max_steps=n,
        check_interval=n + 1, **kw,
    )
    assert got.time_series.shape == ref.time_series.shape == (n, len(probes))
    assert int(got.state.step) == int(ref.state.step) == n
    _record_metrics(got, ref, record_property, comparison="run_forced_n")
    old = run_until_decay_reference(
        grid, materials, decay_by=0.0, min_steps=n, max_steps=n,
        check_interval=n + 1, **kw,
    )
    assert int(old.state.step) == n
    _record_metrics(got, old, record_property, comparison="old_loop_forced_n")
    record_property("assertion_bar", "exact stop; vs run() per-step <=9 ULP at peak")
    arrays = {"time_series": (ref.time_series, got.time_series)}
    arrays.update({name: (getattr(ref.state, name), getattr(got.state, name))
                   for name in ("ex", "ey", "ez", "hx", "hy", "hz")})
    ulps = {name: _ulp_at_peak(a, b) for name, (a, b) in arrays.items()}
    record_property("ulp_at_peak_by_array", json.dumps(ulps, sort_keys=True))
    record_property("max_ulp_at_peak", max(ulps.values()))
    record_property("array_equal", all(np.array_equal(a, b) for a, b in arrays.values()))
    assert max(ulps.values()) <= MAX_ULP_AT_PEAK, ulps
