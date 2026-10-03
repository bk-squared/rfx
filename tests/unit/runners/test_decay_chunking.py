"""Uniform decay scans: exact stop steps and cross-trace output contracts.

The reference is the pre-change loop, frozen on 4725b748 in
_until_decay_reference.py. These are driver contracts, not accuracy claims.
The PI's 2026-09-29 decision uses the existing A/B lock's derived floors
for outputs against both the old loop and forced-N run(). Stop steps remain
exact, and report on/off remains bit-exact. ULP values are measurements only.
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
from tests.locks.test_run_until_decay_ab_identity import (
    _E_COMPS,
    _H_COMPS,
    _RTOL,
    _field_reassoc_atols,
    _reassoc_atol,
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
    branch, driver, monkeypatch, *, cap=241, interval=17, snapshot_interval=7, report_every=None,
    record_dft=False,
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
        if record_dft:
            kwargs["record_dft"] = True
        captured["grid"], captured["materials"] = args[:2]
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
    return result, captured["low"], captured["grid"], captured["materials"]


def _assert_floor_arrays(left, right, n_steps, grid, materials, record_property, *, comparison):
    """Reuse the lock's floors; no independently calibrated output tolerance.

    Yee fields and snapshots use the lock's E/H curl-coupled component floors.
    Each other floating array uses its own compared magnitudes, through the
    lock's scalar-observable helper. Kahan pairs compare both the accumulator
    and acc + comp at the accumulator's floor, never compensation alone.
    Metadata and empty arrays stay exact.
    """
    assert left.keys() == right.keys()
    for name, b in right.items():
        assert left[name].shape == b.shape, name
        assert left[name].dtype == b.dtype, name

    atols = {}
    for pattern in (".state.{}", ".snapshots[{!r}]", "api.state.{}"):
        names = {c: pattern.format(c) for c in _E_COMPS + _H_COMPS}
        arrays = {c: [left[name], right[name]] for c, name in names.items()
                  if name in right and right[name].size}
        if arrays:
            floors, _, _ = _field_reassoc_atols(arrays, n_steps, grid, materials)
            atols.update({names[c]: floor for c, floor in floors.items()})

    measured, failures = {}, {}
    for name, b in right.items():
        a = left[name]
        scale = [a, b]
        # Product structure: NTFFData's c_* partners (rfx/farfield.py:123-146)
        # and (acc, comp) current moments (rfx/current_moments.py:580-592).
        accumulator = None
        if name.startswith(".ntff_data.c_"):
            accumulator = name.replace(".ntff_data.c_", ".ntff_data.", 1)
        elif name == ".current_moment_data[1]":
            accumulator = ".current_moment_data[0]"
        if accumulator is not None:
            acc_a, acc_b = left[accumulator], right[accumulator]
            assert a.shape == acc_a.shape and b.shape == acc_b.shape, name
            assert a.dtype == acc_a.dtype and b.dtype == acc_b.dtype, name
            scale = [acc_a, acc_b]
            # rfx reads acc alone (farfield.py:1063-1074;
            # current_moments.py:938-941). Keep that check and additionally
            # compare acc + comp in the accumulator dtype, without promotion.
            a = np.add(acc_a, a, dtype=acc_a.dtype)
            b = np.add(acc_b, b, dtype=acc_b.dtype)
            name = f"{accumulator} + {name}"
        if (b.dtype.kind not in "fc" or not b.size or name in (".dt", "api.dt")
                or name.startswith(".snapshot_axes")):
            np.testing.assert_array_equal(a, b, err_msg=name)
            continue
        atol = atols[name] if name in atols else _reassoc_atol(scale, n_steps)
        passed = bool(np.allclose(a, b, rtol=_RTOL, atol=atol))
        delta = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128))))
        measured[name] = dict(atol=atol, rtol=_RTOL, max_abs_difference=delta, passed=passed)
        if not passed:
            failures[name] = measured[name]
    record_property(f"reassociation_floors_{comparison}", json.dumps(measured, sort_keys=True))
    assert not failures, failures


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
    record_property("assertion_bar", "exact stop; existing A/B-lock derived floors and _RTOL")

    # Low-level checks include accumulators and their acc + comp pairs.
    # API S and derived flux spectra are additional observable outputs.
    a, b = dict(_arrays(got[1])), dict(_arrays(ref[1]))
    api_a, api_b = _observables(got[0]), _observables(ref[0])
    api_a["dt"], api_b["dt"] = np.asarray(got[0].dt), np.asarray(ref[0].dt)
    for name in ref[0].flux_monitors:
        api_a[f"flux_spectrum[{name}]"] = np.asarray(flux_spectrum(got[0].flux_monitors[name]))
        api_b[f"flux_spectrum[{name}]"] = np.asarray(flux_spectrum(ref[0].flux_monitors[name]))
    a.update({f"api.{name}": value for name, value in api_a.items()})
    b.update({f"api.{name}": value for name, value in api_b.items()})
    _assert_floor_arrays(a, b, ref_steps, got[2], got[3], record_property, comparison="old_loop")
    if ref_steps > 1:
        for name in ("time_series", "s_params", "dft[pz]", "flux[fx].e1_dft", "state.ez"):
            assert np.max(np.abs(api_b[name])) > 0, name


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


def test_decay_reference_retains_requested_dft_records(monkeypatch, record_property):
    kw = dict(cap=15, record_dft=True)
    got = _run("min-cap", simulation.run_until_decay, monkeypatch, **kw)
    ref = _run("min-cap", run_until_decay_reference, monkeypatch, **kw)
    assert got[1].dft_time_records and ref[1].dft_time_records
    assert got[1].dft_time_records[0].shape[0] == 15
    _assert_contract(got, ref, record_property)


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
    """Forced N agrees with run() AND the frozen loop under the A/B floors."""
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
    record_property("assertion_bar", "exact stop; existing A/B-lock derived floors and _RTOL")
    for comparison, reference in (("run_forced_n", ref), ("old_loop_forced_n", old)):
        _assert_floor_arrays(
            dict(_arrays(got)), dict(_arrays(reference)), n, grid, materials,
            record_property, comparison=comparison,
        )
