"""Uniform decay scans preserve the old loop's outputs and check cadence.

The reference is the pre-change loop, frozen on 4725b748 in
_until_decay_reference.py. These are driver contracts, not accuracy claims.
"""

from __future__ import annotations

import time

import jax
import numpy as np
import pytest

from rfx import SnapshotSpec
from rfx import simulation
from tests.unit.runners._until_decay_reference import run_until_decay_reference
from tests.unit.runners.test_snapshot_interval_and_axes import _loaded_sim, _observables
from tests.unit.sparams.test_ringdown_run import _box


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


def _assert_same(got, ref):
    # The API S and dt plus every low-level accumulator (including Kahan
    # compensation) are observable separately from the final six fields.
    a, b = _observables(got[0]), _observables(ref[0])
    a["dt"], b["dt"] = np.asarray(got[0].dt), np.asarray(ref[0].dt)
    for name in (
        "ntff_data",
        "current_moment_data",
        "wire_port_sparams",
        "snapshots",
        "snapshot_axes",
    ):
        left = jax.tree_util.tree_leaves(getattr(got[1], name))
        right = jax.tree_util.tree_leaves(getattr(ref[1], name))
        assert len(left) == len(right), name
        for i, (x, y) in enumerate(zip(left, right)):
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
                a[f"{name}.{i}"], b[f"{name}.{i}"] = np.asarray(x), np.asarray(y)
    assert got[0].time_series.shape == ref[0].time_series.shape
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    if ref[0].time_series.shape[0] > 1:
        for name in ("time_series", "s_params", "dft[pz]", "flux[fx].e1_dft", "state.ez"):
            assert np.max(np.abs(b[name])) > 0, name


@pytest.mark.parametrize("branch", ["point", "energy", "flux", "min-cap", "zero-cap"])
@pytest.mark.parametrize("snapshot_interval", [None, 7])
def test_decay_chunks_match_old_loop_per_stop_branch(branch, snapshot_interval, monkeypatch):
    interval = 37 if branch == "point" else 17
    ref = _run(
        branch,
        run_until_decay_reference,
        monkeypatch,
        snapshot_interval=snapshot_interval,
        interval=interval,
    )
    got = _run(
        branch,
        simulation.run_until_decay,
        monkeypatch,
        snapshot_interval=snapshot_interval,
        interval=interval,
    )
    _assert_same(got, ref)
    n = ref[0].time_series.shape[0]
    if branch.endswith("cap"):
        assert n == 241
    else:
        assert 35 <= n < 241
        assert (n - 1) % interval == 0
        if branch == "point":
            # Reading peaks only at check steps would stop at 186, not 112.
            assert n == 112


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
    cap, interval, snapshot_interval, report_every, monkeypatch
):
    kw = dict(
        cap=cap, interval=interval, snapshot_interval=snapshot_interval, report_every=report_every
    )
    ref = _run("zero-cap", run_until_decay_reference, monkeypatch, **kw)
    got = _run("zero-cap", simulation.run_until_decay, monkeypatch, **kw)
    _assert_same(got, ref)
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


def test_decay_scan_preserves_standalone_step_arithmetic():
    """The existing larger PEC A/B fixture also pins standalone-JIT rounding."""
    from tests.locks.test_run_until_decay_ab_identity import _build

    grid, materials, n, sources, probes = _build()
    kw = dict(sources=sources, probes=probes, decay_by=0.0, max_steps=n, check_interval=50)
    ref = run_until_decay_reference(grid, materials, **kw)
    got = simulation.run_until_decay(grid, materials, **kw)
    np.testing.assert_array_equal(got.time_series, ref.time_series)
    for name in ref.state._fields:
        np.testing.assert_array_equal(
            getattr(got.state, name), getattr(ref.state, name), err_msg=name
        )
