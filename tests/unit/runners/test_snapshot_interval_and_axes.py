"""Recorded field snapshots: the interval is honoured (#1258) and every sample
has a public position and time (#1259).

What a user receives. ``SnapshotSpec(interval=m)`` now records the fields
after steps ``m, 2m, ...`` -- ``n_steps // m`` frames -- where it used to
record every step whatever ``m`` said. ``Result.snapshot_axes[comp]`` says
where each recorded sample sits (metres, in the model frame, CPML cells
included) and when each frame was taken (seconds; E at ``s * dt``, H half a
step earlier). Recording must not move any other number the run returns, so
the time series, the DFT planes, the flux monitors, the wire-port S-parameters
and the final fields are compared BIT FOR BIT with and without a snapshot and
across intervals.

The position test does not recompute a coordinate with the helper that
produces it. It places a PEC box at declared physical corners inside a CPML
domain and uses the boundary condition: every E and H sample whose own
location lies in the closed conductor region is exactly zero (tangential E
and normal H vanish on the walls; the solver zeroes E there and H is built
from that E), and every sample within a cell outside it is live. A reported
coordinate half a cell or a CPML pad off moves samples across that boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation, SnapshotSpec, snapshot_axes
from rfx.checkpoint import load_snapshots, save_snapshots
from rfx.simulation import run as low_level_run, make_source
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.sources.sources import GaussianPulse

_COMPS = ("ex", "ey", "ez", "hx", "hy", "hz")
_DOM = (0.012, 0.012, 0.012)
_DX = 1e-3


def _loaded_sim(boundary="cpml"):
    """Small box with every accumulator a snapshot could disturb.

    The wire port takes the single-port path whose S-parameters come from the
    SAME scan as the snapshot (not from a separate extraction run), so S is a
    real witness here. ``boundary="pec"`` is the model a GPU runs through the
    fused H+E kernel (``update_he_fast``; checked by forcing run()'s backend
    flag and counting its traces); ``"cpml"`` takes the absorber path.
    """
    sim = Simulation(freq_max=10e9, domain=_DOM, dx=_DX, boundary=boundary,
                     cpml_layers=6)
    sim.add(Box((0.004, 0.003, 0.002), (0.008, 0.007, 0.006)), material="fr4")
    sim.add_port((0.006, 0.006, 0.005), "ez", impedance=50.0, extent=0.002)
    sim.add_probe((0.009, 0.004, 0.006), "ez")
    sim.add_probe((0.003, 0.008, 0.005), "hx")
    sim.add_dft_plane_probe(axis="z", coordinate=0.006, component="ez",
                            n_freqs=3, name="pz")
    sim.add_flux_monitor(axis="x", coordinate=0.010, n_freqs=3, name="fx")
    return sim


def _observables(res) -> dict[str, np.ndarray]:
    out = {"time_series": np.asarray(res.time_series),
           "s_params": np.asarray(res.s_params)}
    for name, plane in res.dft_planes.items():
        out[f"dft[{name}]"] = np.asarray(plane.accumulator)
    for name, fm in res.flux_monitors.items():
        for acc in ("e1_dft", "e2_dft", "h1_dft", "h2_dft"):
            out[f"flux[{name}].{acc}"] = np.asarray(getattr(fm, acc))
    for comp in _COMPS:
        out[f"state.{comp}"] = np.asarray(getattr(res.state, comp))
    return out


def _slice_index(sim):
    return sim._build_grid().position_to_index((0.0, 0.0, 0.006))[2]


@pytest.fixture(scope="module")
def runs():
    """``{(boundary, n_steps, interval or None): Result}`` for the loaded
    model, with CPML (absorber path) and with PEC walls (the model a GPU runs
    through the fused H+E kernel). ``runs[(n, m)]`` is the CPML entry."""
    out = {}
    for boundary in ("cpml", "pec"):
        sim = _loaded_sim(boundary)
        kz = _slice_index(sim)
        for n in (40, 41):
            out[(boundary, n, None)] = sim.run(
                n_steps=n, compute_s_params=True, skip_preflight=True)
            for m in (1, 3, 10):
                spec = SnapshotSpec(interval=m, components=("ez", "hy"),
                                    slice_axis=2, slice_index=kz)
                out[(boundary, n, m)] = sim.run(
                    n_steps=n, snapshot=spec, compute_s_params=True,
                    skip_preflight=True)
    for (boundary, n, m), res in list(out.items()):
        if boundary == "cpml":
            out[(n, m)] = res
    return out


@pytest.mark.parametrize("n", [40, 41])
@pytest.mark.parametrize("m", [1, 3, 10])
@pytest.mark.parametrize("boundary", ["cpml", "pec"])
def test_frame_count_is_n_steps_over_interval(runs, boundary, n, m):
    res = runs[(boundary, n, m)]
    for comp in ("ez", "hy"):
        frames = np.asarray(res.snapshots[comp])
        assert frames.shape[0] == n // m, (
            f"interval={m}, n_steps={n}: {frames.shape[0]} frames of "
            f"{comp}, expected {n // m} (#1258)")
        axes = res.snapshot_axes[comp]
        np.testing.assert_array_equal(
            axes.steps, np.arange(m, (n // m) * m + 1, m))
        assert axes.interval == m and axes.n_steps == n


@pytest.mark.parametrize("n", [40, 41])
@pytest.mark.parametrize("m", [3, 10])
@pytest.mark.parametrize("boundary", ["cpml", "pec"])
def test_frames_are_every_mth_step_of_the_interval_1_run(runs, boundary, n, m):
    every = runs[(boundary, n, 1)].snapshots
    got = runs[(boundary, n, m)].snapshots
    for comp in ("ez", "hy"):
        want = np.asarray(every[comp])[m - 1::m][: n // m]
        assert np.array_equal(np.asarray(got[comp]), want), (
            f"interval={m} {comp} frames are not the interval-1 frames after "
            f"steps {m}, {2 * m}, ... bit for bit")


@pytest.mark.parametrize("n", [40, 41])
@pytest.mark.parametrize("m", [1, 3, 10])
@pytest.mark.parametrize("boundary", ["cpml", "pec"])
def test_every_other_output_is_bit_identical_to_the_run_without_snapshot(
        runs, boundary, n, m):
    """n=41 is not a multiple of 3 or 10, so the remainder scan runs."""
    ref = _observables(runs[(boundary, n, None)])
    got = _observables(runs[(boundary, n, m)])
    # Liveness: an identity over zeros would pass for the wrong reason.
    for key in ("time_series", "s_params", "dft[pz]", "flux[fx].e1_dft",
                "state.ez", "state.hy"):
        assert np.abs(ref[key]).max() > 0.0, f"{key} is identically zero"
    moved = [k for k in ref if not np.array_equal(ref[k], got[k])]
    assert not moved, (
        f"{boundary}: recording snapshots at interval={m} moved {moved} "
        f"(n_steps={n})")


def test_chunked_progress_run_records_the_same_frames(runs, capsys):
    """``report_every`` chunks that do not start on a multiple of the
    interval (16 is not a multiple of 3) still record at global steps
    3, 6, ... and still leave every other output bit-identical."""
    sim = _loaded_sim()
    spec = SnapshotSpec(interval=3, components=("ez", "hy"), slice_axis=2,
                        slice_index=_slice_index(sim))
    res = sim.run(n_steps=41, snapshot=spec, compute_s_params=True,
                  skip_preflight=True, report_every=16)
    assert capsys.readouterr().out.count("PROGRESS") == 3
    ref = runs[(41, 3)]
    for comp in ("ez", "hy"):
        assert np.array_equal(np.asarray(res.snapshots[comp]),
                              np.asarray(ref.snapshots[comp]))
    base = _observables(runs[(41, None)])
    got = _observables(res)
    assert not [k for k in base if not np.array_equal(base[k], got[k])]


def _box_grid_run(interval, n_steps, *, slice_axis=None, slice_index=None,
                  checkpoint_segments=None):
    """Low-level run on a bare grid (no port): a point source in a PEC box.

    The drive is sampled once for 64 steps and truncated, so runs of
    different lengths see the same bits. (``make_source`` evaluated for a
    different ``n_steps`` differs in the last bit of some samples: XLA
    vectorises the waveform differently per array length.)
    """
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                cpml_layers=0)
    mats = init_materials(grid.shape)
    src = make_source(grid, (0.007, 0.011, 0.009), "ez",
                      GaussianPulse(f0=5e9, bandwidth=0.8), 64)
    src = src._replace(waveform=src.waveform[:n_steps])
    spec = (None if interval is None else
            SnapshotSpec(interval=interval, components=("ez", "hx"),
                         slice_axis=slice_axis, slice_index=slice_index))
    return low_level_run(grid, mats, n_steps, boundary="pec", sources=[src],
                         snapshot=spec,
                         checkpoint_segments=checkpoint_segments)


def test_checkpoint_segments_record_the_same_frames():
    ref = _box_grid_run(None, 48)
    every = _box_grid_run(1, 48)
    seg = _box_grid_run(4, 48, checkpoint_segments=3)   # segments of 16
    for comp in ("ez", "hx"):
        assert np.array_equal(np.asarray(seg.snapshots[comp]),
                              np.asarray(every.snapshots[comp])[3::4])
        assert np.array_equal(np.asarray(getattr(seg.state, comp)),
                              np.asarray(getattr(ref.state, comp)))
    with pytest.raises(ValueError, match="does not divide the checkpoint"):
        _box_grid_run(5, 48, checkpoint_segments=3)


@pytest.mark.parametrize("bad", [0, -2, 2.5, True, "10"])
def test_interval_below_one_or_not_an_integer_is_refused(bad):
    with pytest.raises(ValueError, match="interval must be an integer >= 1"):
        _box_grid_run(bad, 8)


def test_frame_times_follow_the_scans_own_step_counter():
    """Frame k is the state after ``steps[k]`` updates: the same bits as a
    run of that many steps, whose own step counter reads ``steps[k]``. E is
    at ``steps * dt``; H was last updated half a step earlier."""
    m, n = 7, 30
    res = _box_grid_run(m, n)
    ez_axes, hx_axes = res.snapshot_axes["ez"], res.snapshot_axes["hx"]
    np.testing.assert_array_equal(ez_axes.steps, [7, 14, 21, 28])
    for k in (0, len(ez_axes.steps) - 1):
        s = int(ez_axes.steps[k])
        short = _box_grid_run(None, s)
        assert int(short.state.step) == s
        assert np.array_equal(np.asarray(res.snapshots["ez"])[k],
                              np.asarray(short.state.ez))
        assert np.array_equal(np.asarray(res.snapshots["hx"])[k],
                              np.asarray(short.state.hx))
    dt = float(res.grid.dt)
    assert ez_axes.dt == dt
    np.testing.assert_array_equal(ez_axes.times_s, ez_axes.steps * dt)
    np.testing.assert_array_equal(hx_axes.times_s, (hx_axes.steps - 0.5) * dt)


def test_slice_index_is_a_padded_lattice_index():
    """``slice_index`` indexes the padded arrays of ``Result.state`` directly
    (CPML cells counted), and ``slice_coord`` is the physical plane of each
    component's samples on it."""
    sim = _loaded_sim()
    grid = sim._build_grid()
    assert grid.pad_z_lo > 0, "the test needs a CPML pad to tell the two apart"
    z_phys = 0.006
    kz = grid.position_to_index((0.0, 0.0, z_phys))[2]
    spec = SnapshotSpec(interval=5, components=("ez", "ex"), slice_axis=2,
                        slice_index=kz)
    res = sim.run(n_steps=20, snapshot=spec, skip_preflight=True)
    assert np.array_equal(np.asarray(res.snapshots["ez"])[-1],
                          np.asarray(res.state.ez)[:, :, kz])
    assert np.array_equal(np.asarray(res.snapshots["ex"])[-1],
                          np.asarray(res.state.ex)[:, :, kz])
    ax_ez, ax_ex = res.snapshot_axes["ez"], res.snapshot_axes["ex"]
    assert ax_ez.slice_index == kz and ax_ez.slice_axis == "z"
    assert ax_ez.dims == ("frame", "x", "y")
    assert ax_ex.slice_coord == pytest.approx(z_phys, abs=1e-12)
    assert ax_ez.slice_coord == pytest.approx(z_phys + 0.5 * _DX, abs=1e-12)
    # The helper that needs no run agrees with the run's own axes.
    pre = snapshot_axes(grid, spec, 20)
    for comp in ("ez", "ex"):
        assert pre[comp].slice_coord == res.snapshot_axes[comp].slice_coord
        for a in ("x", "y"):
            assert np.array_equal(pre[comp].coords[a],
                                  res.snapshot_axes[comp].coords[a])


def test_sample_positions_put_the_pec_walls_where_the_box_was_declared():
    lo, hi = (0.005, 0.006, 0.007), (0.010, 0.011, 0.011)
    sim = Simulation(freq_max=10e9, domain=(0.016, 0.016, 0.016), dx=_DX,
                     boundary="cpml", cpml_layers=4)
    sim.add(Box(lo, hi), material="pec")
    sim.add_source((0.003, 0.004, 0.005), "ez")
    sim.add_source((0.013, 0.012, 0.0135), "ex")
    res = sim.run(n_steps=80, skip_preflight=True,
                  snapshot=SnapshotSpec(interval=20, components=_COMPS))
    tol = 1e-9
    for comp in _COMPS:
        axes = res.snapshot_axes[comp]
        assert axes.dims == ("frame", "x", "y", "z")
        field = np.asarray(res.snapshots[comp])[-1]
        X, Y, Z = np.meshgrid(axes.coords["x"], axes.coords["y"],
                              axes.coords["z"], indexing="ij")
        inside = np.ones(field.shape, bool)
        near = np.ones(field.shape, bool)
        for C, a, b in zip((X, Y, Z), lo, hi):
            inside &= (C >= a - tol) & (C <= b + tol)
            near &= (C >= a - 1.01 * _DX) & (C <= b + 1.01 * _DX)
        zero = field == 0.0
        assert inside.sum() > 0 and (near & ~inside).sum() > 0
        wrong_zero = zero & near & ~inside
        wrong_live = ~zero & inside
        assert not wrong_live.any() and not wrong_zero.any(), (
            f"{comp}: {int(wrong_live.sum())} samples reported inside the "
            f"PEC box are live, {int(wrong_zero.sum())} reported just outside "
            f"it are exactly zero -- the reported positions do not put the "
            f"walls at the declared corners {lo} .. {hi}")


def test_checkpoint_round_trip_keeps_positions_and_times(tmp_path):
    pytest.importorskip("h5py")
    res = _box_grid_run(5, 23, slice_axis=1, slice_index=9)
    path = tmp_path / "snaps.h5"
    save_snapshots(path, res.snapshots, grid=res.grid, dt=float(res.grid.dt),
                   axes=res.snapshot_axes)
    snaps, meta = load_snapshots(path)
    assert meta["n_frames"] == 4
    for comp in ("ez", "hx"):
        assert np.array_equal(snaps[comp], np.asarray(res.snapshots[comp]))
        a, b = res.snapshot_axes[comp], meta["axes"][comp]
        for field in ("component", "dims", "stagger", "slice_axis",
                      "slice_index", "slice_coord", "interval", "n_steps",
                      "dt"):
            assert getattr(a, field) == getattr(b, field), field
        assert np.array_equal(a.steps, b.steps)
        assert np.array_equal(a.times_s, b.times_s)
        assert a.coords.keys() == b.coords.keys()
        for name in a.coords:
            assert np.array_equal(a.coords[name], b.coords[name])


def test_until_decay_loop_records_at_the_interval():
    sim = Simulation(freq_max=10e9, domain=_DOM, dx=_DX, boundary="cpml",
                     cpml_layers=6)
    sim.add_source((0.006, 0.006, 0.006), "ez")
    sim.add_probe((0.008, 0.006, 0.006), "ez")
    spec = SnapshotSpec(interval=9, components=("ez",), slice_axis=2,
                        slice_index=_slice_index(sim))
    kw = dict(until_decay=1e-2, decay_check_interval=10, decay_min_steps=40,
              decay_max_steps=400, skip_preflight=True)
    res = sim.run(snapshot=spec, **kw)
    every = sim.run(snapshot=spec._replace(interval=1), **kw)
    n = np.asarray(res.time_series).shape[0]
    assert n == np.asarray(every.time_series).shape[0]
    axes = res.snapshot_axes["ez"]
    assert axes.n_steps == n
    np.testing.assert_array_equal(axes.steps, np.arange(9, n + 1, 9))
    frames = np.asarray(res.snapshots["ez"])
    assert frames.shape[0] == n // 9 > 0, (
        f"the decay loop recorded {frames.shape[0]} frames over {n} steps at "
        f"interval 9, expected {n // 9}")
    assert np.asarray(every.snapshots["ez"]).shape[0] == n
    assert np.abs(frames).max() > 0.0
    assert np.array_equal(frames,
                          np.asarray(every.snapshots["ez"])[8::9][: n // 9])


def test_animation_labels_frames_with_the_times_they_were_taken(runs):
    """``save_field_animation(interval=3)`` strides over RECORDED frames: on
    a run recorded every 10 steps it shows steps 10 and 40 of 40, and titles
    them with those times (E at s*dt, H half a step earlier)."""
    from rfx.animation import _animation_frame_times

    res = runs[(40, 10)]
    dt = float(res.dt)
    np.testing.assert_array_equal(
        _animation_frame_times(res, "ez", 3), np.array([10, 40]) * dt)
    np.testing.assert_array_equal(
        _animation_frame_times(res, "hy", 3), (np.array([10, 40]) - 0.5) * dt)
    assert _animation_frame_times(dict(res.snapshots), "ez", 3) is None
