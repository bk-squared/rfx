"""B2: independent one-device oracle, named records, and determined witnesses.

S error is max_ij |S_multi-S_one| per bin, absolute and divided by
max_ij |S_one| in that bin. Each trace error is max |multi-one| divided
by that trace's own one-device peak over all steps/cells. Witness shares
are compared per bin against the one-device error_per_bin, without scaling.
Both sides use identical scenes, FREQS, lengths and public default options.
"""
import json
import time

import jax
import numpy as np
import pytest

from rfx import Simulation
from tests._multi_device_scene import host_devices
from tests.unit.runners._msl_multidevice_scene import FREQS, build_scene, report_realized, _json


CASES = [pytest.param(scene, lane, n, length,
                     id=f'{scene}-{lane}-n{n}-{length}',
                     marks=[] if n == 2 and length == 1050 else [pytest.mark.slow])
         for scene, lane, n in [('T1A', 'uniform', 2), ('Y1A', 'uniform', 2),
                                ('T1A', 'graded', 2), ('T1A', 'uniform', 3),
                                ('Y1A', 'uniform', 3)]
         for length in (1050, 1400, 2100)]


def assert_owners(scene, n):
    row = report_realized(scene, n)
    assert not row['ownership_flags'], row
    if scene.name.startswith('T1A'):
        assert row['grid_shape'][0] == 79
        assert row['seams'] == ([40] if n == 2 else [27, 54])
        assert [p['feed']['x_indices'] for p in row['ports']] == [[18], [58]]
        assert [p['feed']['x_owners'] for p in row['ports']] == [[0], [n - 1]]
        if n == 2:
            assert [[q['prop_index'] for q in p['ladder']] for p in row['ports']] == [
                [30, 33, 36, 39, 42], [42, 39, 36, 33, 30]]
            assert [[q['x_owners'][0] for q in p['ladder']] for p in row['ports']] == [
                [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
    elif scene.name == 'Y1A':
        assert row['seams'] == ([29] if n == 2 else [20, 40])
        for port in row['ports']:
            assert port['feed']['counts_per_rank'] == ([24, 16] if n == 2 else [0, 40, 0])
    return row


def trace_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    peak = float(np.max(np.abs(expected)))
    delta = float(np.max(np.abs(actual - expected)))
    return dict(peak=peak, delta=delta, relative=delta / peak if peak else (0. if delta == 0 else float('inf')))


def compare_s(actual, expected):
    a, b = np.asarray(actual.S), np.asarray(expected.S)
    assert np.array_equal(np.isnan(a), np.isnan(b)), 'B2-a NaN pattern'
    assert np.any(np.isfinite(b)), 'B2-a requires finite bins'
    absolute = np.nanmax(np.abs(a - b), axis=(0, 1))
    peak = np.nanmax(np.abs(b), axis=(0, 1))
    relative = np.divide(absolute, peak, out=np.zeros_like(absolute), where=peak != 0)
    row = dict(absolute=absolute, relative=relative, peak=peak, nan_pattern=np.isnan(b))
    print('B2-a ' + json.dumps(_json(row)), flush=True)
    assert np.all(absolute <= 9e-6), ('B2-a absolute', row)
    assert np.all(absolute <= 1e-4 * peak), ('B2-a relative', row)
    return row


def compare_records(actual, expected, actual_runs, expected_runs):
    rows = []
    for drive, (a, b) in enumerate(zip(actual_runs, expected_runs, strict=True)):
        assert a.dft_time_records.keys() == b.dft_time_records.keys()
        records = {name: trace_error(a.dft_time_records[name], value)
                   for name, value in b.dft_time_records.items()}
        assert np.asarray(a.time_series).shape == np.asarray(b.time_series).shape
        records.update({f'probe{i}': trace_error(a.time_series[:, i], b.time_series[:, i])
                        for i in range(b.time_series.shape[1])})
        wa, wb = actual.settling_witness[drive], expected.settling_witness[drive]
        delta = np.abs(np.asarray(wa['share_per_bin']) - np.asarray(wb['share_per_bin']))
        row = dict(drive=drive, records=records, status=[wa['status'], wb['status']],
                   share_delta=delta, error_per_bin=wb['error_per_bin'])
        print('B2-b ' + json.dumps(_json(row)), flush=True)
        assert wb['status'] in ('pass', 'fail'), ('B2-b cannot judge', row)
        assert all(r['relative'] <= 9e-6 for r in records.values()), ('B2-b records', row)
        assert wa['status'] == wb['status'], ('B2-b status', row)
        assert np.all(delta <= np.asarray(wb['error_per_bin'])), ('B2-b share', row)
        if all(r['delta'] == 0 for r in records.values()):
            assert _json(wa) == _json(wb), ('B2-b exact witness', row)
        rows.append(row)
    return rows


def parity(scene_name, lane, n, length, monkeypatch):
    started = time.monotonic()
    scene = build_scene(scene_name + ('_graded' if lane == 'graded' else ''), excite_all=True)
    owners = assert_owners(scene, n)
    devices = host_devices(n)
    runs, calls = [], []
    original = Simulation.run

    def capture(sim, **kwargs):
        result = original(sim, **kwargs)
        runs.append(result)
        calls.append(kwargs.get('devices'))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Simulation, 'run', capture)
        expected = scene.sim.compute_msl_s_matrix(freqs=FREQS, n_steps=length)
        expected_runs = runs[:]
        runs.clear()
        actual = scene.sim.compute_msl_s_matrix(freqs=FREQS, n_steps=length, devices=devices)
    assert len(calls) == 4, ('one scan per drive', calls)
    assert calls == [None, None, devices, devices], ('M4 devices reaches variant.run', calls)
    for result in runs:
        for plane in result.dft_planes.values():
            assert len(plane.accumulator.sharding.device_set) == n, 'M4 accumulator device count'
    s = compare_s(actual, expected)
    records = compare_records(actual, expected, runs, expected_runs)
    row = dict(scene=scene_name, lane=lane, n=n, length=length, owners=owners,
               s=s, drives=records, duration_seconds=time.monotonic() - started)
    print('B2-case ' + json.dumps(_json(row)), flush=True)


@pytest.mark.parametrize('scene,lane,n,length', CASES)
def test_msl_s_matrix(scene, lane, n, length, monkeypatch):
    parity(scene, lane, n, length, monkeypatch)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('name', ['eps_override', 'checkpoint_every', 'checkpoint_segments'])
def test_msl_devices_refuses_forward_options(graded, name):
    scene = build_scene('T1A_graded' if graded else 'T1A', excite_all=True)
    with pytest.raises(NotImplementedError, match=name + '.*devices.*(forward|gradient)'):
        scene.sim.compute_msl_s_matrix(devices=host_devices(2), **{name: 1})
