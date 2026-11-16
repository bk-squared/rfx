"""Closed-box TF/SF wrapped-step per-step fields and records."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, Simulation, SnapshotSpec, simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners import distributed_v2


COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
pytestmark = pytest.mark.distributed


def model(boundary, eps_r=2.25):
    """The same 30-cube spans both cuts; its high TF face is row 20."""
    wall = Boundary('cpml', 'cpml', 0, 0)
    boundary_spec = BoundarySpec(x=wall, y=wall, z=wall) if boundary == 'pec' else 'cpml'
    domain = .029 if boundary == 'pec' else .025
    sim = Simulation(freq_max=15e9, domain=(domain,) * 3, dx=.001,
                     boundary=boundary_spec, cpml_layers=2, precision='float32')
    grid = sim._build_grid()
    assert grid.shape == (30, 30, 30)
    if boundary == 'pec':
        assert all(face.realized == 0 for face in grid.boundary_depths)
    pad = grid.pad_x_lo

    def position(nodes):
        return tuple((n - pad) * grid.dx for n in nodes)

    sim.add_material('dielectric', eps_r=eps_r)
    sim.add(Box(position((12, 12, 12)), position((17, 17, 17))), material='dielectric')
    sim.add_tfsf_source(f0=7.5e9, bandwidth=.8, closed_box=True, margin=7)
    # These are fixed global nodes, independent of the implementation's cuts.
    for nodes in ((14, 14, 14), (15, 16, 13), (20, 14, 17), (21, 13, 15)):
        sim.add_vector_probe(position(nodes))
    sim.add_dft_plane_probe(axis='y', coordinate=position((14, 14, 14))[1],
                            component='ez', freqs=[5e9, 7.5e9, 10e9], name='plane')
    return sim


def evaluate(boundary, entry, steps, devices):
    """Observe forward's actual solve before it discards its field history."""
    sim = model(boundary)
    spec = SnapshotSpec(interval=1, components=COMPONENTS)
    options = dict(n_steps=steps, skip_preflight=True)
    selected = jax.devices('cpu')[:devices]
    if devices > 1:
        options['devices'] = selected
    if entry == 'run':
        result = sim.run(**options, snapshot=spec, compute_s_params=False)
        fields = result.snapshots
    else:
        observed = []
        owner, name = ((simulation, 'run') if devices == 1 else
                       (distributed_v2, 'run_distributed'))
        original = getattr(owner, name)

        def capture(*args, **kwargs):
            kwargs['snapshot'] = spec
            result = original(*args, **kwargs)
            observed.append(result)
            return result

        with patch.object(owner, name, capture):
            result = sim.forward(**options, checkpoint=False, distributed=devices > 1)
        assert len(observed) == 1
        fields = observed[0].snapshots
    traces = np.asarray(result.time_series)
    assert traces.dtype == np.float32 and np.isfinite(traces).all()
    fields = {name: np.asarray(fields[name]) for name in COMPONENTS}
    assert all(value.shape == (steps, 30, 30, 30) for value in fields.values())
    spectrum = np.asarray(result.dft_planes['plane'].accumulator)
    return fields, traces, spectrum


def metric(reference, actual):
    """Largest per-step error relative to the reference family peak."""
    peak = max(float(np.max(np.abs(value))) for value in reference)
    delta = max(float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
                for want, got in zip(reference, actual))
    return dict(relative=delta / peak if peak else (0. if delta == 0 else float('inf')),
                peak=peak)


def report(row):
    print(json.dumps(row, sort_keys=True), flush=True)
    target = os.environ.get('RFX_TFSF_REPORT')
    if target:
        with Path(target).open('a') as output:
            output.write(json.dumps(row, sort_keys=True) + '\n')


@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('steps', [12, 60, 240])
def test_t1_t2(boundary, entry, steps):
    if jax.device_count() < 3:
        pytest.skip('requires three forced CPU devices')
    one = evaluate(boundary, entry, steps, 1)
    two = None
    for devices in (2, 3):
        many = evaluate(boundary, entry, steps, devices)
        scores = {}
        for family, components in (('E', COMPONENTS[:3]), ('H', COMPONENTS[3:])):
            scores[family + '_fields'] = metric([one[0][c] for c in components],
                                                 [many[0][c] for c in components])
            indices = [i for i in range(one[1].shape[1]) if COMPONENTS[i % 6] in components]
            scores[family + '_probes'] = metric([one[1][:, indices]], [many[1][:, indices]])
        # DFT is complex: preserve both quadratures in the error norm.
        peak = float(np.max(np.abs(one[2])))
        delta = float(np.max(np.abs(many[2].astype(np.complex128) - one[2])))
        dft = delta / peak if peak else (0. if delta == 0 else float('inf'))
        row = dict(bar='T1', boundary=boundary, entry=entry, steps=steps,
                   devices=devices, scores=scores, dft_relative=dft)
        report(row)
        assert all(np.isfinite(v['relative']) and v['relative'] <= 1e-5
                   for v in scores.values()), row
        assert dft <= 1e-4, row
        if two is not None:
            equal = {c: bool(np.array_equal(two[0][c].view(np.uint32),
                                           many[0][c].view(np.uint32))) for c in COMPONENTS}
            equal['probes'] = bool(np.array_equal(two[1].view(np.uint32), many[1].view(np.uint32)))
            row = dict(bar='T2', boundary=boundary, entry=entry, steps=steps, equal=equal)
            report(row)
            assert all(equal.values()), row
        else:
            two = many
    jax.clear_caches()


def evaluate_ntff(devices):
    """Sample the corrected Hx ghost at the transverse TF/SF boundary."""
    sim = model('cpml')
    grid = sim._build_grid()
    lo = tuple((8 - getattr(grid, f'pad_{axis}_lo')) * grid.dx for axis in 'xyz')
    hi = tuple((22 - getattr(grid, f'pad_{axis}_lo')) * grid.dx for axis in 'xyz')
    # At x=14 the face-centre average reads x=15, the left slab's high ghost.
    # The y=8 face includes the Hx correction just outside the TF box [9,20].
    sim.add_ntff_box(lo, hi, freqs=[5e9, 7.5e9, 10e9])
    options = dict(n_steps=60, skip_preflight=True, compute_s_params=False)
    if devices > 1:
        options['devices'] = jax.devices('cpu')[:devices]
    return sim.run(**options)


def ntff_quantities(result):
    from rfx.antenna import antenna_gain
    from rfx.measurement.farfield import compute_far_field

    angles = (np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37))
    far = compute_far_field(result.ntff_data, result.ntff_box, result.grid, *angles)
    return dict(zip(result.ntff_data._fields, result.ntff_data),
                E_theta=far.E_theta, E_phi=far.E_phi, gain=antenna_gain(far))


def test_tfsf_ntff_seam():
    if jax.device_count() < 2:
        pytest.skip('requires two forced CPU devices')
    one, two = (evaluate_ntff(devices) for devices in (1, 2))
    assert one.ntff_box.face_centre and two.ntff_box.face_centre
    assert (two.ntff_box.i_lo, two.ntff_box.i_hi,
            two.ntff_box.j_lo, two.ntff_box.j_hi) == (8, 22, 8, 22)
    reference, actual = ntff_quantities(one), ntff_quantities(two)
    errors, compensation = {}, {}
    for name, value in reference.items():
        want, got = np.asarray(value), np.asarray(actual[name])
        assert np.isfinite(want).all() and np.isfinite(got).all(), name
        delta = float(np.max(np.abs(got.astype(np.complex128) - want)))
        peak = float(np.max(np.abs(want)))
        errors[name] = delta / peak if peak else delta
        if name in one.ntff_data._fields[6:]:
            sum_peak = float(np.max(np.abs(np.asarray(reference[name[2:]]))))
            compensation[name] = delta / sum_peak if sum_peak else delta
    row = dict(bar='NTFF', devices=2, steps=60, errors=errors,
               compensation_error_over_sum=compensation)
    report(row)
    # Match test_distributed_ntff.py's CPML bars, including compensation.
    assert max(value for name, value in errors.items()
               if name not in compensation) <= 5e-5, row
    assert max(compensation.values()) <= 2 * np.finfo(np.float32).eps, row
    jax.clear_caches()
