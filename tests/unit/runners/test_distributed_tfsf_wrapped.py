"""Closed-box TF/SF wrapped-step fields, records and material gradients."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
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
    """Absolute error in float32 ULP of the independent reference family peak."""
    peak = max(float(np.max(np.abs(value))) for value in reference)
    delta = max(float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
                for want, got in zip(reference, actual))
    return dict(ulp=delta / float(np.spacing(np.float32(peak))) if peak else
                (0. if delta == 0 else float('inf')),
                relative=delta / peak if peak else (0. if delta == 0 else float('inf')),
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
        assert all(np.isfinite(v['ulp']) and v['ulp'] <= 9 for v in scores.values()), row
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


@pytest.mark.parametrize('steps', [60, 240])
def test_t4_material_gradient(steps):
    if jax.device_count() < 2:
        pytest.skip('requires two forced CPU devices')

    def objective(eps_r, devices):
        sim = model('pec', eps_r)
        options = dict(n_steps=steps, skip_preflight=True, checkpoint=False)
        if devices > 1:
            options.update(distributed=True, devices=jax.devices('cpu')[:devices])
        result = sim.forward(**options)
        return jnp.sum(result.time_series[:, 2] ** 2)

    one, two = (np.asarray(jax.grad(lambda eps: objective(eps, devices))(
        jnp.float32(2.25))) for devices in (1, 2))
    score = metric([one], [two])
    row = dict(bar='T4', steps=steps, devices=2, reference=float(one),
               actual=float(two), **score)
    report(row)
    assert one != 0 and np.isfinite(one) and np.isfinite(two), row
    assert score['ulp'] <= 9 and score['relative'] <= 1e-4, row
    jax.clear_caches()
