"""Declared-face PMC images agree with the native 60-step field record."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import CurlBoundary, h_neighbor
from rfx import Simulation, Box, DebyePole
from rfx.boundaries.spec import Boundary, BoundarySpec

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def build(scene):
    cells = (4 if scene == 's3' else 12, 10, 8)
    boundary = dict(x='pec', y='pec', z='pec')
    if scene == 's2':
        boundary['x'] = Boundary(lo='pec', hi='cpml')
    if scene in ('s4', 's6'):
        boundary['z'] = 'cpml'
    if scene == 's1':
        boundary.update(y=Boundary('pmc', 'pec'), z=Boundary('pec', 'pmc'))
    elif scene == 's2':
        boundary['x'] = Boundary('pmc', 'cpml')
    elif scene == 's3':
        boundary['x'] = Boundary('pec', 'pmc')
    elif scene in ('s4', 's6'):
        boundary.update(x=Boundary('pmc', 'pec'), y=Boundary('pec', 'pmc'))
    elif scene == 's5':
        boundary['y'] = Boundary('pec', 'pmc')
    profiles = {}
    if scene in ('s4', 's6'):
        for axis, n in (('x', cells[0]), ('z', cells[2])):
            profile = np.full(n, .001)
            profile[2:4] = [.0009, .0011]
            profiles[f'd{axis}_profile'] = profile
    sim = Simulation(freq_max=15e9, domain=tuple(n*.001 for n in cells), dx=.001,
                     boundary=BoundarySpec(**boundary), cpml_layers=2 if scene in ('s2','s4','s6') else 0,
                     **profiles)
    sim.add_source((.001, .003, .002), 'ez', amplitude_kind='field',
                   waveform=lambda t: jnp.exp(-((t-1.2e-11)/5e-12)**2))
    sim.add_probe((.003, .006, .005), 'ez')
    if scene in ('s5', 's6'):
        sim.add_material('ade', eps_r=2.3, debye_poles=[DebyePole(delta_eps=1.7, tau=1e-11)])
        sim.add(Box((.004, .002, .001), (.009, .007, .006)), material='ade')
    return sim


def test_framed_neighbors_match_global_physical_rows():
    h = jnp.asarray(np.random.default_rng(42).normal(size=(7, 4, 3)), dtype=jnp.float32)
    boundary = CurlBoundary(pmc_faces=frozenset(('x_lo', 'x_hi')))
    want = h_neighbor(h, 0, boundary=boundary)
    # Two four-row slabs, one alignment row, and one exchanged ghost each side.
    slabs = (jnp.pad(h[:5], ((1, 0), (0, 0), (0, 0))),
             jnp.pad(h[3:], ((0, 2), (0, 0), (0, 0))))
    for rank, slab in enumerate(slabs):
        def read(local, owner):
            return h_neighbor(local, 0, boundary=boundary, frame=(1, 4),
                              owned={'x_lo': owner == 0, 'x_hi': owner == 1})
        got = jax.jit(read)(slab, jnp.int32(rank))
        count = 4 if rank == 0 else 3
        np.testing.assert_array_equal(got[1:1+count], want[rank*4:rank*4+count])
        got_static = h_neighbor(slab, 0, boundary=boundary, frame=(1, 4),
                               owned={'x_lo': rank == 0, 'x_hi': rank == 1})
        np.testing.assert_array_equal(got_static, got)


@pytest.mark.parametrize('kwargs', ({'frame': (0, 3)}, {'owned': {'x_lo': False}}))
def test_indexed_neighbor_refuses_slab_frame(kwargs):
    with pytest.raises(ValueError, match='indexed h_neighbor'):
        h_neighbor(jnp.ones((3, 3, 3)), 0, index=(0, 0, 0), **kwargs)


def measure(scene, record_property):
    from tests.contracts.path_equivalence.comparison import component_peaks, compare
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2
    records = [{f: [] for f in FIELDS} for _ in range(2)]
    for steps in range(1, 61):
        results = [build(scene).run(n_steps=steps, skip_preflight=True, compute_s_params=False, **kw)
                   for kw in ({}, {'devices': devices})]
        for record, result in zip(records, results):
            for f in FIELDS:
                record[f].append(np.asarray(getattr(result.state, f)))
        assert results[1].grid.nx % 2 == 1
        if scene == 's3':
            assert (results[1].grid.nx + 1)//2 - 1 == 2

    for record, result in zip(records, results):
        record.update({f: np.stack(record[f]) for f in FIELDS})
        record['probe'] = np.asarray(result.time_series)
    peaks = component_peaks(*records)
    measurements = []
    failures = []
    for f in (*FIELDS, 'probe'):
        try:
            compare(records[0][f], records[1][f], record=f, kind='step',
                    measurements=measurements, peak=peaks['ez'] if f == 'probe' else peaks[f])
        except AssertionError as exc:
            failures.append(str(exc))
    summary = {m['record']: m['difference']/float(np.spacing(np.float32(m['peak'])))
               for m in measurements}
    traces = {}
    for f in (*FIELDS, 'probe'):
        a, b = records[0][f], records[1][f]
        peak = peaks['ez'] if f == 'probe' else peaks[f]
        differences = np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)).reshape(60, -1), axis=1)
        traces[f] = [dict(step=i+1, difference=float(d), peak=peak,
                          ulp=float(d)/float(np.spacing(np.float32(peak)))) for i, d in enumerate(differences)]
    probe_peak = max(float(np.max(np.abs(r['probe']))) for r in records)
    own_ulp = max(t['difference'] for t in traces['probe'])/float(np.spacing(np.float32(probe_peak)))
    row = dict(traces=traces, probe_own_peak_ulp=own_ulp, scene=scene, steps=60, nx=int(results[1].grid.nx), ulps=summary,
               measurements=measurements, failures=failures)
    record_property('pmc_record', json.dumps(row))
    assert not failures, failures
    jax.clear_caches()


@pytest.mark.parametrize('scene', ('s1', 's2', 's3', 's4', 's5', 's6'))
def test_pmc_records_match_single_device(scene, record_property):
    measure(scene, record_property)
