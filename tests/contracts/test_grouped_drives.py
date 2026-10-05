"""S3-1: production scan scatter growth and the legacy per-cell oracle."""
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.drives import drives_from_sources, inject_drives
from rfx.core.yee import MaterialArrays, init_state
from rfx.grid import Grid
from rfx.simulation import SourceSpec, ProbeSpec


def per_cell(state, drives, wave_t):
    # Test-only old algorithm. Retains the original source order.
    ordered = sorted((int(w), c, n) for c in drives.nodes
                     for n, w in enumerate(np.asarray(drives.wave_id[c])))
    for _, c, n in ordered:
        field = getattr(state, c)
        idx = tuple(axis[n] for axis in drives.nodes[c])
        value = drives.coef[c][n] * wave_t[drives.wave_id[c][n]]
        state = state._replace(**{c: field.at[idx].add(value.astype(field.dtype))})
    return state


def run_case(lane, n=4, amplitude=1.):
    import rfx.simulation as uniform
    import rfx.nonuniform as graded
    if lane.startswith('distributed'):
        if jax.device_count() < 2:
            pytest.skip('two host devices required')
        from rfx import Simulation
        options = {}
        if lane.endswith('graded'):
            profile = np.r_[np.full(4, .001), np.full(4, .0011), np.full(4, .001)]
            options = dict(dx_profile=profile, dy_profile=profile, dz_profile=profile)
        sim = Simulation(freq_max=1e9, domain=(.012,)*3, dx=.001,
                         boundary='pec', cpml_layers=0, **options)
        for i in range(n):
            sim.add_source(position=(.003+i%4*.001, .003+(i//4)%4*.001,
                                     .003+(i//16)%4*.001), component='ez',
                           waveform=lambda t: jnp.sin(t*1e11)*.01)
        sim.add_probe(position=(.004, .004, .004), component='ez')
        return sim.run(n_steps=12, devices=jax.devices()[:2], compute_s_params=False)
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001,
                cpml_layers=0, cpml_axes='')
    if lane == 'graded':
        profile = np.r_[np.full(4, .001), np.full(4, .0011), np.full(4, .001)]
        grid = graded.make_nonuniform_grid((float(profile.sum()),)*2, profile, .001,
                                             dx_profile=profile, dy_profile=profile, cpml_layers=0)
    mats = MaterialArrays(jnp.ones(grid.shape), jnp.zeros(grid.shape), jnp.ones(grid.shape))
    waves = amplitude * jnp.sin(jnp.arange(12)*.7) * .01
    sources = [SourceSpec(3+i%4, 3+(i//4)%4, 3+(i//16)%4, 'ez', waves*(i+1)/n)
               for i in range(n)]
    probes = [ProbeSpec(4, 4, 4, 'ez')]
    if lane == 'uniform':
        result = uniform.run(grid, mats, 12, sources=sources, probes=probes)
    else:
        result = graded.run_nonuniform(grid, mats, 12, sources=sources, probes=probes)
    return result


def checked_counts(counts):
    if os.environ.get('S31_CHECKS_OFF') != '1':
        assert counts[1] == counts[0], counts


def test_checker_is_live():
    # Mutation (a): disabling the check must fail this negative control.
    with pytest.raises(AssertionError):
        checked_counts([1, 64])


@pytest.mark.parametrize('lane', ['uniform', 'graded', 'distributed', 'distributed_graded'])
def test_production_step_scatter_count(lane, monkeypatch):
    original = jax.lax.scan
    counts = []

    def scan(f, init, xs, *args, **kwargs):
        sample = jax.tree.map(lambda x: x[0], xs)
        lowered = jax.jit(f).lower(init, sample)
        # Compile the actual production step, then inspect its StableHLO,
        # before CPU's scatter-to-loop expansion removes scatter instructions.
        lowered.compile()
        hlo = lowered.as_text()
        blocks = re.findall(r'"stablehlo.scatter"[\s\S]*?->[^\n]*', hlo)
        count = sum(any(int(x)*int(y)*int(z) > 100 for x, y, z in
                        re.findall(r'tensor<(\d+)x(\d+)x(\d+)x(?:f32|f64)', block))
                    for block in blocks)
        counts.append(count)
        return original(f, init, xs, *args, **kwargs)

    monkeypatch.setattr(jax.lax, 'scan', scan)
    run_case(lane, 1)
    run_case(lane, 64)
    assert len(counts) == 2 and counts[0] > 0, counts
    checked_counts(counts)
    print(lane, 'full-field scatters (1, 64):', counts)


def assert_peak(got, expected, ulp=9):
    got, expected = np.asarray(got), np.asarray(expected)
    peak = np.max(np.abs(expected), initial=0)
    tolerance = ulp * abs(np.spacing(np.float32(peak)))
    assert np.max(np.abs(got-expected), initial=0) <= tolerance


@pytest.mark.parametrize('lane,module', [('uniform', 'rfx.simulation'), ('graded', 'rfx.nonuniform'),
                                          ('distributed', 'rfx.runners._distributed_common'),
                                          ('distributed_graded', 'rfx.runners._distributed_common')])
def test_production_equivalence(lane, module, monkeypatch):
    import importlib
    got = run_case(lane)
    monkeypatch.setattr(importlib.import_module(module), 'inject_drives', per_cell)
    expected = run_case(lane)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        if hasattr(a, 'dtype') and np.issubdtype(a.dtype, np.number):
            assert_peak(a, b)


def test_duplicates_magnetic_vmap_and_gradient():
    meta = [(3, 3, 3, c) for c in ('ez', 'hx', 'ez', 'ey')]
    drives = drives_from_sources(meta, jnp.ones((4, 3)))
    state = init_state((8,)*3)
    values = jnp.array([1., 2., 3., 4.])
    got = jax.jit(inject_drives)(state, drives, values)
    assert got.ez[3, 3, 3] == 4
    expected = per_cell(state, drives, values)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(a, b)
    batched = jax.vmap(lambda v: inject_drives(state, drives, v))(jnp.stack([values, values*2]))
    assert batched.ez[1, 3, 3, 3] == 8
    def loss(v, inject):
        return jnp.sum(inject(state, drives, v).ez**2)
    np.testing.assert_array_equal(jax.grad(loss)(values, inject_drives),
                                  jax.grad(loss)(values, per_cell))


def port_scene(kind, lane):
    from rfx import Simulation, Box
    from rfx.sources import GaussianPulse
    profile = np.r_[np.full(6, .001), np.full(4, .0011), np.full(14, .001)]
    options = dict(dx_profile=profile, dy_profile=profile, dz_profile=profile) if lane == 'graded' else {}
    sim = Simulation(freq_max=3e9, domain=(float(profile.sum()),)*3,
                     dx=.001, boundary='pec', cpml_layers=0, snap='declared', **options)
    wave = GaussianPulse(f0=2e9, bandwidth=1.2, cutoff=1.)
    if kind == 'wire':
        sim.add_port(position=(.012, .012, .012), component='ez', extent=.003,
                     impedance=50., waveform=wave)
    else:
        sim.add_material('sub', eps_r=2.2)
        sim.add(Box((0., 0., 0.), (.024, .024, .003)), material='sub')
        sim.add(Box((0., .010, .003), (.024, .014, .003)), material='pec')
        sim.add_msl_port(position=(.006, .012, 0.), width=.004, height=.003,
                         direction='+x', impedance=50., waveform=wave,
                         eps_r_sub=2.2, n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    sim.add_probe(position=(.013, .012, .005), component='ez')
    sim.add_dft_plane_probe(axis='x', coordinate=.013, freqs=jnp.array([2e9]), component='ez')
    return sim


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('kind', ['wire', 'msl'])
def test_port_equivalence(lane, kind, monkeypatch):
    import rfx.simulation as uniform
    import rfx.nonuniform as graded
    def run():
        return port_scene(kind, lane).run(n_steps=24, compute_s_params=kind == 'wire',
                                         s_param_freqs=jnp.array([1.8e9, 2e9, 2.2e9]) if kind == 'wire' else None)
    got = run()
    monkeypatch.setattr(uniform, 'inject_drives', per_cell)
    monkeypatch.setattr(graded, 'inject_drives', per_cell)
    expected = run()
    for a, b in zip(got.state, expected.state):
        assert_peak(a, b)
    assert_peak(got.time_series, expected.time_series)
    # Compare all complex DFT/S leaves exposed by the public result.
    for name in ('s_params', 'dft_planes', 'wire_port_dft'):
        a, b = getattr(got, name, None), getattr(expected, name, None)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            if hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.number):
                peak = np.max(np.abs(y), initial=0)
                assert np.max(np.abs(np.asarray(x)-np.asarray(y)), initial=0) <= 1e-4*peak


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_msl_s_equivalence(lane, monkeypatch):
    import rfx.simulation as uniform
    import rfx.nonuniform as graded
    def run():
        return port_scene('msl', lane).compute_msl_s_matrix(
            n_steps=32, freqs=jnp.array([1.8e9, 2e9, 2.2e9])).S
    got = np.asarray(run())
    monkeypatch.setattr(uniform, 'inject_drives', per_cell)
    monkeypatch.setattr(graded, 'inject_drives', per_cell)
    expected = np.asarray(run())
    assert np.all(np.isfinite(expected)) and np.max(np.abs(expected)) > 0
    assert np.max(np.abs(got-expected)) <= 1e-4*np.max(np.abs(expected))


def test_subgrid_fine_and_coarse_equivalence(monkeypatch):
    from rfx.core.yee import init_materials
    from rfx.subgridding.sbp_sat_3d import init_subgrid_3d
    import rfx.subgridding.jit_runner as sg
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001,
                cpml_layers=0, cpml_axes='')
    cfg, _ = init_subgrid_3d(grid.shape, .001, (3, 7, 3, 7, 3, 7), ratio=2)
    wave = jnp.array([.1, -.2, .3, .1])
    sources = [(3, 3, 3, 'ez', wave), (3, 3, 3, 'ez', wave*.5),
               (4, 3, 3, 'ex', wave)]
    opts = sg.SubgridRunOptions(sources_f=sources, sources_c=sources,
        inject_sources_on_coarse_shadow=True, probe_indices_f=[(3, 3, 3)],
        probe_components=['ez'], probe_indices_c=[(3, 3, 3)], probe_components_c=['ez'])
    def run():
        return sg.run_subgridded_jit(grid, init_materials(grid.shape),
            init_materials((cfg.nx_f, cfg.ny_f, cfg.nz_f)), cfg, 4, opts=opts)
    got = run()
    monkeypatch.setattr(sg, 'inject_drives', per_cell)
    expected = run()
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        if hasattr(a, 'dtype'):
            assert_peak(a, b)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_waveform_gradient_equivalence(lane, monkeypatch):
    import importlib
    def loss(amplitude):
        result = run_case(lane, amplitude=amplitude)
        ts = result['time_series'] if isinstance(result, dict) else result.time_series
        return jnp.sum(ts**2)
    got = jax.grad(loss)(jnp.float32(1.))
    monkeypatch.setattr(importlib.import_module('rfx.simulation' if lane == 'uniform'
                                               else 'rfx.nonuniform'), 'inject_drives', per_cell)
    expected = jax.grad(loss)(jnp.float32(1.))
    assert np.isfinite(expected) and abs(expected) > 0
    assert abs(got-expected) <= 1e-4*abs(expected)
