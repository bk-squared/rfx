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
from tests.contracts.path_equivalence.comparison import array_peak, component_peaks, container


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


def use_per_cell_reference(monkeypatch):
    import importlib
    from tests.unit.runners._until_decay_reference import _reference_drives
    def layout(meta, dtype, **_kwargs):
        return _reference_drives(meta, np.empty(0, dtype=dtype))
    for name in ('rfx.simulation', 'rfx.nonuniform', 'rfx.vmap_sweep',
                 'rfx.subgridding.jit_runner', 'rfx.runners.distributed_v2',
                 'rfx.runners.distributed_nu', 'rfx.runners._distributed_common'):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, 'drive_layout', layout)
        if hasattr(module, 'inject_drives'):
            monkeypatch.setattr(module, 'inject_drives', per_cell)


def run_case(lane, n=4, amplitude=1., component="ez"):
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
                                     .003+(i//16)%4*.001), component=component,
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


@pytest.mark.parametrize('lane', ['uniform', 'graded', 'distributed', 'distributed_graded', 'vmap', 'subgridded'])
def test_production_step_scatter_count(lane, monkeypatch):
    import importlib
    module = importlib.import_module({
        'uniform': 'rfx.simulation', 'graded': 'rfx.nonuniform',
        'distributed': 'rfx.runners._distributed_common',
        'distributed_graded': 'rfx.runners._distributed_common',
        'vmap': 'rfx.vmap_sweep', 'subgridded': 'rfx.subgridding.jit_runner',
    }[lane])
    inject = module.inject_drives
    def checked_inject(state, drives, values):
        assert drives.waves is None
        assert all(isinstance(a, np.ndarray) for a in jax.tree.leaves(drives))
        return inject(state, drives, values)
    monkeypatch.setattr(module, 'inject_drives', checked_inject)
    original = jax.lax.scan
    counts = []

    def scan(f, init, xs, *args, **kwargs):
        sample = jax.tree.map(lambda x: x[0], xs)
        lowered = jax.jit(f).lower(init, sample)
        # Compile the actual production step, then inspect its StableHLO,
        # before CPU's scatter-to-loop expansion removes scatter instructions.
        lowered.compile()
        hlo = lowered.as_text()
        count = full_field_scatters(hlo)
        counts.append(count)
        return original(f, init, xs, *args, **kwargs)

    monkeypatch.setattr(jax.lax, 'scan', scan)
    for n in (1, 64):
        if lane == 'vmap':
            # Inspect the actual compiled batched production scan, including
            # rank-four field operands after batching, not an unbatched proxy.
            with monkeypatch.context() as patch:
                patch.setattr(jax.lax, 'scan', original)
                fn, args = vmap_case(n)
                lowered = jax.jit(jax.vmap(fn)).lower(*args)
                lowered.compile()
                counts.append(full_field_scatters(lowered.as_text()))
        elif lane == 'subgridded':
            subgrid_case(n)
        else:
            run_case(lane, n)
    assert len(counts) == 2 and counts[0] > 0, counts
    checked_counts(counts)
    print(lane, 'full-field scatters (1, 64):', counts)


def assert_peak(got, expected, ulp=9, *, peak=None):
    got, expected = np.asarray(got), np.asarray(expected)
    peak = array_peak(got, expected) if peak is None else peak
    tolerance = ulp * abs(np.spacing(np.float32(peak)))
    assert np.max(np.abs(got-expected), initial=0) <= tolerance


def assert_peak_tree(got, expected, *, peak=None):
    """Retain component names until E/H peaks have been computed."""
    got, expected = container(got), container(expected)
    if isinstance(got, dict):
        assert got.keys() == expected.keys()
        peaks = component_peaks(got, expected)
        for key in got:
            assert_peak_tree(got[key], expected[key], peak=peaks.get(key))
    elif isinstance(got, (tuple, list)):
        assert len(got) == len(expected)
        for a, b in zip(got, expected):
            assert_peak_tree(a, b)
    elif hasattr(got, 'dtype') and np.issubdtype(got.dtype, np.number):
        assert_peak(got, expected, peak=peak)


@pytest.mark.parametrize('lane', ['uniform', 'graded', 'distributed', 'distributed_graded'])
def test_production_equivalence(lane, monkeypatch):
    got = run_case(lane)
    use_per_cell_reference(monkeypatch)
    expected = run_case(lane)
    assert_peak_tree(got, expected)


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
    def run():
        return port_scene(kind, lane).run(n_steps=24, compute_s_params=kind == 'wire',
                                         s_param_freqs=jnp.array([1.8e9, 2e9, 2.2e9]) if kind == 'wire' else None)
    got = run()
    use_per_cell_reference(monkeypatch)
    expected = run()
    assert_peak_tree(got.state, expected.state)
    assert_peak(got.time_series, expected.time_series)
    assert got.dft_planes and expected.dft_planes
    if kind == 'wire':
        assert got.s_params is not None
        if lane == 'graded':
            assert got.wire_port_sparams is not None
    # Compare all complex DFT/S leaves exposed by the public result.
    for name in ('s_params', 'dft_planes', 'wire_port_sparams'):
        a, b = getattr(got, name, None), getattr(expected, name, None)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            if hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.number):
                peak = np.max(np.abs(y), initial=0)
                assert np.max(np.abs(np.asarray(x)-np.asarray(y)), initial=0) <= 1e-4*peak


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_msl_s_equivalence(lane, monkeypatch):
    def run():
        return port_scene('msl', lane).compute_msl_s_matrix(
            n_steps=32, freqs=jnp.array([1.8e9, 2e9, 2.2e9])).S
    got = np.asarray(run())
    use_per_cell_reference(monkeypatch)
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
    use_per_cell_reference(monkeypatch)
    expected = run()
    assert_peak_tree(got, expected)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_waveform_gradient_equivalence(lane, monkeypatch):
    def loss(amplitude):
        result = run_case(lane, amplitude=amplitude)
        ts = result['time_series'] if isinstance(result, dict) else result.time_series
        return jnp.sum(ts**2)
    got = jax.grad(loss)(jnp.float32(1.))
    use_per_cell_reference(monkeypatch)
    expected = jax.grad(loss)(jnp.float32(1.))
    assert np.isfinite(expected) and abs(expected) > 0
    assert abs(got-expected) <= 1e-4*abs(expected)


def test_uniform_magnetic_source_equivalence(monkeypatch):
    import rfx.simulation as uniform
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001,
                cpml_layers=0, cpml_axes='')
    mats = MaterialArrays(jnp.ones(grid.shape), jnp.zeros(grid.shape), jnp.ones(grid.shape))
    wave = jnp.sin(jnp.arange(12)*.7)*.01
    mags = [uniform.MagneticSourceSpec(4, 4, 4, c, wave*(i+1))
            for i, c in enumerate(('hx', 'hy', 'hx'))]
    def run():
        return uniform.run(grid, mats, 12, mag_sources=mags,
                           probes=[ProbeSpec(4, 4, 4, 'hx')])
    got = run()
    use_per_cell_reference(monkeypatch)
    expected = run()
    assert_peak_tree(got, expected)


def test_decay_preserves_unequal_waveform_padding(monkeypatch):
    import rfx.simulation as uniform
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001,
                cpml_layers=0, cpml_axes='')
    mats = MaterialArrays(jnp.ones(grid.shape), jnp.zeros(grid.shape), jnp.ones(grid.shape))
    sources = [SourceSpec(4, 4, 4, 'ez', jnp.array([.1, .2])),
               SourceSpec(4, 4, 4, 'ez', jnp.array([.3, -.1, .2]))]
    mags = [uniform.MagneticSourceSpec(4, 4, 4, 'hx', jnp.array([.1])),
            uniform.MagneticSourceSpec(4, 4, 4, 'hx', jnp.array([.2, -.1]))]
    def run():
        return uniform.run_until_decay(grid, mats, sources=sources, mag_sources=mags,
            probes=[ProbeSpec(4, 4, 4, 'ez')], min_steps=12, max_steps=12, check_interval=4)
    got = run()
    use_per_cell_reference(monkeypatch)
    expected = run()
    assert_peak_tree(got, expected)


def full_field_scatters(hlo):
    blocks = re.findall(r'"stablehlo.scatter"[\s\S]*?->[^\n]*', hlo)
    return sum(any(np.prod([int(x) for x in dims.rstrip('x').split('x')][-3:]) > 100
                   for dims in re.findall(r'tensor<((?:\d+x){3,})(?:f32|f64)', block))
               for block in blocks)


def vmap_case(n, *, mixed=False):
    from rfx.vmap_sweep import _build_vmap_scan_fn
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001, cpml_layers=0, cpml_axes='')
    sources = [SourceSpec(3+i%4, 3+(i//4)%4, 3+(i//16)%4, 'ez',
                          jnp.array([.1, -.2, .3], dtype=jnp.float32)) for i in range(n)]
    fn = _build_vmap_scan_fn(grid, 3, sources=sources,
        j_source_meta=[(5, 5, 5, 'ez')] if mixed else [], probes=[ProbeSpec(4, 4, 4, 'ez')])
    shape = (2,) + grid.shape
    mats = MaterialArrays(jnp.ones(shape), jnp.zeros(shape), jnp.ones(shape))
    waves = jnp.ones((2, 3, 1), dtype=jnp.float64) if mixed else jnp.zeros((2, 3, 0))
    return fn, (mats, waves)


def subgrid_case(n, *, component='ez', coarse=False):
    from rfx.core.yee import init_materials
    from rfx.subgridding.sbp_sat_3d import init_subgrid_3d
    import rfx.subgridding.jit_runner as sg
    grid = Grid(freq_max=1e9, domain=(.012,)*3, dx=.001, cpml_layers=0, cpml_axes='')
    cfg, _ = init_subgrid_3d(grid.shape, .001, (3, 7, 3, 7, 3, 7), ratio=2)
    sources = [(3+i%4, 3+(i//4)%4, 3+(i//16)%4, component,
                jnp.array([.1, -.2, .3])) for i in range(n)]
    opts = sg.SubgridRunOptions(sources_f=[] if coarse else sources,
        sources_c=sources if coarse or component == 'ez' else [],
        inject_sources_on_coarse_shadow=True)
    return sg.run_subgridded_jit(grid, init_materials(grid.shape),
        init_materials((cfg.nx_f, cfg.ny_f, cfg.nz_f)), cfg, 3, opts=opts)


def test_distinct_waveforms_match_independent_source_table():
    # This handwritten oracle never reads nodes, coefficients or wave IDs from
    # the production Drives. Different cells, components and overlapping cells
    # bind the assembler's mapping as well as the arithmetic.
    sources = [(2, 3, 4, 'ez'), (4, 3, 2, 'ey'), (2, 3, 4, 'ez'), (5, 4, 3, 'ez')]
    waves = jnp.array([[1., 2., -3.], [5., -7., 11.],
                       [-13., 17., 19.], [23., 29., -31.]])
    drives = drives_from_sources(sources, waves)
    got = expected = init_state((8,)*3)
    for step in range(3):
        got = jax.jit(inject_drives)(got, drives, waves[:, step])
        for source_index, (i, j, k, component) in enumerate(sources):
            field = getattr(expected, component)
            expected = expected._replace(**{component: field.at[i, j, k].add(waves[source_index, step])})
        for actual, reference in zip(got, expected):
            np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize('lane', ['distributed', 'distributed_graded', 'subgridded', 'subgridded_coarse'])
@pytest.mark.parametrize('component', ['hx', 'hy', 'hz'])
def test_electric_only_paths_refuse_h_sources(lane, component):
    with pytest.raises(ValueError, match=rf'{lane.split("_")[0]}.*{component}'):
        if lane.startswith('subgridded'):
            subgrid_case(1, component=component, coarse=lane.endswith('coarse'))
        elif lane == 'distributed_graded':
            from rfx.nonuniform import make_nonuniform_grid
            from rfx.runners.distributed_nu import build_sharded_nu_grid, run_nonuniform_distributed_pec
            profile = np.full(12, .001)
            grid = make_nonuniform_grid((.012, .012), profile, .001, cpml_layers=0)
            sg = build_sharded_nu_grid(grid, 2)
            mats = MaterialArrays(jnp.ones(grid.shape), jnp.zeros(grid.shape), jnp.ones(grid.shape))
            run_nonuniform_distributed_pec(sg, mats, None, 3, n_devices=2,
                sources=[SourceSpec(3, 3, 3, component, jnp.ones(3))])
        else:
            from dataclasses import replace
            from rfx import Simulation
            from rfx.runners.distributed_v2 import run_distributed
            sim = Simulation(freq_max=1e9, domain=(.012,)*3, dx=.001,
                             boundary='pec', cpml_layers=0)
            sim.add_source(position=(.003,)*3, component='ez')
            # Public add_source already refuses H. Exercise the runner's
            # admission of a malformed realized source, before tracing.
            sim._ports[0] = replace(sim._ports[0], component=component)
            run_distributed(sim, n_steps=3, devices=jax.devices()[:2])


def test_vmap_casts_mixed_sources_before_concatenation(monkeypatch):
    from tests._x64_compat import enable_x64
    import rfx.vmap_sweep as sweep
    original = sweep.inject_drives
    dtypes = []
    def checked(state, drives, values):
        dtypes.append(values.dtype)
        assert values.dtype == state.ex.dtype == jnp.float32
        assert drives.waves is None
        return original(state, drives, values)
    monkeypatch.setattr(sweep, 'inject_drives', checked)
    with enable_x64():
        fn, args = vmap_case(1, mixed=True)
        jax.jit(jax.vmap(fn))(*args)
    assert dtypes


@pytest.mark.parametrize('site', ['production', 'port', 'subgrid', 'magnetic', 'decay'])
@pytest.mark.parametrize('field', ['e', 'h'])
def test_field_peak_mutations(site, field):
    from rfx.core.yee import FDTDState
    strong = np.array([2.**20], dtype=np.float32)
    tiny = np.array([1e-8], dtype=np.float32)
    components = {c: tiny.copy() for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')}
    components[field + 'x'] = strong
    reference = FDTDState(**components, step=np.int32(1))

    def wrap(state):
        if site == 'port':
            return state  # test_port_equivalence compares state directly.
        if site == 'subgrid':
            return dict(state_c=state, state_f=state)
        return dict(state=state, time_series=tiny)

    ulp = np.spacing(strong[0])
    for count in (1, 2, 10):
        changed = tiny + count * ulp
        candidate = reference._replace(**{field + 'z': changed})
        with pytest.raises(AssertionError):
            assert_peak(changed, tiny)  # Old component-only call is red.
        if count == 10:
            with pytest.raises(AssertionError):
                assert_peak_tree(wrap(candidate), wrap(reference))
        else:
            assert_peak_tree(wrap(candidate), wrap(reference))
