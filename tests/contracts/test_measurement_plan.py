"""M1 J1: compare frozen plans with actual runner setup, before any scan.

S0 builders are reused unchanged. The oracle captures the arguments the path
hands to its stepping function, not a second call to the plan's helpers.
"""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.plan import build_measurement_plan, measurement_plan
from tests.contracts.path_equivalence.builders import BUILDERS, FREQS, build

ROWS = [('_probes', 'probe'), ('_dft_planes', 'dft_plane'), ('_flux_monitors', 'flux'),
        ('_ports', 'lumped_port'), ('_ports', 'passive_port'), ('_ports', 'wire_port'),
        ('_waveguide_ports', 'waveguide_port'), ('_ntff', 'ntff_box'),
        ('_current_moments', 'block_moments')]


class SetupCaptured(Exception):
    pass


def setup(sim, forward=False, frequencies=FREQS):
    captured = {}

    def stop(grid, *args, **kwargs):
        captured.update(kwargs)
        captured['grid'] = grid
        raise SetupCaptured

    with patch('rfx.simulation.run', stop), patch('rfx.runners.nonuniform.run_nonuniform', stop):
        with pytest.raises(SetupCaptured):
            if forward:
                sim.forward(n_steps=12, checkpoint=False, skip_preflight=True,
                            **({'port_s11_freqs': frequencies} if any(p.impedance for p in sim._ports) else {}))
            else:
                sim.run(n_steps=12, skip_preflight=True,
                        compute_s_params=any(p.extent is not None for p in sim._ports),
                        **({'s_param_freqs': frequencies} if any(p.extent is not None for p in sim._ports) else {}))
    return captured


def node_map(channel):
    result = {}
    for n in channel.nodes:
        key = (n.component, n.i, n.j, n.k)
        result[key] = result.get(key, 0.0)+n.weight
    return {key: value for key, value in result.items() if value != 0}


def observe(channel, fields):
    return sum(n.weight*getattr(fields, n.component)[n.i, n.j, n.k] for n in channel.nodes)


def rounding_bound(channel, fields):
    """float32 accumulation bound of observe(): eps32 * len * sum |w f| (a cancelling sum can be far smaller than its terms)."""
    terms = [abs(float(n.weight*getattr(fields, n.component)[n.i, n.j, n.k])) for n in channel.nodes]
    return float(np.finfo(np.float32).eps) * max(len(terms), 1) * sum(terms)


def judge(plan, actual, row, nu):
    from tests.contracts.measurement_oracles import flux_samples, jacobian_nodes, judge_metadata
    judge_metadata(plan, actual)
    owners = {o.id: o for o in plan.owners}
    expected = {f'probe:{i}' for i in range(len(actual['probes']))}
    for i, spec in enumerate(actual['probes']):
        a, b, c, component = spec
        assert node_map(owners[f'probe:{i}'].channels[0]) == {(component, a, b, c): 1.0}
    for i, cfg in enumerate(actual.get('dft_planes') or ()):
        key = f'dft_plane:{i}'
        expected.add(key)
        channel = owners[key].channels[0]
        axes = [a for a in range(3) if a != cfg.axis]
        region = cfg.region or (0, actual['grid'].shape[axes[0]], 0, actual['grid'].shape[axes[1]])
        oracle = {}
        for u in range(region[0], region[1]):
            for v in range(region[2], region[3]):
                p = [0, 0, 0]
                p[cfg.axis], p[axes[0]], p[axes[1]] = cfg.index, u, v
                oracle[(cfg.component, *p)] = 1.0
        assert node_map(channel) == oracle
        assert owners[key].frequencies == tuple(np.asarray(cfg.freqs))
    # NU waveguide setup adds its own flux witnesses; these are not declared owners.
    for i, cfg in enumerate((actual.get('flux_monitors') or ())[:1] if row[0] == '_flux_monitors' else ()):
        key = f'flux_plane:{i}'
        expected.add(key)
        owner = owners[key]
        shape = cfg.e1_dft.shape[1:]
        for c, channel in enumerate(owner.channels):
            area = cfg.dA if c in (0, 3) or cfg.dA2 is None else cfg.dA2
            np.testing.assert_array_equal([n.area for n in channel.nodes],
                                          np.tile(np.broadcast_to(area, shape).ravel(), 1 if c < 2 else 2))
            assert node_map(channel) == jacobian_nodes(
                lambda fields: jnp.sum(flux_samples(nu, cfg, fields)[c]), actual['grid'].shape)
    if row[0] == '_ports':
        ids = [key for key in owners if key.startswith('port:')]
        expected.update(ids)
        assert len(ids) == 1
        owner = owners[ids[0]]
        if nu:
            specs = actual.get('wire_ports') or ()
        else:
            specs = actual.get('wire_port_sparams') or actual.get('lumped_port_sparams') or ()
        if specs:
            spec = specs[0]
            get = spec.get if isinstance(spec, dict) else lambda k, d=None: getattr(spec, k, d)
            mid = tuple(get('mid_'+a, get(a)) for a in 'ijk')
            comp = get('component')
            axis = 'xyz'.index(comp[1])
            # Oracle uses the path's realized metadata and its metric API.
            from rfx.nonuniform import port_metric_axes, e_node_dual_spacing_at
            widths = ([np.asarray(m[0]) for m in port_metric_axes(actual['grid'])] if nu else
                      [np.full(actual['grid'].shape[a], actual['grid'].dx) for a in range(3)])
            assert node_map(owner.channels[0]) == {(comp, *mid): -float(widths[axis][mid[axis]])}
            if not nu:
                from tests.contracts.measurement_monitors import judge_v_ref
                judge_v_ref(owner, spec, actual['grid'])
            live = get('live_cells') or (mid,)
            gap = next((c for c in owner.channels if c.name == 'V_port'), None)
            if gap:
                assert node_map(gap) == {(comp, *p): -float(widths[axis][p[axis]]) for p in live}
            fields = SimpleNamespace(**{c: np.random.default_rng(i).normal(size=actual['grid'].shape)
                                        for i, c in enumerate(('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))})
            if nu:
                from rfx.nonuniform import wire_port_current
                duals = [e_node_dual_spacing_at(w, p) for w, p in zip(widths, mid)]
                oracle = wire_port_current(fields.hx, fields.hy, fields.hz, comp, *mid, *duals)
            else:
                from rfx.probes.probes import _ampere_loop
                oracle = _ampere_loop(fields, mid, comp, actual['grid'].dx, (False,)*3)
            np.testing.assert_allclose(observe(owner.channels[1], fields), oracle, rtol=2e-6, atol=1e-12)
            from rfx.probes.probes import _ampere_loop_components
            ca, aa, cb, ab = _ampere_loop_components(comp)
            expected_current = {}
            for component, back_axis, sign in ((ca, aa, 1), (cb, ab, -1)):
                a = 'xyz'.index(component[1])
                length = float(duals[a]) if nu else actual['grid'].dx
                back = list(mid)
                back[back_axis] -= 1
                expected_current[component, *mid] = sign*length
                expected_current[component, *back] = -sign*length
            assert node_map(owner.channels[1]) == expected_current

    if row[0] == '_waveguide_ports':
        from rfx.sources.waveguide_port import modal_voltage, modal_current
        for i, cfg in enumerate(actual['waveguide_ports']):
            key = f'waveguide_port:0:mode:{i}'
            expected.add(key)
            owner = owners[key]
            fields = SimpleNamespace(**{c: jnp.asarray(np.random.default_rng(j).normal(size=actual['grid'].shape))
                                        for j, c in enumerate(('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))})
            for channel, p in zip(owner.channels, (cfg.probe_x, cfg.probe_x, cfg.ref_x, cfg.ref_x)):
                oracle = (modal_voltage if channel.kind == 'E' else modal_current)(fields, cfg, p, 0.)
                np.testing.assert_allclose(observe(channel, fields), oracle, rtol=0,
                                           atol=rounding_bound(channel, fields))
            # Linear sampling Jacobian is an independent exact node/weight oracle.
            import jax
            components = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
            zeros = tuple(jnp.zeros(actual['grid'].shape) for _ in components)
            for channel, p in zip(owner.channels, (cfg.probe_x, cfg.probe_x, cfg.ref_x, cfg.ref_x)):
                sampler = modal_voltage if channel.kind == 'E' else modal_current
                def functional(*arrays):
                    return sampler(SimpleNamespace(**dict(zip(components, arrays))), cfg, p, 0.)
                gradients = jax.grad(functional, argnums=tuple(range(6)))(*zeros)
                coefficients = {}
                for component, gradient in zip(components, gradients):
                    array = np.asarray(gradient)
                    for index in zip(*np.nonzero(array)):
                        coefficients[component, *index] = float(array[index])
                assert node_map(channel) == coefficients
            assert owner.reference_planes[1].index == cfg.ref_x
            assert owner.reference_planes[2].index == cfg.probe_x
    if row[0] == '_flux_monitors' and not actual.get('flux_monitors'):
        assert owners['flux_plane:0'].availability == 'result missing'
        expected.add('flux_plane:0')
    from tests.contracts.measurement_monitors import judge_monitors
    expected.update(judge_monitors(plan, actual))
    assert set(owners) == expected, 'missing or extra measurement owner'


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r[1])
@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
@pytest.mark.parametrize('forward', [False, True], ids=['run', 'forward'])
def test_j1_actual_path_setup(row, mesh, forward):
    lane = ('fwd_' if forward else 'run_')+('uniform' if mesh == 'uniform' else 'nonuniform')
    sim = build(row, lane, graded=mesh == 'graded')
    if row[0] == '_current_moments' and mesh == 'graded':
        with pytest.raises(NotImplementedError) as refusal:
            setup(sim, forward)
        plan = measurement_plan(sim, n_steps=12, path=lane)
        owner = next(o for o in plan.owners if o.kind == 'current_moment')
        assert owner.channels == ()
        assert owner.availability == f'unavailable: {refusal.value}'
        assert plan == measurement_plan(sim, n_steps=12, path=lane)
        return
    actual = setup(sim, forward)
    plan = build_measurement_plan(sim, actual['grid'], n_steps=12, path=lane, frequencies=FREQS)
    judge(plan, actual, row, mesh != 'uniform')
    assert plan == build_measurement_plan(sim, actual['grid'], n_steps=12, path=lane, frequencies=FREQS)
    assert plan.time_base.dt == float(actual['grid'].dt)
    assert plan.time_base.time(7, 'E') == 8*plan.time_base.dt
    assert plan.time_base.time(7, 'H') == 7.5*plan.time_base.dt
    with pytest.raises(FrozenInstanceError):
        plan.owners = ()


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r[1])
def test_constant_profile_equal_nodes(row):
    uniform = build(row, 'run_uniform')
    a = measurement_plan(uniform, n_steps=12, frequencies=FREQS)
    b = measurement_plan(build(row, 'run_nonuniform', dt=a.time_base.dt), n_steps=12, frequencies=FREQS)
    for x, y in zip(a.owners, b.owners):
        assert x.id == y.id
        for channel in x.channels:
            other = next((c for c in y.channels if c.name == channel.name), None)
            if other is None:
                assert channel.name == 'V_ref' and 'NU missing pre-injection V_ref' in y.known_differences
            else:
                assert channel.nodes == other.nodes


def test_j1_mutations_reject_weight_and_dropped_owner():
    row = ('_ports', 'wire_port')
    sim = build(row, 'run_nonuniform', graded=True)
    actual = setup(sim)
    plan = measurement_plan(sim, n_steps=12, frequencies=FREQS)
    owner = next(o for o in plan.owners if o.kind == 'wire')
    v = owner.channels[0]
    boundary = float(actual['grid'].cells(2)[0])
    # Move the source to a deliberately graded edge; the S0 midpoint can land
    # beyond the two graded cells, so perturb the whole-gap weight there.
    gap = next(c for c in owner.channels if c.name == 'V_port')
    candidates = [n for n in gap.nodes if n.weight != -boundary]
    assert candidates, 'mutation must hit an unequal cell'
    bad = replace(candidates[0], weight=-boundary)
    mutated = replace(gap, nodes=tuple(bad if n == candidates[0] else n for n in gap.nodes))
    bad_owner = replace(owner, channels=tuple(mutated if c == gap else c for c in owner.channels))
    bad_plan = replace(plan, owners=tuple(bad_owner if o == owner else o for o in plan.owners))
    with pytest.raises(AssertionError):
        judge(bad_plan, actual, row, True)
    with pytest.raises((AssertionError, KeyError)):
        judge(replace(plan, owners=plan.owners[1:]), actual, row, True)
    assert v.kind == 'E'
    from rfx.measurement import plan as module
    original = module._port_cells

    def boundary_cells(grid, axis, nu):
        values = original(grid, axis, nu)
        return np.full_like(values, values[0])

    with patch.object(module, '_port_cells', boundary_cells):
        bad_plan = measurement_plan(sim, n_steps=12, frequencies=FREQS)
    with pytest.raises(AssertionError):
        judge(bad_plan, actual, row, True)


@pytest.mark.parametrize('row', list(BUILDERS), ids=lambda r: ':'.join(r))
def test_s0_owner_inventory(row):
    sim = build(row, 'run_nonuniform' if row[0] in ('_dt_pin', '_dt_min_cell') else 'run_uniform')
    plan = measurement_plan(sim, n_steps=12)
    assert sum(o.kind == 'probe' for o in plan.owners) == len(sim._probes)
    assert sum(o.kind == 'ntff' for o in plan.owners) == int(sim._ntff is not None)
    assert sum(o.kind == 'current_moment' for o in plan.owners) == int(sim._current_moments is not None)
    for attr, kind in (('_dft_planes', 'dft_plane'), ('_flux_monitors', 'flux'),
                       ('_msl_ports', 'msl'), ('_coaxial_ports', 'coax')):
        assert sum(o.kind == kind for o in plan.owners) == len(getattr(sim, attr))


def msl_sim(mesh, direction='+x'):
    from rfx import Box, Simulation
    from tests.contracts.path_equivalence.builders import DX, waveform
    shape = (30, 12, 8) if direction[1] == 'x' else (12, 30, 8)
    kw = {}
    if mesh != 'uniform':
        for a, n in zip('xyz', shape):
            cells = np.full(n, DX)
            if mesh == 'graded':
                cells[3:5] = (.9*DX, 1.1*DX)
            kw[f'd{a}_profile'] = cells
    sim = Simulation(freq_max=15e9, domain=tuple(n*DX for n in shape), dx=DX,
                     boundary='cpml', cpml_layers=2, **kw)
    sim.add(Box((0, 0, DX), (shape[0]*DX, shape[1]*DX, DX)), material='pec')
    bounds = ((0, 4*DX, 3*DX), (30*DX, 8*DX, 3*DX)) if direction[1] == 'x' else (
        (4*DX, 0, 3*DX), (8*DX, 30*DX, 3*DX))
    sim.add(Box(*bounds), material='pec')
    position = [6*DX, 6*DX, DX]
    position['xy'.index(direction[1])] = (2 if direction[0] == '+' else 28)*DX
    sim.add_msl_port(position=tuple(position), width=4*DX, height=2*DX, direction=direction,
                     n_probe_offset=5, n_probe_spacing=2, n_probes=3, eps_r_sub=2., waveform=waveform)
    return sim


@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
@pytest.mark.parametrize('direction', ['+x', '-x', '+y', '-y'])
def test_msl_calculator_setup_and_projector(mesh, direction):
    import sys
    from rfx.sources.msl_port import msl_loop_current
    from rfx.sparams._common import msl_modal_voltage
    sim = msl_sim(mesh, direction)
    plan = measurement_plan(sim, n_steps=12, path='msl_uniform' if mesh == 'uniform' else 'msl_nonuniform',
                                frequencies=FREQS)
    actual = {}

    def stop(**kwargs):
        # Read the actual calculator setup at its stepping boundary, before
        # finally restores internal registrations. No plan helper is called.
        frame = sys._getframe(1)
        actual.update(frame.f_locals)
        raise SetupCaptured

    with patch.object(sim, 'run', stop), pytest.raises(SetupCaptured):
        sim.compute_msl_s_matrix(n_steps=12, freqs=FREQS)
    from tests.contracts.measurement_oracles import judge_metadata
    judge_metadata(plan, actual)
    owner = next(o for o in plan.owners if o.kind == 'msl')
    meta = actual['port_idx_meta'][0]
    lo, hi = actual['trace_k_per_port'][0]
    grid = actual['grid']
    fields = SimpleNamespace(**{c: np.random.default_rng(i+21).normal(size=grid.shape)
                                for i, c in enumerate(('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))})
    prop = meta['prop_idx']
    for channel in owner.channels[:-1]:
        # The calculator's resolved plane coordinate is the oracle.
        position = actual['probe_xs'][0][int(channel.name.split(':')[1])]
        index = grid.index_of(prop, position)
        assert all((n.i, n.j, n.k)[prop] == index for n in channel.nodes)
        plane = np.take(fields.ez, index, axis=prop)[None]
        oracle = msl_modal_voltage(plane, j_centre=meta['j_centre'], k_lo=meta['k_lo'],
                                   k_hi=lo, dz_arr=actual['dz_arr'])
        np.testing.assert_allclose(observe(channel, fields), oracle[0], rtol=2e-6)
        assert len(channel.nodes) == lo-meta['k_lo']
        assert [n.weight for n in channel.nodes] == list(actual['dz_arr'][meta['k_lo']:lo])
    stencil = actual['h_stencils'][0]
    ha, hb = (sum(w*np.take(getattr(fields, c), p, axis=prop) for p, w in
                  zip(stencil['h_indices'], stencil['weights']))[None] for c in (meta['h_a'], meta['h_b']))
    if meta['a_is_width']:
        al, ah, bl, bh = meta['j_lo'], meta['j_hi'], lo, hi
    else:
        ha, hb = ha.transpose(0, 2, 1), hb.transpose(0, 2, 1)
        al, ah, bl, bh = lo, hi, meta['j_lo'], meta['j_hi']
    oracle = msl_loop_current(ha, hb, j_lo=al, j_hi=ah, k_trace_lo=bl, k_trace_hi=bh,
                              dy_arr=meta['a_arr'], dz_arr=meta['b_arr'], direction=direction)
    np.testing.assert_allclose(observe(owner.channels[-1], fields), oracle[0], rtol=2e-6, atol=1e-12)
    assert {n.component for n in owner.channels[-1].nodes} == {meta['h_a'], meta['h_b']}
    assert {getattr(n, 'ijk'[prop]) for n in owner.channels[-1].nodes} == set(stencil['h_indices'])


def test_coax_calculator_planes_and_voltage_weights():
    from rfx import Simulation
    from rfx.sources.sources import GaussianPulse
    from rfx.sources.coaxial_port import coaxial_line_plane_voltage
    from rfx.measurement.ports import coax_owner
    sim = Simulation(domain=(.008, .008, .040), freq_max=40e9, boundary='cpml')
    sim.add_coaxial_port((.004, .004, .020), face='top', pin_length=.005,
                         waveform=GaussianPulse(f0=8e9, bandwidth=1.2))
    actual = {}

    def stop(grid, *args, **kwargs):
        import sys
        actual.update(sys._getframe(1).f_locals)
        actual.update(kwargs)
        actual['grid'] = grid
        raise SetupCaptured

    with patch('rfx.simulation.run', stop), pytest.raises(SetupCaptured):
        sim.compute_coaxial_line_reflection(n_steps=12, freqs=FREQS, probe_count=3,
                                            probe_start_cells=3, probe_spacing_cells=2)
    grid = actual['grid']
    indices = tuple(p.index for p in actual['dft_planes'][::2])
    owner = coax_owner(grid, port_id=0, plane_indices=indices, center_xy=actual['center_xy'],
                       pin_radius=actual['a'], outer_radius=actual['b'], frequencies=FREQS,
                       reference_index=actual['z_dut'])
    plan = build_measurement_plan(sim, grid, n_steps=12, calculator_owners=(owner,))
    assert plan.owners == (owner,)
    from tests.contracts.measurement_oracles import judge_metadata
    judge_metadata(plan, actual)
    for channel, index in zip(owner.channels, indices):
        assert {n.k for n in channel.nodes} == {index}
        fields = SimpleNamespace(ex=np.random.default_rng(83).normal(size=grid.shape))
        oracle = coaxial_line_plane_voltage(grid, fields.ex[:, :, index], fields.ex[:, :, index],
                                            center_xy=actual['center_xy'], pin_radius=actual['a'],
                                            outer_radius=actual['b'])
        np.testing.assert_allclose(observe(channel, fields), oracle, rtol=2e-14, atol=1e-17)
    assert owner.reference_planes[-1].coordinate_m == (actual['z_dut']-grid.pad_z_lo)*grid.dx


@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
def test_live_gap_excludes_pec_edges(mesh):
    from rfx import Box
    from tests.contracts.path_equivalence.builders import point
    row = ('_ports', 'wire_port')
    sim = build(row, 'run_uniform' if mesh == 'uniform' else 'run_nonuniform', graded=mesh == 'graded')
    sim.add(Box(point(4, 2, 2), point(6, 4, 3)), material='pec')
    actual = setup(sim)
    plan = measurement_plan(sim, n_steps=12, frequencies=FREQS)
    judge(plan, actual, row, mesh != 'uniform')
    gap = next(c for o in plan.owners if o.kind == 'wire' for c in o.channels if c.name == 'V_port')
    assert len(gap.nodes) == 1


@pytest.mark.parametrize('mesh', ['uniform', 'constant', 'graded'])
def test_multimode_waveguide_owners(mesh):
    from rfx.measurement.ports import waveguide_owner
    from rfx.runners.nonuniform import _build_waveguide_port_config_nu
    sim = build(('_waveguide_ports', 'waveguide_port'),
                'run_uniform' if mesh == 'uniform' else 'run_nonuniform', graded=mesh == 'graded')
    mode_freqs = jnp.asarray([30e9, 50e9])
    sim._waveguide_ports[0] = replace(sim._waveguide_ports[0], n_modes=2, f0=40e9, freqs=mode_freqs)
    grid = sim._build_realized_grid()
    plan = measurement_plan(sim, n_steps=12)
    configs = (sim._build_waveguide_port_config(sim._waveguide_ports[0], grid, mode_freqs, 12) if mesh == 'uniform'
               else _build_waveguide_port_config_nu(sim, sim._waveguide_ports[0], grid, mode_freqs, 12))
    owners = [o for o in plan.owners if o.kind == 'waveguide']
    assert len(owners) == len(configs) == 2
    assert len({o.id for o in owners}) == 2
    for i, (owner, cfg) in enumerate(zip(owners, configs)):
        assert owner.mode == cfg.mode_indices
        assert owner == waveguide_owner(grid, cfg, port_id=0, mode_id=i)
    judge(plan, dict(grid=grid, probes=[(n.i, n.j, n.k, n.component)
                                      for o in plan.owners if o.kind == 'probe' for n in o.channels[0].nodes],
                     waveguide_ports=configs), ('_waveguide_ports', 'waveguide_port'), mesh != 'uniform')


def test_wire_reference_planes_use_forward_setup():
    from tests.unit.runners.test_distributed_wire_ports import _model
    sim = _model(planes=True, n_devices=1)
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    actual = {}

    def stop(grid, *args, **kwargs):
        actual.update(kwargs)
        actual['grid'] = grid
        raise SetupCaptured

    with patch('rfx.simulation.run', stop), pytest.raises(SetupCaptured):
        compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=12)
    plan = measurement_plan(sim, n_steps=12, path='fwd_uniform', frequencies=FREQS)
    for spec in actual['wire_refplane_sparams']:
        owner = next(o for o in plan.owners if o.id == f'port:{spec.port_index}:mode:0')
        plane = owner.reference_planes[1+spec.plane_slot]
        assert (plane.axis, plane.index) == (spec.line_axis, spec.plane_index)
        assert plane.coordinate_m == (spec.plane_index-actual['grid'].axis_pads[spec.line_axis])*actual['grid'].dx


@pytest.mark.parametrize('lane', ['fwd_uniform', 'fwd_nonuniform'])
def test_port_frequencies_retain_setup_float32_rounding(lane):
    sim = build(('_ports', 'wire_port'), lane)
    freqs = np.array([3e9+17, 5e9+19], dtype=np.float64)
    actual = setup(sim, forward=True, frequencies=freqs)
    plan = measurement_plan(sim, n_steps=12, path=lane, frequencies=freqs)
    owner = next(o for o in plan.owners if o.kind == 'wire')
    expected = actual['s_param_freqs'] if lane == 'fwd_nonuniform' else actual['wire_port_sparams'][0].freqs
    assert owner.frequencies == tuple(np.asarray(expected))
