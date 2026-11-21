"""Declared conductor shorts are refused before any time step."""
from dataclasses import replace
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation, Sphere
from rfx._internal._grid_metric import field_index
from rfx.errors import AccuracyWarning, ConfigurationWarning, RfxRefusal
from rfx.model import conductors as products
from rfx.model.source_admission import refuse_shorted_lumped_rlc

DX = 1 / 1024
KINDS = ('pinned', 'drawn', 'thin', 'volume', 'volume_face', 'wire', 'wall')
ENTRIES = ('run', 'forward', 'occupancy')


def point(*values):
    return tuple(v * DX for v in values)


def model(kind='pinned', graded=False, form='fold', *, element=True):
    kw = {'dy_profile': np.array([1, .9, 1.1] + [1] * 9) * DX} if graded else {}
    sim = Simulation(15e9, point(12, 12, 12), dx=DX, boundary='pec',
                     snap='declared', **kw)
    position, component = point(6.5, 6, 6), 'ex'
    if kind == 'pinned':
        sim.add_pinned_sheet(normal_axis=2, plane_index=6,
                             i_range=(4, 9), j_range=(4, 9))
    elif kind in ('drawn', 'thin', 'lossy'):
        shape = Box(point(4, 4, 6), point(9, 9, 6))
        if kind == 'drawn':
            sim.add(shape, material='pec')
        else:
            sim.add_thin_conductor(shape, **({'sigma_bulk': 1e4, 'thickness': DX / 16}
                                            if kind == 'lossy' else {}))
    elif kind in ('volume', 'volume_face'):
        sim.add(Box(point(4, 4, 4), point(9, 9, 8)), material='pec')
        if kind == 'volume_face':
            position = point(6.5, 6, 8)
    elif kind == 'wire':
        sim.add(PolylineWire((point(4, 6, 6), point(9, 6, 6)), radius=0), material='pec')
    elif kind == 'wall':
        position = point(6.5, 6, 0)
    sim.add_source(point(2.5, 2, 2), 'ex', amplitude_kind='field')
    sim.add_probe(point(3.5, 2, 2), 'ex')
    if element:
        sim.add_lumped_rlc(position, component, R=50.,
                           **({'L': 1e-9, 'topology': 'series'} if form == 'ade' else {}))
    return sim


def grid_of(sim):
    return sim._build_nonuniform_grid() if sim._uses_nonuniform_mesh else sim._build_grid()


def expected(sim, owner='pinned_sheet[0]', *, faces=None, index=0):
    spec = sim._lumped_rlc[index]
    edge = tuple(map(int, field_index(grid_of(sim), spec.position, spec.component)))
    prefix = (f'lumped_rlc[{index}] at {tuple(spec.position)} ({spec.component}), '
              f'grid index {edge}: its own E edge is PEC, so ')
    if faces:
        return (prefix + f'the declared PEC wall face(s) {faces} short the element '
                'and it would have no effect. Move it at least one cell inside '
                'the domain, off the wall.')
    return (prefix + 'the conductor shorts the element and it would have no effect. '
            f'Realized conductor provenance (assembly): {owner}. Cut a gap in '
            'the conductor at the element, or move the element off it.')


class FirstStep(Exception):
    pass


@pytest.fixture
def step_hook(monkeypatch):
    from rfx.stepping import prepared
    seen = []

    def stop(product, *args, **kwargs):
        seen.append((product, kwargs))
        raise FirstStep

    monkeypatch.setattr(prepared, 'advance', stop)
    return seen


def execute(sim, entry, **kwargs):
    kw = dict(n_steps=1, skip_preflight=True, **kwargs)
    if entry == 'run':
        return sim.run(compute_s_params=False, **kw)
    if entry == 'occupancy':
        kw['pec_occupancy_override'] = jnp.zeros(grid_of(sim).shape)
    return sim.forward(checkpoint=False, **kw)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('kind,form', [(k, 'fold') for k in KINDS] + [('pinned', 'ade')])
def test_conductor_refusal(kind, form, entry, graded, step_hook):
    sim = model(kind, graded, form)
    owner = ('pinned_sheet[0]' if kind == 'pinned' else
             'thin_conductor[0]' if kind == 'thin' else 'geometry[0]')
    with pytest.raises(RfxRefusal) as caught:
        execute(sim, entry)
    assert str(caught.value) == expected(sim, owner, faces='z_lo' if kind == 'wall' else None)
    assert not step_hook


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('quiet', ['normal', 'off_sheet', 'lossy', 'port_release'])
def test_quiet_edges(quiet, entry, graded, step_hook):
    """Declaration presence is insufficient to establish a free realized edge."""
    sim = model('lossy' if quiet == 'lossy' else 'pinned', graded, element=False)
    if quiet == 'normal':
        sim.add_pinned_sheet(normal_axis=2, plane_index=3,
                             i_range=(4, 9), j_range=(4, 9))
        position, component = point(6, 6, 5.5), 'ez'
    elif quiet == 'off_sheet':
        position, component = point(6.5, 6, 7), 'ex'
    else:
        position, component = point(6.5, 6, 6), 'ex'
    if quiet == 'port_release':
        sim.add_port(position, component, impedance=50., excite=False)
    sim.add_lumped_rlc(position, component, R=50.)
    grid = grid_of(sim)
    root = products.realized_conductors(sim, grid, nonuniform=graded)
    edge = tuple(map(int, field_index(grid, position, component)))
    if quiet == 'normal':
        assert bool(np.asarray(root.pec_edges[0])[6, 6, 6])
        assert not bool(np.asarray(root.pec_edges[2])[edge])
        print(f'Q1 graded={graded}: ex(6,6,6)=True; ez{edge}=False')
    if quiet == 'port_release':
        assert bool(np.asarray(root.pec_edges[0])[edge])
        released = products.lumped_port_stage(root, sim._ports[-1], 'port[1]')
        assert not bool(np.asarray(released.pec_edges[0])[edge])
    with pytest.raises(FirstStep):
        execute(sim, entry)
    assert len(step_hook) == 1


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('design_kind', ['occupancy', 'region'])
def test_design_metal_keeps_element_edge_free(graded, design_kind, step_hook):
    sim = model('empty', graded)
    grid = grid_of(sim)
    root = products.realized_conductors(sim, grid, nonuniform=graded)
    assert root.pec_edges is None
    if design_kind == 'occupancy':
        kw = {'pec_occupancy_override': jnp.ones(grid.shape)}
    else:
        refusal = NotImplementedError if graded else RfxRefusal
        with pytest.raises(refusal) as caught:
            sim.forward(n_steps=1, skip_preflight=True,
                        design_box=(point(4, 4, 4), point(8, 8, 8)),
                        design_occupancy_override=jnp.ones((5, 5, 5)))
        if graded:
            assert str(caught.value) == (
                'forward(design_box=...) is supported on the fwd_uniform forward lane(s), '
                "not 'fwd_nonuniform'. Use eps_override / pec_occupancy_override there.")
        else:
            assert str(caught.value).startswith('lumped_rlc[0]: the design occupancy box')
        assert not step_hook
        return
    with pytest.warns(UserWarning, match='Port occupancy release:.*lumped_rlc'):
        with pytest.raises(FirstStep):
            sim.forward(n_steps=1, skip_preflight=True, checkpoint=False, **kw)
    product, args = step_hook[0]
    assert product.pec_edges is None
    occupancy = np.asarray(args['pec_occupancy'])
    assert all(occupancy[6, j, k] == 0 for j in (5, 6) for k in (5, 6))


def test_preflight_reports_assembly_refusal(step_hook):
    sim = model()
    report = sim.preflight()
    findings = report.by_code('assembly_refusal')
    assert len(findings) == 1
    assert str(findings[0]) == 'The model cannot be assembled: ' + expected(sim)
    assert not report.ok
    assert not step_hook


def test_distributed_v2_uses_element_refusal(step_hook):
    from tests.unit.runners.test_distributed_admission_refusals import _devices
    sim = model()
    with pytest.raises(RfxRefusal) as caught:
        execute(sim, 'run', devices=_devices())
    assert str(caught.value) == expected(sim)
    assert not step_hook


def test_vmap_sweep_uses_sequential_element_admission(monkeypatch, step_hook):
    import rfx.vmap_sweep as sweep
    sim = model()
    sim.add_material('design', eps_r=2.)
    sim.add(Box(point(1, 8, 1), point(2, 9, 2)), material='design')
    calls = []
    original = sweep._sequential_fallback

    def record(*args, **kw):
        calls.append(True)
        return original(*args, **kw)

    monkeypatch.setattr(sweep, '_sequential_fallback', record)
    with pytest.raises(RfxRefusal) as caught:
        sweep.vmap_material_sweep(sim, 'design.eps_r', [2., 3.], n_steps=1)
    assert str(caught.value) == expected(sim)
    assert calls == [True]
    assert not step_hook


def test_dual_average_existing_refusal(step_hook):
    sim = model('empty', graded=True)
    sim._interface_eps = 'dual_average'
    with pytest.raises(ValueError) as caught:
        execute(sim, 'run')
    assert str(caught.value) == "interface_eps='dual_average' cannot combine with thin conductors or lumped RLC"
    assert not step_hook


def test_conformal_staircase_surface_refusal(applied_hook):
    sim = model('empty', element=False)
    sim.add(Sphere(point(6, 6, 6), 2.5 * DX), material='pec')
    sim.add_lumped_rlc(point(6.5, 8, 6), 'ex', R=50.)
    root, inverse, weights = volume_operands(sim, 'conformal')
    edge = (6, 8, 6)
    assert bool(np.asarray(root.pec_edges[0])[edge])
    assert not bool(np.asarray(root.pec_edges[0])[6, 9, 6])
    weight = float(weights[0][edge])
    print(f'Row K: weight={weight}; predicted={"refused" if weight == 0 else "admitted"}', file=sys.stderr)
    if weight == 0:
        with pytest.raises(RfxRefusal) as caught:
            execute(sim, 'run', conformal_pec=True)
        assert str(caught.value) == expected(sim, 'geometry[0]')
        assert not applied_hook['steps']
    else:
        with pytest.raises(FirstStep):
            execute(sim, 'run', conformal_pec=True)
        assert len(applied_hook['steps']) == 1
    assert len(applied_hook['checks']) == 1
    assert applied_hook['checks'][0]['conformal_zero_masks'][0][edge] == (weight == 0)


@pytest.mark.parametrize('kind', ['pinned', 'volume'])
def test_rootless_kernel_product(kind, step_hook):
    sim = model(kind)
    root = products.realized_conductors(sim, grid_of(sim))
    with pytest.raises(RfxRefusal) as caught:
        products.kernel_conductors(sim, root.grid, root.materials, root.pec_cells,
                                   root.sheets, root.wires, periodic=root.periodic)
    assert str(caught.value) == expected(sim, 'pinned_sheet[0]' if kind == 'pinned'
                                         else 'unattributed realized PEC edge')
    assert not step_hook


def test_coax_union_refuses_before_calculator(step_hook):
    """Public coax calculators fence all registered RLC, so the conductor union owner is exercised directly."""
    sim = model('empty')
    root = products.realized_conductors(sim, grid_of(sim))
    edges = tuple(np.zeros(root.grid.shape, bool) for _ in range(3))
    edges[0][6, 6, 6] = True
    with pytest.raises(RfxRefusal) as caught:
        products.coax_kernel_conductors(sim, root.grid, root.materials,
            np.zeros(root.grid.shape, bool), edges, (), skip_preflight=True)
    assert str(caught.value) == expected(sim, 'unattributed realized PEC edge')
    assert not step_hook


def test_public_coax_existing_refusal(step_hook):
    from tests.unit.sparams.test_coax_two_port_smatrix import _sim
    sim = _sim()
    sim.add_lumped_rlc((.004, .004, .010), R=50.)
    with pytest.raises(ValueError) as caught:
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)
    assert str(caught.value) == 'compute_coaxial_two_port() does not support registered lumped RLC elements.'
    assert not step_hook


def test_mixed_microstrip_edge_refusal(step_hook):
    from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_msl, _H_SUB
    sim, yc = _base_sim()
    sim.add_port((2e-3, yc, 0.), 'ez', impedance=50.)
    _add_msl(sim, yc, n_probe_offset=10, n_probe_spacing=4)
    sim.add_lumped_rlc((3e-3, yc, _H_SUB), 'ex', R=50.)
    root = products.realized_conductors(sim, grid_of(sim))
    edge = field_index(root.grid, sim._lumped_rlc[0].position, 'ex')
    assert bool(np.asarray(root.pec_edges[0])[edge])
    with pytest.raises(RfxRefusal) as caught:
        sim.compute_mixed_s_matrix(freqs=np.array([1e9, 2e9]), num_periods=1.,
                                   skip_preflight=True)
    assert str(caught.value) == expected(sim, 'unattributed realized PEC edge')
    assert not step_hook


def test_topology_rootless_refusal(step_hook):
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim = model()
    region = TopologyDesignRegion(corner_lo=point(1, 8, 1), corner_hi=point(2, 9, 2),
                                  material_bg='air', material_fg='fr4')
    with pytest.raises(RfxRefusal) as caught:
        topology_optimize(sim, region, lambda result: -jnp.sum(result.time_series**2),
                          n_iterations=1, skip_preflight=True, verbose=False)
    assert str(caught.value) == expected(sim, 'unattributed realized PEC edge')
    assert not step_hook


@pytest.mark.parametrize('kind', ['mesh', 'edges', 'wall_mesh'])
def test_traced_classification_warns_once(kind):
    sim = model('empty' if kind == 'wall_mesh' else 'pinned', graded=True)
    root = products.realized_conductors(sim, grid_of(sim), nonuniform=True)

    def check(value):
        changed = (replace(root, pec_edges=(value,) + root.pec_edges[1:]) if kind == 'edges'
                   else replace(root, grid=root.grid._replace(dy_arr=value)))
        refuse_shorted_lumped_rlc(sim, changed)
        return value

    value = root.pec_edges[0] if kind == 'edges' else root.grid.dy_arr
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        jax.make_jaxpr(check)(value)
    hits = [w for w in caught if isinstance(w.message, AccuracyWarning)]
    assert [str(w.message) for w in hits] == [
        'Lumped RLC PEC admission check cannot run on a traced mesh or traced conductor '
        'edges; no concrete realized edge classification is available.']


def test_traced_values_still_refuse_geometry():
    sim = model()
    root = products.realized_conductors(sim, grid_of(sim))

    def check(value):
        sim._lumped_rlc[0] = replace(sim._lumped_rlc[0], R=value)
        with pytest.raises(RfxRefusal) as caught:
            refuse_shorted_lumped_rlc(sim, root)
        assert str(caught.value) == expected(sim)
        return value

    jax.make_jaxpr(check)(50.)


def test_wall_takes_precedence(step_hook):
    sim = model('wall')
    sim.add_pinned_sheet(normal_axis=2, plane_index=0, i_range=(4, 9), j_range=(4, 9))
    with pytest.raises(RfxRefusal) as caught:
        execute(sim, 'run')
    assert str(caught.value) == expected(sim, faces='z_lo')
    assert not step_hook


def test_lumped_s_scan_with_preflight_uses_solve_admission(step_hook):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = model()
    sim._ports.clear()
    sim.add_port(point(2.5, 2, 2), 'ex', impedance=50.)
    with pytest.raises(RfxRefusal) as caught:
        compute_lumped_wire_s_matrix_via_scan(sim, freqs=np.array([1e9, 2e9]), n_steps=1)
    assert str(caught.value) == expected(sim)
    assert not step_hook


def test_lumped_s_scan_carried_root_refusal(step_hook):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = model()
    sim._ports.clear()
    sim.add_port(point(2.5, 2, 2), 'ex', impedance=50.)
    with pytest.raises(RfxRefusal) as caught:
        compute_lumped_wire_s_matrix_via_scan(
            sim, freqs=np.array([1e9, 2e9]), n_steps=1, skip_preflight=True)
    print(f'S-scan observed: {type(caught.value).__name__}: {caught.value}', file=sys.stderr)
    assert str(caught.value) == expected(sim)
    assert not step_hook


@pytest.mark.xfail(strict=True, reason="#2761 a switched pinned_sheet_layout keeps no element edge free; the stopgap refuses the scene instead of keeping the edge free with a warning")
def test_switched_sheet_keeps_lumped_rlc_edge_free():
    from tests.contracts.test_pinned_sheet_layout import BINS, DZ, FULL, board, edge_arrays
    with_element, without_element = board(), board()
    with_element.add_lumped_rlc((.0045, .005, float(DZ[:4].sum())), 'ex',
                                R=50., C=1e-13, topology='parallel')

    def run(sim):
        result = sim.forward(n_steps=60, skip_preflight=True,
                             pinned_sheet_layout=(edge_arrays((FULL,)),),
                             port_s11_freqs=BINS)
        return tuple(np.asarray(a) for a in (result.time_series, result.s_params[:, :1, :]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = run(with_element)
    reference = run(without_element)
    releases = [w for w in caught if isinstance(w.message, ConfigurationWarning)
                and str(w.message).startswith('Port occupancy release: lumped_rlc[0]')]
    max_difference = max(float(np.max(np.abs(a - b)))
                         for a, b in zip(actual, reference, strict=True))
    bytes_differ = any(a.tobytes() != b.tobytes()
                       for a, b in zip(actual, reference, strict=True))
    max_byte_difference = max(int(np.max(np.abs(
        np.frombuffer(a.tobytes(), dtype=np.uint8).astype(np.int16)
        - np.frombuffer(b.tobytes(), dtype=np.uint8).astype(np.int16))))
        for a, b in zip(actual, reference, strict=True))
    print(f'Row L: release_warning={bool(releases)}, release_count={len(releases)}, '
          f'bytes_differ={bytes_differ}, max_abs_difference={max_difference}, '
          f'max_abs_byte_difference={max_byte_difference}', file=sys.stderr)
    assert len(releases) == 1
    assert bytes_differ


@pytest.fixture
def applied_hook(monkeypatch):
    """Observe the applied masks; stop after setup and before any step body."""
    from rfx.model import source_admission
    from rfx.stepping import uniform
    seen = {'checks': [], 'steps': []}
    original = source_admission.refuse_shorted_lumped_rlc_applied

    def check(sim, conductors, **masks):
        seen['checks'].append(masks)
        return original(sim, conductors, **masks)

    def stop(ctx, invariants, **kwargs):
        seen['steps'].append(ctx)
        raise FirstStep

    monkeypatch.setattr(source_admission, 'refuse_shorted_lumped_rlc_applied', check)
    monkeypatch.setattr(uniform, 'make_uniform_step', stop)
    return seen


def volume_model(mode, *, curved=False):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    boundary = (BoundarySpec(x=Boundary(lo='pec', hi='pec', conformal=True),
                             y='pec', z='pec') if mode == 'conformal' else 'pec')
    sim = Simulation(15e9, (.012, .012, .012), dx=.001,
                     boundary=boundary, snap='declared')
    shape = (Sphere((.006, .006, .006), .0025) if curved else
             Box((.0023, .0023, .0023), (.0057, .0087, .0087)))
    sim.add(shape, material='pec')
    sim.add_port((.009, .006, .006), 'ex', impedance=50.)
    sim.add_probe((.010, .006, .006), 'ex')
    return sim


def volume_operands(sim, mode):
    from rfx.model.tensor_operands import form_tensor_operands
    root = products.realized_conductors(sim, grid_of(sim))
    _, inverse, weights, _, _ = form_tensor_operands(
        sim, root.grid, root.materials, subpixel_smoothing='kottke_pec' if mode == 'kottke' else False,
        conformal=mode == 'conformal', conformal_min_weight=.1, pec_shapes=root.assembly[4])
    return root, inverse, weights


def execute_volume(sim, entry, mode):
    options = {'subpixel_smoothing': 'kottke_pec'} if mode == 'kottke' else {}
    if entry == 'scan':
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        return compute_lumped_wire_s_matrix_via_scan(
            sim, freqs=np.array([1e9, 2e9]), n_steps=1, skip_preflight=True, **options)
    return execute(sim, entry, **options)


@pytest.mark.parametrize('entry', ['run', 'forward', 'scan'])
@pytest.mark.parametrize('form', ['fold', 'ade'])
@pytest.mark.parametrize('mode', ['conformal', 'kottke'])
@pytest.mark.parametrize('location', ['inside', 'surface'])
def test_volume_operator_applied_masks(entry, form, mode, location, applied_hook):
    from rfx.boundaries.pec import PEC_INV_THRESHOLD, kottke_fenced_edge_masks
    sim = volume_model(mode, curved=mode == 'kottke' and location == 'surface')
    root, inverse, weights = volume_operands(sim, mode)
    edge = (4, 6, 6) if location == 'inside' else (5, 6, 6)
    hard = None
    if mode == 'kottke':
        hard = kottke_fenced_edge_masks(root.pec_edges, inverse,
            sheets=root.sheets, wires=root.wires, periodic=root.periodic)
        if location == 'surface':
            candidates = np.argwhere(np.asarray(root.pec_edges[0]) & ~np.asarray(hard[0])
                                     & (np.asarray(inverse[0]) >= PEC_INV_THRESHOLD))
            assert len(candidates), 'No volume edge outside the Kottke hard mask in this sphere'
            edge = tuple(map(int, candidates[len(candidates) // 2]))
    position = tuple(float(root.grid.node_of(a, e)) for a, e in enumerate(edge))
    sim.add_lumped_rlc(position, 'ex', R=50.,
        **({'C': 1e-13} if form == 'fold' else {'L': 1e-9, 'topology': 'series'}))
    assert tuple(map(int, field_index(root.grid, position, 'ex'))) == edge
    assert bool(np.asarray(root.pec_edges[0])[edge])
    refused = location == 'inside'
    if mode == 'conformal':
        weight = float(weights[0][edge])
        assert weight == 0 if refused else 0 < weight < 1
        print(f'D9 {entry} {form} {mode} {location}: edge={edge}, weight={weight}', file=sys.stderr)
    else:
        membership = bool(np.asarray(hard[0])[edge])
        frozen = bool(np.asarray(inverse[0])[edge] < PEC_INV_THRESHOLD)
        assert membership == refused
        assert frozen == refused
        print(f'D9 {entry} {form} {mode} {location}: edge={edge}, staircase=True, '
              f'hard={membership}, inverse={float(inverse[0][edge])}, frozen={frozen}', file=sys.stderr)
    if refused:
        with pytest.raises(RfxRefusal) as caught:
            execute_volume(sim, entry, mode)
        assert str(caught.value) == expected(sim, 'geometry[0]')
        assert not applied_hook['steps']
    else:
        with pytest.raises(FirstStep):
            execute_volume(sim, entry, mode)
        assert len(applied_hook['steps']) == 1
    if refused:
        assert len(applied_hook['checks']) == 1
    if applied_hook['checks']:
        actual = applied_hook['checks'][0]
        if mode == 'conformal':
            assert bool(np.asarray(actual['conformal_zero_masks'][0])[edge]) == refused
        else:
            assert bool(np.asarray(actual['hard_edges'][0])[edge]) == refused
            assert bool(np.asarray(actual['frozen_masks'][0])[edge]) == refused


@pytest.mark.parametrize('mode', ['conformal', 'kottke'])
@pytest.mark.parametrize('kind', ['volume', 'pinned', 'wire', 'wall'])
def test_operator_preflight_defers_only_volumes(mode, kind, step_hook):
    """Kottke is a call option carried by automatic preflight's stored mode."""
    from rfx.boundaries.spec import Boundary, BoundarySpec
    sim = model(kind)
    if mode == 'conformal':
        sim._boundary_spec = BoundarySpec(x=Boundary(lo='pec', hi='pec', conformal=True), y='pec', z='pec')
        report = sim.preflight()
    else:
        report = sim._preflight_impl(_conductor_operator='kottke_pec')
    findings = report.by_code('assembly_refusal')
    if kind == 'volume':
        assert not findings
    else:
        assert len(findings) == 1
        owner = 'pinned_sheet[0]' if kind == 'pinned' else 'geometry[0]'
        assert str(findings[0]) == 'The model cannot be assembled: ' + expected(
            sim, owner, faces='z_lo' if kind == 'wall' else None)
    assert not step_hook


@pytest.mark.parametrize('mode', ['conformal', 'kottke_pec'])
@pytest.mark.parametrize('carried', [False, True])
@pytest.mark.parametrize('kind', ['volume', 'pinned', 'wire', 'wall'])
def test_kernel_operator_defers_only_volumes(mode, carried, kind):
    sim = model(kind)
    root = products.realized_conductors(sim, grid_of(sim))

    def build():
        return products.kernel_conductors(sim, root.grid, root.materials, root.pec_cells,
            root.sheets, root.wires, periodic=root.periodic,
            root=root if carried else None, operator=mode)

    if kind == 'volume':
        build()
    else:
        with pytest.raises(RfxRefusal) as caught:
            build()
        owner = ('pinned_sheet[0]' if kind == 'pinned' else
                 'geometry[0]' if carried else 'unattributed realized PEC edge')
        assert str(caught.value) == expected(sim, owner, faces='z_lo' if kind == 'wall' else None)
