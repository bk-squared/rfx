"""G1: production conductor products, persistent stages, and consumption replay."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rfx.model.conductors import clear_conductor_edges, realized_conductors
from .builders import build


GEOMETRY_ROWS = [('_geometry', 'pec_volume'), ('_geometry', 'pec_sheet'),
                 ('_geometry', 'pec_wire'), ('_thin_conductors', 'pec_sheet'),
                 ('_pinned_sheets', 'pec_sheet')]


@pytest.mark.parametrize('row', GEOMETRY_ROWS)
def test_conductors_constant_grid_equivalence(row):
    a = build(row, 'run_uniform')
    ga = a._build_grid()
    b = build(row, 'run_nonuniform', dt=ga.dt)
    gb = b._build_nonuniform_grid()
    ca = realized_conductors(a, ga)
    cb = realized_conductors(b, gb, nonuniform=True)
    for x, y in zip(ca.edges, cb.edges, strict=True):
        np.testing.assert_array_equal(x, y)
    assert (ca.pec_cells is None) == (cb.pec_cells is None)
    if ca.pec_cells is not None:
        np.testing.assert_array_equal(ca.pec_cells, cb.pec_cells)
    assert len(ca.sheets) == len(cb.sheets)
    for x, y in zip(ca.sheets, cb.sheets, strict=True):
        assert x.entity_id and x.entity_id == y.entity_id
        assert x.solved_spans == y.solved_spans
        assert all(span is not None for axis, span in enumerate(x.solved_spans)
                   if axis != x.normal_axis)
    assert len(ca.wires) == len(cb.wires)


def test_port_clearing_is_persistent():
    sim = build(('_geometry', 'pec_volume'), 'run_uniform')
    c = realized_conductors(sim, sim._build_grid())
    cell = tuple(int(k) for k in np.argwhere(c.edges[2])[0])
    before = tuple(np.array(m) for m in c.edges)
    after = clear_conductor_edges(c, [cell], component='ez', entity_id='port[0]')
    assert after is not c
    assert not after.edges[2][cell]
    for old, original in zip(c.edges, before, strict=True):
        np.testing.assert_array_equal(old, original)
    assert after.provenance[-1].entity_ids == ('port[0]',)
    assert after.provenance[-1].stage == 'port-edge-clearing'
    with pytest.raises(FrozenInstanceError):
        c.pec_edges = after.pec_edges


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform', 'fwd_uniform', 'fwd_nonuniform'])
def test_preflight_and_run_share_one_assembly(monkeypatch, lane, kernel_objects):
    sim = build(('_geometry', 'pec_sheet'), lane)
    sim._snap = 'declared'
    method = '_assemble_materials_nu' if lane.endswith('nonuniform') else '_assemble_materials'
    original = getattr(sim, method)
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, method, count)
    audit_context = sim._campaign_ctx()
    audit = audit_context.realized()
    assert len(audit.sheets) == 1
    assert any(edge.any() for edge in audit.edges)
    assert audit.mode == 'audit'
    def forbidden_cache():
        raise AssertionError("execution must not read the diagnostic cache")

    monkeypatch.setattr(sim, "_campaign_ctx", forbidden_cache)
    calls.clear()
    seen = []
    preflight = sim._preflight_impl

    def read(*args, **kwargs):
        root = kwargs['_conductors']
        assert root.mode == 'solve'
        assert len(root.sheets) == 1
        seen.append(root)
        return preflight(*args, **kwargs)

    monkeypatch.setattr(sim, '_preflight_impl', read)
    runner = sim.forward if lane.startswith('fwd_') else sim.run
    runner(n_steps=2)
    assert calls == [1]
    assert len(seen) == 1
    assert seen[0].pec_edges is kernel_objects[-1].pec_edges
    assert audit_context.realized() is audit
    assert kernel_objects[-1] is not audit


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform'])
def test_record_reads_kernel_after_port_clearing(monkeypatch, lane, kernel_objects):
    from rfx import Box
    from .builders import point
    sim = build(('_ports', 'lumped_port'), lane)
    sim.add(Box(point(4, 2, 2), point(6, 4, 4)), material='pec')
    grid = sim._build_nonuniform_grid() if lane == 'run_nonuniform' else sim._build_grid()
    original = realized_conductors(sim, grid, nonuniform=lane == 'run_nonuniform')
    idx = tuple(grid.index_of(a, x) for a, x in enumerate(point(5, 3, 3)))
    assert original.edges[2][idx]
    import rfx.realized_geometry as records
    recorded = []
    original_record = records.record_from_conductors

    def record_operand(sim, conductors, **kwargs):
        recorded.append(conductors)
        return original_record(sim, conductors, **kwargs)

    monkeypatch.setattr(records, 'record_from_conductors', record_operand)
    result = sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    final = kernel_objects[-1]
    assert result.realized_geometry.conductors is None
    assert not final.edges[2][idx]
    assert final.provenance[-1].stage == 'port-edge-clearing'
    assert recorded[-1] is final
    assert not recorded[-1].edges[2][idx]
    assert sim._campaign_ctx().realized().edges[2][idx]
    np.testing.assert_array_equal(sim.realized_geometry().edge_masks[2], final.edges[2])


@pytest.mark.parametrize('graded', [False, True])
def test_two_device_conductors_are_the_same_object_fields(graded, two_device_test, kernel_objects):
    import jax
    if len(jax.devices()) < 2:
        return two_device_test()
    lane = 'run_nonuniform' if graded else 'run_uniform'
    a = build(('_geometry', 'pec_volume'), lane, graded=graded)
    b = build(('_geometry', 'pec_volume'), lane, graded=graded)
    a.run(n_steps=2, skip_preflight=True)
    ca = kernel_objects[-1]
    b.run(n_steps=2, skip_preflight=True, devices=jax.devices()[:2])
    cb = kernel_objects[-1]
    for x, y in zip(ca.pec_edges, cb.pec_edges, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(ca.pec_cells, cb.pec_cells)
    assert ca.provenance == cb.provenance
    assert b._pf_campaign_ctx is None


@pytest.mark.parametrize('lane,distributed', [('run_uniform', False),
    ('run_nonuniform', False), ('run_uniform', True), ('run_nonuniform', True)])
def test_dump_replay_moves_fields_and_identity_is_exact(lane, distributed, two_device_test):
    import jax
    import jax.numpy as jnp
    from rfx import _realized
    from .builders import point
    if distributed and len(jax.devices()) < 2:
        return two_device_test()
    sim = build(('_geometry', 'pec_volume'), lane, graded=lane == 'run_nonuniform')
    grid = sim._build_nonuniform_grid() if lane == 'run_nonuniform' else sim._build_grid()
    index = tuple(grid.index_of(a, x) for a, x in enumerate(point(2.1, 2.3, 2.2)))
    kwargs = dict(n_steps=12, skip_preflight=True, compute_s_params=False)
    if distributed:
        kwargs['devices'] = jax.devices()[:2]
    with _realized.capture() as dumped:
        baseline = sim.run(**kwargs)
    rows = [r for r in dumped.records if r['site'] == 'conductors']
    assert len(rows) == 1
    saved = tuple(jnp.asarray(m) for m in rows[0]['pec_edges'])
    assert not saved[2][index]

    def replay(changed):
        def transform(site, quantity, values):
            if site != 'conductors' or quantity != 'pec_edges':
                return values
            return saved[:2] + (saved[2].at[index].set(True),) if changed else saved
        with _realized.capture(transform=transform):
            return sim.run(**kwargs)

    same = replay(False)
    changed = replay(True)
    for field in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        np.testing.assert_array_equal(getattr(baseline.state, field), getattr(same.state, field))
    np.testing.assert_array_equal(baseline.time_series, same.time_series)
    assert not np.array_equal(baseline.time_series, changed.time_series)
    assert changed.realized_geometry.conductors is None


def test_kernel_rejects_port_edits_outside_object():
    from rfx.model.conductors import at_kernel
    sim = build(('_geometry', 'pec_volume'), 'run_uniform')
    c = realized_conductors(sim, sim._build_grid())
    arrays = tuple(np.array(edge) for edge in c.pec_edges)
    arrays[2][0, 0, 0] = True
    with pytest.raises(ValueError, match='outside a conductor stage'):
        at_kernel(sim, c, lane='run_uniform', pec_edges=arrays)


def test_sheet_spans_are_read_from_conductors(monkeypatch):
    import rfx.mesh_edges as shared
    sim = build(('_geometry', 'pec_sheet'), 'run_uniform')
    c = sim._campaign_ctx().realized()
    assert any(span is not None for span in c.sheets[0].solved_spans)

    def unexpected(*args, **kwargs):
        raise AssertionError('a report recomputed the solved sheet span')

    monkeypatch.setattr(shared, 'solved_sheet_span', unexpected)
    record = sim.realized_geometry()
    sim.preflight()
    assert record.conductors is c


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform'])
def test_standalone_record_matches_port_cleared_kernel(lane, kernel_objects):
    from rfx import Box
    from .builders import point
    sim = build(('_ports', 'lumped_port'), lane)
    sim.add(Box(point(4, 2, 2), point(6, 4, 4)), material='pec')
    standalone = sim.realized_geometry()
    sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    for a, b in zip(standalone.edge_masks, kernel_objects[-1].edges, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('geometry', ['pec_volume', 'pec_sheet'])
def test_forward_does_not_cache_traced_conductors(geometry):
    import jax
    sim = build(('_geometry', geometry), 'fwd_uniform')
    traced = jax.jit(lambda: sim.forward(n_steps=2, skip_preflight=True).time_series)()
    eager = sim.forward(n_steps=2, skip_preflight=True).time_series
    np.testing.assert_array_equal(traced, eager)


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform'])
def test_sheet_operator_is_carried_by_kernel_object(monkeypatch, lane, kernel_objects):
    import rfx.materials.thin_conductor as thin
    sim = build(('_thin_conductors', 'surface_impedance'), lane)
    outputs = []
    original = thin.build_sheet_impedance_ctx

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        outputs.append(result)
        return result

    monkeypatch.setattr(thin, 'build_sheet_impedance_ctx', capture)
    sim.run(n_steps=2, skip_preflight=True)
    conductors = kernel_objects[-1]
    assert len(conductors.sheet_impedance) == 1
    assert conductors.sheet_operator is outputs[-1]


@pytest.fixture
def two_device_test(request, tmp_path):
    """Run the same named cell on two CPUs when the parent has one device."""
    def run():
        import os
        from pathlib import Path
        import subprocess
        import sys
        env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='false',
            XLA_FLAGS='--xla_force_host_platform_device_count=2',
            PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(tmp_path),
            XDG_CACHE_HOME=str(tmp_path / 'cache'), MPLCONFIGDIR=str(tmp_path / 'mpl'))
        process = subprocess.Popen([sys.executable, '-m', 'pytest', request.node.nodeid,
            '-q', '-o', 'addopts=', '--basetemp=' + str(tmp_path / 'child')],
            cwd=Path(__file__).resolve().parents[3], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            pytest.fail('two-device conductor cell timed out')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        assert process.returncode == 0, output
    return run


@pytest.fixture
def kernel_objects(monkeypatch):
    """Test-only retention; production Results must never own these arrays."""
    import rfx.model.conductors as products
    captured = []
    original = products.at_kernel

    def observe(*args, **kwargs):
        c, record = original(*args, **kwargs)
        captured.append(c)
        return c, record

    monkeypatch.setattr(products, 'at_kernel', observe)
    return captured


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform'])
def test_preflight_is_independent_of_call_order(lane):
    from rfx import Box
    from .builders import point
    sim = build(('_ports', 'lumped_port'), lane)
    sim.add(Box(point(4, 2, 2), point(6, 4, 4)), material='pec')

    def report():
        return [(str(issue), issue.code, issue.severity, issue.loc, issue.source)
                for issue in sim.preflight()]

    fresh = report()
    assert any(row[1] == 'port_in_pec' for row in fresh)
    sim.realized_geometry()
    assert report() == fresh
    sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    assert report() == fresh


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform'])
def test_result_has_no_dense_conductor_reference(lane):
    from dataclasses import fields, is_dataclass
    from rfx.model.conductors import RealizedConductors
    sim = build(('_geometry', 'pec_volume'), lane)
    record = sim.run(n_steps=2, skip_preflight=True).realized_geometry

    def inspect(value):
        assert not isinstance(value, RealizedConductors)
        if hasattr(value, 'shape'):
            assert len(value.shape) <= 1
        elif is_dataclass(value):
            assert set(vars(value)) == {f.name for f in fields(value)}
            for v in vars(value).values():
                inspect(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                inspect(v)

    inspect(record)
    assert record.conductors is None
    assert sim.realized_geometry().conductors is not None


def test_runner_refuses_an_audit_object():
    from rfx.model.conductors import assembled_materials, at_kernel
    sim = build(('_geometry', 'pec_volume'), 'run_uniform')
    with pytest.raises(ValueError, match='audit-mode'):
        assembled_materials(sim._campaign_ctx().realized())
    audit = sim._campaign_ctx().realized()
    with pytest.raises(ValueError, match='audit-mode'):
        at_kernel(sim, audit, lane='run_uniform', pec_edges=audit.pec_edges)


@pytest.mark.parametrize('graded,entry', [(False, 'run'), (True, 'run'), (True, 'forward')])
def test_distributed_releases_dense_products_before_staging(monkeypatch, graded, entry, two_device_test):
    import jax
    import weakref
    import rfx.model.conductors as products
    import rfx.runners._distributed_common as common
    import rfx.runners.distributed_nu as nu
    if len(jax.devices()) < 2:
        return two_device_test()
    sim = build(('_geometry', 'pec_volume'), 'run_nonuniform' if graded else 'run_uniform',
                graded=graded)
    # A diagnostic query before the solve must not pin a second full domain.
    sim.realized_geometry()
    roots = []
    original_build = products.realized_conductors
    original_kernel = products.at_kernel

    def build_product(*a, **kw):
        root = original_build(*a, **kw)
        roots.append(weakref.ref(root))
        return root

    def kernel(*a, **kw):
        root, record = original_kernel(*a, **kw)
        roots.append(weakref.ref(root))
        return root, record

    monkeypatch.setattr(products, 'realized_conductors', build_product)
    monkeypatch.setattr(products, 'at_kernel', kernel)
    module, name = (nu, 'stage_concrete_forward_array') if graded else (common, 'shard_x_slabs')
    original_stage = getattr(module, name)
    staged = []

    def stage(*a, **kw):
        assert len(roots) == 2 and all(ref() is None for ref in roots)
        assert sim._pf_campaign_ctx is None
        assert sim._realized_geometry_record is None
        staged.append(1)
        return original_stage(*a, **kw)

    monkeypatch.setattr(module, name, stage)
    if entry == 'forward':
        sim.forward(n_steps=2, distributed=True, devices=jax.devices()[:2])
    else:
        sim.run(n_steps=2, compute_s_params=False, devices=jax.devices()[:2])
    assert staged


@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform', 'fwd_uniform', 'fwd_nonuniform'])
def test_equal_ports_have_distinct_provenance(lane, kernel_objects):
    from rfx import Box
    from .builders import point
    sim = build(('_ports', 'lumped_port'), lane)
    sim.add(Box(point(4, 2, 2), point(6, 4, 4)), material='pec')
    sim._ports.append(sim._ports[0])
    if lane.startswith('run_'):
        sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    else:
        sim.forward(n_steps=2, skip_preflight=True)
    labels = [stage.entity_ids for stage in kernel_objects[-1].provenance
              if stage.stage == 'port-edge-clearing']
    assert labels == [('port[0]',), ('port[1]',)]


@pytest.mark.parametrize('scoped', [False, True])
def test_assembly_warning_keeps_origin_and_is_not_dropped(monkeypatch, scoped):
    import inspect
    import warnings
    from contextlib import nullcontext
    from rfx.preflight.realization import assembly_warning_scope
    sim = build(('_geometry', 'pec_sheet'), 'run_uniform')
    product = realized_conductors(sim, sim._build_grid(), mode='audit')
    ctx = sim._campaign_ctx()
    origin = []

    def warned(*a, **kw):
        origin.append(inspect.currentframe().f_lineno + 1)
        warnings.warn('assembly witness', UserWarning)
        return product

    monkeypatch.setattr(sim, '_assemble_realized', warned)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with assembly_warning_scope() if scoped else nullcontext():
            ctx.entry_realizations()
            # Deliberately no subsequent ctx.realized() read.
        assert len(caught) == 1
        assert str(caught[0].message) == 'assembly witness'
        assert caught[0].filename == __file__
        assert caught[0].lineno == origin[0]
        ctx.realized()
        assert len(caught) == 1


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_ringdown_preflight_sees_only_declared_probes(monkeypatch, lane, entry):
    """The channel probes belong to stepping, not declaration preflight."""
    import rfx.model.conductors as products
    from rfx.ringdown import RingdownSpec
    from tests.unit.sparams.test_ringdown_run import _box
    sim = _box(lane)
    declared = tuple(map(id, sim._probes))
    assert len(declared) == 1
    events = []
    preflight = sim._preflight_impl
    method = '_assemble_materials_nu' if lane == 'graded' else '_assemble_materials'
    assemble = getattr(sim, method)
    kernel = products.at_kernel

    def assemble_once(*a, **kw):
        events.append(('assembly', tuple(map(id, sim._probes))))
        return assemble(*a, **kw)

    def declared_preflight(**kw):
        probes = tuple(map(id, sim._probes))
        events.append(('preflight', probes))
        assert probes == declared
        assert kw['_conductors'].mode == 'solve'
        return preflight(**kw)

    def at_kernel(*a, **kw):
        events.append(('kernel', tuple(map(id, sim._probes))))
        return kernel(*a, **kw)

    monkeypatch.setattr(sim, method, assemble_once)
    monkeypatch.setattr(sim, '_preflight_impl', declared_preflight)
    monkeypatch.setattr(products, 'at_kernel', at_kernel)
    kwargs = {'compute_s_params': True} if entry == 'run' else {}
    if entry == 'forward' and lane == 'uniform':
        kwargs['port_s11_freqs'] = np.linspace(8e9, 18e9, 11)
    result = getattr(sim, entry)(n_steps=600, ringdown=RingdownSpec(), **kwargs)
    assert [event for event, _ in events] == ['assembly', 'preflight', 'kernel']
    assert events[0][1] == declared
    assert events[2][1][:len(declared)] == declared
    assert len(events[2][1]) > len(declared)
    assert tuple(map(id, sim._probes)) == declared
    assert result.time_series.shape == (600, len(declared))


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('fence', ['_check_stencil_order_supported', '_dispatch_plan'])
@pytest.mark.parametrize('skip', [False, True])
def test_preflight_precedes_lane_refusals(monkeypatch, entry, graded, fence, skip):
    """Preserve the public error priority, including the explicit skip escape."""
    from rfx.preflight._common import PreflightIssue, PreflightReport
    lane = ('fwd_' if entry == 'forward' else 'run_') + ('nonuniform' if graded else 'uniform')
    sim = build(('_geometry', 'pec_volume'), lane)
    seen = []
    method = '_assemble_materials_nu' if graded else '_assemble_materials'
    assemble = getattr(sim, method)

    def assembly(*a, **kw):
        seen.append('assembly')
        return assemble(*a, **kw)

    def preflight(**kw):
        seen.append('preflight')
        assert kw['_conductors'].mode == 'solve'
        return PreflightReport([PreflightIssue('declaration refusal witness', severity='error')])

    def lane_refusal(*a, **kw):
        seen.append('lane')
        raise NotImplementedError('lane refusal witness')

    monkeypatch.setattr(sim, method, assembly)
    monkeypatch.setattr(sim, '_preflight_impl', preflight)
    monkeypatch.setattr(sim, fence, lane_refusal)
    error, reason = ((NotImplementedError, 'lane refusal witness') if skip
                     else (ValueError, 'declaration refusal witness'))
    with pytest.raises(error, match=reason):
        getattr(sim, entry)(n_steps=2, skip_preflight=skip)
    assert seen == (['lane'] if skip else ['assembly', 'preflight'])


def test_direct_runner_without_collectors_neither_reassembles_nor_reads_cache(monkeypatch):
    from rfx.runners.uniform import run_uniform
    sim = build(('_geometry', 'pec_volume'), 'run_uniform')
    grid = sim._build_grid()
    materials, debye, lorentz, cells, shapes, _, kerr = sim._assemble_materials(grid)
    expected = sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('direct array entry must not reassemble or read audit caches')

    monkeypatch.setattr(sim, '_assemble_materials', forbidden)
    monkeypatch.setattr(sim, '_campaign_ctx', forbidden)
    actual = run_uniform(sim, n_steps=2, grid=grid, base_materials=materials,
        debye_spec=debye, lorentz_spec=lorentz, pec_mask=cells, pec_shapes=shapes,
        kerr_chi3=kerr, compute_s_params=False)
    assert actual.realized_geometry is None
    assert expected.realized_geometry is not None
    np.testing.assert_array_equal(actual.time_series, expected.time_series)
    for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        np.testing.assert_array_equal(getattr(actual.state, name), getattr(expected.state, name))
