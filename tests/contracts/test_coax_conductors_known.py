"""The three coax calculators lend their actual stamped conductors to readers."""
from dataclasses import replace

import numpy as np
import pytest

from rfx.geometry.csg import Cylinder
from rfx.model.conductors import coax_kernel_conductors
from tests.locks.test_sparams_split_bit_identity import (
    _coax_line_result, _coax_two_port_result, _coax_msl_result,
)


class KernelInspected(Exception):
    pass


@pytest.mark.parametrize('lane,call', [
    ('reflection', _coax_line_result), ('two_port', _coax_two_port_result),
    ('transition', _coax_msl_result),
])
def test_kernel_reads_owner_and_record_keeps_stamp(monkeypatch, lane, call):
    import rfx.sparams.coax as calculators
    captured = {}

    def handover(sim, grid, materials, cells, edges, entities, **kwargs):
        root = coax_kernel_conductors(sim, grid, materials, cells, edges, entities, **kwargs)
        if lane != 'transition':
            assert kwargs.get('root') is None, 'line calculator received a board stage'
            assert root.pec_cells is cells
            assert root.pec_edges is edges, 'line calculator merged additional edges'
            for actual, expected in zip(edges, calculators._coax_pec_edge_masks(cells), strict=True):
                np.testing.assert_array_equal(actual, expected)
        # Equal values in a fresh tuple distinguish reading the returned owner
        # from bypassing it with the original local masks (even if equal).
        root = replace(root, pec_edges=tuple(np.array(e, copy=True) for e in root.pec_edges))
        sim._coax_geometry = (sim._coax_geometry[0], root)
        captured.update(sim=sim, root=root, declared=kwargs.get('root'))
        return root

    def kernel(grid, materials, n_steps, **kwargs):
        sim, root = captured['sim'], captured['root']
        assert kwargs['pec_edge_masks'] is root.pec_edges, 'kernel bypassed conductor owner'
        assert materials is root.materials
        assert root.provenance[-1].stage == 'coax stamp'
        record = sim.realized_geometry()
        assert record.conductors is root
        assert sim.realized_geometry() is record
        entities = {e.label: e for e in record.entities if e.provenance == 'coax stamp'}
        assert set(entities) == {f'coaxial_port[0]/{name}' for name in ('shell', 'pin', 'dielectric')}
        by_name = {e.entity_id.rsplit('/', 1)[1]: e for e in root.stamped_entities}
        pin, shell = by_name['pin'], by_name['shell']
        bore = Cylinder(shell.shape.center, shell.declared_radii_m[0], shell.shape.height)
        outer = np.asarray(shell.shape.mask(grid))
        hole = np.asarray(bore.mask(grid))
        pin_mask = np.asarray(pin.shape.mask(grid))
        area = np.asarray(grid.cells(0))[:, None] * np.asarray(grid.cells(1))[None, :]

        def radius(mask):
            return np.sqrt(area[np.asarray(mask).any(axis=2)].sum() / np.pi)

        expected = dict(shell=(radius(hole), radius(outer)), pin=(0., radius(pin_mask)),
                        dielectric=(radius(pin_mask), radius(hole)))
        for name, mask in (('pin', pin_mask), ('shell', outer & ~hole), ('dielectric', hole & ~pin_mask)):
            emitted = entities[f'coaxial_port[0]/{name}']
            np.testing.assert_array_equal(emitted.mask, mask)
            np.testing.assert_array_equal(emitted.radii_m, expected[name])
            assert emitted.port_id == 'coaxial_port[0]'
            assert emitted.sampling == 'closed cylinders at nodes, owned as cells'
            indices = np.flatnonzero(mask.any(axis=(0, 1)))
            assert emitted.axes[2].cell_range == (indices[0], indices[-1] + 1)
            assert emitted.axes[2].bounds_m == (record.nodes[2][indices[0]], record.nodes[2][indices[-1]+1])
        for actual, stored in zip(root.pec_edges, record.edge_masks, strict=True):
            np.testing.assert_array_equal(actual, stored)
        from rfx.preflight.realization import context_from_conductors
        ctx = context_from_conductors(sim, root)
        assert {e.label for e in ctx.interior_pec_entries()} >= {
            'coaxial_port[0]/shell', 'coaxial_port[0]/pin'}
        findings = dict(root.stamp_check_findings)
        if lane == 'transition':
            assert root.stamp_check_findings == ()
            declared = captured['declared']
            assert len(record.entities) == len(sim._geometry) + 3
            assert root.assembly_entries == declared.assembly_entries
            assert root.provenance[:-1] == declared.provenance
            for owned, drawn in zip(root.pec_edges, declared.pec_edges, strict=True):
                assert np.all(owned[np.asarray(drawn)])
        else:
            assert set(findings) == {'sheet-size', 'pad-fill'}
            assert not findings['pad-fill']
            assert not findings['sheet-size']
        sim.add_material('invalidate_coax_view', eps_r=2.)
        assert not any(e.provenance == 'coax stamp' for e in sim.realized_geometry().entities)
        raise KernelInspected

    monkeypatch.setattr(calculators, 'coax_kernel_conductors', handover)
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(KernelInspected):
        call()


@pytest.mark.parametrize('overlap,offset', [('pin', 0), ('shell', 8)])
def test_transition_restored_state_and_board_readers(monkeypatch, overlap, offset):
    import rfx.sparams.coax as calculators
    import rfx.sources.coaxial_port as coax
    import rfx.sources.msl_port as msl
    import rfx.probes.msl_wave_decomp as probe
    import tests._coax_msl_instrument_fixture as fixture

    # Synthetic stage instrumentation: put the stamp under the existing feed;
    # widen y to retain the required wall-to-CPML clearance. No RF claim.
    monkeypatch.setattr(fixture, 'DOMAIN', (12.5e-3, 3.6e-3, 3.9e-3))
    sim = fixture.build_instrument_junction()
    sim._coaxial_ports[0] = sim._coaxial_ports[0]._replace(
        position=(fixture.FEED_X-offset*fixture.DX, fixture.Y_C, fixture.GROUND))
    seen = {}
    calls = []
    declared = calculators.coax_declared_conductors
    stamp = coax.stamp_coaxial_line
    resistor = coax.stamp_coaxial_annular_resistor
    source = coax.build_coaxial_tem_plane_source_specs
    setup = msl.setup_msl_port
    validate = msl.validate_msl_port_geometry
    trace = probe.realized_trace_planes_on_column

    def board(*args, **kwargs):
        root = declared(*args, **kwargs)
        seen['board'] = root
        return root

    def line(grid, materials, **kwargs):
        calls.append('stamp')
        result = stamp(grid, materials, **kwargs)
        seen.update(pre=materials, stamp=result, stamp_cells=result[2].copy(),
                    junction=kwargs['z_hi_index']+1)
        return result

    def stamp_arguments(kwargs):
        assert kwargs['shell_inner_radius'] is seen['stamp'][1]
        assert kwargs['pec_cell_mask'] is seen['stamp'][2]
        np.testing.assert_array_equal(kwargs['pec_cell_mask'], seen['stamp_cells'])

    def load(*args, **kwargs):
        calls.append('resistor')
        stamp_arguments(kwargs)
        seen['loaded'] = resistor(*args, **kwargs)
        return seen['loaded']

    def tem(**kwargs):
        calls.append('source')
        stamp_arguments(kwargs)
        return source(**kwargs)

    def setup_msl(grid, port, materials, **kwargs):
        calls.append('setup')
        k = seen['junction']
        for field in ('eps_r', 'sigma'):
            actual = np.asarray(getattr(materials, field))
            np.testing.assert_array_equal(actual[:, :, k:], getattr(seen['pre'], field)[:, :, k:])
            np.testing.assert_array_equal(actual[:, :, :k], getattr(seen['loaded'], field)[:, :, :k])
        seen['final'] = setup(grid, port, materials, **kwargs)
        return seen['final']

    def validate_board(*args, **kwargs):
        calls.append('validate')
        assert kwargs['pec_edge_masks'] is seen['board'].pec_edges, 'validation needs board-only edges'
        result = validate(*args, **kwargs)
        assert result is None  # main e46544a on both overlap fixtures
        return result

    def trace_board(edges, *args, **kwargs):
        calls.append('trace')
        assert edges is seen['board'].pec_edges, 'trace reader needs board-only edges'
        result = trace(edges, *args, **kwargs)
        assert result == (36, 36)  # main e46544a: k_trace_lo and upper plane
        return result

    def kernel(grid, materials, n_steps, **kwargs):
        root = sim._coax_geometry[1]
        record = sim.realized_geometry()
        assert record.conductors is root
        assert kwargs['pec_edge_masks'] is root.pec_edges
        assert root.materials is materials
        assert calls == ['stamp', 'resistor', 'source', 'validate', 'setup', 'trace']
        for field in ('eps_r', 'sigma'):
            np.testing.assert_array_equal(getattr(root.materials, field), getattr(seen['final'], field))
        k = seen['junction']
        assert k == 33
        entities = {e.entity_id.rsplit('/', 1)[1]: e for e in root.stamped_entities}
        recorded = {e.label.rsplit('/', 1)[1]: e for e in record.entities if e.provenance == 'coax stamp'}
        stamped = entities['shell'].cells | entities['pin'].cells
        np.testing.assert_array_equal(stamped, seen['stamp_cells'])
        assert stamped[:, :, k:].sum() == 0
        i, j, _ = grid.position_to_index((fixture.FEED_X, fixture.Y_C, fixture.GROUND))
        assert entities[overlap].cells[i, j, :k].sum() == 23
        for name, entity in entities.items():
            np.testing.assert_array_equal(recorded[name].mask, entity.cells)
            assert entity.cells[:, :, k:].sum() == 0
            assert entity.provenance == 'coax stamp'
        np.testing.assert_array_equal(root.pec_cells, seen['board'].pec_cells | stamped)
        coax_edges = calculators._coax_pec_edge_masks(stamped)
        differences = []
        for owned, emitted, board_edges, coax_edges in zip(
                root.pec_edges, record.edge_masks, seen['board'].pec_edges, coax_edges, strict=True):
            np.testing.assert_array_equal(owned, board_edges | coax_edges)
            np.testing.assert_array_equal(emitted, owned)
            differences.append(int(np.count_nonzero(owned != board_edges)))
        assert all(differences), 'fixture must distinguish board edges from the kernel union'
        raise KernelInspected

    monkeypatch.setattr(calculators, 'coax_declared_conductors', board)
    monkeypatch.setattr(coax, 'stamp_coaxial_line', line)
    monkeypatch.setattr(coax, 'stamp_coaxial_annular_resistor', load)
    monkeypatch.setattr(coax, 'build_coaxial_tem_plane_source_specs', tem)
    monkeypatch.setattr(msl, 'setup_msl_port', setup_msl)
    monkeypatch.setattr(msl, 'validate_msl_port_geometry', validate_board)
    monkeypatch.setattr(probe, 'realized_trace_planes_on_column', trace_board)
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(KernelInspected):
        sim.compute_coax_msl_transition(**fixture.instrument_kwargs(1))


@pytest.mark.parametrize('reader', ['stamp_check_findings', 'context_from_conductors', '_campaign_ctx'])
def test_diagnostic_failure_reaches_kernel_and_records_finding(monkeypatch, reader):
    import warnings
    import rfx.sparams.coax as calculators
    seen = {}

    def fail(*args, **kwargs):
        raise RuntimeError('injected diagnostic failure')

    def handover(sim, *args, **kwargs):
        # Inject only into the new handover's reads, not existing admission.
        with monkeypatch.context() as patch:
            if reader == '_campaign_ctx':
                patch.setattr(sim, reader, fail)
            else:
                module = 'rfx.model.coax' if reader == 'stamp_check_findings' else 'rfx.preflight.realization'
                patch.setattr(f'{module}.{reader}', fail)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                root = coax_kernel_conductors(sim, *args, **kwargs)
            assert not emitted
        seen['root'] = root
        assert root.stamp_check_findings == (('read-failed', ('RuntimeError', 'injected diagnostic failure')),)
        if reader == '_campaign_ctx':
            assert sim._coax_geometry is None
        else:
            assert sim._coax_geometry[1] is root
        return root

    def kernel(*args, **kwargs):
        assert kwargs['pec_edge_masks'] is seen['root'].pec_edges
        raise KernelInspected

    monkeypatch.setattr(calculators, 'coax_kernel_conductors', handover)
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(KernelInspected):
        _coax_line_result()


def test_skip_preflight_has_no_findings_or_sheet_reads(monkeypatch):
    import rfx.sparams.coax as calculators
    seen = {}

    def forbidden(*args, **kwargs):
        pytest.fail('skip_preflight must not read sheet-size findings')

    def handover(sim, *args, **kwargs):
        assert kwargs['skip_preflight'] is True
        root = coax_kernel_conductors(sim, *args, **kwargs)
        seen['root'] = root
        assert root.stamp_check_findings == ()
        return root

    def kernel(*args, **kwargs):
        assert kwargs['pec_edge_masks'] is seen['root'].pec_edges
        raise KernelInspected

    monkeypatch.setattr('rfx.preflight.pec_geometry._warn_sheet_effective_size', forbidden)
    monkeypatch.setattr(calculators, 'coax_kernel_conductors', handover)
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(KernelInspected):
        _coax_msl_result()


def test_traced_handover_does_not_publish_tracers():
    import jax
    import jax.numpy as jnp
    from rfx import Simulation
    from rfx.sources.coaxial_port import stamp_coaxial_line
    from rfx.sparams.coax import _coax_pec_edge_masks
    sim = Simulation(freq_max=10e9, domain=(.01, .01, .01), dx=.0005, boundary='cpml')
    grid = sim._build_grid()
    entities = []
    materials, _, cells = stamp_coaxial_line(grid, sim._build_materials(grid)[0],
        center_xy=(.005, .005), z_lo_index=grid.pad_z_lo+2,
        z_hi_index=grid.pad_z_lo+8, realized_entities=entities)
    edges = _coax_pec_edge_masks(cells)

    def objective(scale):
        root = coax_kernel_conductors(sim, grid,
            materials._replace(eps_r=materials.eps_r * scale), cells, edges, entities)
        assert sim._coax_geometry is None
        return jnp.sum(root.materials.eps_r)

    assert np.isfinite(jax.grad(objective)(1.))
