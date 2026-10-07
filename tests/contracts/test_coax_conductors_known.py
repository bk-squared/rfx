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

    def handover(sim, *args, **kwargs):
        root = coax_kernel_conductors(sim, *args, **kwargs)
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
        assert set(findings) == {'sheet-size', 'pad-fill'}
        assert not findings['pad-fill']
        if lane == 'transition':
            declared = captured['declared']
            assert len(record.entities) == len(sim._geometry) + 3
            assert root.assembly_entries == declared.assembly_entries
            assert root.provenance[:-1] == declared.provenance
            for owned, drawn in zip(root.pec_edges, declared.pec_edges, strict=True):
                assert np.all(owned[np.asarray(drawn)])
        else:
            assert not findings['sheet-size']
        print(f'{lane}: radii={expected}; checks={findings}')
        sim.add_material('invalidate_coax_view', eps_r=2.)
        assert not any(e.provenance == 'coax stamp' for e in sim.realized_geometry().entities)
        raise KernelInspected

    monkeypatch.setattr(calculators, 'coax_kernel_conductors', handover)
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(KernelInspected):
        call()


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
