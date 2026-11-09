"""Coax diagnostic faults preserve S and warnings; traced stamps stay usable.

RF calibration stays in test_coaxial_line_reflection.py. The 400-step lock
fixture intentionally emits a settling warning, compared with its control.
"""
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.model.conductors import coax_kernel_conductors
from tests.locks.test_sparams_split_bit_identity import _coax_line_result


def _measure(row, quantity, measured, bar=0):
    print(f"B3.1 row={row} quantity={quantity} measured={measured} bar={bar}", file=sys.stderr)


def _equal(row, quantity, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    mismatch = int(np.count_nonzero(actual != expected))
    _measure(row, quantity + '.unequal', mismatch)
    np.testing.assert_array_equal(actual, expected, err_msg=f"{row}: {quantity}; unequal={mismatch}, bar=0")


def _edges(cells):
    """Four incident cells per E edge, zero outside the nonperiodic box."""
    result = []
    for component in range(3):
        mask = np.zeros_like(cells, dtype=bool)
        transverse = [a for a in range(3) if a != component]
        for first in (0, 1):
            for second in (0, 1):
                src, dst = [slice(None)] * 3, [slice(None)] * 3
                for axis, shift in zip(transverse, (first, second)):
                    if shift:
                        src[axis], dst[axis] = slice(None, -1), slice(1, None)
                mask[tuple(dst)] |= cells[tuple(src)]
        result.append(mask)
    return tuple(result)


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
def test_diagnostic_fault_preserves_s_and_warnings(monkeypatch, reader):
    import rfx.sparams.coax as calculators

    with warnings.catch_warnings(record=True) as control:
        warnings.simplefilter('always')
        baseline = _coax_line_result()

    def fail(*args, **kwargs):
        raise RuntimeError('injected diagnostic failure')

    def fault_at_handover(owner, *args, **kwargs):
        # Fault injection is restricted to the diagnostic read: admission and
        # time stepping remain real. There are no observations inside this wrapper.
        with monkeypatch.context() as patch:
            if reader == '_campaign_ctx':
                patch.setattr(owner, reader, fail)
            else:
                module = 'rfx.model.coax' if reader == 'stamp_check_findings' else 'rfx.preflight.realization'
                patch.setattr(f'{module}.{reader}', fail)
            return coax_kernel_conductors(owner, *args, **kwargs)

    monkeypatch.setattr(calculators, 'coax_kernel_conductors', fault_at_handover)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        result = _coax_line_result()
    _equal(reader, 'S11.bytes', np.asarray(result.s11).view(np.uint8),
           np.asarray(baseline.s11).view(np.uint8))
    assert np.isfinite(np.asarray(result.s11)).all()
    expected = [(w.category, str(w.message)) for w in control]
    actual = [(w.category, str(w.message)) for w in emitted]
    _measure(reader, 'warnings_control', expected, 'same category/message/order')
    _measure(reader, 'warnings_faulted', actual, 'same category/message/order')
    assert actual == expected, f'{reader}: control={expected!r}, faulted={actual!r}'


class KernelInspected(Exception):
    pass


def test_skip_preflight_forms_sheet_rows_once_and_hands_the_kernel_its_edges(monkeypatch):
    import rfx.sparams.coax as calculators
    from rfx.preflight.pec_geometry import _warn_sheet_effective_size
    from tests._coax_msl_instrument_fixture import (
        build_instrument_junction, instrument_kwargs,
    )

    sim = build_instrument_junction()
    sim._snap = "declared"
    seen = {}
    sheet_reads = []

    def counted(*args, **kwargs):
        sheet_reads.append(True)
        return _warn_sheet_effective_size(*args, **kwargs)

    def handover(sim, *args, **kwargs):
        assert kwargs['skip_preflight'] is True
        root = coax_kernel_conductors(sim, *args, **kwargs)
        seen['root'] = root
        assert root.stamp_check_findings == ()
        return root

    def kernel(*args, **kwargs):
        assert kwargs['pec_edge_masks'] is seen['root'].pec_edges
        raise KernelInspected

    monkeypatch.setattr('rfx.preflight.pec_geometry._warn_sheet_effective_size', counted)
    monkeypatch.setattr(calculators, 'coax_kernel_conductors', handover)
    monkeypatch.setattr('rfx.simulation.run', kernel)
    with pytest.raises(KernelInspected):
        sim.compute_coax_msl_transition(**instrument_kwargs(200))
    assert len(sheet_reads) == 1


def test_traced_stamp_leaves_public_geometry_and_eager_s_concrete():
    from rfx.sources.coaxial_port import stamp_coaxial_line

    sim = Simulation(freq_max=10e9, domain=(.01, .01, .01), dx=.0005,
                     boundary='cpml', snap='declared')
    sim.add_coaxial_port((.005, .005, .005), face='top')
    declared = sim.realized_geometry()
    grid = declared.conductors.grid
    entities = []
    materials, _, cells = stamp_coaxial_line(
        grid, declared.materials, center_xy=(.005, .005),
        z_lo_index=grid.pad_z_lo+2, z_hi_index=grid.pad_z_lo+8, realized_entities=entities)

    def objective(scale):
        root = coax_kernel_conductors(sim, grid,
            materials._replace(eps_r=materials.eps_r * scale), cells, _edges(cells), entities)
        return jnp.sum(root.materials.eps_r)

    derivative = float(jax.grad(objective)(1.))
    _measure('traced', 'gradient', derivative, 'finite')
    assert np.isfinite(derivative)
    record = sim.realized_geometry()
    arrays = [*record.edge_masks, record.materials.eps_r, record.materials.sigma]
    _measure('traced', 'concrete_arrays', sum(isinstance(a, np.ndarray) for a in arrays), len(arrays))
    assert all(isinstance(a, np.ndarray) for a in arrays)
    result = sim.compute_coaxial_line_reflection(termination='short', n_steps=16,
        freqs=np.array([8e9]), probe_count=3, probe_start_cells=2, probe_spacing_cells=2)
    _measure('traced', 'eager_S11_finite', int(np.isfinite(np.asarray(result.s11)).all()), 1)
    assert np.isfinite(np.asarray(result.s11)).all()
