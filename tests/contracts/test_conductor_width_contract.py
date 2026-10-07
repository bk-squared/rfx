"""One conductor owner, stage-specific context, and the admitted width cells."""
import ast
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, _realized
from rfx.model.conductors import at_kernel, realized_conductors
from rfx.model.conductors import RealizedConductors
from rfx.runners._admission import admit
from tests.unit.materials.test_sheet_footprint_width import (
    DX, ETA0, F0, MU0, N_STEPS, THICKNESS, _build,
)


def _periodic_fixture():
    sim = _build('f0', 8, False, sources=False, y_boundary='periodic', span=(5, 10))
    sim.add(Box((DX, 2*DX, 30*DX), (2*DX, 3*DX, 30*DX)), material='pec')
    return sim


def test_periodic_override_preserves_sheet_arrays():
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx
    sim = _periodic_fixture()
    root = realized_conductors(sim, sim._build_grid())
    expected = build_sheet_impedance_ctx(root.sheet_specs, root.pec_edges, root.periodic)
    with _realized.capture(transform=lambda site, quantity, values: tuple(
            np.array(v, copy=True) for v in values)):
        final, _ = at_kernel(sim, root, lane='fwd_uniform', pec_edges=root.pec_edges,
                             sheet_operator=expected)
    for field in fields(expected):
        a, b = getattr(expected, field.name), getattr(final.sheet_operator, field.name)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b)


def _monitor_fixture():
    sim = Simulation(freq_max=10e9, domain=(12e-3,)*3, dx=1e-3,
                     boundary='cpml', cpml_layers=4, snap='declared')
    sim.add_thin_conductor(Box((3e-3, 3e-3, 6e-3), (8e-3, 8e-3, 6e-3)),
                           sigma_bulk=1e4, thickness=35e-6, surface_impedance_f0=9e9)
    sim.add(Box((3e-3, 3e-3, 6e-3), (5e-3, 5e-3, 6e-3)), material='pec')
    sim.add_source((6e-3, 6e-3, 5e-3), 'ex', amplitude_kind='field',
                   waveform=GaussianPulse(f0=9e9, bandwidth=.8))
    sim.add_current_moment_monitor((1e-3, 1e-3, 4e-3), (10e-3, 10e-3, 8e-3),
                                   block_size=2e-3, freqs=np.array([9e9]))
    return sim


# M cells execute three 2500-step traces; pytest reports each cell's duration.
@pytest.mark.parametrize('lane', ['run_uniform', 'run_nonuniform', 'fwd_uniform',
                                   'fwd_nonuniform', 'run_periodic', 'fwd_periodic'])
def test_measured_finite_resistance_width(lane):
    spectra = {}
    for model in ('ref', 'dc', 'f0'):
        sim = _build(model, 8, lane.endswith('nonuniform'),
                     y_boundary='periodic' if lane.endswith('periodic') else 'pmc')
        if lane.startswith('fwd'):
            result = sim.forward(n_steps=N_STEPS, skip_preflight=True, checkpoint=False)
            probe = result.dft_planes['trans']
        else:
            result = sim.run(n_steps=N_STEPS, compute_s_params=False, skip_preflight=True)
            probe = result.dft_planes['trans']
        spectra[model] = np.asarray(probe.accumulator).mean(axis=(1, 2))
    dc, f0 = (np.abs(spectra[m] / spectra['ref']) for m in ('dc', 'f0'))
    assert np.max(np.abs(dc - 1 / 1.4)) < 1e-3
    assert np.max(np.abs(f0 - 1 / 1.4)) < 1e-3
    assert np.max(np.abs(dc - f0)) < 1e-3


def _record_model(model, lane='run_uniform'):
    kw = {'dz_profile': np.full(10, DX)} if lane.endswith('nonuniform') else {}
    sim = Simulation(freq_max=10e9, domain=(10*DX,)*3, dx=DX,
                     boundary='pec', snap='declared', **kw)
    sigma = {'pec': 5.8e7, 'dc': 1/(ETA0*THICKNESS), 'f0': np.pi*F0*MU0/ETA0**2}[model]
    sim.add_thin_conductor(Box((2*DX, 3*DX, 5*DX), (7*DX, 8*DX, 5*DX)),
                           sigma_bulk=sigma, thickness=THICKNESS,
                           surface_impedance_f0=F0 if model == 'f0' else None)
    return sim


_RECORD_LANES = ['run_uniform', 'run_nonuniform', 'fwd_uniform', 'fwd_nonuniform',
                 'run_periodic', 'fwd_periodic', 's_matrix_scan', 'waveguide_s_matrix',
                 'msl_validator', 'current_moments']


@pytest.mark.parametrize('lane', _RECORD_LANES)
@pytest.mark.parametrize('model', ['pec', 'dc', 'f0'])
def test_record_and_context_cells(lane, model):
    sim = _record_model(model, lane)
    if lane.endswith('periodic'):
        from rfx.boundaries.spec import BoundarySpec
        sim._boundary_spec = BoundarySpec(x='pec', y='periodic', z='pec')
        sim._periodic_axes = 'y'
    admission = {'msl_validator': 'run_uniform', 'current_moments': 'run_uniform',
                 'run_periodic': 'run_uniform', 'fwd_periodic': 'fwd_uniform'}.get(lane, lane)
    admit(sim, admission)
    rec = sim.realized_geometry()
    axis = rec.entities[0].axes[1]
    assert axis.extent_m == pytest.approx((5.7 if model == 'pec' else 5)*DX, rel=1e-9)
    root = rec.conductors
    ctx = root.sheet_context(root.pec_edges)
    if model == 'f0':
        assert np.count_nonzero(ctx.mask_ex) == 30
        assert np.count_nonzero(ctx.end_ex) == 10
    else:
        assert ctx is None


@pytest.mark.parametrize('model', ['pec', 'dc', 'f0'])
@pytest.mark.parametrize('lane', ['run_distributed', 'run_distributed_nu',
                                   'fwd_distributed_nu', 'run_adi', 'fwd_adi', 'run_subgridded'])
def test_refused_cells_and_distributed_dc_record(lane, model):
    sim = _record_model(model)
    # A DC lossy conductor is admitted on the multi-device paths, uniform and graded
    # (rfx/runners/_admission.py, '_thin_conductors'/'lossy_sheet' rows); ADI and subgridded refuse it.
    if model == 'dc' and lane in ('run_distributed', 'run_distributed_nu', 'fwd_distributed_nu'):
        admit(sim, lane)
        record = sim.realized_geometry()
        assert record.entities[0].axes[1].extent_m == pytest.approx(5*DX, rel=1e-9)
        assert record.conductors.sheet_context(record.conductors.pec_edges) is None
    else:
        word = {'pec': 'PEC thin conductor', 'dc': 'lossy thin conductor',
                'f0': 'surface-impedance sheet'}[model]
        with pytest.raises(NotImplementedError, match=word):
            admit(sim, lane)


def test_only_conductors_builds_sheet_contexts():
    root = Path(__file__).resolve().parents[2] / 'rfx'
    offenders = []
    for path in root.rglob('*.py'):
        if path == root / 'model/conductors.py':
            continue
        tree = ast.parse(path.read_text())
        names = {'build_sheet_impedance_ctx', '_build_sheet_ctx'}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names
                             if a.name == 'build_sheet_impedance_ctx')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
                if name in names:
                    offenders.append((str(path.relative_to(root)), node.lineno))
    assert not offenders, offenders


@pytest.mark.parametrize('model', ['pec', 'dc', 'f0'])
def test_record_consumes_owner_width_product(monkeypatch, model):
    sim = _record_model(model)
    original = RealizedConductors.width_axes
    def shifted(root, *args, **kwargs):
        return tuple(replace(p, lo=p.lo-DX) for p in original(root, *args, **kwargs))
    monkeypatch.setattr(RealizedConductors, 'width_axes', shifted)
    axis = sim.realized_geometry().entities[0].axes[1]
    assert axis.extent_m == pytest.approx((6.7 if model == 'pec' else 6)*DX, rel=1e-9)


def test_monitor_reads_excluded_sheet_context_without_reassembly(monkeypatch):
    from rfx.current_moments import monitor_for_simulation
    sim = _monitor_fixture()
    root = realized_conductors(sim, sim._build_grid())
    original = RealizedConductors.sheet_context
    seen = []
    def observe(self, edges):
        ctx = original(self, edges)
        seen.append(ctx)
        return ctx
    def forbidden(*args, **kwargs):
        raise AssertionError('the monitor must borrow the solve assembly')
    monkeypatch.setattr(RealizedConductors, 'sheet_context', observe)
    monkeypatch.setattr(sim, '_assemble_materials', forbidden)
    monitor = monitor_for_simulation(sim, root.grid, root.periodic, conductors=root)
    assert monitor is not None and len(seen) == 1
    ctx = seen[0]
    assert [np.count_nonzero(getattr(ctx, 'mask_e'+a)) for a in 'xy'] == [24, 24]
    assert [np.count_nonzero(getattr(ctx, 'end_e'+a)) for a in 'xy'] == [8, 8]
    for mask, hard in zip((ctx.mask_ex, ctx.mask_ey, ctx.mask_ez), root.pec_edges):
        assert not np.any(np.asarray(mask) & np.asarray(hard))


def test_direct_forward_lends_complete_monitor_products():
    sim = _monitor_fixture()
    root = realized_conductors(sim, sim._build_grid())
    materials, debye, lorentz = root.assembly[:3]
    result = sim._forward_from_materials(root.grid, materials, debye, lorentz,
        n_steps=4, checkpoint=False, pec_mask=root.pec_cells, pec_sheets=root.sheets,
        pec_wires=root.wires, sheet_impedance=root.sheet_context(root.pec_edges))
    assert result.current_moment_data is not None
    assert np.isfinite(np.asarray(result.current_moment_data[0])).all()
