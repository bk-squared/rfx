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
def test_record_and_context_cells(monkeypatch, lane, model):
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
    if lane in ('s_matrix_scan', 'waveguide_s_matrix', 'msl_validator', 'current_moments'):
        _extractor_context_cell(monkeypatch, lane, model)


@pytest.mark.parametrize('model', ['pec', 'dc', 'f0'])
@pytest.mark.parametrize('lane', ['run_distributed', 'run_distributed_nu',
                                   'fwd_distributed_nu', 'run_adi', 'fwd_adi', 'run_subgridded'])
def test_refused_cells_and_distributed_dc_record(lane, model):
    sim = _record_model(model)
    # A DC lossy conductor is admitted on the multi-device paths, uniform and graded
    # (rfx/runners/_admission.py, '_thin_conductors'/'lossy_sheet' rows); ADI and subgridded refuse it.
    if (model == 'dc' and lane in ('run_distributed', 'run_distributed_nu', 'fwd_distributed_nu')
            or model == 'pec' and lane in ('run_distributed', 'run_distributed_nu', 'fwd_distributed_nu')):
        admit(sim, lane)
        record = sim.realized_geometry()
        assert record.entities[0].axes[1].extent_m == pytest.approx((5.7 if model == 'pec' else 5)*DX, rel=1e-9)
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


@pytest.mark.parametrize('lent', ['none', 'kernel', 'traced_assembly'])
def test_traced_materials_monitor_value_and_gradient(lent):
    """Topology's bare forward and a lent kernel root keep the census concrete."""
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from rfx.core.yee import MaterialArrays
    from rfx.model.conductors import kernel_conductors

    sim = Simulation(freq_max=1.2e10, domain=(24e-3,)*3, dx=1e-3,
                     cpml_layers=6, boundary='cpml')
    sim.add_source((12e-3,)*3, 'ez', amplitude_kind='current')
    sim.add_current_moment_monitor((6e-3,)*3, (18e-3,)*3,
                                   block_size=6e-3, freqs=np.array([3e9]))
    root = realized_conductors(sim, sim._build_grid())
    base, debye, lorentz, pec = root.assembly[:4]
    i0, i1 = (root.grid.position_to_index((p,)*3) for p in (10e-3, 14e-3))
    region = SimpleNamespace(bounds=tuple(v for a in range(3) for v in (i0[a], i1[a]+1)))
    slices = tuple(slice(i0[a], i1[a]+1) for a in range(3))

    def objective(rho):
        materials = MaterialArrays(base.eps_r.at[slices].set(1+5*rho), base.sigma, base.mu_r)
        borrowed = None
        if lent == 'kernel':
            borrowed = kernel_conductors(sim, root.grid, materials, pec, periodic=root.periodic)
        elif lent == 'traced_assembly':
            borrowed = replace(root, materials=materials, assembly=(materials, *root.assembly[1:]))
        result = sim._forward_from_materials(root.grid, materials, debye, lorentz,
            n_steps=4, checkpoint=True, pec_mask=pec, conductors=borrowed,
            monitor_overrides={'design_region': region})
        return jnp.sum(jnp.abs(result.current_moment_data[0])**2)

    value, gradient = jax.value_and_grad(objective)(jnp.float32(.5))
    assert np.isfinite(value) and np.isfinite(gradient)


def test_2d_run_and_audit_keep_grid_axis_width():
    sim = Simulation(freq_max=10e9, domain=(12e-3, 12e-3, 1e-3), dx=1e-3,
                     boundary='cpml', cpml_layers=4, mode='2d_tmz', snap='declared')
    sim.add_thin_conductor(Box((3e-3, 6e-3, 0), (8e-3, 6e-3, 1e-3)),
                           sigma_bulk=1e4, thickness=35e-6, surface_impedance_f0=9e9)
    sim.add_source((6e-3, 3e-3, 0), 'ez', waveform=GaussianPulse(f0=9e9, bandwidth=.8))
    audit = sim.realized_geometry()
    result = sim.run(n_steps=4, compute_s_params=False, skip_preflight=True)
    assert audit.entities[0].axes[2].extent_m == 0
    assert result.realized_geometry.entities[0].axes[2].extent_m == 0


@pytest.mark.parametrize('span', [(5, 10), (0, 10)])
def test_graded_periodic_declaration_keeps_grid_axis_width(span):
    from rfx.boundaries.spec import BoundarySpec
    sim = Simulation(freq_max=10e9, domain=(10e-3,)*3, dx=1e-3,
                     boundary=BoundarySpec(x='pec', y='periodic', z='pec'),
                     snap='declared', dz_profile=np.full(10, 1e-3))
    sim.add_thin_conductor(Box((2e-3, span[0]*1e-3, 5e-3), (7e-3, span[1]*1e-3, 5e-3)),
                           sigma_bulk=1e4, thickness=35e-6, surface_impedance_f0=9e9)
    axis = sim.realized_geometry().entities[0].axes[1]
    assert axis.extent_m == pytest.approx((span[1]-span[0])*1e-3)


def _assert_context_equal(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    for field in fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if right is None:
            assert left is None
        else:
            np.testing.assert_array_equal(left, right)


def _extractor_context_cell(monkeypatch, lane, model):
    """Reach production consumers; stop scan extractors at their kernel handoff."""
    import sys
    from rfx.current_moments import monitor_for_simulation, refuse_current_the_monitor_cannot_see
    from rfx.sources.msl_port import msl_port_from_entry, validate_msl_port_geometry
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    from rfx.sparams import waveguide
    from tests.unit.materials.test_sheet_impedance import _wr90
    from tests.unit.ports.test_msl_realized_port_contract import _model

    original = RealizedConductors.sheet_context
    built, used = [], []

    def observe(owner, edges):
        ctx = original(owner, edges)
        built.append((owner, edges, ctx))
        return ctx

    class HandoffReached(Exception):
        pass

    def handoff(*args, **kwargs):
        used.append(kwargs['sheet_impedance'])
        raise HandoffReached

    def set_model(sim):
        sim._thin_conductors = [replace(tc,
            sigma_bulk=5.8e7 if model == 'pec' else 1e4,
            surface_impedance_f0=tc.surface_impedance_f0 if model == 'f0' else None)
            for tc in sim._thin_conductors]
        return sim

    with monkeypatch.context() as patch:
        patch.setattr(RealizedConductors, 'sheet_context', observe)
        if lane == 's_matrix_scan':
            sim = _record_model(model)
            sim.add_port((5*DX, 5*DX, 3*DX), 'ez')
            patch.setattr(sim, '_forward_from_materials', handoff)
            with pytest.raises(HandoffReached):
                compute_lumped_wire_s_matrix_via_scan(sim, np.array([F0]), n_steps=4)
        elif lane == 'waveguide_s_matrix':
            sim = set_model(_wr90())
            patch.setattr(waveguide, 'extract_waveguide_s_matrix', handoff)
            with pytest.raises(HandoffReached):
                sim.compute_waveguide_s_matrix(n_steps=4)
        else:
            # Capture the context local after the real validator has consumed it.
            # A value overwritten/dropped after the owner call must also fail.
            if lane == 'msl_validator':
                sim, _ = _model(trace_kind='f0' if model == 'f0' else 'pec')
                if model == 'dc':
                    sim.add_thin_conductor(Box((2e-3, 2e-3, 14e-3), (6e-3, 6e-3, 14e-3)),
                                           sigma_bulk=1e4, thickness=35e-6)
                target, local = validate_msl_port_geometry, 'sheet_impedance'
            else:
                sim = set_model(_monitor_fixture())
                target, local = refuse_current_the_monitor_cannot_see, 'ctx'
            root = realized_conductors(sim, sim._build_grid())

            def capture(frame, event, arg):
                if frame.f_code is target.__code__ and event == 'return':
                    used.append(frame.f_locals.get(local))

            previous = sys.getprofile()
            sys.setprofile(capture)
            try:
                if lane == 'msl_validator':
                    validate_msl_port_geometry(root.grid, msl_port_from_entry(sim._msl_ports[0]),
                                               pec_edge_masks=root.pec_edges, conductors=root)
                else:
                    assert monitor_for_simulation(sim, root.grid, conductors=root) is not None
            finally:
                sys.setprofile(previous)

    assert len(used) == 1
    if model == 'f0' or lane != 'current_moments':
        assert len(built) == 1
        owner, edges, ctx = built[0]
        assert used[0] is ctx
        expected = original(owner, edges)
        kernel, _ = at_kernel(sim, owner, lane='fwd_uniform', pec_edges=edges,
                              sheet_operator=expected)
        _assert_context_equal(used[0], kernel.sheet_operator)
        if model == 'f0':
            assert used[0] is not None
            assert any(np.any(getattr(used[0], 'mask_e'+a)) for a in 'xyz')
    else:
        assert not built and used[0] is None


@pytest.mark.parametrize('nonuniform', [False, True])
def test_kernel_only_monitor_censuses_sim_not_caller_arrays(nonuniform):
    from rfx.current_moments import monitor_for_simulation
    from rfx.model.conductors import kernel_conductors
    kw = {'dz_profile': np.full(24, 1e-3)} if nonuniform else {}
    sim = Simulation(freq_max=1.2e10, domain=(24e-3,)*3, dx=1e-3,
                     cpml_layers=6, boundary='cpml', **kw)
    sim.add_current_moment_monitor((6e-3,)*3, (18e-3,)*3,
                                   block_size=6e-3, freqs=np.array([3e9]))
    grid = sim._build_nonuniform_grid() if nonuniform else sim._build_grid()
    root = realized_conductors(sim, grid, nonuniform=nonuniform)
    # Bare callers (including mixed S-matrix) lend solve arrays, not the census.
    caller = root.materials._replace(mu_r=root.materials.mu_r * 2)
    kernel = kernel_conductors(sim, grid, caller, root.pec_cells, periodic=root.periodic)
    assert len(kernel.assembly) == 4
    assert monitor_for_simulation(sim, grid, conductors=kernel) is not None
    # The reverse direction must still see a material in the sim declaration.
    sim.add_material('magnetic', mu_r=2.)
    sim.add(Box((10e-3,)*3, (14e-3,)*3), material='magnetic')
    kernel = kernel_conductors(sim, grid, root.materials, root.pec_cells, periodic=root.periodic)
    with pytest.raises(NotImplementedError, match='mu_r != 1'):
        monitor_for_simulation(sim, grid, conductors=kernel)
