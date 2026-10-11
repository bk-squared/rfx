"""Assembly refusals are errors on the standalone preflight entry too.

Fixtures reuse the existing lattice-ownership, pole-stability, pinned-sheet,
DC-film and realization admission witnesses; no refusal rule is duplicated.
"""
from dataclasses import replace
import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.csg import PolylineWire, Sphere
from tests.unit.materials.test_pole_stability_admission import _box
from tests.locks.test_preflight_split_snapshot import (
    _pec_box_subcell_sim, _pec_realization_refused_sim, _pec_zero_cells_sim,
)
from tests.contracts.test_dc_film_admission import model as dc_model, UnknownArea


def _geometry_model(case, graded):
    builders = {'pec_box_subcell': _pec_box_subcell_sim,
                'pec_zero_cells': _pec_zero_cells_sim,
                'pec_realization_refused': _pec_realization_refused_sim}
    if case in builders:
        sim = builders[case]()
        if graded:
            sim._dz_profile = np.full(round(sim._domain[2] / sim._dx), sim._dx)
        return sim
    # Coordinates from test_lattice_ownership_contract's refused sheet/wire
    # witnesses, with a PEC boundary so the physical node range is explicit.
    sim = Simulation(10e9, (.02, .02, .02), dx=.001, boundary='pec',
                     **({'dz_profile': np.full(20, .001)} if graded else {}))
    if case == 'outside_plane':
        sim.add(Box((-.005, -.005, .024), (.005, .005, .024)), material='pec')
    elif case == 'zero_nodes':
        sim.add(Box((.0022, .0022, .006), (.0028, .0028, .006)), material='pec')
    elif case == 'thick_sheet':
        sim.add_thin_conductor(Box((.002, .002, .006), (.008, .008, .008)))
    elif case == 'diagonal_wire':
        sim.add(PolylineWire(((.001, .001, .001), (.003, .002, .001)), radius=0), material='pec')
    elif case == 'single_wire_node':
        sim.add(PolylineWire(((.001, .001, .001),), radius=0), material='pec')
    elif case == 'wire_radius':
        sim.add(PolylineWire(((.002, .01, .01), (.018, .01, .01)), radius=.0001), material='pec')
    elif case == 'pinned_plane':
        sim.add_pinned_sheet(normal_axis=2, plane_index=99, i_range=(1, 4), j_range=(1, 4))
    elif case == 'pinned_range':
        sim.add_pinned_sheet(normal_axis=2, plane_index=4, i_range=(1, 99), j_range=(1, 4))
    elif case == 'pinned_line':
        sim.add_pinned_sheet(normal_axis=2, plane_index=4, i_range=(1, 1), j_range=(1, 4))
    return sim


def _model(case, graded):
    if case == 'unstable_pole':
        return _box(.45, graded=graded)
    if case == 'oriented_continuation':
        from rfx import OrientedBox
        from tests.unit.geometry.test_oriented_box import DOMAIN, DX, PAD, SIZE, Q
        sim = Simulation(6e9, DOMAIN, dx=DX, cpml_layers=PAD,
                         **({'dz_profile': np.full(50, DX)} if graded else {}))
        sim.add(OrientedBox((0., .045, .05), SIZE, Q), material='pec')
        return sim
    if case == 'f0_without_bounds':
        from rfx.materials.thin_conductor import ThinConductor
        from tests.unit.materials.test_sheet_impedance import (
            MaskOnlyShape, _planar_sheet, _box_sheet, NU_L, NU_DX, NU_DZ,
            NU_Z, NU_FOOT, SIGMA_BULK, THICKNESS, F0,
        )
        sim = Simulation(10e9, (NU_L, NU_L, float(np.sum(NU_DZ))), dx=NU_DX,
                         boundary='cpml', cpml_layers=6,
                         **({'dz_profile': NU_DZ} if graded else {}))
        sim.add_thin_conductor(_box_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS)
        sim._thin_conductors[0] = ThinConductor(
            shape=MaskOnlyShape(_planar_sheet(NU_Z, NU_FOOT)),
            sigma_bulk=SIGMA_BULK, thickness=THICKNESS, surface_impedance_f0=F0)
        return sim
    if case.startswith('dc_'):
        sim, _, shape = dc_model('normal' if graded else 'uniform',
            radius=2.5 if case == 'dc_area' else 10.5,
            centre=(24, 24), height=2.5 if case == 'dc_thick' else 0)
        if case == 'dc_ambiguous':
            shape = Sphere(shape.center, .0025)
        elif case == 'dc_empty':
            shape = replace(shape, center=(.2, .2, shape.center[2]))
        elif case == 'dc_unknown_area':
            shape = UnknownArea(shape)
        sim._thin_conductors[0] = replace(sim._thin_conductors[0], shape=shape)
        return sim
    return _geometry_model(case, graded)


CASES = ('unstable_pole', 'pec_box_subcell', 'pec_zero_cells', 'pec_realization_refused', 'outside_plane',
         'zero_nodes', 'thick_sheet', 'diagonal_wire', 'wire_radius',
         'pinned_plane', 'pinned_range', 'dc_area', 'dc_thick', 'dc_ambiguous',
         'dc_empty', 'dc_unknown_area', 'oriented_continuation', 'f0_without_bounds',
         'single_wire_node', 'pinned_line')


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('case', CASES)
def test_preflight_reports_run_refusal(case, graded, capsys):
    sim = _model(case, graded)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, NotImplementedError)) as refused:
            sim.run(n_steps=1, compute_s_params=False)
    capsys.readouterr()
    report = sim.preflight()
    findings = [issue for issue in report
                if issue.severity == 'error' and str(refused.value) in str(issue)]
    assert len(findings) == 1
    if case in ('pec_box_subcell', 'pec_zero_cells', 'pec_realization_refused'):
        assert findings[0].code == case
        assert "geometry[0] 'pec'" in str(findings[0])
        assert not report.by_code('assembly_refusal')
    assert report.ok is False
    assert 'All checks passed' not in capsys.readouterr().out
    with pytest.raises(ValueError) as strict:
        sim.preflight(strict=True)
    assert str(refused.value) in str(strict.value)


@pytest.mark.parametrize('exc_type', [ValueError, NotImplementedError, TypeError,
                                     KeyError, AttributeError, IndexError])
@pytest.mark.parametrize('stage', ['grid', 'assembly'])
def test_original_exception_is_reported(monkeypatch, exc_type, stage, capsys):
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    exc = exc_type('original: detail')
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise exc

    monkeypatch.setattr(sim, '_build_grid' if stage == 'grid' else '_assemble_realized', fail)
    report = sim.preflight()
    refusal = issubclass(exc_type, (ValueError, NotImplementedError))
    code = 'assembly_refusal' if refusal else 'assembly_fault'
    findings = report.by_code(code)
    assert len(findings) == 1
    assert findings[0].severity == 'error'
    expected = ('The model cannot be assembled: ' + str(exc) if refusal else
                f'The model assembly raised {exc_type.__name__}: {exc}; this is a defect, please report it.')
    assert str(findings[0]) == expected
    assert sim._campaign_ctx()._assembly_exception is exc
    assert not report.ok
    assert 'All checks passed' not in capsys.readouterr().out
    assert len(calls) == 1


def test_general_sparameters_includes_assembly_refusal():
    report = _box(.45).preflight_sparameters(include_general=True)
    assert len(report.by_code('assembly_refusal')) == 1
    assert not report.ok


@pytest.mark.parametrize('exc_type', [ValueError, KeyError])
@pytest.mark.parametrize('other', ['matching_error', 'matching_warning', 'unrelated_error', 'quoting_error'])
def test_assembly_finding_defers_only_to_located_refusal(monkeypatch, exc_type, other):
    from rfx.preflight._common import PreflightWarning

    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    exc = exc_type('ez')
    other_code = 'adi_interior_pec_unsupported' if other == 'quoting_error' else 'pec_realization_refused'

    def fail(*args, **kwargs):
        raise exc

    def located(*args, **kwargs):
        message = str(exc) if other != 'unrelated_error' else 'different failure'
        warnings.warn(PreflightWarning(
            'geometry entry: ' + message,
            severity='warning' if other == 'matching_warning' else 'error',
            code=other_code, loc='geometry[0]',
        ))

    monkeypatch.setattr(sim, '_assemble_realized', fail)
    # This validator runs after the configuration registry's assembly check.
    monkeypatch.setattr(sim, '_validate_ntff_inverse_design', located)
    report = sim.preflight()
    code = 'assembly_refusal' if exc_type is ValueError else 'assembly_fault'
    assert len(report.by_code(code)) == (0 if other == 'matching_error' else 1)
    assert len(report.by_code(other_code)) == 1
    assert not report.ok


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('kind', ['wire', 'dc_film'])
def test_traced_radius_refusal_with_concrete_mesh(kind, graded, capsys):
    # Existing witnesses: test_pec_filament_radius::test_traced_radius_still_
    # refuses_with_static_metrics and test_dc_film_admission::test_traced_radius_
    # refuses_without_raw_tracer_error. The mesh itself remains concrete.
    def check(radius):
        sim = _model('wire_radius' if kind == 'wire' else 'dc_area', graded)
        if kind == 'wire':
            entry = sim._geometry[0]
            sim._geometry[0] = replace(entry, shape=replace(entry.shape, radius=radius))
        else:
            entry = sim._thin_conductors[0]
            sim._thin_conductors[0] = replace(entry, shape=replace(entry.shape, radius=radius))
        with pytest.raises((ValueError, NotImplementedError)) as refused:
            sim.run(n_steps=1, compute_s_params=False)
        capsys.readouterr()
        report = sim.preflight()
        findings = [issue for issue in report
                    if issue.severity == 'error' and str(refused.value) in str(issue)]
        assert len(findings) == 1
        assert not report.ok
        assert 'All checks passed' not in capsys.readouterr().out
        with pytest.raises(ValueError) as strict:
            sim.preflight(strict=True)
        assert str(refused.value) in str(strict.value)
        return radius

    jax.make_jaxpr(check)(.0001 if kind == 'wire' else .0025)


def test_failed_assembly_retains_campaign_unavailable_advisory(monkeypatch):
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    sim.add(Box((.002,) * 3, (.004,) * 3), material='pec')

    def refuse(*args, **kwargs):
        raise ValueError('assembly unavailable')

    monkeypatch.setattr(sim, '_assemble_realized', refuse)
    report = sim.preflight()
    assert len(report.by_code('assembly_refusal')) == 1
    assert len(report.by_code('campaign_statics_unavailable')) == 1


def test_one_assembly_and_lent_product_is_reused(monkeypatch):
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    original = sim._assemble_realized
    calls = []

    def assemble(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, '_assemble_realized', assemble)
    sim.preflight()
    assert len(calls) == 1
    product = sim._campaign_ctx().realized()

    def forbidden(*args, **kwargs):
        pytest.fail('lent product must not assemble again')

    monkeypatch.setattr(sim, '_assemble_realized', forbidden)
    monkeypatch.setattr('rfx.model.source_admission.refuse_dead_soft_sources', forbidden)
    monkeypatch.setattr('rfx.model.pad_fill.check_pad_fill', forbidden)
    report = sim._preflight_impl(_conductors=product)
    assert not report.by_code('assembly_refusal')
    assert not report.by_code('assembly_fault')


def _solve_only_model(case, graded, monkeypatch):
    if case == 'dead_soft_source':
        # Reviewer scen7.py: a soft source at the centre of an interior volume.
        sim = Simulation(10e9, (.024,) * 3, dx=.001, boundary='pec',
                         **({'dz_profile': np.full(24, .001)} if graded else {}))
        sim.add(Box((.006,) * 3, (.018,) * 3), material='pec')
        sim.add_source((.012,) * 3, 'ez', amplitude_kind='field')
        sim.add_probe((.003,) * 3, 'ez')
        return sim
    if case == 'pad_shortfall':
        from tests.unit.geometry.test_pad_fill_paths import _graded_rig, _rig, _size_by_plain_ceil
        _size_by_plain_ceil(monkeypatch)
        return _graded_rig(monkeypatch) if graded else _rig(10)
    if case in ('periodic_interval', 'outside_point'):
        from tests.unit.boundaries.test_periodic_extents import model
        sim = model()
        if case == 'periodic_interval':
            sim.add_port((.009, .003, .002), 'ex', extent=.002, impedance=50., excite=False)
        else:
            sim.add_source((.015, .002, .002), 'ez')
        return sim
    if case == 'dead_wall':
        from tests.contracts.test_source_on_pec_wall import model, DX
        sim = model('graded_run' if graded else 'uniform_run')
        sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
        return sim
    if case == 'dead_wire':
        from tests.contracts.test_port_edge_release import model
        sim, grid, _, _ = model('graded' if graded else 'uniform', 'wire')
        lo = tuple(float(grid.node_of(a, i)) for a, i in enumerate((8, 8, 4)))
        hi = tuple(float(grid.node_of(a, i)) for a, i in enumerate((9, 9, 8)))
        sim.add(Box(lo, hi), material='pec')
        return sim
    from tests.contracts.path_equivalence.test_source_on_pec import model
    return model('graded' if graded else 'uniform',
                 component='hx' if case == 'dead_h_source' else 'ez')


SOLVE_ONLY_CASES = [(case, graded) for case in
                    ('dead_soft_source', 'dead_h_source', 'dead_wire', 'dead_wall', 'pad_shortfall')
                    for graded in (False, True)] + [
                        ('periodic_interval', False), ('outside_point', False)]


@pytest.mark.parametrize('case,graded', SOLVE_ONLY_CASES,
                         ids=[f'{case}-{"graded" if graded else "uniform"}'
                              for case, graded in SOLVE_ONLY_CASES])
def test_solve_only_refusal_is_reported(case, graded, monkeypatch, capsys):
    sim = _solve_only_model(case, graded, monkeypatch)
    with pytest.raises(ValueError) as refused:
        sim.run(n_steps=1, compute_s_params=False)
    capsys.readouterr()
    original = sim._assemble_realized
    calls = []

    def assemble(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, '_assemble_realized', assemble)
    # A cached audit product must retain the admission refusal on repeat calls.
    for _ in range(2):
        report = sim.preflight()
        hits = [i for i in report.errors if str(refused.value) in str(i)]
        assert len(hits) == 1
        assert not report.ok
        assert 'All checks passed' not in capsys.readouterr().out
        with pytest.raises(ValueError) as strict:
            sim.preflight(strict=True)
        assert str(refused.value) in str(strict.value)
    assert len(calls) == 1


def test_adi_quoting_an_assembly_refusal_keeps_both_errors(capsys):
    # Reviewer scen3.py: ADI + unstable Lorentz + interior PEC volume.
    from rfx import GaussianPulse
    from rfx.materials.lorentz import lorentz_pole
    sim = Simulation(10e9, (.024,) * 3, dx=.001, boundary='pec', solver='adi')
    sim.add_material('res', eps_r=2., lorentz_poles=[lorentz_pole(1., 5e12, 1e9)])
    sim.add(Box((.008,) * 3, (.014,) * 3), material='res')
    sim.add(Box((.016, .016, .004), (.020, .020, .010)), material='pec')
    sim.add_source((.005, .006, .006), 'ez', waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.017, .016, .015), 'ez')
    with pytest.raises(ValueError) as refused:
        sim.run(n_steps=1, compute_s_params=False)
    capsys.readouterr()
    report = sim.preflight()
    assert len(report.by_code('assembly_refusal')) == 1
    assert len(report.by_code('adi_interior_pec_unsupported')) == 1
    assert str(refused.value) in str(report.by_code('assembly_refusal')[0])
    assert str(refused.value) in str(report.by_code('adi_interior_pec_unsupported')[0])
    assert len(report.errors) == 2
    assert not report.ok
    assert 'All checks passed' not in capsys.readouterr().out
    with pytest.raises(ValueError) as strict:
        sim.preflight(strict=True)
    assert str(refused.value) in str(strict.value)


def test_same_text_at_a_different_raise_site_is_not_a_duplicate(monkeypatch):
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')

    def assembly(*args, **kwargs):
        raise ValueError('ez')

    def later_validator(*args, **kwargs):
        raise ValueError('ez')

    monkeypatch.setattr(sim, '_assemble_realized', assembly)
    monkeypatch.setattr(sim, '_validate_ntff_inverse_design', later_validator)
    report = sim.preflight()
    assert len(report.by_code('assembly_refusal')) == 1
    assert len(report.by_code('uncoded')) == 1
    assert len(report.errors) == 2
