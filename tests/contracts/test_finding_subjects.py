"""Finding identities are checked against scenes, independently of the annotator."""
import ast
from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import pytest

from rfx.preflight._scopes import CODE_SCOPE

ROOT = Path(__file__).resolve().parents[2]
REACHED = set()
# No platform exclusions: failure seams and a supplied memory budget let the
# CPU battery exercise the unavailable and memory findings without a GPU.
UNREACHABLE = {}

# These callers produce S-parameter setup reports, outside general preflight.
OUTSIDE_GENERAL = {
    'layout_measured_from_band_low_edge': 'S-parameter layout audit only',
    'waveguide_setup_audit_skipped': 'S-parameter setup audit only',
    'record_far_boundary_band_below_cutoff': 'S-parameter record audit only',
    'record_shorter_than_far_boundary_round_trip': 'S-parameter record audit only',
    'port_index_mirror_asymmetry': 'S-parameter mirror audit only',
    'port_index_mirror_known_e_plane_offset': 'S-parameter mirror audit only',
    'port_junction_probe_clearance': 'S-parameter runtime junction audit only',
    'dc_film_traced_pair': 'runtime JAX callback, outside preflight capture',
}


def emitting_sites():
    """Census constructors, constant codes, defaults and the flux adapter."""
    sites = []
    constructors = {'PreflightIssue', 'PreflightWarning', 'PreflightErrorWarning',
                    'PreflightConfigError', 'DCFilmAdmissionError'}
    for path in sorted((ROOT / 'rfx').rglob('*.py')):
        tree = ast.parse(path.read_text())
        constants = {target.id: node.value.value
                     for node in ast.walk(tree) if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                     for target in node.targets if isinstance(target, ast.Name)}

        def codes(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return {node.value}
            if isinstance(node, ast.Name) and node.id in constants:
                return {constants[node.id]}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                return codes(node.args[2]) if len(node.args) > 2 else set()
            return set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
            values = set()
            if name in constructors:
                for kw in node.keywords:
                    if kw.arg == 'code':
                        values |= codes(kw.value)
                if name == 'DCFilmAdmissionError' and not any(k.arg == 'code' for k in node.keywords):
                    values.add('dc_film_area')
            if name == 'finding' and path.name == '_preflight.py' and len(node.args) >= 3:
                values |= codes(node.args[2])
            for code in values:
                sites.append((str(path.relative_to(ROOT)), node.lineno, code))
    # Default for unstructured captured warnings is also a shipped report code.
    sites.append(('rfx/preflight/_impl.py', 0, 'uncoded'))
    return sites


def test_code_completeness():
    sites = emitting_sites()
    found = {code for _, _, code in sites} - OUTSIDE_GENERAL.keys()
    assert found <= CODE_SCOPE.keys(), sorted(found - CODE_SCOPE.keys())
    assert CODE_SCOPE.keys() <= found, sorted(CODE_SCOPE.keys() - found)
    assert len(CODE_SCOPE) == 98
    assert {scope: list(CODE_SCOPE.values()).count(scope) for scope in set(CODE_SCOPE.values())} == {
        'object': 71, 'place': 2, 'model': 23, 'mixed': 2}


def check_report(sim, report):
    REACHED.update(issue.code for issue in report)
    record = sim.realized_geometry()
    labels = [row.label for row in record.realization_table.rows]
    for issue in report:
        expected = CODE_SCOPE[issue.code]
        assert issue.scope == expected or expected == 'mixed'
        assert issue.scope in ('model', 'place', 'object')
        assert tuple(label for label in labels if label in issue.subjects) == issue.subjects
        if issue.scope == 'object':
            assert issue.subjects, issue.code
        if issue.scope == 'model':
            assert issue.subjects == ()
            assert issue.region is None
        if issue.scope == 'place':
            assert issue.region is not None
        if issue.region is not None:
            lo, hi = issue.region['bounds_m']
            for a, values in enumerate(record.nodes):
                nodes = np.asarray(values)
                assert nodes[0] <= lo[a] <= hi[a] <= nodes[-1]
                # Invert the supplied physical bounds using only the node array.
                start = int(np.flatnonzero(np.isclose(nodes, lo[a], rtol=0, atol=1e-13))[0])
                stop = int(np.flatnonzero(np.isclose(nodes, hi[a], rtol=0, atol=1e-13))[0])
                assert issue.region['cells'][a] == (start, stop)
        encoded = issue.to_dict()
        assert (encoded['scope'], encoded['subjects'], encoded['region']) == (
            issue.scope, issue.subjects, issue.region)
    assert [item['subjects'] for item in report.to_dict()['issues']] == [i.subjects for i in report]


# Each row states the owners built into the scene, not a transformation of loc,
# message text, annotation helpers, or stored report output.
SCENES = {
    'absorber_probe_outside': {'absorber_overlap': [('probe[0]',)],
                               'absorber_proximity': [('port[0]',)]},
    'absorber_probe_proximity': {'absorber_proximity': [('probe[0]',), ('port[0]',)]},
    'dispersive_pole_at_face': {'dispersive_pole_at_absorber_face': [('geometry[0]',)]},
    'geometry_in_cpml': {'geometry_in_absorber': [('geometry[0]',)], 'lossless_q': [('geometry[0]',)]},
    'guards_box_alumina': {'lossless_q': [('geometry[0]',)]},
    'guards_lossless_q': {'lossless_q': [('geometry[0]',)]},
    'guards_no_sources': {},
    'guards_under_resolved_dielectric': {'mesh_resolution': [('geometry[0]',)] * 3},
    'guards_upml_refinement': {},
    'nu_source_on_graded_node': {'source_on_graded_node': [('port[0]',)]},
    'nu_thin_conductor_graded_node': {'thin_conductor_graded_node': [('thin_conductor[0]',)]},
    'nu_wire_port_on_graded_node': {'wire_port_on_graded_node': [('port[0]',)] * 2},
    'nu_grading_beyond_cap': {},
    'nu_grading_reaches_absorber': {},
    'nu_cpml_thin': {},
    'nu_tfsf_oblique': {},
    'upml_nonuniform_lane': {},
    'absorber_budget_over_axis': {},
    'floquet_nonuniform_automesh': {},
    'subgrid_unsupported_feature': {},
    'ad_memory_sane': {},
    'pec_zero_cells': {'pec_zero_cells': [('geometry[0]',)]},
    'pec_realization_refused': {'pec_realization_refused': [('geometry[0]',)]},
    'pec_box_subcell': {'mesh_resolution': [('geometry[0]',)], 'pec_box_subcell': [('geometry[0]',)]},
    'source_decoupled': {'source_decoupled': [('domain', 'port[0]')]},
    'source_decoupled_whole_boundary': {'source_decoupled': [('domain', 'port[0]')]},
    'leontovich_thin_film_offband': {
        'absorber_proximity': [('port[0]',)],
        'thin_conductor_leontovich_thin_film': [('thin_conductor[0]',)],
        'thin_conductor_leontovich_band_offset': [('thin_conductor[0]', 'port[0]')],
        'sheet_effective_size': [('thin_conductor[0]',)]},
    'sheet_congruence_off_lattice': {
        'sheet_plane_realized': [('thin_conductor[0]', 'thin_conductor[1]')],
        'congruent_conductor_rasterization_parity': [('thin_conductor[0]', 'thin_conductor[1]')]},
    'conformal_fine_dx': {'port_aperture_snap': [('waveguide_port[0]',), ('waveguide_port[0]',)],
                         'port_freqs_below_cutoff': [('waveguide_port[0]',)]},
    'waveguide_nu_declared_geometry': {'port_evanescent': [('waveguide_port[0]',), ('waveguide_port[1]',)]},
    'ntff_absorber_overlap': {'absorber_overlap': [('ntff',)]},
    'floating_port': {'floating_port': [('geometry[0]', 'port[0]')],
                      'lossless_q': [('geometry[0]',)]},
    'off_lattice_edges': {'mesh_resolution': [('geometry[0]',)] * 2,
                          'off_lattice_design_edges': [('geometry[0]',)]},
    'refplane_near_field': {'refplane_near_field': [('port[0]', 'port[1]')],
                            'sheet_effective_size': [('thin_conductor[0]',)],
                            'off_lattice_design_edges': [('thin_conductor[0]',)],
                            'wire_port_no_radius': [('port[0]', 'port[1]')]},
    'refplane_partial_optin': {'refplane_partial_optin': [('port[0]', 'port[1]')],
                               'sheet_effective_size': [('thin_conductor[0]',)],
                               'off_lattice_design_edges': [('thin_conductor[0]',)],
                               'wire_port_no_radius': [('port[0]', 'port[1]')]},
    'msl_conductor_plane_mismatch': {
        'lossless_q': [('geometry[2]',)], 'sheet_effective_size': [('geometry[1]',)],
        'off_lattice_design_edges': [('geometry[1]',)],
        'msl_port_conductor_planes': [('geometry[0]', 'msl_port[0]')],
        'msl_port_geometry': [('msl_port[0]',), ('msl_port[0]',)]},
    'ntff_pec_overlap_full': {
        'ntff_pec_overlap': [('geometry[0]', 'ntff')],
        'port_in_pec': [('geometry[0]', 'port[0]'), ('geometry[0]', 'probe[0]')],
        'off_lattice_design_edges': [('geometry[0]',)],
        'port_conductor_continues': [('geometry[0]', 'port[0]')]},
    'pec_boundary_open': {'ntff_near_field': [('geometry[0]', 'ntff')]},
    'guards_msl_sheet_probe': {
        'absorber_proximity': [('port[0]',)], 'body_flush_on_absorber': [('geometry[0]',)],
        'lossless_q': [('geometry[0]',)], 'off_lattice_design_edges': [('geometry[1]',)],
        'conductor_in_thin_absorber': [('geometry[1]',), ('geometry[1]',)]},
    'waveguide_refplane_in_slab': {
        'mesh_resolution': [('geometry[0]',)] * 3, 'lossless_q': [('geometry[0]',)],
        'waveguide_reference_plane': [('geometry[0]', 'waveguide_port[0]')],
        'waveguide_port_obstacle_before_reference': [('geometry[0]', 'waveguide_port[0]')]},
    'wire_port_dead_extent': {
        'absorber_proximity': [('port[0]',)],
        'wire_port_dead_extent_cells': [('geometry[0]', 'port[0]')],
        'pec_box_one_cell': [('geometry[0]',)], 'wire_port_no_radius': [('port[0]',)]},
    'wire_port_microstrip': {
        'absorber_proximity': [('port[0]',)],
        'wire_port_midpoint_in_pec': [('geometry[0]', 'port[0]')],
        'pec_box_one_cell': [('geometry[0]',)], 'wire_port_no_radius': [('port[0]',)]},
    'wire_port_end_gap': {
        'mesh_resolution': [('geometry[1]',)],
        'wire_port_end_gap_to_conductor': [('geometry[0]', 'port[0]')],
        'pec_box_one_cell': [('geometry[0]', 'geometry[1]')], 'wire_port_no_radius': [('port[0]',)]},
}


@pytest.mark.parametrize('scene_id', SCENES)
def test_scene_subjects(scene_id):
    from tests.locks.test_preflight_split_snapshot import _FIXTURES
    _, build, kwargs, _ = next(row for row in _FIXTURES if row[0] == scene_id)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = build()
        report = sim.preflight(**kwargs)
        check_report(sim, report)
    actual = defaultdict(list)
    for issue in report:
        if issue.scope == 'object':
            actual[issue.code].append(issue.subjects)
    assert dict(actual) == SCENES[scene_id]


@pytest.mark.parametrize('path', ['uniform_run', 'graded_run', 'forward', 'distributed_v2', 'distributed_nu'])
def test_execution_path(path, monkeypatch):
    import jax
    from tests.contracts.test_realization_one_rule import scene
    graded = path in ('graded_run', 'distributed_nu')
    sim = scene('dielectric_box', 'graded_run' if graded else 'uniform_run')
    from rfx import GaussianPulse
    sim.add_source((.008, .007, .006), 'ez', waveform=GaussianPulse(f0=250e9, bandwidth=.8))
    reports = []
    original = sim._run_preflight_gate

    def gate(report, **kwargs):
        reports.append(report)
        return original(report, **kwargs)

    monkeypatch.setattr(sim, '_run_preflight_gate', gate)
    if path == 'forward':
        result = sim.forward(n_steps=1, skip_preflight=False)
    else:
        kwargs = {'devices': jax.devices('cpu')[:2]} if path.startswith('distributed') else {}
        if kwargs:
            assert len(kwargs['devices']) == 2
        result = sim.run(n_steps=1, skip_preflight=False, compute_s_params=False, **kwargs)
    assert len(reports) == 1
    assert reports[0].realization_table is result.realized_geometry.realization_table
    issues = reports[0].by_code('unresolved_pulse')
    assert issues
    assert {issue.subjects for issue in issues} == {('port[0]',)}
    check_report(sim, reports[0])


def test_traced_mesh_path(monkeypatch):
    import jax
    import jax.numpy as jnp
    from rfx import Box, GaussianPulse, Simulation
    reports = []

    def evaluate(delta):
        profile = jnp.full(10, .001).at[3].add(delta).at[4].add(-delta)
        sim = Simulation(freq_max=1e9, domain=(.01004, .01, .01), dx=.001,
                         boundary='pec', snap='declared', dz_profile=profile,
                         dt=.2*.001/299792458., dt_min_cell=.0005)
        sim.add_material('glass', eps_r=2.)
        sim.add(Box((.00223, .00237, .00219), (.00661, .00673, .00657)), material='glass')
        sim.add_source((.008, .007, .006), 'ez', waveform=GaussianPulse(f0=250e9, bandwidth=.8))
        original = sim._run_preflight_gate

        def gate(report, **kwargs):
            reports.append(report)
            return original(report, **kwargs)

        monkeypatch.setattr(sim, '_run_preflight_gate', gate)
        result = sim.forward(n_steps=1, skip_preflight=False)
        return jnp.sum(result.time_series)

    jax.grad(evaluate)(0.)
    assert len(reports) == 1
    issues = reports[0].by_code('unresolved_pulse')
    assert issues
    assert {issue.subjects for issue in issues} == {('port[0]',)}
    assert all(issue.region is None for issue in issues)
    REACHED.update(issue.code for issue in reports[0])


@pytest.mark.parametrize('scene_id,code,expected', [
    ('congruence_off_lattice', 'congruent_conductor_rasterization_parity', [('geometry[0]', 'geometry[1]')]),
    ('slot_vacuum', 'sheet_slot_vacuum', [('geometry[0]', 'geometry[1]', 'geometry[2]')]),
    ('cavity_faces', 'sheet_cavity_electrical_thickness', [('geometry[0]', 'thin_conductor[0]', 'thin_conductor[1]')]),
    ('stack_snapped_patch', 'sheet_cavity_spacing', [('geometry[0]', 'geometry[2]')]),
    ('tie_stack', 'thin_metal_nu_mesh', [('thin_conductor[0]',), ('thin_conductor[1]',)]),
    ('tie_stack', 'graded_box_rasterization', [('geometry[0]',)]),
    ('off_lattice_edges', 'off_lattice_design_edges', [('geometry[0]',)]),
    ('floating_port', 'floating_port', [('geometry[0]', 'port[0]')]),
    ('refplane_near_field', 'refplane_near_field', [('port[0]', 'port[1]')]),
    ('refplane_partial_optin', 'refplane_partial_optin', [('port[0]', 'port[1]')]),
    ('pec_face_short_of_wall', 'pec_face_short_of_domain_wall', [('geometry[0]', 'domain')]),
    ('wire_port_dead_extent', 'wire_port_dead_extent_cells', [('geometry[0]', 'port[0]')]),
    ('wire_port_dead_cell_nu', 'wire_port_dead_cell_classification_unavailable', [('port[0]',)]),
    ('wire_port_end_gap', 'wire_port_end_gap_to_conductor', [('geometry[0]', 'port[0]')]),
    ('wire_port_microstrip', 'wire_port_midpoint_in_pec', [('geometry[0]', 'port[0]')]),
    ('wire_port_microstrip', 'wire_port_no_radius', [('port[0]',)]),
    ('waveguide_refplane_in_slab', 'waveguide_reference_plane', [('geometry[0]', 'waveguide_port[0]')]),
    ('waveguide_refplane_in_slab', 'waveguide_port_obstacle_before_reference', [('geometry[0]', 'waveguide_port[0]')]),
    ('guards_msl_sheet_probe', 'body_flush_on_absorber', [('geometry[0]',)]),
    ('guards_msl_sheet_probe', 'conductor_in_thin_absorber', [('geometry[1]',)] * 2),
    ('coax_junction_short', 'coaxial_port_junction_short', [('geometry[0]', 'coaxial_port[0]')]),
    ('msl_conductor_plane_mismatch', 'msl_port_conductor_planes', [('geometry[0]', 'msl_port[0]')]),
    ('ntff_pec_overlap_full', 'ntff_pec_overlap', [('geometry[0]', 'ntff')]),
    ('ntff_pec_overlap_full', 'port_in_pec', [('geometry[0]', 'port[0]'), ('geometry[0]', 'probe[0]')]),
    ('ntff_small_ground_plane', 'ntff_small_ground_plane', [('geometry[0]', 'ntff')]),
    ('pec_boundary_open', 'pec_boundary_open', [()]),
    ('pec_boundary_open', 'ntff_near_field', [('geometry[0]', 'ntff')]),
    ('ntff_pec_overlap_full', 'port_conductor_continues', [('geometry[0]', 'port[0]')]),
    ('congruence_off_lattice', 'pec_box_one_cell', [('geometry[0]', 'geometry[1]')]),
    ('pec_face_short_of_wall', 'pec_faces_finite_pec', [('geometry[0]', 'domain')]),
])
def test_multi_owner_scene(scene_id, code, expected):
    from tests.locks.test_preflight_split_snapshot import _FIXTURES
    _, build, kwargs, _ = next(row for row in _FIXTURES if row[0] == scene_id)
    sim = build()
    sim._snap = 'declared'
    report = sim.preflight(**kwargs)
    check_report(sim, report)
    assert [issue.subjects for issue in report.by_code(code)] == expected


@pytest.mark.parametrize('kind', ['clamped', 'invalid'])
def test_flux_subjects(kind):
    from dataclasses import replace
    from tests.unit.preflight.test_flux_region_preflight import _sim, _add, U
    sim = _sim(graded=True)
    _add(sim, size=(60*U, 8*U) if kind == 'clamped' else (4*U, 8*U))
    if kind == 'invalid':
        sim._flux_monitors[0] = replace(sim._flux_monitors[0], size=(-U, U))
    report = sim.preflight(check_ntff=False)
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('flux_region_' + kind)] == [('window',)]


@pytest.mark.parametrize('kind', ['area', 'layers', 'permittivity'])
def test_dc_film_subjects(kind):
    from dataclasses import replace
    from tests.contracts.test_dc_film_admission import model
    sim, _, _ = model(radius=2.5 if kind == 'area' else 10.5,
                      centre=(24., 24.), height=2.5 if kind == 'layers' else 0.)
    if kind == 'permittivity':
        sim._thin_conductors[0] = replace(sim._thin_conductors[0], eps_r=2.)
    report = sim.preflight()
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('dc_film_' + kind)] == [('thin_conductor[0]',)]


def test_thickness_subjects():
    from tests.unit.preflight.test_thin_conductor_thickness import _sim
    sim = _sim(50e-6, .0005)
    report = sim.preflight()
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('thin_conductor_thickness_unmodelled')] == [
        ('geometry[0]', 'thin_conductor[0]')]


def test_half_node_subjects():
    from rfx import Box
    from tests.unit.preflight.test_half_node_split import _sim
    sim = _sim('uniform', dom=(.02, .019, .013))
    sim.add(Box((.005, .005, .0035), (.015, .014, .0035)), material='pec')
    sim.add_port((.010, .009, 0.), component='ez', extent=.0035 + .95e-9)
    report = sim.preflight()
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('half_node_split')] == [('geometry[0]', 'port[0]')]


def test_rlc_label_rows_are_non_geometric():
    from tests.contracts.test_realization_one_rule import scene
    plain = scene('dielectric_box', 'uniform_run')
    with_rlc = scene('dielectric_box', 'uniform_run')
    with_rlc.add_lumped_rlc((.008, .007, .006), 'ez', R=50.)
    original = plain.realized_geometry().realization_table
    record = with_rlc.realized_geometry().realization_table
    row = record.row('lumped_rlc[0]')
    assert row.status == 'not represented'
    assert row.axes == row.regions == row.traced_inputs == ()
    assert row.occupancy_count == 0
    assert [r for r in record.rows if r.label != 'lumped_rlc[0]'] == list(original.rows)
    before, after = plain.preflight(), with_rlc.preflight()
    # Existing locks and every realization finding remain unchanged with rows present.
    assert [i.to_dict() for i in before] == [i.to_dict() for i in after]


@pytest.mark.parametrize('scene_id', [
    'pec_zero_cells', 'pec_realization_refused', 'pec_box_subcell',
    'wire_port_end_gap', 'sheet_congruence_off_lattice', 'slot_vacuum',
    'cavity_faces', 'stack_snapped_patch', 'off_lattice_edges',
])
def test_rlc_rows_preserve_realization_findings(scene_id):
    from tests.locks.test_preflight_split_snapshot import _FIXTURES
    _, build, kwargs, _ = next(row for row in _FIXTURES if row[0] == scene_id)
    sim = build()
    before = [i.to_dict() for i in sim.preflight(**kwargs)]
    rows = sim.realized_geometry().realization_table.rows
    sim.add_lumped_rlc(tuple(float(v)/2 for v in sim._domain), 'ez', R=50.)
    after = [i.to_dict() for i in sim.preflight(**kwargs)]
    assert after == before
    assert tuple(r for r in sim.realized_geometry().realization_table.rows
                 if r.label != 'lumped_rlc[0]') == rows


def test_warning_carrier_serialization():
    from rfx.preflight._common import PreflightWarning
    warning = PreflightWarning('unchanged', code='example', loc='old-location',
                               source='old-source', scope='object', subjects=('geometry[0]',))
    assert str(warning) == 'unchanged'
    assert warning.to_dict() == {
        'message': 'unchanged', 'code': 'example', 'severity': 'warning',
        'loc': 'old-location', 'source': 'old-source', 'scope': 'object',
        'subjects': ('geometry[0]',), 'region': None}


def test_readable_record_fault_still_propagates(monkeypatch):
    import rfx.realized_geometry as records
    from tests.contracts.test_realized_geometry_record import _model

    def fail(*args, **kwargs):
        raise ValueError('record producer defect')

    monkeypatch.setattr(records, '_record_from_assembly', fail)
    with pytest.raises(ValueError, match='record producer defect'):
        _model().realized_geometry()


def assert_target(sim, code, expected, **kwargs):
    report = sim.preflight(**kwargs)
    check_report(sim, report)
    assert [i.subjects for i in report.by_code(code)] == expected
    return report


def test_msl_declaration_identity():
    from tests.unit.ports.test_msl_realized_port_contract import _model
    sim, _ = _model(port_top=11.)
    report = assert_target(sim, 'msl_port_conductor_planes', [('geometry[0]', 'msl_port[0]')])
    # The remaining two checks concern launch clearance and the probe ladder;
    # neither measures the substrate or conductor interval.
    assert [i.subjects for i in report.by_code('msl_port_geometry')] == [
        ('msl_port[0]',), ('msl_port[0]',)]


@pytest.mark.parametrize('kind', ['stub', 'uninspectable', 'unrealizable'])
def test_line_subjects(kind):
    from dataclasses import replace
    from tests.unit.preflight.test_line_stub import make_sim
    sim = make_sim()
    code, expected = 'line_stub_behind_port', [('geometry[2]', 'msl_port[0]')]
    if kind == 'uninspectable':
        class NoBounds:
            def mask(self, grid):
                raise NotImplementedError

            def mask_on_coords(self, x, y, z):
                raise NotImplementedError

        sim._geometry.append(replace(sim._geometry[-1], shape=NoBounds()))
        code, expected = 'line_stub_inspection_unavailable', [('geometry[3]',)]
    if kind == 'unrealizable':
        from tests.unit.ports.test_msl_port_preflight import _build_sim, W_TRACE, H_SUB
        sim = _build_sim(dx=.001, ly=W_TRACE + 8*H_SUB)
        code, expected = 'line_stub_realization', [()]
    assert_target(sim, code, expected)


@pytest.mark.parametrize('kind', ['pec_face', 'pmc', 'invariant'])
def test_film_boundary_subjects(kind):
    from rfx import Box, Simulation
    from rfx.boundaries.spec import BoundarySpec
    sim = Simulation(10e9, (.012, .012, .001 if kind == 'invariant' else .012),
                     dx=.001, boundary=BoundarySpec(x='pmc', y='pmc', z='pmc') if kind == 'pmc' else 'pec',
                     mode='2d_tmz' if kind == 'invariant' else '3d')
    sim.add_thin_conductor(Box((.002, .002, 0.), (.010, .010, 0.)),
                           sigma_bulk=100., thickness=35e-6)
    expected = ('thin_conductor[0]',) if kind == 'invariant' else ('thin_conductor[0]', 'domain')
    assert_target(sim, 'dc_film_' + kind, [expected])


def test_traced_film_subjects():
    import jax
    import jax.numpy as jnp
    from rfx import Cylinder, Simulation

    def check(profile):
        sim = Simulation(10e9, (.048, .048, .024), dx=.001, boundary='pec', dz_profile=profile)
        sim.add_thin_conductor(Cylinder((.024, .024, .01227), .0025, 0),
                               sigma_bulk=100., thickness=35e-6)
        report = sim.preflight(check_ntff=False, check_resolution=False)
        REACHED.update(i.code for i in report)
        findings = report.by_code('dc_film_unchecked')
        assert [i.subjects for i in findings] == [('thin_conductor[0]',)]
        assert all(i.region is None for i in findings)
        assert report.by_code('campaign_statics_traced_mesh')
        return jnp.sum(profile)

    jax.jit(check)(jnp.full(24, .001))


@pytest.mark.parametrize('kind', ['refusal', 'fault'])
def test_unavailable_assembly(kind, monkeypatch):
    from rfx import Box, Simulation
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    sim.add(Box((.002,) * 3, (.004,) * 3), material='pec')

    def fail(*args, **kwargs):
        raise (ValueError if kind == 'refusal' else KeyError)('unavailable assembly')

    monkeypatch.setattr(sim, '_assemble_realized', fail)
    report = sim.preflight()
    REACHED.update(i.code for i in report)
    for code in ('assembly_' + kind, 'campaign_statics_unavailable'):
        assert report.by_code(code)
        assert all(i.scope == 'model' and i.subjects == () and i.region is None for i in report.by_code(code))


def test_uncoded_transport(monkeypatch):
    from rfx import Simulation
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')

    def legacy(*args, **kwargs):
        warnings.warn('legacy extension warning', UserWarning)

    monkeypatch.setattr(sim, '_validate_ntff_inverse_design', legacy)
    assert_target(sim, 'uncoded', [()])


@pytest.mark.parametrize('kind', ['cutoff', 'aperture'])
def test_waveguide_subjects(kind):
    if kind == 'cutoff':
        from tests.unit.ports.test_waveguide_port_spectrum_guard import _build
        sim = _build(.10, .002, np.array([9e9, 10e9, 11e9]), f0=5e9)
        code, expected = 'port_source_below_cutoff', [('waveguide_port[0]',), ('waveguide_port[1]',)]
    else:
        from tests.unit.ports.test_port_aperture_rasterization import _build
        sim = _build(dx=.001, domain=(.10, .02286, .01016),
                     y_range=(0., .02286), z_range=(0., .01016), freqs=np.array([9e9]), f0=9e9)
        code, expected = 'port_aperture_unrasterizable', [('waveguide_port[0]',)]
    assert_target(sim, code, expected)


def test_dielectric_seam_subjects():
    from tests.unit.preflight.test_preflight_absorber import _dp_sim, _seam_sphere
    sim = _dp_sim()
    sim.add_material('d', eps_r=4.)
    sim.add(_seam_sphere(), material='d')
    assert_target(sim, 'dielectric_at_absorber_seam', [('geometry[0]',)])


def test_declared_span_subjects(monkeypatch):
    from tests.unit.geometry.test_declared_span_reaches_padded_face import _rig, _size_by_plain_ceil
    _size_by_plain_ceil(monkeypatch)
    assert_target(_rig(10, cylinder=True), 'declared-span-short-of-padded-face', [('geometry[0]',)])


def test_imported_feature_subjects():
    from rfx import Box, Simulation

    class ImportedPlate(Box):
        def min_feature_size(self):
            return .0005

    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec', snap='declared')
    sim.add(ImportedPlate((.002, .002, .002), (.008, .008, .0025)), material='pec')
    assert_target(sim, 'mesh_import_underresolved', [('geometry[0]',)])


def test_trace_footprint_subjects():
    from rfx import Box
    from tests.contracts.test_one_node_tie_rule import model
    sim, _, _, _ = model('uniform')
    sim.add(Box((.003, .007, .003), (.0117, .011, .003)), material='pec')
    sim.add_port((.0117, .009, .002), 'ez')
    assert_target(sim, 'trace_port_footprint', [('geometry[0]', 'port[0]')])


def test_undriven_subjects():
    from rfx import GaussianPulse, Simulation
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    sim.add_source((.006, .005, .004), 'ez', waveform=GaussianPulse(f0=5e9, bandwidth=8e9),
                   amplitude_kind='current')
    assert_target(sim, 'undriven_pulse', [('port[0]',)])


def test_tfsf_model_scope():
    from tests.contracts.test_tfsf_boundary_admission import plane
    assert_target(plane(), 'tfsf_transverse_periodic', [()])


def test_precision_model_scope():
    from tests._x64_compat import enable_x64
    from rfx import Simulation
    with enable_x64(False):
        sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec', dz_profile=np.full(10, .001))
        sim._precision = 'float64'
        report = assert_target(sim, 'precision_float64_without_x64', [()])
        assert [i.subjects for i in report.by_code('precision_nonuniform_lane_unsupported')] == [()]


def test_refinement_model_scope():
    from tests.locks.test_preflight_split_snapshot import _subgrid_unsupported_feature_sim
    sim = _subgrid_unsupported_feature_sim()
    sim._dz_profile = np.full(round(sim._domain[2] / sim._dx), sim._dx)
    assert_target(sim, 'nonuniform_refinement', [()])


def test_flux_unavailable_subjects(monkeypatch):
    from tests.unit.preflight.test_flux_region_preflight import _sim, _add, U
    sim = _sim(graded=True)
    _add(sim, size=(4*U, 8*U))
    # The record remains readable while this individual audit is unavailable.
    record = sim.realized_geometry()

    def fail(*args, **kwargs):
        raise NotImplementedError('flux window audit unavailable')

    monkeypatch.setattr(sim, '_build_realized_grid', fail)
    monkeypatch.setattr(sim, 'realized_geometry', lambda: record)
    assert_target(sim, 'flux_region_unavailable', [('window',)], check_ntff=False)


def test_probe_index_hole():
    from tests.unit.preflight.test_preflight_absorber import _absorber_placement_sim
    sim = _absorber_placement_sim(-.001)
    sim.add_probe((-.002, .006, .007), 'ez')
    sim._internal_probe_indices = {0}
    report = sim.preflight()
    # The hidden witness is omitted from both the report check and record.
    findings = [i for i in report.by_code('absorber_overlap') if i.subjects == ('probe[1]',)]
    assert len(findings) == 1
    assert 'probe[0]' not in {row.label for row in sim.realized_geometry().realization_table.rows}


def test_all_lossless_declarations_survive_summary():
    from rfx import Box, Simulation
    sim = Simulation(1e9, (.03, .02, .01), dx=.001, boundary='cpml', cpml_layers=4)
    sim.add_material('same_material', eps_r=2.)
    for i in range(8):
        x = .002 + i*.003
        sim.add(Box((x, .004, .003), (x+.001, .007, .005)), material='same_material')
    assert_target(sim, 'lossless_q', [('geometry[0]', 'geometry[1]', 'geometry[2]', 'geometry[3]',
                                    'geometry[4]', 'geometry[5]', 'geometry[6]', 'geometry[7]')])


def test_scene_code_coverage(request):
    if len([item for item in request.session.items if item.path == Path(__file__)]) < 90:
        pytest.skip('coverage census applies to the complete scene battery')
    assert REACHED | UNREACHABLE.keys() == CODE_SCOPE.keys(), sorted(CODE_SCOPE.keys() - REACHED)
