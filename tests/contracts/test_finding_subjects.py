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
    """Census constructors, retained outcomes, defaults and the flux adapter."""
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
            if name in constructors or name == 'legacy_outcome':
                for kw in node.keywords:
                    if kw.arg == 'code':
                        values |= codes(kw.value)
                if name == 'DCFilmAdmissionError' and not any(k.arg == 'code' for k in node.keywords):
                    values.add('dc_film_area')
            if name == 'legacy_outcome' and len(node.args) >= 5:
                values |= codes(node.args[4])
            if name == 'finding' and path.name in ('_preflight.py', '_flux_subjects.py') and len(node.args) >= 3:
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
    # Retained outcomes add declared_z; existing codes keep their class.
    assert len(CODE_SCOPE) == 101
    assert {scope: list(CODE_SCOPE.values()).count(scope) for scope in set(CODE_SCOPE.values())} == {
        'object': 73, 'place': 2, 'model': 24, 'mixed': 2}


def check_report(sim, report, *, unavailable_labels=None):
    REACHED.update(issue.code for issue in report)
    record = sim.realized_geometry() if unavailable_labels is None else None
    labels = ([row.label for row in record.realization_table.rows]
              if record is not None else unavailable_labels)
    for issue in report:
        assert isinstance(issue._subjects, tuple)
        assert issue._region is None or isinstance(issue._region, dict)
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
        if issue.region is not None and record is not None:
            lo, hi = issue.region['bounds_m']
            for a, values in enumerate(record.nodes):
                nodes = np.asarray(values)
                assert nodes[0] <= lo[a] <= hi[a] <= nodes[-1]
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
    # S2 snapshot predeclarations: test_preflight_split_snapshot.py:110-113.
    'guards_box_alumina': {'lossless_q': [('geometry[0]',)], 'realization': [('geometry[0]',)]},
    'guards_lossless_q': {'lossless_q': [('geometry[0]',)], 'realization': [('geometry[0]',)]},
    'guards_no_sources': {},
    'guards_under_resolved_dielectric': {'mesh_resolution': [('geometry[0]',)] * 3,
                                        'realization': [('geometry[0]',)]},
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
        # S2 snapshot predeclaration: test_preflight_split_snapshot.py:103.
        'realization': [('thin_conductor[0]',), ('thin_conductor[1]',)],
        'sheet_plane_realized': [('thin_conductor[0]', 'thin_conductor[1]')],
        'congruent_conductor_rasterization_parity': [('thin_conductor[0]', 'thin_conductor[1]')]},
    'conformal_fine_dx': {'port_aperture_snap': [('waveguide_port[0]',), ('waveguide_port[0]',)],
                         'port_freqs_below_cutoff': [('waveguide_port[0]',)]},
    # S2 snapshot predeclaration: test_preflight_split_snapshot.py:119.
    'waveguide_nu_declared_geometry': {'port_evanescent': [('waveguide_port[0]',), ('waveguide_port[1]',)],
                                      'port_aperture_snap': [('waveguide_port[0]',)]},
    'ntff_absorber_overlap': {'absorber_overlap': [('ntff',)]},
    'floating_port': {'floating_port': [('geometry[0]', 'port[0]')],
                      'lossless_q': [('geometry[0]',)]},
    'off_lattice_edges': {'mesh_resolution': [('geometry[0]',)] * 2,
                          # S2 snapshot predeclaration: test_preflight_split_snapshot.py:106.
                          'realization': [('geometry[0]',)],
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
        # S2 snapshot predeclaration: test_preflight_split_snapshot.py:115.
        'realization': [('geometry[0]',)],
        'ntff_pec_overlap': [('geometry[0]', 'ntff')],
        'port_in_pec': [('geometry[0]', 'port[0]'), ('geometry[0]', 'probe[0]')],
        'off_lattice_design_edges': [('geometry[0]',)],
        'port_conductor_continues': [('geometry[0]', 'port[0]')]},
    'pec_boundary_open': {'ntff_near_field': [('geometry[0]', 'ntff')]},
    'guards_msl_sheet_probe': {
        # S2 snapshot predeclaration: test_preflight_split_snapshot.py:112.
        'realization': [('geometry[0]',), ('geometry[1]',)],
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


def test_thin_absorber_region_one_face():
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    d = 1 / 1024
    sim = Simulation(10e9, (8*d, 6*d, 4*d), dx=d, cpml_layers=4,
                     boundary=BoundarySpec(x=Boundary('pec', 'cpml'), y='pec', z='pec'))
    record = sim.realized_geometry()
    report = sim.preflight(check_ntff=False)
    check_report(sim, report)
    finding, = report.by_code('thin_absorber')
    # Eight interior x cells followed by four high-face pad cells; no y/z pads.
    assert finding.region['cells'] == ((8, 12), (0, 6), (0, 4))
    x, y, z = record.nodes
    assert finding.region['bounds_m'] == ((x[8], y[0], z[0]), (x[12], y[6], z[4]))


def test_grading_runway_region():
    from rfx import Simulation
    d = 1 / 1024
    # Four high-face interior cells total 5*d; the first eight are uniform.
    profile = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]) * d
    sim = Simulation(10e9, (8*d, 8*d, 13*d), dx=d, cpml_layers=4,
                     boundary='cpml', dz_profile=profile)
    record = sim.realized_geometry()
    report = sim.preflight(check_ntff=False)
    check_report(sim, report)
    finding, = report.by_code('nu_grading_reaches_absorber')
    # Each axis has four low-pad cells. The high runway starts at interior z=8.
    assert finding.region['cells'] == ((4, 12), (4, 12), (12, 16))
    x, y, z = record.nodes
    assert finding.region['bounds_m'] == ((x[4], y[4], z[12]), (x[12], y[12], z[16]))


def test_pec_gap_pair_subjects_and_region():
    from rfx import Box, Simulation
    sim = Simulation(10e9, (.02, .02, .02), dx=.001, boundary='pec')
    sim.add(Box((.004, .006, .006), (.008, .012, .012)), material='pec')
    sim.add(Box((.0095, .006, .006), (.013, .012, .012)), material='pec')
    record = sim.realized_geometry()
    report = sim.preflight(check_ntff=False)
    check_report(sim, report)
    # The 1.5-cell x gap is below the three-cell resolution bar.
    findings = report.by_code('mesh_resolution')
    assert [finding.subjects for finding in findings] == [
        ('geometry[0]',), ('geometry[1]',), ('geometry[0]', 'geometry[1]')]
    finding = findings[2]
    assert finding.region['cells'] == ((4, 13), (6, 12), (6, 12))
    x, y, z = record.nodes
    assert finding.region['bounds_m'] == ((x[4], y[6], z[6]), (x[13], y[12], z[12]))


@pytest.mark.parametrize('scene', ['wr90-aperture', 'msl-no-bounds'])
def test_unrasterizable_aperture_public_calls_raise(scene, monkeypatch):
    if scene == 'wr90-aperture':
        from tests.unit.ports.test_port_aperture_rasterization import _build
        sim = _build(dx=.001, domain=(.10, .02286, .01016),
                     y_range=(0., .02286), z_range=(0., .01016), freqs=np.array([9e9]), f0=9e9)
        error, match = ValueError, 'resolves to an invalid physical aperture span'
        code, expected = 'port_aperture_unrasterizable', [('waveguide_port[0]',)]
    else:
        from dataclasses import replace
        from tests.unit.preflight.test_line_stub import make_sim
        from tests.unit.ports.test_msl_reflector_scan_conductors import _NoBBox
        sim = make_sim()
        sim._geometry.append(replace(sim._geometry[-1], shape=_NoBBox()))
        error, match = NotImplementedError, None
        code, expected = 'line_stub_inspection_unavailable', [('geometry[3]',)]
    # Exception types and the aperture fragment are origin/main's public contract.
    for read in (sim.realized_geometry, lambda: sim.fidelity_report(print_report=False)):
        with pytest.raises(error, match=match):
            read()
    from rfx.model import realization_unavailable
    builds = []
    original = realization_unavailable.identity_record

    def counted(*args):
        builds.append(None)
        return original(*args)

    monkeypatch.setattr(realization_unavailable, 'identity_record', counted)
    report = sim.preflight()
    assert len(builds) == 1
    for _ in range(2):
        # A refused public record still names the finding's objects, built once.
        assert [i.subjects for i in report.by_code(code)] == expected
        for issue in report:
            issue.to_dict()
    assert len(builds) == 1
    # Resolving finding identities must not populate any public geometry reader.
    for read in (sim.realized_geometry, lambda: sim.fidelity_report(print_report=False),
                 lambda: report.realization_table):
        with pytest.raises(error, match=match):
            read()


def test_off_lattice_place_region():
    from rfx import Simulation
    from rfx.preflight._common import PreflightIssue
    from rfx.preflight._subjects import finding_subjects, resolve_findings
    sim = Simulation(10e9, (.02, .02, .02), dx=.001, boundary='pec')
    # Supply a physical region: geometry-row bounds have already been snapped
    # and cannot exercise a lower face lying strictly inside a grid cell.
    finding = PreflightIssue('Off-lattice region', **finding_subjects(
        sim, region=((.0042, .0062, .0062), (.0132, .0122, .0122))))
    resolve_findings([finding])
    assert finding.scope == 'place' and finding.subjects == ()
    assert finding.region['cells'] == ((4, 14), (6, 13), (6, 13))
    np.testing.assert_allclose(finding.region['bounds_m'],
                               ((.004, .006, .006), (.014, .013, .013)), rtol=0, atol=1e-15)


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
    from rfx.preflight._subjects import _finding_view
    reports = []

    def evaluate(delta):
        profile = jnp.full(10, .001).at[3].add(delta).at[4].add(-delta)
        sim = Simulation(freq_max=1e9, domain=(.01004, .01, .01), dx=.001,
                         boundary='pec', snap='declared', dz_profile=profile,
                         dt=.2*.001/299792458., dt_min_cell=.0005)
        sim.add_material('glass', eps_r=2.)
        sim.add(Box((.00223, .00237, .00219), (.00661, .00673, .00657)), material='glass')
        sim.add_source((.008, .007, .006), 'ez', waveform=GaussianPulse(f0=250e9, bandwidth=.8))
        sim.add_probe((.007, .006, .006), 'ez')
        original = sim._run_preflight_gate

        def gate(report, **kwargs):
            reports.append(report)
            return original(report, **kwargs)

        monkeypatch.setattr(sim, '_run_preflight_gate', gate)
        result = sim.forward(n_steps=1, skip_preflight=False)
        record = result.realized_geometry
        assert [row.label for row in record.realization_table.rows] == [
            'geometry[0]', 'domain', 'probe[0]']
        labels = [row.label for row in _finding_view(sim, record).realization_table.rows]
        assert 'port[0]' in labels
        check_report(sim, reports[-1], unavailable_labels=labels)
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
    if kind == 'stub':
        assert_target(sim, code, expected)
    else:
        error = NotImplementedError if kind == 'uninspectable' else ValueError
        with pytest.raises(error):
            sim.realized_geometry()
        report = sim.preflight()
        labels = (('geometry[0]', 'geometry[1]', 'geometry[2]', 'geometry[3]', 'domain', 'msl_port[0]')
                  if kind == 'uninspectable' else ('geometry[0]', 'geometry[1]', 'domain', 'msl_port[0]'))
        check_report(sim, report, unavailable_labels=labels)
        assert [i.subjects for i in report.by_code(code)] == expected


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
    if kind == 'aperture':
        report = sim.preflight()
        REACHED.update(i.code for i in report)
        assert all(i.scope == 'object' for i in report.by_code(code))
        assert [i.subjects for i in report.by_code(code)] == expected
    else:
        assert_target(sim, code, expected)


def test_dielectric_seam_subjects():
    from tests.unit.preflight.test_preflight_absorber import _dp_sim, _seam_sphere
    sim = _dp_sim()
    sim.add_material('d', eps_r=4.)
    sim.add(_seam_sphere(), material='d')
    assert_target(sim, 'dielectric_at_absorber_seam', [('geometry[0]',)])


def test_thin_conductor_seam_subjects():
    from rfx import Box, Cylinder, Simulation
    sim = Simulation(10e9, (.03, .02, .02), dx=.001, boundary='cpml', cpml_layers=8)
    sim.add_material('d', eps_r=2.)
    sim.add(Box((.010, .006, .006), (.014, .010, .010)), material='d')
    sim.add_thin_conductor(Cylinder((.004, .010, .010), .004, 0., axis='z'),
                           sigma_bulk=1e8, thickness=1e-6)
    report = sim.preflight()
    check_report(sim, report)
    assert [(i.code, i.severity, i.subjects) for i in report.by_code('dielectric_at_absorber_seam')] == [
        ('dielectric_at_absorber_seam', 'warning', ('thin_conductor[0]',))]
    # The existing geometry-continuation warning remains uncoded on main.
    assert all(i.severity == 'warning' for i in report.by_code('uncoded'))


@pytest.mark.parametrize('owner', [('_unknown_collection', 0), 'missing[0]'])
def test_unresolved_subject_is_only_a_contract_failure(monkeypatch, owner):
    from rfx import Simulation
    from rfx.preflight._common import PreflightWarning
    from rfx.preflight._subjects import finding_subjects
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')

    def emit(*args, **kwargs):
        warnings.warn(PreflightWarning('original warning', code='dielectric_at_absorber_seam',
                                      **finding_subjects(sim, objects=(owner,))))

    monkeypatch.setattr(sim, '_validate_ntff_inverse_design', emit)
    report = sim.preflight()
    hit, = report.by_code('dielectric_at_absorber_seam')
    assert (str(hit), hit.severity) == ('original warning', 'warning')
    assert hit.subjects == ('<unresolved finding subject>',)
    assert not report.by_code('uncoded')
    report.to_dict()
    with pytest.raises(AssertionError):
        check_report(sim, report)


def test_missing_subjects_follow_resolved_labels_once():
    from rfx import Box, Simulation
    from rfx.preflight._common import PreflightIssue
    from rfx.preflight._subjects import finding_subjects, resolve_findings
    sim = Simulation(10e9, (.01,) * 3, dx=.001, boundary='pec')
    sim.add(Box((.002, .002, .002), (.006, .006, .006)), material='pec')
    finding = PreflightIssue('original warning', code='dielectric_at_absorber_seam',
                             **finding_subjects(sim, objects=(
                                 'missing[0]', 'domain', 'geometry[0]', 'missing[1]', 'missing[0]')))
    resolve_findings([finding])
    assert finding.subjects == ('geometry[0]', 'domain', '<unresolved finding subject>')
    assert (finding.code, finding.severity, str(finding)) == (
        'dielectric_at_absorber_seam', 'warning', 'original warning')


def test_msl_sparameters_report_survives_mesh_change(monkeypatch):
    from tests.unit.ports.test_msl_clearance_diagnostic import _base, _candidate
    sim, _, _ = _base()
    _candidate(sim, '+x', 18)
    report = sim.preflight_sparameters(calculator='msl')
    finding, = [i for i in report if i.source == 'preflight_sparameters']
    assert finding.code == 'msl_port_geometry'
    assert finding.subjects == ('geometry[3]', 'msl_port[0]')
    before = (finding.subjects, finding.region)
    assert before[1] is None
    encoded = report.to_dict()
    sim._dx /= 2
    sim._resolved_mesh = None

    def no_assembly():
        pytest.fail('old MSL finding tried to assemble the changed model')

    monkeypatch.setattr(sim, 'realized_geometry', no_assembly)
    assert (finding.subjects, finding.region) == before
    assert report.to_dict() == encoded
    assert isinstance(finding._subjects, tuple)
    assert finding._region is None


def test_old_report_survives_mesh_change(monkeypatch):
    from rfx import Box, Simulation
    sim = Simulation(10e9, (.02,) * 3, boundary='cpml', cpml_layers=4)
    sim.add_material('d', eps_r=2.)
    sim.add(Box((.004, .006, .006), (.007, .012, .012)), material='d')
    report = sim.preflight(check_ntff=False)
    finding, = report.by_code('lossless_q')
    assert finding.subjects == ('geometry[0]',)
    before = (finding.subjects, finding.region)
    assert before[1] is not None
    encoded = report.to_dict()
    old_dx = sim._dx
    sim.add(Box((.014, .006, .006), (.016, .012, .012)), material='d')
    sim.realized_geometry()
    assert sim._dx != old_dx
    assert (finding.subjects, finding.region) == before
    assert report.to_dict() == encoded

    def no_assembly():
        pytest.fail('old finding tried to assemble the changed model')

    monkeypatch.setattr(sim, 'realized_geometry', no_assembly)
    assert (finding.subjects, finding.region) == before
    assert report.to_dict() == encoded


def test_ntff_nearest_owner_is_port():
    from rfx import Box, Simulation
    sim = Simulation(10e9, (.06,) * 3, dx=.002, boundary='cpml', cpml_layers=8)
    sim.add(Box((.025,) * 3, (.035,) * 3), material='pec')
    sim.add_port((.012, .03, .03), 'ez')
    sim.add_ntff_box((.01,) * 3, (.05,) * 3)
    report = sim.preflight()
    check_report(sim, report)
    assert [(i.severity, i.subjects) for i in report.by_code('ntff_near_field')] == [
        ('warning', ('port[0]', 'ntff'))]


def test_thickness_nearest_dielectric_is_second():
    from rfx import Box, Simulation
    sim = Simulation(30e9, (.004,) * 3, dx=.0002, boundary='pec', snap='declared')
    sim.add_material('substrate', eps_r=3.4)
    sim.add(Box((.0004, .0004, .003), (.0036, .0036, .0032)), material='substrate')
    sim.add(Box((.0004, .0004, .0015), (.0036, .0036, .002)), material='substrate')
    sim.add_thin_conductor(Box((.001, .001, .002), (.003, .003, .002)),
                           sigma_bulk=5.8e7, thickness=50e-6)
    report = assert_target(sim, 'thin_conductor_thickness_unmodelled', [
        ('geometry[1]', 'thin_conductor[0]')])
    hit, = report.by_code('thin_conductor_thickness_unmodelled')
    assert hit.severity == 'warning'
    assert 'h = 500µm' in str(hit) and 't/h = 10 %' in str(hit)


def test_half_node_skips_internal_probe():
    from rfx import Box
    from tests.unit.preflight.test_half_node_split import _sim
    sim = _sim('uniform', dom=(.02, .019, .013))
    sim.add(Box((.005, .005, .0035), (.015, .014, .0035)), material='pec')
    for x in (.012, .013):
        sim.add_probe((x, .009, .0035 + .95e-9), 'ex')
    sim._internal_probe_indices = {0}
    report = sim.preflight(check_ntff=False)
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('half_node_split')] == [('geometry[0]', 'probe[1]')]


def test_half_node_user_probe_at_internal_position():
    from rfx import Box
    from tests.unit.preflight.test_half_node_split import _sim
    sim = _sim('uniform', dom=(.02, .019, .013))
    sim.add(Box((.005, .005, .0035), (.015, .014, .0035)), material='pec')
    for _ in range(2):
        sim.add_probe((.012, .009, .0035 + .95e-9), 'ex')
    sim._internal_probe_indices = {0}
    report = sim.preflight(check_ntff=False)
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('half_node_split')] == [('geometry[0]', 'probe[1]')]


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


def _s2_subject_scene(snap='declared', *, sheet=False, lumped=False):
    from rfx import Box, Simulation
    sim = Simulation(10e9, (.012, .011, .010), dx=.001, boundary='pec', snap=snap)
    sim.add(Box((.00223, .00337, .00419),
                (.00761, .00873, .00419 if sheet else .00757)), material='pec')
    if lumped:
        sim.add_lumped_rlc((.009, .005, .002), 'ez', R=50.)
    return sim


@pytest.mark.parametrize('snap,severity', [('strict', 'error'), ('declared', 'warning')])
def test_s2_table_route_subjects(snap, severity):
    sim = _s2_subject_scene(snap)
    report = sim.preflight(check_ntff=False, check_resolution=False)
    check_report(sim, report)
    findings = report.by_code('realization')
    assert findings
    assert [(i.scope, i.subjects, i.severity) for i in findings] == [
        ('object', ('geometry[0]',), severity)]
    assert findings[0].source == 'judge_realization'
    assert findings[0].region is not None


@pytest.mark.parametrize('snap,severity', [('strict', 'error'), ('declared', 'warning')])
def test_s2_sheet_replay_subjects(snap, severity):
    from rfx.model.realization_verdict import replay_legacy_findings
    sim = _s2_subject_scene(snap, sheet=True)
    report = sim.preflight(check_ntff=False, check_resolution=False)
    check_report(sim, report)
    original, = report.by_code('sheet_effective_size')
    assert (original.scope, original.subjects, original.severity) == (
        'object', ('geometry[0]',), severity)
    assert original.region is not None
    # Inspect replay itself: report substitution can otherwise hide a broken carrier.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        replay_legacy_findings(warnings, sim._campaign_ctx(), 'sheet_effective_size')
    replayed, = [w.message for w in caught]
    assert replayed.to_dict() == original.to_dict()


def test_s2_delivery_preserves_subjects():
    from rfx.model.realization_verdict import deliver_findings
    sim = _s2_subject_scene()
    report = sim.preflight(check_ntff=False, check_resolution=False)
    original, = report.by_code('realization')
    table = sim._campaign_ctx()._realized.realization_table
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        deliver_findings(table, (original,))
    delivered, = [w.message for w in caught]
    assert delivered.scope == 'object'
    assert delivered.subjects == ('geometry[0]',)
    assert delivered.to_dict() == original.to_dict()


def test_s2_aperture_table_subjects():
    from rfx import Simulation
    sim = Simulation(100e9, (.012, .011, .010), dx=.001,
                     dz_profile=np.full(10, .001),
                     boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'},
                     cpml_layers=4, snap='declared')
    sim.add_waveguide_port(.00523, direction='+x', y_range=(.00217, .00873),
                          z_range=(.00129, .00761), freqs=[80e9], f0=80e9,
                          ref_offset=1, probe_offset=2)
    report = sim.preflight(check_ntff=False, check_resolution=False)
    check_report(sim, report)
    findings = report.by_code('port_aperture_snap')
    assert [(i.source, i.scope, i.subjects) for i in findings] == [
        ('judge_realization', 'object', ('waveguide_port[0]',))]
    assert findings[0].region is not None


def test_s2_coax_gate_retains_sheet_subjects(monkeypatch):
    from rfx.preflight import pec_geometry
    from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs

    sim = build_instrument_junction()
    original = pec_geometry._warn_sheet_effective_size
    retained = []

    def capture(warn, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            original(warnings, *args, **kwargs)
        for item in caught:
            retained.append(item.message)
            warn.warn(item.message)

    class ReachedGate(Exception):
        pass

    def gate(issues, **kwargs):
        finding, = [i for i in issues if i.code == 'sheet_effective_size']
        warning, = [w for w in retained if w.code == 'sheet_effective_size']
        assert finding.scope == warning.scope == 'object'
        assert finding.subjects == warning.subjects
        assert finding.subjects
        assert finding._subjects is warning._subjects
        assert finding._region is warning._region
        raise ReachedGate

    monkeypatch.setattr(pec_geometry, '_warn_sheet_effective_size', capture)
    monkeypatch.setattr(sim, '_run_preflight_gate', gate)
    kwargs = instrument_kwargs(n_steps=1)
    kwargs['skip_preflight'] = False
    with pytest.raises(ReachedGate):
        sim.compute_coax_msl_transition(**kwargs)


def test_s2_lumped_label_rows_do_not_change_findings(monkeypatch):
    from rfx.model import realization_placements

    def findings():
        sim = _s2_subject_scene(lumped=True)
        sim.preflight(check_ntff=False, check_resolution=False)
        table = sim._campaign_ctx()._realized.realization_table
        return table, [(f.code, f.severity, str(f)) for f in table.findings]

    with_rows, before = findings()
    assert any(code == 'realization' for code, _, _ in before)
    row = with_rows.row('lumped_rlc[0]')
    assert row.status == 'not represented'
    assert row.axes == row.regions == row.traced_inputs == ()
    assert row.family == 'lumped_rlc'
    monkeypatch.setattr(realization_placements, 'label_rows', lambda sim: [])
    without_rows, after = findings()
    assert all(r.label != 'lumped_rlc[0]' for r in without_rows.rows)
    assert after == before


@pytest.mark.parametrize('key', ['<unresolved finding subject>', 'absent[0]'])
def test_bind_retained_unknown_key_keeps_finding(key):
    from rfx.preflight._common import PreflightIssue
    from rfx.preflight._subjects import _Selection, _Region
    from rfx.preflight._realization_subjects import bind_retained_subjects
    sim = _s2_subject_scene()
    sim.preflight(check_ntff=False, check_resolution=False)
    table = sim._campaign_ctx()._realized.realization_table
    selection = _Selection(sim, (key,))
    finding = PreflightIssue('retained sheet finding', code='sheet_effective_size',
        severity='warning', scope='object', subjects=selection, region=_Region(selection, None))
    bind_retained_subjects(finding, table)
    assert (finding.code, finding.severity, str(finding)) == (
        'sheet_effective_size', 'warning', 'retained sheet finding')
    assert finding.scope == 'object'
    assert finding.subjects == ('<unresolved finding subject>',)
    assert finding.region is None


def test_declared_z_model_scope():
    from rfx import Box, Simulation
    sim = Simulation(10e9, (.012, .011, .0105), dx=.001,
                     dz_profile=np.full(10, .001), boundary='pec', snap='declared')
    sim.add(Box((.002, .002, .002), (.008, .008, .008)), material='pec')
    report = sim.preflight(check_ntff=False, check_resolution=False)
    check_report(sim, report)
    finding, = report.by_code('declared_z')
    assert (finding.scope, finding.subjects, finding.region) == ('model', (), None)
    assert finding.source == 'judge_declared_z'


def _coax_stamp_findings(monkeypatch):
    import rfx.sparams.coax as calculators
    from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
    sim = build_instrument_junction()
    original = calculators.coax_kernel_conductors
    captured = []

    class StampBuilt(Exception):
        pass

    def handover(*args, **kwargs):
        root = original(*args, **kwargs)
        captured.append(root.stamp_check_findings)
        raise StampBuilt

    monkeypatch.setattr(calculators, 'coax_kernel_conductors', handover)
    kwargs = instrument_kwargs(n_steps=1)
    kwargs['skip_preflight'] = False
    with pytest.raises(StampBuilt):
        sim.compute_coax_msl_transition(**kwargs)
    return dict(captured[0])


def test_coax_stamp_retains_sheet_subjects(monkeypatch):
    findings = _coax_stamp_findings(monkeypatch)
    assert 'read-failed' not in findings
    finding, = findings['sheet-size']
    assert finding.code == 'sheet_effective_size'
    assert finding.scope == 'object'
    # The ground tiles adjoining the clearance hole and the narrow trace.
    assert finding.subjects == ('geometry[1]', 'geometry[3]', 'geometry[4]', 'geometry[5]')


def test_coax_stamp_ignores_foreign_warning(monkeypatch):
    from rfx.preflight import pec_geometry
    original = pec_geometry._warn_sheet_effective_size

    def with_foreign_warning(*args, **kwargs):
        warnings.warn('x', UserWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(pec_geometry, '_warn_sheet_effective_size', with_foreign_warning)
    findings = _coax_stamp_findings(monkeypatch)
    assert 'read-failed' not in findings
    finding, = findings['sheet-size']
    assert (finding.code, finding.severity) == ('sheet_effective_size', 'warning')
    assert '4 conductor sheet dimension(s)' in str(finding)


def test_f0_sheet_backed_gap_subjects():
    from tests.contracts.test_f0_sheet_backed_gap import _cavity
    sim = _cavity(1)
    report = sim.preflight()
    check_report(sim, report)
    assert [i.subjects for i in report.by_code('f0_sheet_backed_gap')] == [
        ('thin_conductor[0]',), ('thin_conductor[1]',)]


def test_scene_code_coverage(request):
    if len([item for item in request.session.items if item.path == Path(__file__)]) < 90:
        pytest.skip('coverage census applies to the complete scene battery')
    assert REACHED | UNREACHABLE.keys() == CODE_SCOPE.keys(), sorted(CODE_SCOPE.keys() - REACHED)
