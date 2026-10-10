"""One-plane film contracts; expected conductivities follow G and the grid by hand."""
from pathlib import Path
import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx import _realized
from rfx.core.yee import MaterialArrays, component_e_materials, init_materials
from rfx.model.materials import realize_components


H = .001
G = .01


def model(graded=False, plane=9, film=True, boundary='pec', **film_options):
    widths = np.full(16, H)
    if graded:
        widths[plane - 1:plane + 1] = ([.00075, .0015] if graded == 'up'
                                            else [.0015, .00075])
    sim = Simulation(freq_max=10e9, domain=(float(widths.sum()), .008, .008),
                     dx=H, boundary=boundary,
                     **({'dx_profile': widths} if graded else {}))
    x = float(widths[:plane].sum())
    if film:
        sim.add_thin_conductor(Box((x, 0, 0), (x, .008, .008)),
            sigma_bulk=1000., thickness=1e-5, **film_options)
    sim.add_source((x, .004, .004), component='ez',
                   waveform=lambda t: jnp.ones_like(t), amplitude_kind='current')
    sim.add_probe((float(widths[:3].sum()), .004, .004), component='ez')
    return sim, widths


def assert_expected(sigma, eps, widths, plane, *, start=0, count=None):
    """Full-width footprint gives weight one away from transverse boundaries."""
    dual = (widths[plane - 1] + widths[plane]) / 2
    for c in range(3):
        actual = np.asarray(sigma[c])
        epsilon = np.asarray(eps[c])
        if count is not None:
            actual = actual[1:1 + count]
            epsilon = epsilon[1:1 + count]
        expected = np.zeros(actual[:, 2:-2, 2:-2].shape, dtype=np.float32)
        if c != 0 and start <= plane < start + len(actual):
            expected[plane - start] = G / dual
        np.testing.assert_array_max_ulp(actual[:, 2:-2, 2:-2], expected, maxulp=1)
        np.testing.assert_array_equal(epsilon[:, 2:-2, 2:-2], 1.)


@pytest.mark.parametrize('graded', [False, 'up', 'down'])
@pytest.mark.parametrize('plane', [8, 9, 4])
def test_c1_arrays(graded, plane):
    sim, widths = model(graded, plane)
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    np.testing.assert_array_equal(mats.sigma, 0.)
    np.testing.assert_array_equal(mats.eps_r, 1.)
    assert mats.sigma_film[0] is None
    c = realize_components(mats, grid, periodic=(False,) * 3)
    assert_expected(c.sigma_update, c.eps_update, widths, plane)
    eps, sigma = component_e_materials(mats, cell_sizes=(None, None, None)
                                       if not graded else (grid.dx_arr, None, None))
    assert_expected(sigma, eps, widths, plane)


@pytest.mark.parametrize('lane', ['uniform_run', 'graded_run', 'graded_forward'])
@pytest.mark.parametrize('plane', [8, 9, 4])
def test_c1_two_devices_consumed(lane, plane):
    assert jax.local_device_count() >= 2, 'run with two CPU devices'
    sim, widths = model('up' if lane.startswith('graded') else False, plane)
    with _realized.capture() as capture:
        if lane.endswith('forward'):
            sim.forward(n_steps=3, distributed=True)
        else:
            sim.run(n_steps=3, devices=jax.devices()[:2], compute_s_params=False)
    records = [r for r in capture.records if 'sigma_e' in r and 'owned_start' in r]
    assert records, [r['site'] for r in capture.records]
    reached = False
    for r in records:
        start, count = int(r['owned_start']), int(r['owned_count'])
        assert_expected(r['sigma_e'], r['eps_e'], widths, plane, start=start, count=count)
        reached |= start <= plane < start + count
    assert reached


@pytest.mark.parametrize('zeros', [False, True])
def test_c3_forward_override(zeros):
    sim, widths = model()
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    with _realized.capture() as capture:
        sim.forward(n_steps=3, sigma_override=jnp.zeros(grid.shape) if zeros else mats.sigma)
    records = [r for r in capture.records if 'sigma_e' in r]
    assert records
    for r in records:
        assert_expected(r['sigma_e'], r['eps_e'], widths, 9)


@pytest.mark.parametrize('graded', [False, 'up'])
def test_c4_permittivity(graded):
    sim, _ = model(graded, eps_r=2.)
    with pytest.raises(ValueError, match='a film on one node plane has no volume; its eps_r is not modelled'):
        sim._assemble_materials(sim._build_realized_grid())


def test_c5_upml_operands():
    sim, widths = model()
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    c = realize_components(mats, grid, periodic=(False,) * 3)
    assert_expected(c.upml_sigma, c.upml_eps, widths, 9)
    for a, b in zip(c.upml_sigma, c.sigma_update):
        np.testing.assert_array_equal(a, b)


def test_c2_film_shape_and_identity():
    from rfx.model.materials import with_components, validate_components
    mats = init_materials((3, 4, 5))
    with pytest.raises(ValueError, match="sigma_film arrays must have the cells' shape"):
        realize_components(mats._replace(sigma_film=(jnp.zeros((2, 4, 5)), None, None)),
                           None, periodic=(False,) * 3)
    ready = with_components(mats, None, periodic=(False,) * 3)
    with pytest.raises(ValueError, match='sigma_film'):
        validate_components(ready._replace(sigma_film=(jnp.zeros((3, 4, 5)), None, None)),
                            periodic=(False,) * 3)


def test_c7_mapping_refuses_unclassified_carrier():
    from rfx.stepping.slab import Slab, cut
    mats = init_materials((9, 4, 4))
    mapping = dict(eps_r='eps_r', sigma='sigma', mu_r='mu_r')
    cut(mats._replace(sigma_film=(None,) * 3), Slab(9, 2), mapping)
    with pytest.raises(ValueError, match='field sigma_film of the record is not cut: name its kind'):
        cut(mats._replace(sigma_film=(None, jnp.ones((9, 4, 4)), None)), Slab(9, 2), mapping)


@pytest.mark.parametrize('graded', [False, 'up'])
@pytest.mark.parametrize('kind', ['pmc', 'invariant', 'design'])
def test_c4_geometry_refusals(graded, kind):
    from rfx.boundaries.spec import BoundarySpec
    if kind == 'pmc':
        boundary = BoundarySpec(x='pmc', y='pec', z='pec')
        sim, _ = model(graded, boundary=boundary)
        from dataclasses import replace
        tc = sim._thin_conductors[0]
        sim._thin_conductors[0] = replace(tc, shape=Box((0, 0, 0), (0, .008, .008)))
        with pytest.raises(ValueError, match='a film on a PMC domain face'):
            sim._assemble_materials(sim._build_realized_grid())
    elif kind == 'invariant':
        sim, _ = model(graded, film=False)
        sim._mode = '2d_tmz'
        sim.add_thin_conductor(Box((.002, .002, 0), (.006, .006, 0)),
                               sigma_bulk=1000., thickness=1e-5)
        with pytest.raises(ValueError, match="invariant axis"):
            sim._assemble_materials(sim._build_realized_grid())
    else:
        sim, widths = model(graded)
        # The write window includes the upper context row, hence this box
        # reaches the film even though its own last cell is below it.
        x = float(widths[:9].sum())
        box = ((x - widths[8], .001, .001), (x - widths[8] / 4, .002, .002))
        with pytest.raises(ValueError, match='contains edges of thin conductor 0'):
            sim.forward(n_steps=2, design_box=box,
                        design_eps_override=jnp.ones((2, 2, 2)))


@pytest.mark.parametrize('kind', ['closed', 'plane', 'oblique'])
def test_c4_tfsf_refusals(kind):
    sim = Simulation(freq_max=10e9, domain=(.018, .016, .014), dx=H,
                     boundary='cpml', cpml_layers=4)
    sim.add_tfsf_source(f0=10e9, bandwidth=.5, margin=3, closed_box=kind == 'closed',
        angle_deg=30. if kind == 'oblique' else 0.,
        method='methodB' if kind == 'oblique' else 'bloch')
    sim.add_thin_conductor(Box((.003, -1, -1), (.003, 1, 1)),
                           sigma_bulk=1000., thickness=1e-5)
    with pytest.raises(ValueError, match='requires vacuum'):
        sim.run(n_steps=2, compute_s_params=False)


@pytest.mark.parametrize('graded', [False, 'up'])
def test_c4_f0_overlap(graded):
    sim, _ = model(graded)
    sim.add_thin_conductor(sim._thin_conductors[0].shape, sigma_bulk=5.8e7,
                           thickness=35e-6, surface_impedance_f0=10e9)
    with pytest.raises(ValueError, match='an f0 sheet and a lossy film share edges'):
        sim.run(n_steps=2, compute_s_params=False)


@pytest.mark.parametrize('graded', [False, True])
def test_c4_original_mixed_overlap_refused(graded):
    from dataclasses import replace
    from tests.unit.materials.test_shared_thin_conductor_fold import build, products
    h = 1 / 1024
    sim, grid = build('mixed', nu=graded, h=h)
    sim._thin_conductors[-1] = replace(sim._thin_conductors[-1], shape=Box(
        tuple(v * h for v in (5.4, 2.2, 1.3)),
        tuple(v * h for v in (5.4, 6.2, 5.3))))
    with pytest.raises(ValueError, match='an f0 sheet and a lossy film share edges'):
        products(sim, grid, nu=graded)


@pytest.mark.parametrize('inside', [False, True], ids=['outside', 'inside'])
def test_c4_topology_design_film(inside):
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim, _ = model()
    sim.add_material('design_dielectric', eps_r=2.2)
    lo, hi = (8 * H, 9 * H) if inside else (2 * H, 3 * H)
    region = TopologyDesignRegion((lo, 2 * H, 2 * H), (hi, 3 * H, 3 * H),
                                  material_fg='design_dielectric')
    def objective(result):
        return jnp.sum((result.time_series * 1e-8) ** 2)
    if inside:
        with pytest.raises(ValueError, match='contains edges of thin conductor 0'):
            topology_optimize(sim, region, objective, n_iterations=2, verbose=False)
    else:
        pytest.importorskip("optax")
        with _realized.capture() as capture:
            _realized.enter(sim, 'topology_optimize')
            result = topology_optimize(sim, region, objective, n_iterations=2, verbose=False)
        assert len(result.loss_history) == 2
        assert np.all(np.isfinite(result.loss_history))
        records = [r for r in capture.records if 'sigma_e' in r]
        assert records
        for record in records:
            for c in range(3):
                expected = np.zeros((3, 4, 4), dtype=np.float32)
                if c != 0:
                    expected[1] = G / H
                np.testing.assert_array_max_ulp(
                    record['sigma_e'][c][8:11, 2:6, 2:6], expected, maxulp=1)
            assert record['materials'].sigma_film[1] is not None


@pytest.mark.parametrize('graded', [False, 'up'])
@pytest.mark.parametrize('impedance', [0., 50.])
def test_c6_source_port_coefficient(graded, impedance):
    from rfx.core.yee import e_update_coeffs, cell_component_e_materials
    from rfx.nonuniform import current_source_volume
    sim, widths = model(graded)
    from dataclasses import replace
    sim._ports[0] = replace(sim._ports[0], impedance=impedance)
    grid = sim._build_realized_grid()
    with _realized.capture() as capture:
        sim.run(n_steps=3, compute_s_params=False)
    electric = [r for r in capture.records if 'sigma_e' in r]
    sources = [r for r in capture.records if 'drive_scale' in r]
    assert electric and sources
    cell = (9, 4, 4)
    r = electric[-1]
    eps, sig = (r[name][2][cell] for name in ('eps_e', 'sigma_e'))
    cb = float(e_update_coeffs(eps, sig, grid.dt)[1])
    volume = current_source_volume(grid, cell, 'ez')[0] if graded else H ** 3
    # A current moment scales by Cb/dV; a Thevenin port by
    # Cb*sigma_port/edge_length, sigma_port=H/(R*d_dual_x*H).
    dual = (widths[8] + widths[9]) / 2
    expected_scale = cb / volume if impedance == 0 else cb / (impedance * dual * H)
    assert float(sources[-1]['drive_scale'][0]) == pytest.approx(expected_scale, rel=1e-6)
    mats = sources[-1]['materials']
    _, point = cell_component_e_materials(mats, cell, 'ez',
        cell_sizes=(grid.dx_arr, None, None) if graded else None)
    assert float(point) == pytest.approx(float(sig), rel=1e-6)


def test_c6_graded_sigma_gradient():
    from dataclasses import replace
    sim, _ = model('up')
    tc = sim._thin_conductors[0]
    def objective(sigma):
        sim._thin_conductors[0] = replace(tc, sigma_bulk=sigma)
        result = sim.forward(n_steps=20)
        return jnp.sum(result.time_series ** 2)
    try:
        derivative = float(jax.grad(objective)(1000.))
        central = float((objective(1001.) - objective(999.)) / 2)
        assert np.isfinite(derivative) and central != 0
        assert derivative == pytest.approx(central, rel=1e-3)
    finally:
        sim._thin_conductors[0] = tc


# Constructors that intentionally omit an explicit carrier keyword. Counts also
# catch a new unclassified construction added inside an already reviewed function.
CONSTRUCTOR_EXCEPTIONS = {
    ('rfx/adi.py', 'adi_step_2d'): (1, 'ADI refuses thin-conductor declarations before stepping'),
    ('rfx/adi.py', 'adi_step_3d'): (2, 'ADI refuses thin-conductor declarations before stepping'),
    ('rfx/adi.py', 'apply_adi_cpml_2d'): (1, 'ADI refuses thin-conductor declarations before stepping'),
    ('rfx/checkpoint.py', 'load_materials'): (1, '**records loads sigma_film per component from HDF5'),
    ('rfx/core/yee.py', 'init_materials'): (1, 'initial cell construction precedes the shared film fold'),
    ('rfx/geometry/rasterize_grid.py', 'rasterize_geometry'): (1, 'initial cell construction precedes the shared film fold'),
    ('rfx/model/electric_metrics.py', 'stage_forward_materials'): (1, '**staged includes sigma_film with the lumped cut policy'),
    ('rfx/model/materials.py', 'assemble_cells'): (1, 'initial cell construction precedes the shared film fold'),
    ('rfx/model/materials.py', 'kernel_materials'): (1, 'KernelComponents already contains the film in sigma_update'),
    ('rfx/rcs.py', 'compute_rcs'): (2, 'vacuum reference, with sigma explicitly zero'),
    ('rfx/runners/_distributed_common.py', '_apply_cpml_h_distributed'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/runners/_distributed_common.py', 'cpml_coeff_h_vacuum'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/runners/_distributed_common.py', 'update_h_nu_shmap._h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/runners/distributed_nu.py', '_apply_cpml_h_local_nu'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/runners/distributed_nu.py', 'run_nonuniform_distributed_pec._update_e_dispersive_shmap._e_disp'): (1, 'only called with dispersion; shared means already enter Debye/Lorentz coefficients'),
    ('rfx/runners/distributed_v2.py', 'run_distributed._update_h_shmap._h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/runners/distributed_v2.py', '_shard_materials'): (1, 'refuses a non-None film before constructing'),
    ('rfx/sources/coaxial_port.py', 'build_coaxial_tem_plane_source_specs'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/msl_port.py', 'make_msl_port_sources_jm'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf.py', '_apply_closed_box_h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf.py', 'apply_tfsf_h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf.py', 'update_tfsf_1d_h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf_2d.py', '_update_h_tez'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf_2d.py', '_update_h_tmz'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf_2d.py', 'apply_tfsf_2d_h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/tfsf_oblique_open.py', 'apply_methodB_h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/sources/waveguide_port.py', 'apply_waveguide_port_h'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/stepping/slab.py', '_vacuum_dispersion_values'): (1, 'vacuum reference, with sigma explicitly zero'),
    ('rfx/subgridding/disjoint_3d.py', ''): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/subgridding/sbp_sat_1d.py', '_update_h_1d'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/subgridding/sbp_sat_2d.py', '_update_hx_2d'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/subgridding/sbp_sat_2d.py', '_update_hy_2d'): (1, 'H-only operand: no electric conductivity consumed'),
    ('rfx/subgridding/sbp_sat_3d.py', '_make_mats'): (1, 'vacuum reference, with sigma explicitly zero'),
}

# Every exact cell-sigma read, including typed exclusions, is classified.
SIGMA_READS = {
    ('rfx/_realized.py', 'scalar_electric', 'materials'): (1, 'arithmetic into coefficient / observed scalar operands'),
    ('rfx/adi.py', 'adi_step_3d', 'observed'): (1, 'arithmetic into ADI damping and curl coefficients'),
    ('rfx/api/_compile.py', '_CompileMixin._has_pec_to_conform', 'self._resolve_material(e.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/api/_compile.py', '_CompileMixin.conductor_mask', 'materials'): (1, 'threshold / conductor cell footprint'),
    ('rfx/api/_execute.py', '_ExecuteMixin._auto_configure_mesh', 'spec'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/api/_execute.py', '_ExecuteMixin._run_adi_from_materials', 'materials'): (5, 'interface/refusal test, tracer test, broadcasting and slicing to ADI'),
    ('rfx/artifacts.py', '_material_summary', 'material'): (1, 'report text / safe scalar conversion'),
    ('rfx/auto_config.py', 'analyze_features', 'mat'): (1, 'documented dict of MaterialSpec or dict at :136; no assembled grid input'),
    ('rfx/boundaries/cpml.py', '_flip_profile', 'p'): (1, 'CPMLParams profile record, returned explicitly as CPMLParams in this function'),
    ('rfx/boundaries/cpml.py', '_pad_profile_at_end', 'p'): (2, 'CPMLParams profile record, returned explicitly as CPMLParams in this function'),
    ('rfx/boundaries/cpml.py', '_pad_profile_at_start', 'p'): (2, 'CPMLParams profile record, returned explicitly as CPMLParams in this function'),
    ('rfx/checkpoint.py', 'save_materials', 'materials'): (1, 'passing on to HDF5 sigma dataset'),
    ('rfx/core/yee.py', 'cell_component_e_materials', 'materials'): (1, 'arithmetic: one-edge mean and own-component stamp'),
    ('rfx/core/yee.py', 'cell_owned_component_materials', 'materials'): (1, 'arithmetic: own-component cell conductivity'),
    ('rfx/core/yee.py', 'component_e_materials', 'materials'): (1, 'arithmetic: subtract lumped mirror, edge mean, add own-edge stamp'),
    ('rfx/fidelity.py', '_assembled_as_pec', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/fidelity.py', '_declared_material', 'spec'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/geometry/port_termination.py', 'conductor_entries', 'sim._resolve_material(entry.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/geometry/rasterize_grid.py', 'rasterize_geometry', 'mat'): (4, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/geometry/smoothing.py', 'smoothed_shape_pairs', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/interop/_design.py', 'simulation_from_design', 'spec'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/io.py', 'export_geometry_json', 'm'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/materials/__init__.py', 'set_material', 'materials'): (1, 'passing on: where(mask,new_sigma,old_sigma)'),
    ('rfx/materials/thin_conductor.py', 'sheet_update_coeffs', 'materials'): (2, 'arithmetic: cell sigma + sheet sigma into exponential A/B'),
    ('rfx/runners/distributed_nu.py', 'run_nonuniform_distributed_pec.run_fn', 'materials'): (1, 'carrier staged or passed alongside volume sigma to the shared edge reader'),
    ('rfx/model/materials.py', '_design_box_edge_coeffs', 'materials'): (1, 'slicing/passing on to window MaterialArrays'),
    ('rfx/model/materials.py', 'assemble_cells', 'mat'): (4, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/model/materials.py', 'assemble_cells', 'sim._resolve_material(entry.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/model/materials.py', 'electric_materials_traced', 'materials'): (1, 'source trace detection includes the film record'),
    ('rfx/model/materials.py', 'realize_components', 'materials'): (1, 'arithmetic: volume sigma / lumped subtraction and edge averaging'),
    ('rfx/model/overrides.py', 'apply_material_overrides', 'materials'): (2, 'slicing/passing on or retaining unoverridden cell sigma'),
    ('rfx/model/thin_conductors.py', '_fold_dc_plane', 'materials'): (2, 'film fold shapes or diagnostic envelope / carrier-preserving junction splice'),
    ('rfx/model/thin_conductors.py', 'conductivity_envelope', 'materials'): (1, 'film fold shapes or diagnostic envelope / carrier-preserving junction splice'),
    ('rfx/model/thin_conductors.py', 'fold_thin_conductor', 'materials'): (1, 'passing on: ones_like shape for f0, where overwrite for DC'),
    ('rfx/model/thin_conductors.py', 'splice_film', 'materials'): (1, 'film fold shapes or diagnostic envelope / carrier-preserving junction splice'),
    ('rfx/model/thin_conductors.py', 'splice_junction_materials', 'junction'): (1, 'film fold shapes or diagnostic envelope / carrier-preserving junction splice'),
    ('rfx/model/thin_conductors.py', 'splice_junction_materials', 'materials'): (1, 'film fold shapes or diagnostic envelope / carrier-preserving junction splice'),
    ('rfx/preflight/absorber.py', '_conductor_in_thin_absorber_findings', 'self._resolve_material(mat_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/absorber.py', '_validate_cfg_lossless_resonator_in_absorber._resolve', 'mspec'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/line_stub.py', '_permittivity', 'material'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/mesh.py', '_validate_cfg_graded_box_rasterization', 'self._resolve_material(entry.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/mesh.py', '_validate_mesh_quality', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/msl_reflector.py', 'msl_nearest_downstream_reflector._is_conductor', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/ntff.py', '_validate_ntff_inverse_design', 'self._resolve_material(e.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/pec_geometry.py', '_validate_cfg_campaign_statics', 'self._resolve_material(e.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/pec_geometry.py', '_validate_cfg_sheet_cavity_thickness', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/pec_geometry.py', '_validate_cfg_sheet_cavity_thickness', 'self._resolve_material(g.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/ports.py', '_validate_cfg_port_inside_pec', 'self._resolve_material(e.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/ports.py', 'half_node_split_findings', 'sim._resolve_material(entry.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/preflight/realization.py', '_CampaignStaticsContext.entry_realizations', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/rcs.py', 'compute_rcs', 'materials'): (1, 'passing on: zeros_like for vacuum reference shape'),
    ('rfx/runners/_admission.py', '', 'm'): (1, 'explicit sigma operand; see classified constructor and shared realization reach'),
    ('rfx/runners/_admission.py', '_adi_homogeneous', 'materials'): (1, 'arithmetic/interface refusal: homogeneous cell eps/sigma'),
    ('rfx/runners/_admission.py', '_pec_shapes', 'sim._resolve_material(entry.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/runners/_admission.py', '_placed', 'material'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/runners/_distributed_common.py', 'slab_e_component_materials', 'materials'): (1, 'slicing/passing on: replicated x-low view into edge mean'),
    ('rfx/runners/_distributed_common.py', 'slab_e_materials_shmap', 'mat'): (1, 'passing on: checkpointed slab mean operands'),
    ('rfx/runners/_distributed_common.py', 'update_e_nu_shmap', 'mat'): (1, 'passing on into slab E coefficient construction'),
    ('rfx/runners/_distributed_common.py', 'update_h_nu_shmap', 'mat'): (1, 'passing on into MaterialArrays for H update'),
    ('rfx/runners/distributed_nu.py', 'run_nonuniform_distributed_pec._update_e_dispersive_shmap', 'mat'): (1, 'passing on to dispersive E shard-map'),
    ('rfx/runners/distributed_v2.py', '_shard_materials', 'materials'): (1, 'slicing/passing on: device_put into MaterialArrays'),
    ('rfx/runners/distributed_v2.py', 'run_distributed', 'materials'): (1, 'slicing/passing on: cut cell sigma into slabs'),
    ('rfx/runners/distributed_v2.py', 'run_distributed._update_e_shmap', 'mat'): (1, 'passing on to E shard-map'),
    ('rfx/runners/distributed_v2.py', 'run_distributed._update_h_shmap', 'mat'): (1, 'passing on to H shard-map'),
    ('rfx/runners/nonuniform.py', 'assemble_interface_eps_nu', 'sim._resolve_material(entry.material_name)'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/simulation.py', '_resolve_design_box', 'materials'): (1, 'slicing/passing on: default box sigma from assembled cells'),
    ('rfx/simulation.py', '_resolve_design_box', 'spec'): (4, 'DesignBox specification from _design_box_spec, not assembled materials'),
    ('rfx/sources/coaxial_port.py', 'stamp_coaxial_annular_resistor', 'materials'): (1, 'arithmetic/write preparation: host copy of cell sigma'),
    ('rfx/sources/coaxial_port.py', 'stamp_coaxial_line', 'materials'): (1, 'arithmetic/write preparation: host copy of cell sigma'),
    ('rfx/subgridding/validation.py', 'validate_subgrid_setup', 'materials'): (3, 'arithmetic: material plane deltas and interface-clearance test'),
    ('rfx/subgridding/validation.py', 'validate_subgrid_setup', 'spec'): (1, 'resolved declared material in static geometry validation'),
    ('rfx/surrogate.py', '_is_pec_entry', 'mat'): (1, 'declared material returned by _resolve_material / specification mapping in this file; not assembled cells'),
    ('rfx/topology.py', 'density_to_eps', 'fields'): (1, 'TopologyMaterialFields from density_to_material_fields; definition and producer in this file'),
    ('rfx/topology.py', 'topology_optimize', 'mat_bg'): (1, 'resolved design-region MaterialSpec, local _resolve_material assignments at :462/:466'),
    ('rfx/topology.py', 'topology_optimize', 'mat_fg'): (1, 'resolved design-region MaterialSpec, local _resolve_material assignments at :462/:466'),
    ('rfx/topology.py', 'topology_optimize.forward', 'base_materials'): (2, 'slicing/passing on: base sigma plus local design override; dtype'),
    ('rfx/topology.py', 'topology_optimize.forward', 'fields'): (1, 'TopologyMaterialFields from density_to_material_fields; definition and producer in this file'),
    ('rfx/visualize.py', 'plot_geometry_2d_slice', '_geo_mats'): (1, 'threshold / conductor overlay'),
    ('rfx/vmap_sweep.py', '_apply_batched_thin_conductors._one', 'mats'): (1, 'passing on: folded sigma returned per batch'),
    ('rfx/vmap_sweep.py', '_build_batched_materials', 'base_materials'): (1, 'slicing/passing on; global non-zero mask and batch replacement'),
    ('rfx/vmap_sweep.py', '_build_batched_materials', 'pre_materials'): (1, 'slicing/passing on; global non-zero mask and batch replacement'),
    ('rfx/vmap_sweep.py', 'vmap_material_sweep', 'batched_materials'): (2, 'slicing/passing on: batch slice / batched scan argument'),
}


def test_c2_carrier_and_cell_read_census():
    from collections import Counter
    constructors, reads = Counter(), Counter()
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / 'rfx').rglob('*.py')):
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        aliases = {'MaterialArrays'}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases.update(a.asname or a.name for a in node.names if a.name == 'MaterialArrays')
        def scope(node):
            names = []
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
            return '.'.join(reversed(names))
        file = str(path.relative_to(root))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in aliases or name.endswith('.MaterialArrays'):
                    if not any(k.arg == 'sigma_film' for k in node.keywords):
                        constructors[file, scope(node)] += 1
                if name == 'cut' or name.endswith('.cut'):
                    kind = node.args[2] if len(node.args) > 2 else next(
                        (k.value for k in node.keywords if k.arg == 'kind'), None)
                    # Resolve a locally named mapping as well as inline forms.
                    if isinstance(kind, ast.Name):
                        assignments = [n.value for n in ast.walk(tree)
                            if isinstance(n, ast.Assign) and scope(n) == scope(node)
                            and any(isinstance(t, ast.Name) and t.id == kind.id for t in n.targets)]
                        if len(assignments) == 1:
                            kind = assignments[0]
                    if isinstance(kind, ast.Dict):
                        assert 'sigma_film' in [getattr(k, 'value', None) for k in kind.keys], (file, scope(node))
                    elif isinstance(kind, ast.Call) and ast.unparse(kind.func) == 'dict':
                        assert any(k.arg == 'sigma_film' for k in kind.keywords), (file, scope(node))
                if name == 'getattr' and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'sigma':
                    reads[file, scope(node), ast.unparse(node.args[0])] += 1
            if isinstance(node, ast.Attribute) and node.attr == 'sigma' and isinstance(node.ctx, ast.Load):
                reads[file, scope(node), ast.unparse(node.value)] += 1
    assert constructors == Counter({k: v[0] for k, v in CONSTRUCTOR_EXCEPTIONS.items()})
    assert reads == Counter({k: v[0] for k, v in SIGMA_READS.items()})
    assert all(reason for _, reason in (*CONSTRUCTOR_EXCEPTIONS.values(), *SIGMA_READS.values()))


@pytest.mark.parametrize('plane,expected_plane', [(11., 11), (11.6, 0), (12., 0), (5.5, 5)])
def test_c1_periodic_plane_and_lower_tie(plane, expected_plane):
    from rfx.boundaries.spec import BoundarySpec
    sim = Simulation(freq_max=10e9, domain=(12*H, 8*H, 8*H), dx=H,
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec'))
    sim.add_thin_conductor(Box((plane*H, 0, 0), (plane*H, 8*H, 8*H)),
                           sigma_bulk=1000., thickness=1e-5)
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    for c in (1, 2):
        expected = np.zeros(grid.shape, dtype=np.float32)
        expected[expected_plane, 2:-2, 2:-2] = G/H
        np.testing.assert_array_max_ulp(np.asarray(mats.sigma_film[c])[:, 2:-2, 2:-2],
                                        expected[:, 2:-2, 2:-2], maxulp=1)


def test_c1_split_materials_consumed_on_two_devices():
    from rfx.runners._distributed_common import _split_materials, slab_e_component_materials
    sim, widths = model()
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    slabs = _split_materials(mats, 2)
    nx_per = (grid.shape[0] + 1) // 2
    def consume(local, rank):
        return slab_e_component_materials(local, nx_per, grid.shape[0], rank=rank)
    eps, sigma = jax.pmap(consume)(slabs, jnp.arange(2))
    for rank in range(2):
        count = min(nx_per, grid.shape[0] - rank * nx_per)
        assert_expected(tuple(a[rank] for a in sigma), tuple(a[rank] for a in eps),
                        widths, 9, start=rank * nx_per, count=count)


def test_c6_two_device_graded_source():
    from rfx.core.yee import e_update_coeffs
    sim, widths = model('up')
    grid = sim._build_realized_grid()
    with _realized.capture() as capture:
        sim.forward(n_steps=3, distributed=True)
    source = next(r for r in capture.records if r['site'] == 'distributed_nu.sources')
    edge = next(r for r in capture.records if 'owned_start' in r and int(r['owned_start']) == 9)
    eps = edge['eps_e'][2][1, 4, 4]
    sigma = edge['sigma_e'][2][1, 4, 4]
    cb = float(e_update_coeffs(eps, sigma, grid.dt)[1])
    dual = (widths[8] + widths[9]) / 2
    assert float(source['drive_scale'][0]) == pytest.approx(cb / (dual * H * H), rel=1e-6)


@pytest.mark.parametrize('forward', [False, True])
def test_c6_two_device_graded_port_existing_refusal(forward):
    from dataclasses import replace
    sim, _ = model('up')
    sim._ports[0] = replace(sim._ports[0], impedance=50.)
    with pytest.raises((ValueError, NotImplementedError), match='[Pp]ort|lumped'):
        if forward:
            sim.forward(n_steps=2, distributed=True)
        else:
            sim.run(n_steps=2, devices=jax.devices()[:2], compute_s_params=False)


@pytest.mark.parametrize('lane', ['adi', 'subgrid'])
def test_c4_existing_path_refusals(lane):
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .012), dx=H,
                     boundary='pec', **({'solver': 'adi'} if lane == 'adi' else {}))
    sim.add_thin_conductor(Box((.006, .002, .002), (.006, .010, .010)),
                           sigma_bulk=1000., thickness=1e-5)
    if lane == 'subgrid':
        sim.add_refinement(z_range=(0., .008), ratio=2, validation='research')
    with pytest.raises((ValueError, NotImplementedError), match='thin.conductor|subgridded|thin_conductors'):
        sim.run(n_steps=2, compute_s_params=False)


def test_c2_checkpoint_roundtrip(tmp_path):
    from rfx.checkpoint import save_materials, load_materials
    sim, widths = model()
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    save_materials(tmp_path / 'film.h5', mats)
    restored = load_materials(tmp_path / 'film.h5')
    eps, sigma = component_e_materials(restored)
    assert_expected(sigma, eps, widths, 9)


def test_c3_global_sigma_sweep_retains_film():
    from rfx.vmap_sweep import _build_batched_materials
    sim, widths = model()
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    batch = _build_batched_materials(sim, grid, mats, 'sigma', jnp.asarray([0., 7.]))
    for row in range(2):
        local = jax.tree.map(lambda a: a[row], batch)
        eps, sigma = component_e_materials(local)
        assert_expected(sigma, eps, widths, 9)


@pytest.mark.parametrize('normal', range(3))
def test_c1_rectangular_coverage_and_volume_preservation(normal):
    from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
    from rfx.grid import Grid
    grid = Grid(freq_max=10e9, domain=(.008,) * 3, dx=H, cpml_layers=0)
    cells = init_materials(grid.shape)._replace(
        eps_r=jnp.full(grid.shape, 4.), sigma=jnp.full(grid.shape, .025))
    tangents = [a for a in range(3) if a != normal]
    lo, hi = [.002] * 3, [.006] * 3
    lo[normal] = hi[normal] = .004
    tc = ThinConductor(shape=Box(tuple(lo), tuple(hi)), sigma_bulk=1000., thickness=1e-5)
    result, _ = apply_thin_conductor(grid, tc, cells)
    assert result.eps_r is cells.eps_r and result.sigma is cells.sigma
    assert result.sigma_film[normal] is None
    for component in tangents:
        transverse = next(a for a in tangents if a != component)
        expected = np.zeros(grid.shape, dtype=np.float32)
        # Four cells along the E edge direction; at the transverse ends,
        # a single occupied incident cell gives half coverage.
        for along in range(2, 6):
            for across, weight in [(2, .5), (3, 1.), (4, 1.), (5, 1.), (6, .5)]:
                index = [0, 0, 0]
                index[normal], index[component], index[transverse] = 4, along, across
                expected[tuple(index)] = G * weight / H
        np.testing.assert_array_max_ulp(result.sigma_film[component], expected, maxulp=1)


@pytest.mark.parametrize('lane', ['uniform', 'graded', 'periodic'])
def test_c1_abutting_films_equal_spanning_film(lane):
    from rfx.boundaries.spec import BoundarySpec
    widths = np.full(12, H)
    if lane == 'graded':
        widths[5:7] = [.00075, .0015]
    nodes = np.r_[0., np.cumsum(widths)]
    periodic = lane == 'periodic'
    # The periodic junction is node zero: [8, 12) and [0, 4).
    pieces = [(8, 12), (0, 4)] if periodic else [(2, 6), (6, 10)]
    whole = (8, 16) if periodic else (2, 10)

    def assemble(intervals):
        sim = Simulation(freq_max=10e9, domain=(float(widths.sum()), .012, .008),
                         dx=H, boundary=(BoundarySpec(x='periodic', y='pec', z='pec')
                                         if periodic else 'pec'),
                         **({'dx_profile': widths} if lane == 'graded' else {}))
        for low, high in intervals:
            xlow = float(nodes[low])
            xhigh = high*H if high > 12 else float(nodes[high])
            sim.add_thin_conductor(Box((xlow, .002, .004), (xhigh, .010, .004)),
                                   sigma_bulk=1000., thickness=1e-5)
        grid = sim._build_realized_grid()
        return grid, sim._assemble_materials(grid)[0]

    grid, spanning = assemble([whole])
    expected_x, expected_y = np.zeros(grid.shape), np.zeros(grid.shape)
    xcells = [8, 9, 10, 11, 0, 1, 2, 3] if periodic else list(range(2, 10))
    xnodes = ([8, 9, 10, 11, 0, 1, 2, 3, 4] if periodic else list(range(2, 11)))
    for x in xcells:
        expected_x[x, 2:11, 4] = G/H
        expected_x[x, [2, 10], 4] /= 2
    for x in xnodes:
        expected_y[x, 2:10, 4] = G/H * (.5 if x in (xnodes[0], xnodes[-1]) else 1.)
    # At the graded junction the incident widths give 1/3 + 2/3 = 1.
    # At the periodic junction they give 1/2 + 1/2 = 1.
    records = [spanning]
    for order in (pieces, pieces[::-1]):
        _, result = assemble(order)
        records.append(result)
    for result in records:
        assert result.sigma_film[2] is None
        for actual, expected in zip(result.sigma_film[:2], (expected_x, expected_y)):
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
        for actual, reference in zip(result.sigma_film[:2], spanning.sigma_film[:2]):
            np.testing.assert_array_max_ulp(actual, reference, maxulp=1)
    for a, b in zip(records[1].sigma_film[:2], records[2].sigma_film[:2]):
        np.testing.assert_array_equal(a, b)


def test_c1_overlapping_films_add_in_either_order():
    from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor
    from rfx.grid import Grid
    grid = Grid(freq_max=10e9, domain=(.008,) * 3, dx=H, cpml_layers=0)
    shape = Box((.002, .002, .004), (.006, .006, .004))
    a, b = (ThinConductor(shape, sigma, 1e-5) for sigma in (1000., 2000.))
    records = []
    for order in ((a, b), (b, a)):
        result = init_materials(grid.shape)
        for conductor in order:
            result, _ = apply_thin_conductor(grid, conductor, result)
        records.append(result)
        for component in (0, 1):
            expected = np.zeros(grid.shape)
            for along in range(2, 6):
                for across in range(2, 7):
                    index = (along, across, 4) if component == 0 else (across, along, 4)
                    expected[index] = (G + 2*G)/H * (.5 if across in (2, 6) else 1.)
            np.testing.assert_allclose(result.sigma_film[component], expected, rtol=2e-7)
        assert result.sigma_film[2] is None
        np.testing.assert_array_equal(result.eps_r, 1.)
        np.testing.assert_array_equal(result.sigma, 0.)
    for a, b in zip(records[0].sigma_film[:2], records[1].sigma_film[:2]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('graded', [False, 'up'])
@pytest.mark.parametrize('transform', [jax.grad, jax.value_and_grad])
def test_c4_traced_permittivity_refused(graded, transform):
    from dataclasses import replace
    from rfx.model.thin_conductors import DCFilmAdmissionError
    sim, _ = model(graded)
    conductor = sim._thin_conductors[0]

    def objective(eps):
        sim._thin_conductors[0] = replace(conductor, eps_r=eps)
        return jnp.sum(sim.forward(n_steps=2).time_series)

    with pytest.raises(DCFilmAdmissionError, match='a film on one node plane has no volume; its eps_r is not modelled') as error:
        transform(objective)(4.)
    assert error.value.code == 'dc_film_permittivity'


def test_c4_occupancy_design_box_refuses_film():
    sim, widths = model()
    x = float(widths[:9].sum())
    box = ((x - widths[8], .001, .001), (x - widths[8]/4, .002, .002))
    with pytest.raises(ValueError, match='contains edges of thin conductor 0'):
        sim.forward(n_steps=2, design_box=box,
                    design_occupancy_override=jnp.ones((2, 2, 2)))


def test_c2_coax_junction_splices_film_record():
    from rfx.model.thin_conductors import splice_junction_materials
    shape = (2, 3, 4)
    stub = init_materials(shape)._replace(eps_r=jnp.full(shape, 2.), sigma=jnp.full(shape, 3.),
        sigma_film=(jnp.full(shape, 5.), None, jnp.full(shape, 7.)))
    junction = init_materials(shape)._replace(eps_r=jnp.full(shape, 11.), sigma=jnp.full(shape, 13.),
        sigma_film=(jnp.full(shape, 17.), jnp.full(shape, 19.), None))
    result = splice_junction_materials(stub, junction, 2)
    for actual, below, above in [(result.eps_r, 2., 11.), (result.sigma, 3., 13.),
                                (result.sigma_film[0], 5., 17.),
                                (result.sigma_film[1], 0., 19.),
                                (result.sigma_film[2], 7., 0.)]:
        expected = np.empty(shape)
        expected[:, :, :2], expected[:, :, 2:] = below, above
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('param', ['sub.eps_r', 'eps_r'])
def test_c3_sweep_results_retain_film(param):
    from rfx import GaussianPulse
    from rfx.vmap_sweep import vmap_material_sweep

    def build(value=2.2, film=True):
        sim = Simulation(freq_max=10e9, domain=(.024, .024, .004), dx=H,
                         boundary='cpml', cpml_layers=6)
        sim.add_material('sub', eps_r=value if param == 'sub.eps_r' else 2.2)
        sim.add(Box((.018, 0, 0), (.020, .024, .004)), material='sub')
        if param == 'eps_r':
            # Fill the background too: global sweeps replace non-vacuum cells.
            sim.add_material('global', eps_r=value)
            sim.add(Box((0, 0, 0), (.024, .024, .004)), material='global')
        if film:
            sim.add_thin_conductor(Box((.012, -1, -1), (.012, 1, 1)),
                                   sigma_bulk=1e4, thickness=1e-5)
        sim.add_source((.006, .012, .002), component='ez',
                       waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='current')
        sim.add_probe((.016, .012, .002), component='ez')
        return sim

    values = [2.2, 3.]
    sweep = np.asarray(vmap_material_sweep(build(), param, values, n_steps=200).time_series)
    bare = np.asarray(vmap_material_sweep(build(film=False), param, values, n_steps=200).time_series)
    for row, value in enumerate(values):
        reference = np.asarray(build(value).run(n_steps=200, compute_s_params=False).time_series)
        actual = sweep[row].reshape(reference.shape)
        peak = np.max(np.abs(reference))
        assert peak > 0
        assert np.max(np.abs(actual - reference)) <= 1e-4*peak
        assert np.max(np.abs(actual - bare[row].reshape(reference.shape))) > 1e-2*peak


@pytest.mark.parametrize('graded', [False, True])
def test_c4_msl_port_f0_and_film_refused_in_sheet_context(graded, monkeypatch):
    from tests.unit.ports.test_msl_realized_port_contract import _model
    import rfx.sources.msl_port as msl
    sim, _ = _model(nonuniform=graded, trace_kind='f0')
    trace = sim._thin_conductors[0].shape
    sim.add_thin_conductor(trace, sigma_bulk=1000., thickness=1e-5)
    reached = []
    original = msl.validate_msl_port_geometry

    def observed(*args, **kwargs):
        # Both runners already lend the realized materials to this builder;
        # the standalone SheetConductors fallback is therefore not taken.
        assert kwargs['conductors'].materials.sigma_film is not None
        reached.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(msl, 'validate_msl_port_geometry', observed)
    with pytest.raises(ValueError, match='an f0 sheet and a lossy film share edges'):
        sim.run(n_steps=2, compute_s_params=False)
    assert reached == [True]


def test_v2_material_shard_refuses_a_film():
    from rfx.runners.distributed_v2 import _shard_materials
    cells = jnp.ones((2, 2, 2), jnp.float32)
    film = (jnp.zeros((2, 2, 2), jnp.float32),) * 3
    with pytest.raises(ValueError, match='lossy film'):
        _shard_materials(MaterialArrays(eps_r=cells, sigma=cells * 0, mu_r=cells, sigma_film=film), None)
