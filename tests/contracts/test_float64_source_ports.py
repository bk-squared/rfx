"""B-src internal MSL, time samplers, waveguide edges and caller contract."""
import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tests.contracts.float64_source_reference import assert_value, gaussian
from tests.contracts.float64_source_scenes import capture, observe, scene


def test_e13_e14_internal_msl():
    from rfx import GaussianPulse
    from rfx.core.yee import MaterialArrays
    from rfx.sources.msl_port import make_msl_port_sources_jm
    with jax.enable_x64():
        sim, _ = scene('pin')
        grid = sim._build_realized_grid()
        one = jnp.ones(grid.shape, jnp.float64)
        materials = MaterialArrays(one*2.2, jnp.zeros_like(one), one)
        em = SimpleNamespace(j_grid_lo=12, k_grid_lo=12, n_y_grid=1, n_z_grid=1,
            cell_indices=[(12, 12, 12)], ey=np.array([[.137]]), ez=np.array([[.271]]),
            hy=np.array([[.193]]), hz=np.array([[.317]]))
        port = SimpleNamespace(excitation=GaussianPulse(f0=10e9, bandwidth=.8), direction='+x')
        with observe() as rows:
            e, h = make_msl_port_sources_jm(grid, port, materials, 200, em, dtype=jnp.float64)
        local, _ = rows['make_msl_port_sources_jm'][-1]
        for specs, base, profiles, coefficients, signs in (
            (e, local['base_wave_e'], (.193, .317), (local['coeff_E'], local['coeff_Ey']), (1, -1)),
            (h, local['base_wave_h'], (.271, .137), (local['coeff_H'],)*2, (-1, 1))):
            assert len(specs) == 2
            for spec, profile, coefficient, sign in zip(specs, profiles, coefficients, signs):
                reference = sign*np.float64(coefficient)*profile*np.asarray(base, dtype=np.float64)
                assert_value(spec.waveform, reference, 6, scale=np.abs(reference))
        for offset, name, bar in ((0., 'e', 307), (.5, 'h', 309)):
            times = (np.arange(200, dtype=np.float64)+offset)*float(grid.dt)
            assert_value(local['times_'+name], times, 0)
            assert_value(local['base_wave_'+name], gaussian(times, port.excitation), bar)


def test_e14_internal_sample_helpers():
    from rfx import GaussianPulse
    from rfx.model.source_coefficients import field_source_samples, current_source_samples
    with jax.enable_x64():
        sim, _ = scene('pin')
        grid = sim._build_realized_grid()
        waveform = GaussianPulse(f0=10e9, bandwidth=.8)
        reference = gaussian(np.arange(200, dtype=np.float64)*float(grid.dt), waveform)
        assert_value(field_source_samples(waveform, 200, float(grid.dt), dtype=jnp.float64), reference, 307)
        assert_value(current_source_samples(grid, waveform, 200, dtype=jnp.float64), reference, 307)


def test_e17_waveguide_conformal():
    from tests.contracts.float64_value_reference import assert_spacings, conformal_reference, clamp_reference
    with jax.enable_x64():
        rows = capture('conformal-guide')
        observed = [value for _, value in rows['default_conformal_weights'] if value is not None]
        assert observed, 'waveguide admission must read the default conformal weights'
        references = rows['conformal_reference_inputs']
        assert references
        for samples, table, (actual_lengths, actual_areas) in references:
            lengths, areas = conformal_reference(samples)
            lengths = tuple(a[table.sc_lattice_slice] for a in lengths)
            areas = tuple(a[table.sc_lattice_slice] for a in areas)
            expected = clamp_reference(lengths, areas)
            for actual, raw, reference in zip(actual_lengths, lengths, expected):
                physical = table.sc_physical_slice
                bound = (reference < raw)[physical]
                assert_spacings(np.asarray(actual)[~bound], reference[physical][~bound], 8)
                assert_spacings(np.asarray(actual)[bound], reference[physical][bound], 64)
        for arrays in observed:
            assert all(a.dtype == jnp.float64 for a in arrays)


def test_product_callers_supply_dtype():
    """s16: frozen convenience callers alone may omit the fallback keyword."""
    required = {'uniform_source_table', 'port_drive_waveform', 'field_source_samples',
        'current_source_samples', 'drive_table', 'occupancy_from_design',
        'compute_smoothed_eps', 'compute_smoothed_eps_nonuniform',
        'compute_inv_eps_tensor_diag', '_smoothed_eps_uniform', '_to_jax',
        'init_tfsf', 'tfsf_from_declaration', 'init_tfsf_methodB', 'init_plane_wave',
        '_gather_table', 'make_msl_port_sources_jm', 'deferred_msl_sources_jm',
        'floquet_port_source', '_homogeneous_reference_materials',
        'apply_density_filter', 'density_to_material_fields'}
    missing = []
    for path in Path('rfx').rglob('*.py'):
        tree = ast.parse(path.read_text())
        aliases = {n.asname: n.name for n in ast.walk(tree) if isinstance(n, ast.alias) and n.asname}
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else (
                call.func.attr if isinstance(call.func, ast.Attribute) else '')
            name = aliases.get(name, name)
            if name not in required:
                continue
            if str(path) == 'rfx/simulation.py' and name in {'uniform_source_table', 'port_drive_waveform'}:
                continue
            if not any(k.arg == 'dtype' for k in call.keywords):
                missing.append((str(path), call.lineno, name))
    assert not missing, missing
