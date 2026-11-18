"""B-src internal MSL, time samplers, waveguide edges and caller contract."""
import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.float64_source_reference import assert_value, gaussian
from tests.contracts.float64_source_scenes import capture, observe, scene


def test_e13_e14_internal_msl():
    from rfx import GaussianPulse
    from rfx.core.yee import MaterialArrays
    from rfx.sources.msl_port import make_msl_port_sources_jm
    with jax.enable_x64():
        sim, _ = scene('pin')
        grid = sim._build_realized_grid()
        assert type(grid.dt) is float
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
        assert type(grid.dt) is float
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
        'apply_density_filter', 'density_to_material_fields', 'drive_columns',
        'kottke_inv_eps_from_occupancy', 'make_msl_port_sources', 'make_current_source'}
    root = Path(__file__).resolve().parents[2]
    missing, checked = [], 0
    for path in (root / 'rfx').rglob('*.py'):
        tree = ast.parse(path.read_text())
        aliases = {n.asname: n.name for n in ast.walk(tree) if isinstance(n, ast.alias) and n.asname}
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else (
                call.func.attr if isinstance(call.func, ast.Attribute) else '')
            if name == 'defer' and call.args and isinstance(call.args[0], ast.Name):
                name = call.args[0].id
            name = aliases.get(name, name)
            if name not in required:
                continue
            if path.relative_to(root).as_posix() == 'rfx/simulation.py' and name in {'uniform_source_table', 'port_drive_waveform'}:
                continue
            checked += 1
            if not any(k.arg == 'dtype' and not (isinstance(k.value, ast.Constant)
                       and k.value.value is None) for k in call.keywords):
                missing.append((str(path), call.lineno, name))
    assert checked >= 30, checked
    assert not missing, missing


@pytest.mark.parametrize('label', ['floquet', 'U-tfsf'])
def test_record_end_drive_dtype(label):
    from rfx.measurement.record_end import _declared_drives
    with jax.enable_x64():
        sim, _ = scene(label)
        grid = sim._build_realized_grid()
        drives = []
        def refuse(*args):
            pytest.fail(str(args))
        with observe() as rows:
            _declared_drives(sim, 200, grid, False, drives, refuse, None)
        assert len(drives) == 1
        if label == 'floquet':
            assert type(grid.dt) is float
            assert np.asarray(drives[0][1]).dtype == np.float64
            assert rows['uniform_source_table']
            assert all(local['times'].dtype == jnp.float64
                       for local, _ in rows['uniform_source_table'])
        else:
            assert rows['init_tfsf']
            local, _ = rows['init_tfsf'][-1]
            profiles = [local[name] for name in ('_prof', '_prof_hi', 'magnetic_lo', 'magnetic_hi')]
            assert all(profile.b.dtype == profile.c.dtype == jnp.float64
                       for profile in profiles)


def test_rcs_low_level_tfsf_dtype(monkeypatch):
    import rfx.measurement.rcs as rcs
    from rfx.core.yee import MaterialArrays
    class Captured(Exception):
        pass
    real_init = rcs.init_tfsf
    seen = []
    def capture_init(*args, **kwargs):
        seen.append(real_init(*args, **kwargs))
        raise Captured
    monkeypatch.setattr(rcs, 'init_tfsf', capture_init)
    with jax.enable_x64():
        sim, _ = scene('U-tfsf')
        grid = sim._build_realized_grid()
        one = jnp.ones(grid.shape, jnp.float64)
        materials = MaterialArrays(one, jnp.zeros_like(one), one)
        with observe() as rows, pytest.raises(Captured):
            rcs.compute_rcs(grid, materials, 2, f0=10e9, cpml_layers=8)
        assert len(seen) == 1
        assert rows['init_tfsf']
        local, _ = rows['init_tfsf'][-1]
        profiles = [local[name] for name in ('_prof', '_prof_hi', 'magnetic_lo', 'magnetic_hi')]
        assert all(profile.b.dtype == profile.c.dtype == jnp.float32
                   for profile in profiles)


def test_e14_public_msl_pinned():
    from rfx.grid import Grid
    from tests.contracts.test_msl_one_reference_plane import _thru
    with jax.enable_x64():
        sim = _thru()
        sim._precision = 'float64'
        sim._dt_pin = float(.5 * Grid.courant_dt(sim._dx, 3))
        with observe() as rows:
            result = sim.run(n_steps=2, compute_s_params=False, conformal_pec='cut_cell_floor')
            jax.block_until_ready(result)
        calls = rows['port_drive_waveform']
        assert calls
        for local, _ in calls:
            assert type(local['grid'].dt) is float
            times = np.arange(local['n_steps'], dtype=np.float64)*local['grid'].dt
            assert_value(local['times'], times, 0)
            assert_value(local['samples'], gaussian(times, local['excitation']), 307)
