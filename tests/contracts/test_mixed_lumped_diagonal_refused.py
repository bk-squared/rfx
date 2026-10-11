"""Mixed refusal contract; retained entries have same-run independent oracles."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from rfx.probes.probes import PreDecisionLumpedDiagonalWarning
from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_msl, _H_SUB, _DX


def _run(kind, channel):
    sim, yc = _base_sim()
    extent = {'lumped': None, 'wire1': _DX, 'wire3': _H_SUB}[kind]
    sim.add_port(position=(2e-3, yc, 0.0), component='ez', impedance=50.0,
                 **({} if extent is None else {'extent': extent}))
    _add_msl(sim, yc, n_probe_offset=10, n_probe_spacing=4)
    return sim.compute_mixed_s_matrix(
        freqs=np.linspace(1e9, 2e9, 3), num_periods=2.0,
        skip_preflight=True, magnitude_channel=channel, return_diagnostics=True)


@pytest.fixture(scope='module', params=[(k, c) for k in ('lumped', 'wire1', 'wire3')
                                       for c in ('wave', 'flux')],
                ids=[f'{k}-{c}' for k in ('lumped', 'wire1', 'wire3') for c in ('wave', 'flux')])
def measured(request):
    import rfx.sparams.mixed as mixed
    kind, channel = request.param
    projection_calls, live_counts = [], []
    project, assemble = mixed._project_passive, mixed._assemble_mixed_power_wave_s

    def recording_projection(s):
        projection_calls.append(True)
        return project(s)

    def recording_assembly(*args, **kwargs):
        live_counts.append(np.asarray(args[5]).copy())
        return assemble(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings(record=True) as seen:
        patch.setattr(mixed, '_project_passive', recording_projection)
        patch.setattr(mixed, '_assemble_mixed_power_wave_s', recording_assembly)
        warnings.simplefilter('always')
        result, raw = _run(kind, channel)
    np.testing.assert_array_equal(live_counts, [[3 if kind == 'wire3' else 1]])
    return kind, channel, result, raw, seen, projection_calls


def _independent_msl_column(raw, n_live):
    """Documented b/a, written independently using the run's raw phasors."""
    run = raw['drive_plan'].index(('msl', 0))
    z = raw['z0_hj_msl'][0]
    voltage, current = raw['v0_msl'][run, 0], raw['i_msl'][run, 0]
    incident = (voltage + z * current) / (2 * np.sqrt(z))
    reflected = (voltage - z * current) / (2 * np.sqrt(z))
    z_cell = 50.0 / n_live
    receive = (raw['v_lw'][run, 0] - z_cell * raw['i_lw'][run, 0]) / (2 * np.sqrt(z_cell))
    return np.asarray([receive / incident, reflected / incident])


def test_returned_entries_and_witnesses(measured):
    kind, channel, result, raw, _, _ = measured
    s = np.asarray(result.S)
    assert np.isnan(s[:, 0]).all()
    assert np.isnan(result.s21_power_witness[:, 0]).all()
    expected = _independent_msl_column(raw, 3 if kind == 'wire3' else 1)
    np.testing.assert_allclose(s[1, 1], expected[1], rtol=1e-6, atol=1e-7)
    if channel == 'wave':
        if kind == 'wire3':
            assert np.isnan(s[0, 1]).all()
        else:
            np.testing.assert_allclose(s[0, 1], expected[0], rtol=1e-6, atol=1e-7)
        assert result.S_wave is None
    else:
        assert np.isnan(result.S_wave[:, 0]).all()
        np.testing.assert_allclose(result.S_wave[1, 1], expected[1], rtol=1e-6, atol=1e-7)
        if kind == 'wire3':
            assert np.isnan(result.S_wave[0, 1]).all()
        else:
            np.testing.assert_allclose(result.S_wave[0, 1], expected[0], rtol=1e-6, atol=1e-7)
        # Flux magnitudes use recorded flux, not the per-cell b/a magnitude.
        # The fixture faces -x, so its incident net flux is minus the +x plane.
        run = raw['drive_plan'].index(('msl', 0))
        net = np.maximum(-raw['plane_msl_flux'][run, 0], 0.0)
        incident_power = net / np.maximum(1 - np.abs(expected[1])**2, 0.05)
        arriving = np.maximum(-raw['box_lw_flux'][run, 0], 0.0)
        magnitude = np.sqrt(np.divide(arriving, incident_power,
                                     out=np.zeros_like(arriving), where=incident_power > 0))
        flux_receive = magnitude * np.exp(1j * np.angle(expected[0]))
        assert np.isfinite(s[0, 1]).all()
        np.testing.assert_allclose(s[0, 1], flux_receive, rtol=1e-6, atol=1e-7)


def test_projection_is_not_formed_and_warning_names_refusal(measured):
    kind, channel, result, _, seen, calls = measured
    refusals = [w for w in seen if 'refused lumped/wire-driven columns' in str(w.message)]
    assert not any(issubclass(w.category, PreDecisionLumpedDiagonalWarning) for w in seen)
    assert not calls
    assert result.S_raw is None and result.passivity_correction is None
    assert len(refusals) == 1 and refusals[0].category is UserWarning
    message = str(refusals[0].message)
    assert "S[:,0] (port 'lw0', index 0)" in message
    assert 'per-cell drive frame' in message and 'whole-port wave later' in message
    assert 'joint passivity projection is not formed' in message
    assert 'uniform run(compute_s_params=True) S driver on a model without MSL ports' in message
    assert 'or use MSL ports only' in message
    if kind == 'wire3':
        matrix = 'S_wave' if channel == 'flux' else 'S'
        assert f"{matrix}[0,1] ('lw0', index 0 <- 'msl_0', index 1)" in message
        assert ('a multi-cell wire receive wave is formed per cell and reads 1/n_live '
                "of the power; use magnitude_channel='flux'") in message
    else:
        assert 'Refused receive entries' not in message
    assert not any('nonfinite_sparams' in str(w.message) for w in seen)


def test_unrelated_nan_is_reported(monkeypatch):
    import rfx.sparams.mixed as mixed
    assemble = mixed._assemble_mixed_power_wave_s

    def diverged_msl(*args, **kwargs):
        s, witness = assemble(*args, **kwargs)
        return s.at[1, 1].set(np.nan), witness

    monkeypatch.setattr(mixed, '_assemble_mixed_power_wave_s', diverged_msl)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        _run('lumped', 'wave')
    assert any('nonfinite_sparams' in str(w.message) and 'NaN' in str(w.message) for w in seen)


def test_partial_checks_only_exempt_the_refused_positions():
    from rfx.sparams._common import _mixed_reciprocity_deviation, _warn_if_nonpassive_smatrix
    s = np.full((3, 3, 2), np.nan + 0j)
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 1] = mask[1, 0] = mask[:, 2] = False
    s[0, 1], s[1, 0] = 0.2, 0.4
    s[:, 2] = 0.0
    s[:, 2, 1] = [0.8, 0.8, 0.0]
    assert _mixed_reciprocity_deviation(s) == ((0, 1), 0.5)
    result = SimpleNamespace(S=s, freqs=[1e9, 2e9], port_names=('a', 'b', 'c'))
    with pytest.warns(UserWarning, match='driven port 2, frequency index 1') as seen:
        _warn_if_nonpassive_smatrix(result, extractor='mixed', refused_positions=mask)
    assert len(seen) == 1
    s[:, 2, 1] = 0.0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        _warn_if_nonpassive_smatrix(result, extractor='mixed', refused_positions=mask, strict=True)
    assert not seen
    for value, name in ((np.nan, 'NaN'), (np.inf, 'Inf')):
        s[2, 2, 1] = value
        with pytest.warns(UserWarning, match=f'contains 1 {name}'):
            _warn_if_nonpassive_smatrix(result, extractor='mixed', refused_positions=mask)
        with pytest.raises(ValueError, match=f'contains 1 {name}'):
            _warn_if_nonpassive_smatrix(result, extractor='mixed', refused_positions=mask, strict=True)


@pytest.mark.parametrize('wire_mode', [False, True])
def test_every_driven_column_is_refused(wire_mode):
    from rfx.sparams._common import _assemble_mixed_power_wave_s, _mixed_flux_magnitude_override
    # Independently chosen Z=4 phasors give a=1, passive b=1, MSL reflection=0.
    v = np.full((3, 2, 2), 2.0 + 0j)
    current = np.full_like(v, -0.5)
    for k in range(2):
        v[k, k], current[k, k] = -2.0, 0.5
    vm = np.full((3, 1, 2), 2.0 + 0j)
    im = np.full_like(vm, -0.5)
    im[2] = 0.5
    plan = [('lw', 0), ('lw', 1), ('msl', 0)]
    wave, witness = _assemble_mixed_power_wave_s(
        v, current, vm, im, [4.0, 4.0], [1, 1], [4.0], wire_mode, plan)
    expected = np.ones((3, 3, 2), dtype=np.complex64)
    expected[:, :2] = np.nan
    expected[2, 2] = 0.0
    np.testing.assert_array_equal(wave, expected)
    assert np.isnan(witness).all()
    box = np.full((3, 2, 2), -1.0)
    box[0, 0] = box[1, 1] = 1.0
    plane = np.ones((3, 1, 2))
    plane[2] = -1.0
    flux, ill, _ = _mixed_flux_magnitude_override(
        wave, box, plane, plan, [-1], 2, refused_columns=(0, 1))
    np.testing.assert_array_equal(flux, expected)
    assert not ill.any()


def test_masks_cover_each_drive_and_only_multi_cell_wire_receivers():
    from rfx.sparams._mixed_checks import _mixed_refusal_masks
    plan = [('lw', 0), ('lw', 1), ('msl', 0), ('msl', 1)]
    flux, wave = _mixed_refusal_masks(4, 2, [1, 3], True, plan)
    np.testing.assert_array_equal(flux, [[1, 1, 0, 0], [1, 1, 0, 0],
                                       [1, 1, 0, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(wave, [[1, 1, 0, 0], [1, 1, 1, 1],
                                       [1, 1, 0, 0], [1, 1, 0, 0]])


def test_flux_refusal_does_not_change_other_nan_diagonals():
    from rfx.sparams._common import _mixed_flux_magnitude_override
    s = np.ones((2, 2, 1), dtype=np.complex64)
    s[0, 0] = s[1, 1] = np.nan
    box = np.array([[[1.0]], [[-1.0]]])
    plane = np.array([[[1.0]], [[-1.0]]])
    args = (s, box, plane, [('lw', 0), ('msl', 0)], [-1], 1)
    # Unrefused NaN normalizations retain main's fallback; the separate
    # finiteness check must still warn or raise on the unexpected diagonal.
    legacy, _, _ = _mixed_flux_magnitude_override(*args)
    np.testing.assert_array_equal(np.asarray(legacy)[[0, 1], [1, 0]], 0.0)
    refused, _, _ = _mixed_flux_magnitude_override(*args, refused_columns=(0,))
    assert np.isnan(refused[1, 0]).all()
    np.testing.assert_array_equal(refused[0, 1], 0.0)
    assert np.isnan(refused[1, 1]).all()
