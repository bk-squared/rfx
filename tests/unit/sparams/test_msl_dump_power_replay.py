"""MSL replay follows recorded wave references and assembly, not lumped roles."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rfx.validation import load_port_vi_dump_npz, replay_smatrix_from_port_vi_dump

LEGACY_CURRENT = ('line current sign normalized so +x and -x MSL ports '
                  'produce positive characteristic impedance on the validated thru-line envelope')


def _write(tmp_path, version=4, *, assembly='multi_drive_solve', missing_references=False):
    refs = np.array([25., 100.])
    root_r = np.sqrt(refs)
    # A prescribed passive reciprocal power S, with BOTH incident port
    # waves nonzero on each record. A per-drive receiving-role shortcut
    # cannot reproduce this complete wave-system oracle.
    expected = np.array([[.2+.1j, .6-.05j], [.6-.05j, -.1+.2j]])
    a = np.array([[1., .3j], [.2-.1j, .8]])
    b = expected @ a
    v = root_r[:, None] * (a+b)
    i = (a-b) / root_r[:, None]
    order = [1, 0]
    v, i = v[:, order].T[..., None], i[:, order].T[..., None]
    if version == 3:
        expected = root_r[:, None] * expected / root_r[None, :]
    meta = dict(schema='rfx.msl_nprobe_dump', schema_version=version,
                current_convention=LEGACY_CURRENT if version == 3 else 'native_msl_loop_current',
                production_smatrix_assembly=assembly,
                port_definitions=[{'impedance_ohm': 50.}, {'impedance_ohm': 50.}])
    if version == 4:
        meta['s_wave_convention'] = 'power'
    if not missing_references:
        meta['s_reference_impedances_ohm'] = refs.tolist()
    path = tmp_path / f'dump-{version}.npz'
    np.savez(path, metadata_json=json.dumps(meta), raw_v=np.repeat(v[:, :, None, :], 3, axis=2),
             raw_i1=i, raw_z0=np.full((2, 2, 1), 999.), freqs_hz=[2e9],
             port_names=['left', 'right'], driven_port_indices=order,
             production_smatrix=expected[..., None])
    return path, expected


@pytest.mark.parametrize('version', [3, 4])
def test_msl_uses_actual_references_and_full_drive_solve(tmp_path, version):
    path, expected = _write(tmp_path, version)
    dump = load_port_vi_dump_npz(path)
    np.testing.assert_array_equal(dump.port_impedances, [25., 100.])
    result = replay_smatrix_from_port_vi_dump(dump)
    np.testing.assert_allclose(result.s_params[..., 0], expected, rtol=0, atol=1e-14)


def test_source_load_or_fitted_impedance_cannot_substitute_for_missing_reference(tmp_path):
    path, _ = _write(tmp_path, 3, missing_references=True)
    with pytest.raises(ValueError, match='lacks actual S reference impedances'):
        load_port_vi_dump_npz(path)


@pytest.mark.parametrize('key,value', [
    ('schema_version', 2), ('s_wave_convention', 'voltage'),
    ('current_convention', 'positive_out_of_dut'),
    ('production_smatrix_assembly', 'unknown'),
])
def test_replay_refuses_ambiguous_or_conflicting_msl_contracts(tmp_path, key, value):
    path, _ = _write(tmp_path)
    dump = load_port_vi_dump_npz(path)
    dump.metadata[key] = value
    with pytest.raises(ValueError):
        replay_smatrix_from_port_vi_dump(dump)


def test_declared_fallback_replays_its_power_ratios_without_lumped_receive_sign(tmp_path):
    path, _ = _write(tmp_path, assembly='single_ratio_fallback')
    dump = load_port_vi_dump_npz(path)
    z = np.array([25., 100.])[None, :, None]
    a = (dump.voltages + z*dump.currents) / (2*np.sqrt(z))
    b = (dump.voltages - z*dump.currents) / (2*np.sqrt(z))
    expected = np.empty((2, 2, 1), complex)
    for record, drive in enumerate(dump.driven_port_indices):
        expected[:, drive] = b[record] / a[record, drive]
    np.testing.assert_allclose(replay_smatrix_from_port_vi_dump(dump).s_params,
                               expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize('version', [3, 4])
def test_fallback_preserves_the_producers_wave_units_for_its_legacy_floor(tmp_path, version):
    path, _ = _write(tmp_path, version, assembly='single_ratio_fallback')
    dump = load_port_vi_dump_npz(path)
    dump.voltages[:] = 0.5e-30
    dump.currents[:] = 0
    # Voltage waves are 0.25e-30. v4 relative row scales are [1, 0.5].
    # Diagonals keep the voltage-unit guard; receiving rows use the working
    # wave guard. Removing that floor incorrectly makes every entry one.
    expected = np.full((2, 2), 0.2)
    if version == 4:
        expected[0, 1] = 2 / 9
        expected[1, 0] = 0.1
    np.testing.assert_allclose(replay_smatrix_from_port_vi_dump(dump).s_params[..., 0],
                               expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize('refs', [[0, 100], [-25, 100], [np.inf, 100],
                                  [np.nan, 100], [25+1j, 100], [25]])
def test_replay_rejects_invalid_actual_references(tmp_path, refs):
    path, _ = _write(tmp_path)
    dump = load_port_vi_dump_npz(path)
    dump.metadata['s_reference_impedances_ohm'] = refs
    with pytest.raises(ValueError, match='one finite positive real impedance per port'):
        replay_smatrix_from_port_vi_dump(dump)


@pytest.mark.parametrize('version', [3, 4])
def test_spatial_audit_uses_power_metric_even_for_unequal_legacy_voltage_s(tmp_path, version):
    path, expected = _write(tmp_path, version)
    with np.load(path, allow_pickle=False) as saved:
        payload = dict(saved)
    meta = json.loads(str(payload['metadata_json']))
    meta['current_spatial_alignment'] = 'linear_bracketing_H_to_E_node'
    meta['current_plane_stencils'] = [dict(weights=[.4, .6], voltage_coordinate=x)
                                      for x in (0., 1.)]
    payload.update(metadata_json=json.dumps(meta), raw_i1_left=payload['raw_i1'],
                   raw_i1_same_index=payload['raw_i1'])
    np.savez(path, **payload)
    script = (Path(__file__).resolve().parents[3]
              / 'docs/research_notes/issue726/collocation/audit_current_impact.py')
    spec = importlib.util.spec_from_file_location('msl_current_audit', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report, arrays = module.audit(path)
    np.testing.assert_allclose(arrays['centered_s'][..., 0], expected, atol=1e-14)
    power_s = np.array([[.2+.1j, .6-.05j], [.6-.05j, -.1+.2j]])
    assert report['centered_max_column_power'] == pytest.approx(.4125)
    expected_gain = np.linalg.eigvalsh(power_s.conj().T @ power_s).max()
    assert report['centered_max_coherent_power_gain'] == pytest.approx(expected_gain)
    assert report['same_index_max_coherent_power_gain'] == pytest.approx(expected_gain)


@pytest.mark.parametrize('case', ['coupon/call-0-raw-vi.npz', 'ad/call-2-raw-vi.npz'])
def test_recorded_v3_real_fields_replay_without_guessing_reference_or_receiving_role(case):
    root = Path(__file__).resolve().parents[3]
    path = root / 'docs/research_notes/issue726/collocation/gpu-369367260604' / case
    dump = load_port_vi_dump_npz(path)
    result = replay_smatrix_from_port_vi_dump(dump)
    np.testing.assert_allclose(result.s_params, dump.production_smatrix, rtol=1e-6, atol=3e-7)
