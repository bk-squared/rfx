#!/usr/bin/env python3
"""Offline causal audit of the repaired short; never changes acceptance gates.

Input JSONs come from slow_931_advisory_attribution.py --arm current, with
matching *_records.npz files. Numeric run_id.txt is supplied by the submitter.
The committed historical pair establishes exact causal identity; fresh runs
are independently checked and reported, without assuming CPU bitwise identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/design_notes/931_migration/fixture-repair-evidence'
HISTORY = ROOT / 'docs/design_notes/931_migration/slow-consumer-evidence'
TRACE_NAMES = ('v_probe_t', 'v_ref_t', 'i_probe_t', 'i_ref_t')


def load(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def matrix(data):
    value = np.asarray(data['s_real']) + 1j * np.asarray(data['s_imag'])
    assert value.shape == (2, 2, 6) and np.isfinite(value).all()
    return value


def records_path(path):
    target = path.with_name(path.stem + '_records.npz')
    assert target.is_file(), target
    return target


def run_id(path, explicit):
    for parent in path.resolve().parents:
        witness = parent / 'run_id.txt'
        if witness.is_file():
            value = witness.read_text().strip()
            if explicit is not None:
                assert value == explicit, (witness, value, explicit)
            assert re.fullmatch(r'[1-9][0-9]*', value), (witness, value)
            return value, fingerprint(witness)
    assert explicit is not None and re.fullmatch(r'[1-9][0-9]*', explicit), (
        'Supply the submitter-recorded numeric --run-id or retain ancestor run_id.txt')
    return explicit, {'source': '--run-id supplied by submitter'}


def comparison(a, b):
    assert a.shape == b.shape
    delta = np.abs(a - b)
    peak = float(np.max(np.abs(a)))
    return {'exactly_equal': bool(np.array_equal(a, b)),
            'max_abs_delta': float(delta.max()),
            'max_abs_delta_over_reference_peak': float(delta.max() / peak) if peak else None}


def bins(data):
    s = matrix(data)
    power = np.sum(np.abs(s) ** 2, axis=0)
    # This checks capture arithmetic, not a physical-accuracy acceptance gate.
    np.testing.assert_allclose(power, data['column_power'], rtol=1e-6, atol=1e-7)
    return [{'frequency_hz': f,
             's_real': s[:, :, i].real.tolist(), 's_imag': s[:, :, i].imag.tolist(),
             's_abs': np.abs(s[:, :, i]).tolist(),
             'left_reflection_power': float(abs(s[0, 0, i]) ** 2),
             'left_reported_transmission_power': float(abs(s[1, 0, i]) ** 2),
             'right_reflection_power': float(abs(s[1, 1, i]) ** 2),
             'right_reported_transmission_power': float(abs(s[0, 1, i]) ** 2),
             'column_power': power[:, i].tolist()}
            for i, f in enumerate(data['frequencies_hz'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--fine', type=Path, required=True)
    parser.add_argument('--run-id', help='Submitter-recorded run ID, if run_id.txt is unavailable')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    live_paths = {'coarse': args.coarse, 'fine': args.fine}
    live = {name: load(path) for name, path in live_paths.items()}
    identities = {name: run_id(path, args.run_id) for name, path in live_paths.items()}
    assert identities['coarse'][0] == identities['fine'][0]
    assert live['coarse']['source_sha'] == live['fine']['source_sha']
    assert re.fullmatch(r'[0-9a-f]{40}', live['coarse']['source_sha'])

    hist_paths = {name: HISTORY / ('advisory_' + name + '.json')
                  for name in ('current', 'node_volume', 'legacy_sigma')}
    hist = {name: load(path) for name, path in hist_paths.items()}
    committed_paths = {name: EVIDENCE / ('short-' + name + '.json') for name in live}
    committed = {name: load(path) for name, path in committed_paths.items()}
    # The bridge changes occupancy while holding compiled inputs constant.
    for name in ('node_volume', 'legacy_sigma'):
        for key in ('source_sha', 'fixture_sha256', 'material_input_sha256',
                    'port_input_sha256', 'grid_shape', 'dt', 'n_steps'):
            assert hist[name][key] == hist['current'][key], (name, key)
        assert 2.25 < hist[name]['max_column_power'] <= 3.0
    old_s = matrix(hist['current'])
    repaired_s = matrix(committed['coarse'])
    causal_identity = {'s11': comparison(old_s[0, 0], repaired_s[0, 0])}
    assert causal_identity['s11']['exactly_equal']
    old_records_path = records_path(hist_paths['current'])
    committed_records_path = EVIDENCE / 'short-coarse-traces.npz'
    with np.load(old_records_path) as old_records, np.load(committed_records_path) as new_records:
        for name in TRACE_NAMES:
            key = 'drive0_port0_' + name
            causal_identity[key] = comparison(old_records[key], new_records[key])
            assert causal_identity[key]['exactly_equal'], key

    report = {
        'provenance': {'run_id': identities['coarse'][0],
                       'source_sha': live['coarse']['source_sha'],
                       'adjudicator': fingerprint(Path(__file__)),
                       'run_id_witnesses': {k: v[1] for k, v in identities.items()}},
        'conclusion': 'historical interval encoded invalid outgoing-power sum',
        'original_gates': {'coarse': '2.25 < maximum column power <= 3.0',
                           'fine': 'maximum column power <= 2.25',
                           'settling': 'each driven run <= -40 dB'},
        'committed_causal_identity': causal_identity,
        'causal_identity_inputs': [fingerprint(old_records_path), fingerprint(committed_records_path)],
        'historical': {name: {'input': fingerprint(hist_paths[name]), 'bins': bins(data),
                              'port_planes_m': data['port_planes_m'],
                              'realization_census': {k: data[k] for k in ('current', 'node_volume', 'legacy_sigma')}}
                       for name, data in hist.items()},
        'live': {},
        'unqualified': [
            'Coarse absolute reflection accuracy: original record fails -40 dB settling.',
            'Absorber depth is below the documented floor in both retained controls.',
            'Fine control changes band, duration, and physical absorber thickness; it is not mesh convergence.',
            'The unchanged historical coarse advisory test remains red; no qualified replacement witness is asserted.',
        ],
    }
    for name, data in live.items():
        assert data['arm'] == 'current' and data['fine'] == (name == 'fine')
        for key in ('fixture_sha256', 'frequencies_hz', 'grid_shape', 'dx', 'dt',
                    'n_steps', 'num_periods', 'port_planes_m', 'declared_short_faces_m'):
            assert data[key] == committed[name][key], (name, key)
        s = matrix(data)
        assert np.all(s[[0, 1], [1, 0], :] == 0), 'Transmission through closing plate'
        assert np.all(np.abs(s[[0, 1], [0, 1], :]) > 0), 'Unobserved driven reflection'
        assert all(data['driven_incident_nonzero']), 'Unobserved driven incident wave'
        comparisons = {'s_matrix': comparison(matrix(committed[name]), s)}
        trace_path = records_path(live_paths[name])
        archive_path = EVIDENCE / ('short-' + name + '-traces.npz')
        with np.load(trace_path) as traces, np.load(archive_path) as archive:
            for drive in (0, 1):
                for port in (0, 1):
                    for field in TRACE_NAMES:
                        key = f'drive{drive}_port{port}_{field}'
                        record = traces[key]
                        assert np.isfinite(record).all(), key
                        assert np.any(record != 0) if drive == port else np.all(record == 0), key
                        comparisons[key] = comparison(archive[key], record)
        # Live reproduction deltas are evidence, not a widened historical gate.
        # Exactness is asserted only for the original controlled causal pair above.
        measured_max = float(np.sum(np.abs(s) ** 2, axis=0).max())
        original_pass = 2.25 < measured_max <= 3.0 if name == 'coarse' else measured_max <= 2.25
        report['live'][name] = {
            'input': fingerprint(live_paths[name]), 'records': fingerprint(trace_path),
            'committed_reference': fingerprint(committed_paths[name]),
            'committed_records': fingerprint(archive_path),
            'bins': bins(data), 'maximum_column_power_recomputed': measured_max,
            'maximum_column_power_capture': data['max_column_power'],
            'original_interval_pass': original_pass,
            'soft_column_power_advisory': any('ADVISORY' in w and 'max column power' in w
                                              for w in data['warnings']),
            'settling_db': data['settling_db'],
            'settling_pass': bool(np.all(np.asarray(data['settling_db']) <= -40)),
            'port_planes_m': data['port_planes_m'],
            'driven_incident_nonzero': data['driven_incident_nonzero'],
            'wave_and_record_summaries': data['records_summary'],
            'reproduction_comparisons': comparisons, 'warnings': data['warnings'],
        }
    # These are declared attribution falsifiers, not replacement test gates.
    assert not report['live']['coarse']['original_interval_pass'], 'Historical interval unexpectedly restored'
    assert report['live']['fine']['original_interval_pass'], 'Fine control unexpectedly entered the gap'
    assert not report['live']['coarse']['settling_pass'], 'Update the stated coarse qualification limitation'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'run_id': identities['coarse'][0], 'output': str(args.output),
                      'conclusion': report['conclusion'],
                      'live': {k: {n: v[n] for n in ('maximum_column_power_recomputed',
                                                       'original_interval_pass', 'settling_pass')}
                               for k, v in report['live'].items()}}, indent=2))


if __name__ == '__main__':
    main()
