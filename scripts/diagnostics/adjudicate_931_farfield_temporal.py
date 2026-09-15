#!/usr/bin/env python3
"""Offline temporal qualification of the repaired #931 far-field fixture.

Compare independent 600/1200-step, current-operator captures. Predeclared gates:
for EACH mesh, absolute-power relative change and combined complex angular
E_theta/E_phi relative L2 change are strictly below 1%, using the 1200-step
record as reference. Retain local < 5% and local < scalar/2 at BOTH horizons.
Normalized-pattern changes are reported separately, never substituted for
absolute power or complex-field convergence. No phase or amplitude alignment.
The submitter must retain a numeric ancestor run_id.txt for each capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

GEOMETRY_KEYS = (
    'shape', 'dt', 'occupied_cells', 'node_vs_center_occupancy_xor',
    'edge_counts', 'wall_planes_m', 'source_index', 'source_node_m',
    'source_dual_volume_m3', 'ntff_indices', 'ntff_coordinates_m',
)
TEMPORAL_LIMIT = 0.01


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    return {'path': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run_identity(directory, source_sha):
    for parent in (directory.resolve(), *directory.resolve().parents):
        witness = parent / 'run_id.txt'
        if witness.is_file():
            run_id = witness.read_text().strip()
            require(re.fullmatch(r'[1-9][0-9]*', run_id), f'Invalid run identity: {witness}')
            commit = parent / 'commit.txt'
            require(commit.is_file(), f'Missing commit witness beside {witness}')
            require(commit.read_text().strip() == source_sha,
                    f'Commit witness differs from capture source: {commit}')
            return {'run_id': run_id, 'witness': fingerprint(witness),
                    'commit_witness': fingerprint(commit)}
    raise ValueError(f'No submitter run_id.txt found for {directory}')


def load_capture(directory, steps):
    json_path = directory / 'farfield-attribution.json'
    npz_path = directory / 'farfield-attribution.npz'
    data = json.loads(json_path.read_text())
    require(data['fixture'] == 'repaired', 'Expected the repaired fixture')
    require(data['steps'] == steps, f'Expected {steps} steps')
    require(re.fullmatch(r'[0-9a-f]{40}', data['source']), 'Invalid source SHA')
    require(set(data['cases']) == {'current_uniform', 'current_graded'},
            'Expected exactly current uniform and graded captures')
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, array in arrays.items():
        require(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all(),
                f'Nonfinite or nonnumeric array: {key}')
    return data, arrays, {'identity': run_identity(directory, data['source']),
                          'json': fingerprint(json_path), 'npz': fingerprint(npz_path)}


def relative_l2(reference, candidate):
    require(reference.shape == candidate.shape, 'Mismatched array dimensions')
    # Accumulate in float64/complex128 so a weak polarization cannot underflow.
    dtype = np.complex128 if np.iscomplexobj(reference) else np.float64
    reference, candidate = reference.astype(dtype), candidate.astype(dtype)
    denominator = float(np.linalg.norm(reference.ravel()))
    delta = float(np.linalg.norm((candidate - reference).ravel()))
    # An identically zero polarization can be physically legitimate. Its relative
    # norm is undefined, not a manufactured pass; the combined field must be nonzero.
    return {'reference_l2': denominator, 'absolute_l2_delta': delta,
            'relative_l2_delta': delta / denominator if denominator > 0 else None}


def integrated_power(u, theta, phi):
    return float(np.sum(u * np.sin(theta)[None, :, None]
                        * np.gradient(theta)[None, :, None]
                        * np.gradient(phi)[None, None, :]))


def validate_fields(data, arrays, mesh):
    key = 'current_' + mesh
    record = data['cases'][key]
    theta, phi = arrays['theta'], arrays['phi']
    require(theta.ndim == phi.ndim == 1 and theta.size >= 2 and phi.size >= 2,
            'Invalid angular axes')
    require(np.all(np.diff(theta) > 0) and np.all(np.diff(phi) > 0),
            'Angular axes must increase')
    et, ep, u = (arrays[key + suffix] for suffix in ('_E_theta', '_E_phi', '_U'))
    require(et.shape == ep.shape == u.shape == (1, theta.size, phi.size),
            f'{key}: expected one frequency with angular shape (1, theta, phi)')
    computed_u = np.abs(et) ** 2 + np.abs(ep) ** 2
    np.testing.assert_allclose(u, computed_u, rtol=2e-6, atol=0,
                               err_msg=f'{key}: U differs from complex fields')
    power = integrated_power(u, theta, phi)
    require(np.isfinite(power) and power > 0, f'{key}: invalid absolute power')
    np.testing.assert_allclose(power, record['power'], rtol=2e-6, atol=0,
                               err_msg=f'{key}: archived power arithmetic mismatch')
    times, waveform = (arrays[key + suffix] for suffix in ('_source_times', '_source_waveform'))
    require(times.shape == waveform.shape == (data['steps'],), f'{key}: invalid source history')
    require(np.any(waveform != 0), f'{key}: zero excitation')
    require(np.all(np.diff(times) > 0), f'{key}: non-increasing source times')
    require(record['node_vs_center_occupancy_xor'] == 0, f'{key}: occupancy discrepancy')
    return et, ep, u, power


def adjudicate(short_dir, long_dir, output_dir, expected_sha=None):
    captures = {steps: load_capture(directory, steps)
                for steps, directory in ((600, short_dir), (1200, long_dir))}
    short, sa, sp = captures[600]
    long, la, lp = captures[1200]
    require(short['source'] == long['source'], 'Captures use different source SHAs')
    if expected_sha is not None:
        require(long['source'] == expected_sha, 'Capture SHA differs from pinned expected SHA')
    for key in ('theta', 'phi'):
        np.testing.assert_array_equal(sa[key], la[key], err_msg=f'{key}: angular grid differs')
    fields = {steps: {mesh: validate_fields(data, arrays, mesh) for mesh in ('uniform', 'graded')}
              for steps, (data, arrays, _) in captures.items()}
    report = {
        'provenance': {'source_sha': long['source'], 'captures': {'600': sp, '1200': lp},
                       'adjudicator': fingerprint(Path(__file__))},
        'criteria': {'temporal_absolute_power_relative_delta_lt': TEMPORAL_LIMIT,
                     'temporal_combined_complex_angular_relative_l2_lt': TEMPORAL_LIMIT,
                     'temporal_reference_steps': 1200,
                     'original_local_error_lt': 0.05,
                     'original_local_error_lt_scalar_error_times': 0.5,
                     'strict_inequalities': True, 'phase_or_amplitude_alignment': False},
        'meshes': {}, 'horizons': {},
    }
    for mesh in ('uniform', 'graded'):
        key = 'current_' + mesh
        for field in GEOMETRY_KEYS:
            require(short['cases'][key][field] == long['cases'][key][field],
                    f'{key}: geometry changed: {field}')
        for suffix in ('_source_times', '_source_waveform'):
            np.testing.assert_array_equal(sa[key + suffix], la[key + suffix][:600],
                                          err_msg=f'{key}: source prefix differs: {suffix}')
        set_, sep, su, ps = fields[600][mesh]
        let, lep, lu, pl = fields[1200][mesh]
        combined = relative_l2(np.stack((let, lep)), np.stack((set_, sep)))
        require(combined['reference_l2'] > 0, f'{key}: zero complex angular norm')
        power_delta = abs(ps - pl) / pl
        peak_short, peak_long = su / su.max(), lu / lu.max()
        normalized = relative_l2(lu / pl, su / ps)
        record = {
            'geometry_equal': True, 'geometry': long['cases'][key],
            'source_waveform_and_time_prefix_equal': True,
            'source_tail_amplitude': {'600': float(sa[key + '_source_waveform'][-1]),
                                      '1200': float(la[key + '_source_waveform'][-1])},
            'power': {'600': ps, '1200': pl, 'absolute_delta': abs(ps - pl),
                      'relative_delta': power_delta, 'passed': power_delta < TEMPORAL_LIMIT},
            'combined_complex_angular': {**combined,
                'passed': combined['relative_l2_delta'] < TEMPORAL_LIMIT},
            'E_theta': relative_l2(let, set_), 'E_phi': relative_l2(lep, sep),
            'power_normalized_pattern': normalized,
            'peak_normalized_pattern_max_abs_delta': float(np.max(abs(peak_short - peak_long))),
        }
        record['polarization_reference_norm_fraction'] = {
            name: record[name]['reference_l2'] / combined['reference_l2']
            for name in ('E_theta', 'E_phi')}
        record['temporal_qualified'] = record['power']['passed'] and record['combined_complex_angular']['passed']
        report['meshes'][mesh] = record
    for steps, (data, arrays, _) in captures.items():
        uniform = fields[steps]['uniform'][3]
        graded = fields[steps]['graded'][3]
        scalar_power = float(data['cases']['current_graded']['scalar_inplane_power'])
        require(np.isfinite(scalar_power) and scalar_power > 0, 'Invalid scalar-inplane power')
        local, scalar = abs(graded - uniform) / uniform, abs(scalar_power - uniform) / uniform
        np.testing.assert_allclose([local, scalar],
                                   [data['current_errors']['local'], data['current_errors']['scalar']],
                                   rtol=2e-5, atol=1e-9, err_msg='Archived gate arithmetic mismatch')
        report['horizons'][str(steps)] = {
            'local': local, 'scalar': scalar,
            'unchanged_5pct_gate_passed': local < 0.05,
            'local_cell_discriminator_passed': local < scalar / 2,
            'cross_mesh_patterns': {k: data['current_errors'][k] for k in (
                'peak_normalized_pattern_max_abs', 'power_normalized_pattern_relative_l2')},
        }
    report['temporal_qualified'] = all(item['temporal_qualified'] for item in report['meshes'].values())
    report['original_gates_passed_at_both_horizons'] = all(
        item['unchanged_5pct_gate_passed'] and item['local_cell_discriminator_passed']
        for item in report['horizons'].values())
    report['qualification_passed'] = (report['temporal_qualified']
                                      and report['original_gates_passed_at_both_horizons'])
    report['qualification_scope'] = (
        '600 versus 1200 steps at the retained geometry and 30 GHz only; '
        'combined complex angular field, not independent polarization or pointwise convergence')
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_comparison(fields, sa['theta'], output_dir)
    report['figure'] = fingerprint(figure)
    destination = output_dir / 'farfield-temporal-adjudication.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def plot_comparison(fields, theta, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    degrees = np.degrees(theta)
    for row, mesh in enumerate(('uniform', 'graded')):
        for steps, style in ((600, '-'), (1200, '--')):
            et, ep, u, power = fields[steps][mesh]
            # One curve per theta, phi-averaged, while L2 qualification uses all angles.
            axes[row, 0].plot(degrees, np.mean(u[0], axis=1), style, label=f'{steps} steps')
            axes[row, 1].plot(degrees, np.mean(u[0] / power, axis=1), style,
                              label=f'{steps} steps')
        let, lep = fields[1200][mesh][:2]
        set_, sep = fields[600][mesh][:2]
        delta = np.sqrt(np.mean(abs(let[0] - set_[0]) ** 2 + abs(lep[0] - sep[0]) ** 2, axis=1))
        reference = np.sqrt(np.mean(abs(let[0]) ** 2 + abs(lep[0]) ** 2, axis=1))
        peak = float(np.max(reference))
        axes[row, 2].plot(degrees, delta / peak, label='Complex delta / peak field RMS')
        axes[row, 0].set_ylabel(f'{mesh}: angular intensity (absolute)')
        axes[row, 1].set_ylabel('Power-normalized angular intensity')
        axes[row, 2].set_ylabel('Complex angular change')
        for axis in axes[row]:
            axis.set_xlabel('Theta (degrees; phi-averaged)')
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    destination = output_dir / 'farfield-temporal-power-angular.png'
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps-600', type=Path, required=True)
    parser.add_argument('--steps-1200', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-sha')
    args = parser.parse_args()
    report = adjudicate(args.steps_600, args.steps_1200, args.output_dir, args.expected_sha)
    print(json.dumps({key: report[key] for key in ('provenance', 'temporal_qualified',
                     'original_gates_passed_at_both_horizons', 'qualification_passed')}, indent=2))
    raise SystemExit(0 if report['qualification_passed'] else 1)


if __name__ == '__main__':
    main()
