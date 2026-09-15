#!/usr/bin/env python3
"""Inspect a named V173A composition capture against its retained numerical pin.

No solver or Harminv rerun: compare saved raw traces and independently recompute
FFT bins and ringdown-tail arithmetic. Pin tolerances remain frequency 1e-6
relative and spectrum 0.1 dB absolute. This is no Q/convergence/matching claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    fresh_json = args.capture_dir / 'capture.json'
    fresh_npz = args.capture_dir / 'capture.npz'
    prior_npz = ROOT / 'docs/design_notes/931_migration/fixture-repair-evidence/v173a-traces.npz'
    pin_json = ROOT / 'tests/data/v173a_aligned_composition.json'
    data, pin = (json.loads(path.read_text()) for path in (fresh_json, pin_json))
    run_witness = args.capture_dir / 'run_id.txt'
    run_id = run_witness.read_text().strip()
    assert re.fullmatch(r'[1-9][0-9]*', run_id), 'Numeric submitter run ID required'
    assert re.fullmatch(r'[0-9a-f]{40}', data['sha'])
    assert data['sha'] == (args.capture_dir / 'commit.txt').read_text().strip()
    for key in ('fixture', 'n_steps', 'ringdown_start_s', 'ringdown_start_sample', 'dt', 'harness_sha256'):
        assert data[key] == pin[key], key
    with np.load(fresh_npz) as new, np.load(prior_npz) as old:
        trace, dt = new['trace'], float(new['dt'])
        previous = old['trace']
        assert dt == float(old['dt']) == data['dt']
        assert trace.shape == previous.shape == (data['n_steps'],)
        assert np.isfinite(trace).all() and np.any(trace != 0)
        start = data['ringdown_start_sample']
        ringdown = trace[start:] - np.mean(trace[start:])
        frequencies = np.fft.rfftfreq(len(trace), dt)
        amplitude = abs(np.fft.rfft(trace * np.hanning(len(trace))))
        band = (frequencies >= 1.5e9) & (frequencies <= 3.5e9)
        frequencies, band_amplitude = frequencies[band], amplitude[band]
        spectrum = 20 * np.log10(band_amplitude + float(np.max(band_amplitude)) * 1e-6)
        # Archive differences are reproduction evidence, never extra pin gates.
        archive_comparisons = {}
        for name, fresh, previous_array in (
                ('ringdown', ringdown, old['ringdown']),
                ('fft_band_frequencies_hz', frequencies, old['fft_band_frequencies_hz']),
                ('fft_band_log_amplitude_db', spectrum, old['fft_band_log_amplitude_db'])):
            assert fresh.shape == previous_array.shape, name
            delta = fresh.astype(np.float64) - previous_array.astype(np.float64)
            archive_comparisons[name] = {
                'array_equal': bool(np.array_equal(fresh, previous_array)),
                'max_abs_delta': float(np.max(abs(delta))),
                'deltas': delta.tolist() if name != 'ringdown' else None,
            }
    full_tail = float(np.max(abs(trace[-len(trace)//10:])) / np.max(abs(trace)))
    ring_tail = float(np.max(abs(ringdown[-len(ringdown)//10:])) / np.max(abs(ringdown)))
    assert full_tail == data['tail_to_peak'] and ring_tail == data['ringdown_tail_to_peak']
    minimum_index = int(np.argmin(spectrum))
    assert abs(float(spectrum[minimum_index]) - data['probe_spectrum_min_db']) < 1e-10
    assert float(frequencies[minimum_index]) == data['probe_spectrum_min_f_hz']
    freq_delta = abs(data['dominant_probe_mode_hz'] - pin['dominant_probe_mode_hz'])
    spectrum_delta = abs(data['probe_spectrum_min_db'] - pin['probe_spectrum_min_db'])
    raw_delta = trace.astype(np.float64) - previous.astype(np.float64)
    report = {
        'provenance': {'run_id': run_id, 'source_sha': data['sha'],
                       'inputs': [fingerprint(p) for p in (fresh_json, fresh_npz, prior_npz, pin_json, run_witness)],
                       'script': fingerprint(Path(__file__))},
        'scope': 'Aligned synthetic composition drift sentinel only; no physical convergence, Q, mode identification, or S11 claim.',
        'raw_trace': {'array_equal': bool(np.array_equal(trace, previous)),
                      'max_abs_delta': float(np.max(abs(raw_delta))),
                      'relative_l2_delta': float(np.linalg.norm(raw_delta) / np.linalg.norm(previous.astype(np.float64))),
                      'samples': len(trace), 'dt_s': dt},
        'archive_comparisons_not_acceptance_gates': archive_comparisons,
        'fresh_capture_consistency': {
            'full_trace_tail_to_peak': full_tail, 'ringdown_tail_to_peak': ring_tail,
            'probe_spectrum_min_db': float(spectrum[minimum_index]),
            'probe_spectrum_min_f_hz': float(frequencies[minimum_index]),
            'checked_against': 'fresh capture.json, independently recomputed from fresh capture.npz'},
        'frequency_pin': {'reference_hz': pin['dominant_probe_mode_hz'],
                          'fresh_hz': data['dominant_probe_mode_hz'], 'absolute_delta_hz': freq_delta,
                          'relative_tolerance': 1e-6,
                          'allowed_absolute_delta_hz': 1e-6 * pin['dominant_probe_mode_hz'],
                          'passed': freq_delta <= 1e-6 * pin['dominant_probe_mode_hz']},
        'spectrum_pin': {'reference_db': pin['probe_spectrum_min_db'], 'fresh_db': data['probe_spectrum_min_db'],
                         'absolute_delta_db': spectrum_delta, 'absolute_tolerance_db': 0.1,
                         'passed': spectrum_delta <= 0.1},
        'ringdown_start_sample': start, 'full_trace_peak': float(np.max(abs(trace))),
        'ringdown_peak': float(np.max(abs(ringdown))),
        'full_trace_tail_to_peak': full_tail, 'ringdown_tail_to_peak': ring_tail,
        'fft_band_frequencies_hz': frequencies.tolist(), 'fft_band_log_amplitude_db': spectrum.tolist(),
        'fft_minimum_is_uppermost_band_bin': minimum_index == len(frequencies) - 1,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    time_ns = np.arange(len(trace)) * dt * 1e9
    axes[0, 0].plot(time_ns, previous, color='0.7', label='Committed capture')
    axes[0, 0].plot(time_ns, trace, '--', label=f'Run {run_id}')
    axes[0, 0].axvline(start * dt * 1e9, color='black', linewidth=0.8, label='Ringdown start')
    axes[0, 0].set(xlabel='Time (ns)', ylabel='Full probe Ez trace')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(time_ns[start:], ringdown)
    axes[0, 1].set(xlabel='Time (ns)', ylabel='DC-subtracted ringdown Ez')
    axes[0, 1].text(0.03, 0.06, f'Tail / ringdown peak = {ring_tail:.6f}', transform=axes[0, 1].transAxes)
    axes[1, 0].plot(frequencies / 1e9, spectrum, 'o-')
    axes[1, 0].set(xlabel='Frequency (GHz)', ylabel='Unnormalized probe FFT amplitude (dB)')
    axes[1, 1].plot(time_ns, raw_delta)
    axes[1, 1].set(xlabel='Time (ns)', ylabel='Fresh minus committed raw trace')
    for axis in axes.flat: axis.grid(alpha=0.25)
    figure = args.output_dir / 'v173a-live-ringdown-spectrum.png'
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    report['figure'] = fingerprint(figure)
    output = args.output_dir / 'v173a-live-trace-inspection.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    assert report['frequency_pin']['passed'] and report['spectrum_pin']['passed']


if __name__ == '__main__':
    main()
