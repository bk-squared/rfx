#!/usr/bin/env python3
"""Read-only adjudication of four independently named #931 live MSL captures.

Recompute the unchanged physical screens from arrays, verify source/run
provenance, then apply the predeclared all-bin mesh (.02 raw absolute),
confirmation (rtol=.005, atol=.002 projected), and temporal (same tolerance
on raw and projected S) budgets. Never write a golden. Reports and the
companion figure use new output paths only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests._msl_fixture_qualification import qualify_result, quasi_static_line

SOURCE_FILES = ('tests/unit/autodiff/test_msl_sparam_ad.py',
                'tests/_msl_fixture_qualification.py', 'scripts/capture_msl_e2e_golden.py')
ROLES = {'base': (1, 12), 'long': (1, 24), 'refine': (2, 12), 'confirm': (1, 12)}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def fingerprint(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def complex_array(data, name, shape):
    value = np.asarray(data[name], dtype=float)
    require(value.shape == (*shape, 2), f'{name}: invalid complex-pair shape {value.shape}')
    require(np.isfinite(value).all(), f'{name}: nonfinite data')
    return value[..., 0] + 1j * value[..., 1]


def real_array(data, name, shape, *, boolean=False):
    value = np.asarray(data[name])
    require(value.shape == shape, f'{name}: invalid shape {value.shape}, expected {shape}')
    require(value.dtype == np.bool_ if boolean else np.issubdtype(value.dtype, np.number),
            f'{name}: invalid dtype {value.dtype}')
    require(np.isfinite(value).all(), f'{name}: nonfinite data')
    return value


def load_capture(path, role, sha, hashes):
    data = json.loads(path.read_text())
    require(data['source_sha'] == sha, f'{role}: source SHA mismatch')
    require(data['source_hashes'] == hashes, f'{role}: source hashes mismatch pinned commit')
    require(data['fixture_id'] == '931-msl-254um-600um-aligned-v2', f'{role}: wrong fixture')
    require(data['backend'] == 'cpu' and data['dtype'] == 'complex64'
            and data['x64_enabled'] is False, f'{role}: wrong backend/dtype/x64 configuration')
    require((data['refinement'], data['num_periods']) == ROLES[role], f'{role}: mesh/horizon mismatch')
    require(isinstance(data['run_name'], str) and data['run_name'].strip(), f'{role}: unnamed run')
    witness = next((p / 'run_id.txt' for p in path.resolve().parents
                    if (p / 'run_id.txt').is_file()), None)
    require(witness is not None, f'{role}: missing submitter run_id.txt')
    run_id = witness.read_text().strip()
    require(re.fullmatch(r'[1-9][0-9]*', run_id), f'{role}: invalid run ID')
    for declared in (data.get('run_id'), data.get('provenance', {}).get('run_id')):
        require(declared is None or str(declared) == run_id, f'{role}: conflicting embedded run ID')
    commit_file = witness.parent / 'commit.txt'
    require(commit_file.is_file() and commit_file.read_text().strip() == sha,
            f'{role}: missing/mismatched submitter-directory commit witness')
    freqs = real_array(data, 'freqs_hz', (10,))
    require(np.all(np.diff(freqs) > 0), f'{role}: unordered frequencies')
    require(np.all(np.abs(freqs - np.arange(1, 11) * 5e8) <= 512), f'{role}: wrong ten-bin band')
    # Z0 is one frequency-dependent impedance per port; beta is the line fit.
    result = SimpleNamespace(freqs=freqs,
        S=complex_array(data, 'S_projected', (2, 2, 10)),
        S_raw=complex_array(data, 'S_raw', (2, 2, 10)),
        Z0=complex_array(data, 'Z0', (2, 10)), beta=complex_array(data, 'beta', (10,)),
        assembly=data['assembly'], reliable=real_array(data, 'reliable', (2, 10), boolean=True),
        settling_db=real_array(data, 'settling_db', (2,)),
        cond_a=real_array(data, 'cond_a', (10,)),
        beta_railed=real_array(data, 'beta_railed', (2, 10), boolean=True))
    qualification = qualify_result(result)
    # Stored failures are evidence to cross-check, never the authority.
    require(qualification['failures'] == data['qualification']['failures'],
            f'{role}: stored qualification disagrees with recomputation')
    geometry = data['geometry']
    for key, expected in {'substrate_height_m': 254e-6, 'trace_width_m': 600e-6,
                          'eps_r': 3.66, 'domain_m': [14e-3, 3.4e-3, 1.978e-3],
                          's_reference_x_m': [5e-3, 9e-3]}.items():
        require(np.allclose(geometry[key], expected, rtol=0, atol=1e-15), f'{role}: wrong {key}')
    require(isinstance(data['preflight'], list), f'{role}: missing preflight')
    return data, result, {'input': fingerprint(path), 'run_id': run_id,
        'run_id_witness': fingerprint(witness), 'commit_witness': fingerprint(commit_file),
        'run_name': data['run_name'], 'qualification_recomputed': qualification}


def compare(reference, other, *, rtol, atol):
    delta = np.abs(reference - other)
    # np.testing.assert_allclose(base, other) uses the other array in its rtol.
    limit = atol + rtol * np.abs(other)
    passed = delta <= limit
    return {'pass': bool(passed.all()), 'rtol': rtol, 'atol': atol,
            'maximum_abs_delta': float(delta.max()),
            'maximum_budget_ratio': float((delta / limit).max()),
            'per_bin_maximum_abs_delta': delta.max(axis=(0, 1)).tolist(),
            'per_bin_maximum_budget_ratio': (delta / limit).max(axis=(0, 1)).tolist(),
            'per_element_abs_delta': delta.tolist(), 'per_element_limit': limit.tolist(),
            'failing_indices_output_input_frequency': np.argwhere(~passed).tolist()}


def bins(result):
    z0_model, beta_model, _ = quasi_static_line(result.freqs)
    return [{'frequency_hz': float(f), 'S_raw_real': result.S_raw[..., i].real.tolist(),
             'S_raw_imag': result.S_raw[..., i].imag.tolist(),
             'S_raw_abs': np.abs(result.S_raw[..., i]).tolist(),
             'S_projected_real': result.S[..., i].real.tolist(),
             'S_projected_imag': result.S[..., i].imag.tolist(),
             'projection_abs_delta': np.abs(result.S[..., i] - result.S_raw[..., i]).tolist(),
             'Z0_real_ohm': result.Z0[..., i].real.tolist(),
             'Z0_imag_ohm': result.Z0[..., i].imag.tolist(),
             'Z0_model_relative_error': (result.Z0[..., i].real / z0_model - 1).tolist(),
             'beta_real_rad_per_m': float(result.beta[i].real),
             'beta_imag_rad_per_m': float(result.beta[i].imag),
             'beta_model_relative_error': float(result.beta[i].real / beta_model[i] - 1),
             'reliable': result.reliable[..., i].tolist(), 'cond_a': float(result.cond_a[i]),
             'beta_railed': result.beta_railed[..., i].tolist()}
            for i, f in enumerate(result.freqs)]


def figure(results, evidence, comparisons, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), constrained_layout=True)
    for role, result in results.items():
        freq = result.freqs / 1e9
        label = f'{role} (run {evidence[role]["run_id"]})'
        for n, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            ax = axes.flat[n]
            line, = ax.plot(freq, np.abs(result.S_raw[i, j]), '.-', label=label)
            ax.plot(freq, np.abs(result.S[i, j]), '--', color=line.get_color(), alpha=.65)
            ax.set_title(f'S{i+1}{j+1} magnitude: raw solid / projected dashed')
        axes.flat[4].plot(freq, np.angle(result.S_raw[1, 0], deg=True), '.-', label=label)
        axes.flat[5].plot(freq, evidence[role]['qualification_recomputed']['raw_sigma_max'], '.-', label=label)
        for port in (0, 1):
            axes.flat[6].plot(freq, result.Z0[port].real, '.-' if port == 0 else '--', label=f'{label}, port {port+1}')
        axes.flat[7].plot(freq, result.beta.real, '.-', label=label)
        axes.flat[8].plot([1, 2], result.settling_db, 'o-', label=label)
        axes.flat[9].semilogy(freq, result.cond_a, '.-', label=label)
        axes.flat[10].plot(freq, evidence[role]['qualification_recomputed']['raw_s21_phase_error_deg'], '.-', label=label)
    base = results['base']
    z0, beta, s21 = quasi_static_line(base.freqs)
    axes.flat[4].plot(base.freqs / 1e9, np.angle(s21, deg=True), 'k:', label='drawing model')
    axes.flat[6].axhline(z0, color='k', linestyle=':', label='drawing model')
    axes.flat[7].plot(base.freqs / 1e9, beta, 'k:', label='drawing model')
    axes.flat[8].axhline(-40, color='k', linestyle=':', label='-40 dB screen')
    for name, comparison in comparisons.items():
        axes.flat[11].plot(base.freqs / 1e9, comparison['per_bin_maximum_budget_ratio'], '.-', label=name)
    axes.flat[11].axhline(1, color='k', linestyle=':')
    titles = ['Raw S21 phase (degrees)', 'Raw largest singular value', 'Fitted Z0 real (ohm)',
              'Fitted beta real (rad/m)', 'Drive settling (dB)', 'Incident matrix condition',
              'Raw S21 model phase error (degrees)', 'All-bin drift / unchanged budget']
    for ax, title in zip(list(axes.flat)[4:], titles):
        ax.set_title(title)
    for n, ax in enumerate(axes.flat):
        ax.set_xlabel('Driven port' if n == 8 else 'Frequency (GHz)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=6)
    fig.suptitle('#931 aligned 254 um / 600 um live MSL qualification\nFull ten-bin evidence; physics screens remain 3–4.5 GHz')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument('--' + role, required=True, type=Path)
    parser.add_argument('--source-sha', default='b36fc46cdf21d1c57f221e6a057654bcad60bae2')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    plot_path = args.figure or args.output.with_suffix('.png')
    require(not args.output.exists() and not plot_path.exists(), 'Evidence output already exists')
    require(args.output.resolve() != plot_path.resolve(), 'JSON and figure paths must differ')
    require(re.fullmatch(r'[0-9a-f]{40}', args.source_sha), 'Supply full pinned source SHA')
    hashes = {name: hashlib.sha256(subprocess.check_output(
        ['git', 'show', args.source_sha + ':' + name], cwd=ROOT)).hexdigest() for name in SOURCE_FILES}
    require(hashlib.sha256((ROOT / SOURCE_FILES[1]).read_bytes()).hexdigest() == hashes[SOURCE_FILES[1]],
            'Local physical qualification implementation differs from pinned run source')
    data, results, evidence = {}, {}, {}
    for role in ROLES:
        data[role], results[role], evidence[role] = load_capture(getattr(args, role), role, args.source_sha, hashes)
    require(len({v['run_name'] for v in evidence.values()}) == 4, 'Four independent run names required')
    require(len({v['run_id'] for v in evidence.values()}) == 4, 'Four independent VESSL runs required')
    for role in ROLES:
        for key in ('freqs_hz', 'jax_version'):
            require(data[role][key] == data['base'][key], f'{role}: mismatched {key}')
        for key, value in data['base']['geometry'].items():
            other = data[role]['geometry'][key]
            factor = ROLES[role][0]
            if key == 'boundary_spacing_m':
                require(np.allclose(np.asarray(other) * factor, value, rtol=0, atol=1e-15), f'{role}: spacing mismatch')
            elif key == 'dz_profile_m':
                require(np.allclose(other, np.repeat(np.asarray(value) / factor, factor), rtol=0, atol=1e-15), f'{role}: dz mismatch')
            elif key == 'grid_shape':
                # Stored nx/ny/nz count nodes; refinement doubles intervals.
                require(np.array_equal(other, (np.asarray(value) - 1) * factor + 1), f'{role}: grid shape mismatch')
            else:
                require(other == value, f'{role}: drawing/waveform mismatch in {key}')
        evidence[role].update(bins=bins(results[role]), settling_db=results[role].settling_db.tolist())
    comparisons = {
        'base_refine_raw': compare(results['base'].S_raw, results['refine'].S_raw, rtol=0, atol=.02),
        'base_confirm_projected': compare(results['base'].S, results['confirm'].S, rtol=.005, atol=.002),
        'base_long_raw': compare(results['base'].S_raw, results['long'].S_raw, rtol=.005, atol=.002),
        'base_long_projected': compare(results['base'].S, results['long'].S, rtol=.005, atol=.002)}
    failures = [f'{role}: {failure}' for role in ROLES
                for failure in evidence[role]['qualification_recomputed']['failures']]
    failures += [name + ': convergence budget exceeded' for name, value in comparisons.items() if not value['pass']]
    report = {'eligible_for_repin': not failures, 'failures': failures,
              'provenance': {'source_sha': args.source_sha, 'source_hashes': hashes,
                             'adjudicator': fingerprint(Path(__file__)),
                             'run_ids': {k: v['run_id'] for k, v in evidence.items()}},
              'runs': evidence, 'comparisons': comparisons,
              'scope': 'Fixed drawing; unchanged physical, mesh, and confirmation screens; '
                       'predeclared temporal drift budget equals confirmation budget. No golden writes.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure(results, evidence, comparisons, plot_path)
    report['figure'] = fingerprint(plot_path)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'eligible_for_repin': not failures, 'failures': failures,
                      'output': str(args.output), 'figure': str(plot_path)}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
