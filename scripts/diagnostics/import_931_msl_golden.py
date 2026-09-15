#!/usr/bin/env python3
"""Offline transfer of the qualified #931 base capture; never runs a solve.

Requires the four-run adjudicator's passing report and unchanged input files.
Writes complex64 S_projected plus a provenance manifest, then reloads the pin
and checks both base and independent confirmation at the existing tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adjudicate_931_msl_live import ROOT, ROLES, compare, fingerprint, require


def checked_file(witness):
    path = Path(witness['path'])
    require(fingerprint(path)['sha256'] == witness['sha256'], f'Changed evidence: {path}')
    return path


def array(data, name):
    pairs = np.asarray(data[name], dtype=np.float64)
    require(pairs.shape == (2, 2, 10, 2), f'Wrong {name} shape')
    require(np.isfinite(pairs).all(), f'Nonfinite {name}')
    return pairs[..., 0] + 1j * pairs[..., 1]


def prepare(adjudication_path, golden_path):
    adjudication = json.loads(adjudication_path.read_text())
    require(adjudication['eligible_for_repin'] is True and not adjudication['failures'],
            'Four-run adjudication must pass before transfer')
    checked_file(adjudication['provenance']['adjudicator'])
    checked_file(adjudication['figure'])
    captures = {}
    for role in ROLES:
        evidence = adjudication['runs'][role]
        for key in ('run_id_witness', 'commit_witness'):
            checked_file(evidence[key])
        captures[role] = json.loads(checked_file(evidence['input']).read_text())
        require(not evidence['qualification_recomputed']['failures'], f'{role}: unqualified')
    base = captures['base']
    require(base['dtype'] == 'complex64', 'Only the qualified complex64 base may be imported')
    projected = array(base, 'S_projected')
    candidate = projected.astype(np.complex64)
    np.testing.assert_array_equal(candidate.astype(np.complex128), projected)
    expected = {
        'base_refine_raw': compare(array(base, 'S_raw'), array(captures['refine'], 'S_raw'), rtol=0, atol=.02),
        'base_confirm_projected': compare(projected, array(captures['confirm'], 'S_projected'), rtol=.005, atol=.002),
        'base_long_raw': compare(array(base, 'S_raw'), array(captures['long'], 'S_raw'), rtol=.005, atol=.002),
        'base_long_projected': compare(projected, array(captures['long'], 'S_projected'), rtol=.005, atol=.002),
    }
    require(expected == adjudication['comparisons'], 'Recorded comparisons differ from saved arrays')
    require(all(c['pass'] for c in expected.values()), 'Saved arrays exceed unchanged budgets')
    refine_dir = checked_file(adjudication['runs']['refine']['input']).parent
    completion = {name: fingerprint(refine_dir / name)
                  for name in ('runner.rc', 'refine.rc', 'pytest.rc', 'summary.json', 'job-exit.json')}
    for name in ('runner.rc', 'refine.rc', 'pytest.rc'):
        require((refine_dir / name).read_text().strip() == '0', f'Refinement {name} failed')
    summary = json.loads((refine_dir / 'summary.json').read_text())
    job_exit = json.loads((refine_dir / 'job-exit.json').read_text())
    require(summary['returncode'] == job_exit['returncode'] == 0, 'Refinement did not finish cleanly')
    require(summary['source_sha'] == job_exit['source_sha'] == base['source_sha'], 'Completion SHA differs')
    old = np.load(golden_path, allow_pickle=False)
    old_deltas = {role: float(np.max(np.abs(array(captures[role], 'S_projected') - old)))
                  for role in ('base', 'refine')}
    for role, delta in old_deltas.items():
        require(delta == captures[role]['old_golden_max_abs_delta'], 'Current pin differs from pre-repair golden')
    mesh = {}
    for name in ('S_raw', 'S_projected'):
        delta = np.abs(array(base, name) - array(captures['refine'], name))
        mesh[name] = {'entries': int(delta.size), 'maximum_abs_delta': float(delta.max()),
                      'mean_abs_delta': float(delta.mean()), 'entries_over_0_02': int((delta > .02).sum())}
    manifest = dict(base)
    manifest.update(
        import_method='Offline import_931_msl_golden.py; no solve; base S_projected cast losslessly to complex64',
        reason='The repaired 254 um substrate / 600 um trace board passes the predeclared .02 raw-complex '
               'mesh budget. Its graded 50/38.5 um substrate mesh and trace sheet at 254 um replace '
               'the historical 80 um lattice and 320 um trace plane. The approximately .20 old-pin '
               'gap persists under refinement; this is a deliberate fixture change, not temporal '
               'or floating-point drift. The historical replay fixtures remain unchanged.',
        array_axes=['output_port', 'input_port', 'frequency'], complex_encoding='[real, imaginary]',
        golden_shape=list(candidate.shape), golden_dtype=str(candidate.dtype),
        previous_golden=fingerprint(golden_path), old_golden_comparison=old_deltas,
        drift_tolerance={'rtol': .005, 'atol': .002},
        qualification_reports={role: adjudication['runs'][role]['input'] for role in ROLES},
        run_ids=adjudication['provenance']['run_ids'], adjudication_report=fingerprint(adjudication_path),
        adjudication=adjudication, mesh_comparison=mesh, refinement_completion=completion,
        refinement_pytest_counts=summary['pytest']['counts'],
        writer=fingerprint(Path(__file__)))
    return candidate, manifest, captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjudication', required=True, type=Path)
    parser.add_argument('--golden', type=Path, default=ROOT / 'tests/fixtures/msl_s_matrix_golden.npy')
    args = parser.parse_args()
    manifest_path = args.golden.with_suffix('.json')
    require(not manifest_path.exists(), 'Manifest already exists; do not overwrite provenance')
    candidate, manifest, captures = prepare(args.adjudication, args.golden)
    # Finish all validation before the explicit, local golden transfer.
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    np.save(args.golden, candidate, allow_pickle=False)
    saved = np.load(args.golden, allow_pickle=False)
    require(saved.dtype == np.complex64 and saved.shape == (2, 2, 10), 'Wrong saved pin format')
    np.testing.assert_array_equal(saved, array(captures['base'], 'S_projected'))
    np.testing.assert_allclose(saved, array(captures['confirm'], 'S_projected'), rtol=.005, atol=.002)
    manifest['golden_sha256'] = hashlib.sha256(args.golden.read_bytes()).hexdigest()
    manifest['offline_verification'] = {'base_array_equal': True, 'confirmation_allclose': True,
                                        'entries_checked_per_run': int(saved.size)}
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    with manifest_path.open('x') as stream:
        stream.write(manifest_text)
    print(json.dumps({'golden': str(args.golden), 'manifest': str(manifest_path),
                      'verification': manifest['offline_verification']}, indent=2))


if __name__ == '__main__':
    main()
