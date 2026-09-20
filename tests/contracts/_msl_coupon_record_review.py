"""Review the existing coupon's unchanged repeat/refinement gates offline.

Read-only with respect to fixtures: this emits an acceptance report, never
updates a golden. The original 604 physical verdict is extracted from its
preserved log; no missing diagnostics are synthesized for that record.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def load_arrays(path):
    with np.load(path) as record:
        return {name: record[name] for name in record.files}


def review(base_root, qualification_root):
    environment = json.loads((base_root / 'environment.json').read_text())
    baseline = load_arrays(base_root / 'coupon/call-0-result.npz')
    baseline_hash = hashlib.sha256((base_root / 'coupon/call-0-result.npz').read_bytes()).hexdigest()
    prefix = '[WI-1 MSL raw qualification] '
    log = gzip.decompress((base_root / 'coupon.log.gz').read_bytes()).decode()
    verdicts = [ast.literal_eval(line[len(prefix):]) for line in log.splitlines()
                if line.startswith(prefix)]
    if len(verdicts) != 1 or verdicts[0]['failures']:
        raise ValueError('baseline does not have one passing original physical verdict')
    cases = {}
    failures = []
    for label, refinement in (('confirmation', 1), ('refinement', 2)):
        root = qualification_root / label
        inputs = json.loads((root / 'inputs.json').read_text())
        verdict = json.loads((root / 'qualification.json').read_text())
        arrays = load_arrays(root / 'result.npz')
        if inputs['source_sha'] != environment['source_sha']:
            raise ValueError('qualification source revision differs from baseline')
        if inputs['reference_record_sha256'] != baseline_hash:
            raise ValueError('qualification used a different reference record')
        if inputs['refinement'] != refinement or inputs['num_periods'] != 12:
            raise ValueError('qualification mesh or record duration differs from the contract')
        np.testing.assert_array_equal(arrays['freqs'], baseline['freqs'])
        if verdict['failures']:
            failures.append(f'{label}: physical qualification failed: {verdict["failures"]}')
        cases[label] = (inputs, arrays)
    if cases['confirmation'][0]['drawing'] != cases['refinement'][0]['drawing']:
        raise ValueError('refinement changed the physical drawing')
    repeat = cases['confirmation'][1]
    fine = cases['refinement'][1]
    repeat_delta = abs(repeat['S'] - baseline['S'])
    repeat_limit = .002 + .005 * abs(baseline['S'])
    # These are the existing capture tolerances, not measured new bounds.
    if not np.all(repeat_delta <= repeat_limit):
        failures.append('same-mesh confirmation exceeds rtol=.005 / atol=.002')
    # The dump always retains the pre-projection S. Do not substitute the
    # displayed S when a result omits its optional S_raw attribute.
    with np.load(base_root / 'coupon/call-0-raw-vi.npz') as dump:
        base_raw = dump['production_smatrix']
        base_metadata = json.loads(str(dump['metadata_json']))
    with np.load(qualification_root / 'refinement/raw-vi.npz') as dump:
        fine_raw = dump['production_smatrix']
        fine_metadata = json.loads(str(dump['metadata_json']))
    base_planes = [s['voltage_coordinate'] for s in base_metadata['current_plane_stencils']]
    fine_planes = [s['voltage_coordinate'] for s in fine_metadata['current_plane_stencils']]
    # Refinement must retain the actual sampled reference planes, not just
    # the declared drawing. Allow only float64 coordinate accumulation noise.
    np.testing.assert_allclose(fine_planes, base_planes, rtol=1e-12, atol=0)
    if 'S_raw' in baseline:
        np.testing.assert_array_equal(base_raw, baseline['S_raw'])
    if 'S_raw' in fine:
        np.testing.assert_array_equal(fine_raw, fine['S_raw'])
    mesh_delta = abs(fine_raw - base_raw)
    if not np.all(mesh_delta <= .02):
        failures.append('raw-complex mesh difference exceeds existing .02 budget')
    return dict(
        source_sha=environment['source_sha'], failures=failures,
        accepted_for_golden_update=not failures,
        baseline_original_qualification=verdicts[0],
        repeat_max_abs=float(repeat_delta.max()),
        repeat_max_fraction_of_allowance=float(np.max(repeat_delta / repeat_limit)),
        refinement_raw_max_abs=float(mesh_delta.max()),
        repeat_absolute_tolerance=.002, repeat_relative_tolerance=.005,
        refinement_raw_absolute_tolerance=.02,
        base_sampled_reference_planes_m=base_planes,
        refined_sampled_reference_planes_m=fine_planes,
        baseline_result_sha256=baseline_hash,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.base, args.qualification)
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
