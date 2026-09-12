"""Report fixed-input CV06b changes; never declare calibrated accuracy.

The first control's notch bin is fixed before reading the new observations.
Same-field reconstruction distinguishes the current-plane change from any
change in the recorded fields. Retain the producer's low-signal verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_current_impact import audit


def db(value):
    return float(20 * np.log10(max(abs(value), 1e-300)))


def compare(before, after):
    inputs = [json.loads((root / 'inputs.json').read_text()) for root in (before, after)]
    for receipt in inputs:
        receipt.pop('driver_sha256', None)
    if inputs[0] != inputs[1]:
        raise ValueError('recorded physical inputs, source or observation ladders changed')
    with np.load(before / 'clean-result.npz') as record:
        freqs = record['freqs']
        band = np.flatnonzero((freqs >= 3e9) & (freqs <= 5e9))
        k = int(band[np.argmin(abs(record['S'][1, 0, band]))])
    results = {}
    for label in ('clean', 'near'):
        with np.load(before / f'{label}-result.npz') as old:
            old_s = old['S']
            np.testing.assert_array_equal(freqs, old['freqs'])
        with np.load(after / f'{label}-result.npz') as new:
            new_s = new['S']
            np.testing.assert_array_equal(freqs, new['freqs'])
            quality = {name: new[name].tolist() for name in ('settling_db',)}
            quality['reliable_at_notch'] = new['reliable'][:, k].tolist()
            quality['beta_railed_bins_3_to_5_ghz'] = new['beta_railed'][:, band].sum(axis=1).tolist()
        same_field, arrays = audit(after / f'{label}-phasors.npz')
        with (np.load(after / f'{label}-phasors.npz') as dump,
              np.load(before / f'{label}-phasors.npz') as old_dump):
            np.testing.assert_array_equal(new_s, dump['production_smatrix'])
            unchanged = dict(
                voltage_phasors_bit_identical=bool(np.array_equal(
                    dump['raw_v'], old_dump['raw_v'])),
                same_index_current_bit_identical=bool(np.array_equal(
                    dump['raw_i1_same_index'], old_dump['raw_i1'])),
            )
        results[label] = dict(
            before_s11_db=db(old_s[0, 0, k]), after_s11_db=db(new_s[0, 0, k]),
            before_s21_db=db(old_s[1, 0, k]), after_s21_db=db(new_s[1, 0, k]),
            complex_s_change_max_abs=float(np.max(abs(new_s - old_s))),
            new_fields_old_current_vs_before_s_max_abs=float(
                np.max(abs(arrays['same_index_s'] - old_s))),
            same_field_current_impact=same_field, **quality, **unchanged,
        )
    return dict(
        scope='same recorded physical inputs; observation-plane sensitivity, not an exact full-wave error',
        fixed_before_control_notch_hz=float(freqs[k]), arms=results,
        before_near_minus_clean_s11_db=(results['near']['before_s11_db']
                                        - results['clean']['before_s11_db']),
        after_near_minus_clean_s11_db=(results['near']['after_s11_db']
                                       - results['clean']['after_s11_db']),
        producer_verdict=json.loads((after / 'comparison.json').read_text()),
        physical_accuracy_verdict=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
