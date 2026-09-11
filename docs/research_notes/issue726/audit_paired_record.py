"""Audit #726 raw records; reports observations, never a physical-accuracy PASS.

The pinned producer did not save its HJ normalization impedance. Reconstruct
it from that producer's verified homogeneous cv06b substrate (float32), not
from the fitted Z0 or the 50-ohm source/load metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff


def load_arm(root, label):
    with np.load(root / f"{label}-result.npz") as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    with np.load(root / f"{label}-phasors.npz") as data:
        result['v'] = np.asarray(data['raw_v'])[:, :, 0, :]
        result['i'] = np.asarray(data['raw_i1'])
        result['metadata'] = json.loads(str(data['metadata_json']))
        np.testing.assert_array_equal(result['S'], data['production_smatrix'])
        np.testing.assert_array_equal(result['freqs'], data['freqs_hz'])
    result['quality'] = json.loads((root / f"{label}-quality.json").read_text())
    return result


def summary(record, k, band):
    s, v, i = (record[key] for key in ('S', 'v', 'i'))
    zref = np.array([float(hammerstad_jensen_z0_eps_eff(
        port['width_m'], port['height_m'], float(np.float32(3.66)))[0])
        for port in record['metadata']['port_definitions']])
    a, b = (v + zref[None, :, None] * i) / 2, (v - zref[None, :, None] * i) / 2
    replay = np.stack([np.linalg.solve(a[:, :, q], b[:, :, q]).T
                       for q in range(len(record['freqs']))], axis=2)
    column_power = np.sum(abs(s)**2, axis=0)
    return dict(
        quality=record['quality'], settling_db=record['settling_db'].tolist(),
        at_control_notch=dict(
            frequency_hz=float(record['freqs'][k]),
            raw_s11_db=float(20 * np.log10(abs(s[0, 0, k]))),
            raw_s21_db=float(20 * np.log10(abs(s[1, 0, k]))),
            reliable=record['reliable'][:, k].tolist(),
            beta_railed=record['beta_railed'][:, k].tolist(),
            cond_a=float(record['cond_a'][k]),
        ),
        max_column_power_all=float(column_power.max()),
        max_column_power_band=float(column_power[:, band].max()),
        max_cond_a_all=float(record['cond_a'].max()),
        max_cond_a_band=float(record['cond_a'][band].max()),
        unreliable_bins_band=int(np.any(~record['reliable'][:, band], axis=0).sum()),
        beta_railed_bins_band=record['beta_railed'][:, band].sum(axis=1).tolist(),
        inferred_reference_impedances_ohm=zref.tolist(),
        reference_provenance='source130b9071: homogeneous EPS_R=3.66 sampled at float32; not a producer metadata field',
        max_abs_vi_replay_diff=float(np.max(abs(s - replay))),
    )


def audit(root):
    environment = json.loads((root.parent / 'environment.json').read_text())
    if (environment['source_sha'] != '130b9071e5e6aaa0efda0dd1d6a36fb7e8d519bd'
            or environment['x64']):
        raise ValueError('the inferred reference is specific to the recorded float32 producer')
    records = {label: load_arm(root, label) for label in ('clean', 'near')}
    clean, near = records.values()
    np.testing.assert_array_equal(clean['freqs'], near['freqs'])
    for name in ('grid', 'simulation', 'n_probes_per_port', 'current_convention'):
        if clean['metadata'][name] != near['metadata'][name]:
            raise ValueError(f'paired producer metadata differs: {name}')
    for p, (a, b) in enumerate(zip(clean['metadata']['port_definitions'],
                                  near['metadata']['port_definitions'])):
        a, b = dict(a), dict(b)
        if p == 0:
            a.pop('n_probe_offset')
            b.pop('n_probe_offset')
        if a != b:
            raise ValueError(f'physical port or unrequested ladder changed: port {p}')
    f = clean['freqs']
    band = (f >= 3e9) & (f <= 5e9)
    indices = np.flatnonzero(band)
    k = int(indices[np.argmin(abs(clean['S'][1, 0, indices]))])
    result = {label: summary(record, k, band) for label, record in records.items()}
    result['producer_verdict'] = json.loads((root / 'comparison.json').read_text())
    result['raw_delta_s11_db_at_control_notch'] = (
        result['near']['at_control_notch']['raw_s11_db']
        - result['clean']['at_control_notch']['raw_s11_db'])
    result['comparison_scope'] = (
        'raw observation at the same frequency but different probe0 reference planes; '
        'no calibrated error, no passivity projection, no replacement of producer gates')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('artifacts', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.artifacts)
    # Evidence is append-only: do not replace a prior analysis unnoticed.
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
