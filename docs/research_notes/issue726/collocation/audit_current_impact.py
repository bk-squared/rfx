"""Compare spatial-current choices on identical saved MSL fields.

This computes unprojected B A^-1 with NumPy. It reports a numerical change,
not an accuracy verdict. No simulation, fitted impedance or fitted beta is
used. Only dumps carrying both real H-side currents are admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def solve(v, current, zref):
    # Input axes: drive, port, frequency. Rows below are drive, so solving
    # A.T * S.T = B.T gives the usual receiver-by-drive S after transpose.
    a = (v + zref[None, :, None] * current) / 2
    b = (v - zref[None, :, None] * current) / 2
    return np.stack([np.linalg.solve(a[:, :, k], b[:, :, k]).T
                     for k in range(v.shape[-1])], axis=2)


def audit(path):
    with np.load(path, allow_pickle=False) as dump:
        meta = json.loads(str(dump['metadata_json']))
        if meta.get('current_spatial_alignment') != 'linear_bracketing_H_to_E_node':
            raise ValueError('record lacks explicit spatial-current provenance')
        v = np.asarray(dump['raw_v'], dtype=np.complex128)[:, :, 0, :]
        currents = {name: np.asarray(dump[key], dtype=np.complex128)
                    for name, key in (
                        ('centered', 'raw_i1'), ('left', 'raw_i1_left'),
                        ('same_index', 'raw_i1_same_index'))}
        stored = np.asarray(dump['production_smatrix'])
        freqs = np.asarray(dump['freqs_hz'])
    zref = np.asarray(meta['s_reference_impedances_ohm'], dtype=float)
    stencils = meta['current_plane_stencils']
    if v.ndim != 3 or v.shape[0] != v.shape[1] or zref.shape != (v.shape[1],):
        raise ValueError('record is not a complete square multi-drive experiment')
    if any(i.shape != v.shape for i in currents.values()):
        raise ValueError('side currents and voltage have incompatible axes')
    if not np.all(np.isfinite(zref) & (zref > 0)):
        raise ValueError('the recorded real reference impedances must be positive')
    weights = np.asarray([s['weights'] for s in stencils])
    if weights.shape != (v.shape[1], 2):
        raise ValueError('current stencil count does not match ports')
    reconstructed = (weights[None, :, 0, None] * currents['left']
                     + weights[None, :, 1, None] * currents['same_index'])
    centered = solve(v, currents['centered'], zref)
    old = solve(v, currents['same_index'], zref)
    differences = np.abs(centered - old)
    worst = np.unravel_index(np.argmax(differences), differences.shape)
    current_scale = max(float(np.max(abs(currents['centered']))), 1e-300)
    result = dict(
        dump=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        production_assembly=meta['production_smatrix_assembly'],
        scope='same FDTD fields, same V plane and Zref; unprojected extraction comparison only',
        voltage_planes_m=[s['voltage_coordinate'] for s in stencils],
        current_weights=weights.tolist(), reference_impedances_ohm=zref.tolist(),
        centered_vs_stored_s_max_abs=float(np.max(abs(centered - stored))),
        current_interpolation_relative_residual=float(
            np.max(abs(reconstructed - currents['centered'])) / current_scale),
        current_change_relative_to_peak=float(
            np.max(abs(currents['same_index'] - currents['centered'])) / current_scale),
        s_change_max_abs=float(differences[worst]),
        s_change_worst=dict(receiver=int(worst[0]), drive=int(worst[1]),
                            frequency_hz=float(freqs[worst[2]])),
        centered_max_column_power=float(np.max(np.sum(abs(centered)**2, axis=0))),
        same_index_max_column_power=float(np.max(np.sum(abs(old)**2, axis=0))),
        physical_accuracy_verdict=None,
    )
    return result, dict(freqs_hz=freqs, centered_s=centered, same_index_s=old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    arrays_path = args.out.with_suffix('.npz')
    if args.out.exists() or arrays_path.exists():
        parser.error('choose unused output paths; evidence is append-only')
    result, arrays = audit(args.dump)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
