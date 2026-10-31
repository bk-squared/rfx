"""Measure the small graded microstrip fixture with dielectric smoothing on.

The existing driven fixture is tests/unit/nonuniform/
test_nu_drive_sees_override_1267.py::_board(..., port='msl'). Its graded
z air column starts at the substrate's upper face. Geometry, mesh and feed
are kept verbatim; the probe ladder is made explicit to fit the short feed,
and smoothing is enabled on the calculator's internal run.
The smaller graded source-span fixture only assembles a closed PEC box;
this fixture already supplies a driven microstrip and a witness probe.

FACT: S11 is result.S[0, 0]; minimum frequency is the sampled argmin(abs(S11)),
not an interpolated resonance. The 4 ns record is an estimate of sufficient
settling; the output retains settling and reliability diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Eight steps; no physical minimum claimed.')
    args = parser.parse_args()
    # Explicit path makes python -I work with both an archived tree and a worktree.
    sys.path.insert(0, str(args.repo.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import rfx
    from tests.unit.nonuniform.test_nu_drive_sees_override_1267 import _board, EPS_LO, EPS_HI

    print(f'rfx.__file__={rfx.__file__}', flush=True)
    sim = _board(EPS_LO, EPS_HI, port='msl')
    # The fixture's automatic ladder overruns this short domain. Keep the
    # existing feed and use three probes at x = 7, 7.5, 8 mm. A distinct
    # name leaves the old port's automatic-placement bookkeeping inactive.
    sim._msl_ports = [replace(sim._msl_ports[0], name='band', n_probes=3,
                              n_probe_offset=10, n_probe_spacing=1)]
    grid = sim._build_nonuniform_grid()
    profiles = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)
    mesh_hash = hashlib.sha256(b''.join(np.asarray(p, dtype='<f8').tobytes() for p in profiles)).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'mesh.npz', dx=profiles[0], dy=profiles[1], dz=profiles[2])
    n_steps = 8 if args.smoke else int(np.ceil(4e-9 / float(grid.dt)))
    frequencies = np.linspace(4e9, 12e9, 3 if args.smoke else 161)
    calls = []
    original_run = sim.run

    def smoothed_run(**kwargs):
        calls.append(kwargs.get('n_steps'))
        return original_run(**kwargs, subpixel_smoothing=True)

    # compute_msl_s_matrix has no smoothing keyword; its public run entry does.
    sim.run = smoothed_run
    result = sim.compute_msl_s_matrix(n_steps=n_steps, freqs=jnp.asarray(frequencies),
                                     enforce_passivity=False)
    s11 = np.asarray(result.S)[0, 0]
    reliable = getattr(result, 'reliable', None)
    summary = {
        'smoke': args.smoke, 'subpixel_smoothing': True, 'run_calls': len(calls),
        'shape': list(grid.shape), 'nodes': int(np.prod(grid.shape)),
        'dt_s': float(grid.dt), 'n_steps': n_steps, 'record_s': n_steps * float(grid.dt),
        'mesh_sha256': mesh_hash, 'backend': jax.default_backend(),
        'probe_x_m': [.007, .0075, .008],
        'jax_version': jax.__version__, 'rfx_file': rfx.__file__,
        'settling_db': np.asarray(getattr(result, 'settling_db', [])).tolist(),
        'reliable': None if reliable is None else np.asarray(reliable).tolist(),
    }
    if not args.smoke:
        if not np.isfinite(s11).all() or not calls:
            raise RuntimeError('Nonfinite S11 or no smoothed run: no user number is valid.')
        index = int(np.argmin(np.abs(s11)))
        summary.update(minimum_frequency_hz=float(frequencies[index]),
                       minimum_abs_s11=float(abs(s11[index])),
                       frequency_bin_hz=float(frequencies[1] - frequencies[0]),
                       minimum_at_band_edge=index in (0, len(frequencies) - 1))
    np.savetxt(args.output / 's11.csv',
               np.column_stack((frequencies, s11.real, s11.imag, abs(s11))),
               delimiter=',', header='frequency_hz,s11_real,s11_imag,abs_s11', comments='')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if args.smoke:
        print('SMOKE ONLY: eight steps; S11 minimum unmeasured.', flush=True)


if __name__ == '__main__':
    main()
