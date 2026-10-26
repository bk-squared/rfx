"""F-b/F-c facts for a/45; write JSON to argv[1], print only to stderr.

Run with PYTHONPATH=$PWD through mac-serial. The requested output must be
outside the repository. No oracle bar or verdict is changed by this diagnostic.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from tests._filled_waveguide import measure


@contextlib.contextmanager
def capture_table_sizes(rows):
    from rfx.model import waveguide_modes
    original = waveguide_modes.incident_tables_uncut

    def capture(cfg):
        tables = original(cfg)
        if tables:
            rows.append(dict(length=len(tables[0]), bytes=sum(x.nbytes for x in tables),
                             f_cutoff=float(cfg.f_cutoff)))
        return tables

    waveguide_modes.incident_tables_uncut = capture
    try:
        yield
    finally:
        waveguide_modes.incident_tables_uncut = original


def main(path):
    path = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if path == repo or repo in path.parents:
        raise ValueError('diagnostic JSON must be outside the repository')
    started = time.monotonic()
    measured = []
    table_sizes = []
    # Capture library progress too: stdout is deliberately empty.
    with contextlib.redirect_stdout(sys.stderr), capture_table_sizes(table_sizes):
        for n in (16000, 32000, 64000):
            start = time.monotonic()
            row = measure('O2', 45, n_steps=n)
            row['wall_s'] = time.monotonic() - start
            measured.append(row)
            print(f"case B N={n}: {row['wall_s']:.3f} s", file=sys.stderr, flush=True)
        case_a = measure('O1', 45, n_steps=16000)
    settled = np.asarray(measured[-1]['sum'])
    table = []
    for row in measured:
        n = row['n_steps']
        # sum is |S11|^2+|S21|^2, so read the witness for driven column 0.
        witness = row['settling_witness'][0]
        total = np.asarray(witness['share_per_bin'])
        drive = np.asarray(witness['drive_share_per_bin'])
        delta = np.sqrt(n) * 2**-24 * 10
        difference = np.abs(np.asarray(row['sum']) - settled)
        bound = 2 * total + total**2 + 2 * delta
        for k, f in enumerate(row['freqs']):
            table.append(dict(
                n_steps=n, freq_hz=f, sum=row['sum'][k],
                s_poles=float(total[k] - drive[k]), s_drive=float(drive[k]),
                s=float(total[k]), delta=float(delta),
                absolute_difference=float(difference[k]), bound=float(bound[k]),
                bound_pass=bool(difference[k] <= bound[k]),
                drive_pass=bool(drive[k] <= 1e-2),
                witness_status=witness['status'], witness_reason=witness['reason']))
    output = dict(
        case_b_table=table, case_b_runs=measured,
        case_a=dict(n_steps=16000, freqs=case_a['freqs'],
                    settling_witness=case_a['settling_witness']),
        wall_s=time.monotonic() - started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        uncut_tables=table_sizes,
        conclusion='리더가 채움')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + '\n')
    print('N f_Hz sum s_poles s_drive s delta abs_difference bound bound_pass status reason', file=sys.stderr)
    for row in table:
        print(' '.join(str(row[k]) for k in (
            'n_steps', 'freq_hz', 'sum', 's_poles', 's_drive', 's', 'delta',
            'absolute_difference', 'bound', 'bound_pass', 'witness_status', 'witness_reason')), file=sys.stderr)
    print('case A witnesses: ' + json.dumps(output['case_a']), file=sys.stderr)
    print('resources: ' + json.dumps({k: output[k] for k in
                                     ('uncut_tables', 'max_rss_bytes', 'wall_s')}), file=sys.stderr)
    print(f"JSON: {path}", file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1])
