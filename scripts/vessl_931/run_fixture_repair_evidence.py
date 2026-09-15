#!/usr/bin/env python3
"""Serial, read-only fixture evidence with live logs and explicit outcomes."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def commands(lane: str, out: Path, run_name: str):
    py = sys.executable
    # xdist does not forward worker stdout under -s. Capture it so -rA and
    # JUnit preserve the complete per-bin witnesses even for passing tests.
    pytest = [py, '-m', 'pytest', '-o', 'addopts=', '-n', '4', '-v',
              '--capture=tee-sys', '-o', 'junit_logging=all', '-rA',
              '--timeout=7200', '--timeout-method=thread']
    if lane == 'v173a':
        yield 'capture', [py, 'scripts/harnesses/v173a_physics_equivalence.py',
                          '--output', str(out / 'capture.json'), '--run-name', run_name + '-capture']
        nodes = ['tests/unit/misc/test_v173a_physics_equivalence_slow.py']
    elif lane in ('msl', 'msl-long', 'msl-refine'):
        label, refinement, periods = {
            'msl': ('base', 1, 12), 'msl-long': ('long', 1, 24),
            'msl-refine': ('refine', 2, 12),
        }[lane]
        yield label, [py, 'scripts/capture_msl_e2e_golden.py', '--run-name', run_name + '-' + label,
                      '--output', str(out / (label + '.json')), '--refinement', str(refinement),
                      '--num-periods', str(periods)]
        module = 'tests/unit/autodiff/test_msl_sparam_ad.py::'
        nodes = [module + name for name in (
            'test_aligned_coupon_drive_covers_the_declared_measurement_band',
            'test_qualification_checks_the_float32_rounded_band_endpoint',
            'test_migrated_trace_is_a_sheet_on_the_declared_plane',
            'test_aligned_coupon_qualification_accepts_matched_line_limit',
            'test_aligned_coupon_qualification_rejects_wrong_observables')]
        if lane == 'msl':
            nodes.insert(0, module + 'test_compute_msl_s_matrix_end_to_end_matches_historical_base')
    elif lane == 'farfield':
        for steps in (600, 1200):
            yield str(steps), [py, 'scripts/diagnostics/slow_931_farfield_attribution.py',
                               '--fixture', 'repaired', '--operators', 'current', '--steps', str(steps),
                               '--output-dir', str(out / str(steps))]
        nodes = ['tests/unit/farfield/test_farfield_inplane_nonuniform.py']
    else:
        for label, extra in [('coarse', []), ('fine', ['--fine'])]:
            yield label, [py, 'scripts/diagnostics/slow_931_advisory_attribution.py', '--arm', 'current',
                          '--output-dir', str(out / label), *extra]
        nodes = ['tests/contracts/test_pec_short_advisory_geometry.py',
                 'tests/unit/sparams/test_sparam_passivity_guard.py']
    yield 'pytest', pytest + ['--junitxml=' + str(out / 'junit.xml'), *nodes]


def junit_counts(path):
    if not path.exists():
        return {'available': False, 'reason': 'pytest did not produce JUnit XML'}
    cases = list(ET.parse(path).getroot().iter('testcase'))
    counts = dict(total=len(cases), passed=0, failed=0, errors=0, skipped=0)
    failing_ids = []
    for case in cases:
        status = ('failed' if case.find('failure') is not None else
                  'errors' if case.find('error') is not None else
                  'skipped' if case.find('skipped') is not None else 'passed')
        counts[status] += 1
        if status in ('failed', 'errors'):
            failing_ids.append(case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', ''))
    return {'available': True, 'counts': counts, 'failing_ids': failing_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=('v173a', 'msl', 'msl-long', 'msl-refine', 'farfield', 'short'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd().resolve()
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == os.environ['RFX_SHA'], (sha, os.environ['RFX_SHA'])
    assert os.environ['PYTHONPATH'] == str(root)
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    import jax
    import rfx
    assert Path(rfx.__file__).resolve().is_relative_to(root / 'rfx'), rfx.__file__
    assert jax.default_backend() == 'cpu'
    report = {'run_name': out.name, 'lane': args.lane, 'source_sha': sha,
              'rfx_import': rfx.__file__, 'jax_version': jax.__version__,
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'run_id_source': 'run_id.txt written by scripts/vessl_submit.sh', 'commands': []}
    rc = 1
    try:
        for label, command in commands(args.lane, out, out.name):
            started = time.monotonic()
            timeout_s = 86400 if args.lane == 'msl-refine' and label == 'refine' else 21600
            item = {'name': label, 'command': command, 'status': 'running', 'timeout_s': timeout_s}
            report['commands'].append(item)
            (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'RUN {label}: {command}', flush=True)
            # Copy each line to both terminal and file, keeping the child's rc.
            with (out / (label + '.log')).open('w') as log:
                process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=60', str(timeout_s), *command],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            (out / (label + '.rc')).write_text(str(returncode) + '\n')
            item.update(status='complete', returncode=returncode, elapsed_s=time.monotonic() - started)
            if label == 'pytest':
                item['junit'] = junit_counts(out / 'junit.xml')
            print(f'RESULT {label}: rc={returncode}', flush=True)
        rc = int(any(item['returncode'] != 0 for item in report['commands']))
    finally:
        report.update(returncode=rc, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      pytest=junit_counts(out / 'junit.xml'))
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
