"""Record-level expected findings cannot hide a newly failing record."""
from pathlib import Path


class KnownFinding(AssertionError):
    pass


def expectations(entries):
    """Expand the compact per-finding manifest for one record group."""
    known = [entry['fingerprint'] for entry in entries]
    witnesses = {entry['fingerprint'].split(':', 1)[0]: {
        'relative': entry['relative'], 'relative_bar': entry['relative_bar']}
        for entry in entries if 'relative' in entry}
    causes = ', '.join(sorted({entry['cause'] for entry in entries}))
    return known, witnesses, causes


def group(failure):
    if failure.startswith(('dt:', 'result.dt:', 'nodes.', 'geometry.', 'kernel.', 'kernel:')):
        return 'realized'
    if failure.startswith('time_series'):
        return 'probes'
    if failure.startswith(('objective:', 'gradient:')):
        return failure.split(':', 1)[0]
    if failure.startswith('sparam_time_records'):
        return 'port_samples'
    if failure.startswith(('lumped_port_sparams', 'wire_port_sparams', 's_params')):
        return 'port_dft'
    if failure.startswith(('dft_planes', 'flux_monitors', 'ntff_data', 'current_moment_data')):
        return 'observers'
    return 'execution'


def fingerprint(failure):
    # A different measured amplitude of the SAME failed record is still the
    # same finding. Exceptions retain their entire message, so an unrelated
    # builder or runtime error cannot be accepted under an existing xfail.
    if group(failure) == 'execution':
        # The assembler embeds its caller's absolute filename. Keep the caller,
        # line and full error, but make the checkout prefix portable to CI.
        prefix = str(Path(__file__).resolve().parents[3]) + '/'
        return failure.replace(prefix + 'rfx/runners/_admission.py:',
                               'rfx/runners/_admission.py:')
    if ': relative diff ' in failure:
        return failure.split(':', 1)[0] + ': numeric'
    return failure


def assert_record(report, record, known, witnesses=None):
    failures = [f for f in report['failures'] if group(f) in (record, 'execution')]
    unexpected = [f for f in failures if fingerprint(f) not in known]
    if unexpected:
        raise RuntimeError('\n'.join(unexpected))
    # An existing xfail cannot hide m2 changing the SAME failing record.
    # This checks the finding's measured fingerprint with the original bar;
    # the cross-path verdict remains a failure, never a widened pass.
    for measurement in report['measurements']:
        name = measurement['record']
        witness = (witnesses or {}).get(name)
        if witness is not None and measurement['difference'] > measurement['bar']:
            drift = abs(measurement['relative'] - witness['relative'])
            bar = max(measurement['relative_bar'], witness['relative_bar'])
            if drift > bar:
                raise RuntimeError(f'{name}: S0 finding fingerprint changed: {drift:.9g} > bar {bar:.9g}')
    if failures:
        raise KnownFinding('\n'.join(failures))
