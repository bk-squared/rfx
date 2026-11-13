"""Wire-format and host-state contract for progress events; no FDTD compile."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rfx.progress import events
from rfx.stepping import events as transport


def test_derived_chunk_lengths():
    import numpy as np
    from rfx.stepping.progress_scan import scan_with_progress

    for n_steps in range(1, 401):
        # Independent oracle: ceil(n / 100), minimum two, then merge tail 1.
        width = max(2, (n_steps + 99) // 100)
        full, tail = divmod(n_steps, width)
        expected = [width] * full
        if tail == 1 and expected:
            expected[-1] += 1
        elif tail:
            expected.append(tail)
        lengths = []

        def record_chunk(carry, xs_chunk, lo):
            assert lo == sum(lengths)
            lengths.append(len(xs_chunk))
            return carry, None

        with events(lambda row: None), transport.scope('run', 'uniform'):
            scan_with_progress(None, None, np.arange(n_steps),
                               n_steps=n_steps, report_every=None,
                               chunk_scan=record_chunk)
        assert lengths == expected, n_steps
        assert sum(lengths) == n_steps
        assert n_steps == 1 or 1 not in lengths


@pytest.mark.parametrize('mode', ['direct', 'inherited', 'stop'])
def test_explicit_chunk_lengths_unchanged(mode):
    import io
    import numpy as np
    from rfx.stepping.progress_scan import scan_with_progress

    lengths = []

    def record_chunk(carry, xs_chunk, lo):
        lengths.append(len(xs_chunk))
        return carry, None

    with events(lambda row: None), transport.scope('run', 'uniform'):
        if mode == 'inherited':
            transport.current().cadence = 2
        scan_with_progress(None, None, np.arange(5), n_steps=5,
                           report_every=2 if mode == 'direct' else None,
                           stream=io.StringIO(), chunk_scan=record_chunk,
                           stop_fn=(lambda *args: False) if mode == 'stop' else None,
                           chunk=2 if mode == 'stop' else None)
    assert lengths == ([2, 3] if mode == 'stop' else [2, 2, 1])


def encoded(row):
    return 'RFX_EVENT ' + json.dumps(row, separators=(',', ':'), ensure_ascii=True) + '\n'


def test_schema_size_and_unicode_label():
    rows = []
    with events(rows.append), transport.scope('compute_waveguide_s_matrix', 'multi-device', ['p1', 'p2']):
        state = transport.current()
        state.emit('phase_start', phase='step', label='\ud55c' * 200)
        state.emit('progress', phase='step', steps_done=50000, steps_total=50000,
                   total_is_cap=True, rate_steps_per_s=123.45678901234567,
                   eta_s=123.45678901234567, criterion='conserved_energy',
                   criterion_value=1.2345678901234567e-12, criterion_threshold=1e-12)
    start = next(r for r in rows if r.get('label', '').startswith('\ud55c'))
    assert start['label'].endswith('\u2026') and len(start['label']) <= 48
    progress = next(r for r in rows if r['kind'] == 'progress')
    assert set(progress) == {'v', 't', 'kind', 'path', 'entry', 'run_id', 'phase',
                             'steps_done', 'steps_total', 'total_is_cap', 'rate_steps_per_s',
                             'eta_s', 'criterion', 'criterion_value', 'criterion_threshold',
                             'pass_index', 'pass_count'}
    assert all(len(encoded(r).encode()) <= 500 for r in rows)
    assert all(r['run_id'].isascii() and len(r['run_id']) <= 32 for r in rows)


def test_plan_parts_rejoin_and_nested_scope():
    rows = []
    labels = [f'p{i}' for i in range(1, 65)]
    with events(rows.append), transport.scope('run', 'uniform', labels):
        with transport.scope('forward', 'uniform'):
            pass
    plans = [r for r in rows if r['kind'] == 'plan']
    assert len(plans) > 1
    assert [r['part'] for r in plans] == list(range(1, len(plans) + 1))
    assert {r['parts'] for r in plans} == {len(plans)}
    assert sum([r['passes'] for r in plans], []) == labels
    assert sum([r['phases'] for r in plans], []) == ['assemble', 'compile', 'step', 'post-process']
    assert {r['entry'] for r in rows} == {'run'}
    assert len({r['run_id'] for r in rows}) == 1
    assert all(len(encoded(r).encode()) <= 500 for r in rows)


def test_file_flush_and_stderr(tmp_path, capsys):
    path = tmp_path / 'events.jsonl'
    with events(path), transport.scope('run', 'uniform'):
        lines = path.read_text().splitlines()
        assert json.loads(lines[0].removeprefix('RFX_EVENT '))['kind'] == 'plan'
        assert len(lines) == 2  # Read while the sink is still open.
    with events('stderr'), transport.scope('run', 'uniform'):
        pass
    captured = capsys.readouterr()
    assert captured.out == ''
    assert all(line.startswith('RFX_EVENT ') for line in captured.err.splitlines())


def test_rate_limit_and_step_end(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(transport.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(transport.time, 'time', lambda: clock[0])
    rows = []
    with events(rows.append), transport.scope('run', 'uniform'):
        transport.begin_step(None, total=50, ticks=True)
        for step in range(1, 41):
            clock[0] = 100 + step / 10
            transport.tick(step)
        transport.end_step(None, 40)
    ticks = [r for r in rows if r['kind'] == 'progress']
    assert len(ticks) >= 3
    assert all(b['t'] - a['t'] >= 1 for a, b in zip(ticks, ticks[1:]))
    end = next(r for r in rows if r['kind'] == 'phase_end' and r['phase'] == 'step')
    assert end['stop'] == 'steps_done' and end['steps_done'] == 40
    assert 'label' not in end


def test_error_preserves_exception_and_resets_scope():
    rows = []
    failure = RuntimeError('step failed')
    with pytest.raises(RuntimeError) as caught:
        with events(rows.append), transport.scope('run', 'uniform'):
            transport.begin_step(None, total=2, ticks=True)
            raise failure
    assert caught.value is failure
    assert rows[-1]['kind'] == 'phase_end' and rows[-1]['stop'] == 'error'
    assert transport.current() is None


@pytest.mark.parametrize('path', ['multi-device', 'subgridded', 'uniform', 'graded'])
def test_deferred_paths_account_for_every_phase(path):
    rows = []
    with events(rows.append), transport.scope('forward', path, refusal='path_not_wired'):
        assert transport.current() is None
    assert [(r['phase'], r['reason_code']) for r in rows[1:]] == [
        ('assemble', 'path_not_wired'), ('compile', 'compile_not_separated'),
        ('step', 'path_not_wired'), ('post-process', 'path_not_wired')]


def test_phase_end_waits_for_output():
    waited = []
    output = SimpleNamespace(block_until_ready=lambda: waited.append(True))
    rows = []
    def sink(row):
        if row['kind'] == 'phase_end' and row['phase'] == 'step':
            assert waited
        rows.append(row)
    with events(sink), transport.scope('run', 'uniform'):
        transport.begin_step(None, total=1, ticks=False)
        transport.end_step(output, 1)
    assert next(r for r in rows if r.get('phase') == 'step')['ticks'] is False


def test_environment_is_read_at_each_call(monkeypatch, capsys):
    monkeypatch.setenv('RFX_EVENTS', 'stderr')
    with transport.scope('run', 'uniform'):
        pass
    assert 'RFX_EVENT ' in capsys.readouterr().err
    monkeypatch.setenv('RFX_EVENTS', 'bogus')
    with pytest.raises(ValueError, match='RFX_EVENTS'):
        with transport.scope('run', 'uniform'):
            pass


def test_environment_subprocess(monkeypatch):
    monkeypatch.setenv('RFX_EVENTS', 'stderr')
    result = subprocess.run([sys.executable, '-c',
        'from rfx.stepping.events import scope\nwith scope("run", "uniform"): pass'],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3], check=True)
    assert json.loads(result.stderr.splitlines()[0][10:])['kind'] == 'plan'
    assert result.stdout == ''


def test_solve_beyond_plan_keeps_ordinal_without_label():
    rows = []
    with events(rows.append), transport.scope('run', 'uniform', ['p1', 'p2']):
        for _ in range(3):
            transport.next_solve()
            transport.begin_step(None, total=1, ticks=False)
            transport.end_step(None, 1)
            transport.result_ready(None)
    assert len([r for r in rows if r['kind'] == 'plan']) == 1
    third = [{k: v for k, v in r.items() if k not in ('t', 'run_id')}
             for r in rows if r.get('pass_index') == 3]
    assert third == [
        {'v': 1, 'kind': 'phase_start', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'assemble'},
        {'v': 1, 'kind': 'phase_end', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'assemble'},
        {'v': 1, 'kind': 'not_reported', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'compile', 'reported': False,
         'reason_code': 'compile_not_separated', 'detail': 'Compile is included in the first step chunk'},
        {'v': 1, 'kind': 'phase_start', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'step', 'ticks': False},
        {'v': 1, 'kind': 'phase_end', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'step', 'stop': 'steps_done', 'steps_done': 1},
        {'v': 1, 'kind': 'phase_start', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'post-process', 'extractor': 's_matrix'},
        {'v': 1, 'kind': 'phase_end', 'path': 'uniform', 'entry': 'run',
         'pass_index': 3, 'pass_count': 2, 'phase': 'post-process'},
    ]
