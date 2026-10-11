"""Two-stage refusal before allocation, with independent byte estimates."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.sparams._tail_witness import tail_share_witness


def record(n=40000):
    return np.exp((-1 + 2j*np.pi*.4) * (np.arange(n) * .0001))


def witness(y):
    return tail_share_witness([('mode', y)], .0001, 0, [.4], freq_max=.5)


def detail(value):
    """Preserve every float bit and array byte in the complete witness detail."""
    if hasattr(value, '_asdict'):
        return detail(value._asdict())
    if isinstance(value, dict):
        return {key: detail(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [value.dtype.str, list(value.shape), value.tobytes().hex()]
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return value


# The first witness window has 20,000 original samples. FIR q=10 twice
# leaves ceil(ceil(20000/10)-20)/10 - 20 = 178 samples, one channel.
# L=floor(.33*178)=58, M=120, k=58; the centered one-mode signal has rank 2.
STAGE1 = 1.87 * 16 * (3*120*58 + 120*58 + 2*58*58 + 4*58**2)
STAGE2 = 1.71 * 16 * 5 * 20000 * 2


@pytest.mark.parametrize('stage,bound,needed', [
    (1, 1, STAGE1), (2, (STAGE1 + STAGE2)/2, STAGE2),
], ids=['stage1', 'stage2'])
def test_witness_refuses_before_stage_allocation(monkeypatch, stage, bound, needed):
    assert STAGE1 < (STAGE1 + STAGE2)/2 < STAGE2
    pencil = Mock(wraps=rd._pencil)
    fit = Mock(wraps=rd._fit_residues)
    monkeypatch.setattr(rd, '_pencil', pencil)
    monkeypatch.setattr(rd, '_fit_residues', fit)
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    result = witness(record())
    fit.assert_not_called()
    assert pencil.call_count == stage - 1
    assert result.status == 'undetermined'
    assert f'{needed / 2**30:.2f} GiB at stage {stage}' in result.reason
    assert f'above the {bound / 2**30:.2f} GiB bound' in result.reason
    assert 'n_win=20000, Nd=178, C=1, L=58' in result.reason
    assert ('k=58' if stage == 1 else 'rank=2') in result.reason
    assert 'shorten the identification window, identify on fewer channels' in result.reason
    assert 'raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES' in result.reason


@pytest.mark.parametrize('stage,bound,needed', [
    (1, 1, STAGE1), (2, (STAGE1 + STAGE2)/2, STAGE2),
], ids=['stage1', 'stage2'])
def test_refusal_carries_dimensions(monkeypatch, stage, bound, needed):
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    y = record()
    with pytest.raises(rd.IdentificationMemoryRefused) as caught:
        rd.identify(y - y[20000:].mean(), .0001, 20000, 40000, freq_max=.5)
    exc = caught.value
    assert isinstance(exc, ValueError)
    assert (exc.stage, exc.needed_bytes, exc.bound_bytes) == (stage, needed, bound)
    assert (exc.n_win, exc.Nd, exc.C, exc.L, exc.M, exc.k, exc.r) == (
        20000, 178, 1, 58, 120, 58, None if stage == 1 else 2)


def test_default_verdict_is_bit_identical_with_check_disabled(monkeypatch):
    assert rd.IDENTIFY_MEMORY_BOUND_BYTES == 4 * 2**30
    y = record()
    bounded = detail(witness(y))
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 2**80)
    assert detail(witness(y)) == bounded


@pytest.mark.parametrize('stage', [1, 2])
def test_refusing_rss_is_flat_in_fresh_processes(tmp_path, stage):
    # ru_maxrss is bytes on macOS, KiB on Linux. Input and imports precede
    # the measurement. A 10 MiB envelope is declared for all three lengths.
    worker = '''
import json, resource, sys
from pathlib import Path
from tests.contracts.test_identification_memory_bound import rd, record, witness
y = record(int(sys.argv[1]))
stage = int(sys.argv[3])
if stage == 1:
    rd.IDENTIFY_MEMORY_BOUND_BYTES = 1
else:
    # Two q=10 FIR stages on each of these first windows, rank 2.
    n_win = len(y) // 2
    nd = ((n_win + 9)//10 - 20 + 9)//10 - 20
    columns = int(.33 * nd)
    rows = nd - columns
    k = min(rows, columns)
    pencil = 1.87 * 16 * (3*rows*columns + rows*k + 2*k*columns + 4*k*k)
    residue = 1.71 * 16 * 5 * n_win * 2
    assert pencil < residue
    rd.IDENTIFY_MEMORY_BOUND_BYTES = (pencil + residue)/2
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = witness(y)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert result.status == 'undetermined' and f'stage {stage}' in result.reason
increment = (after - before) * (1 if sys.platform == 'darwin' else 1024)
Path(sys.argv[2]).write_text(json.dumps({'stage': stage, 'n': len(y), 'increment_bytes': increment}))
'''
    measurements = []
    lengths = (10000, 20000, 40000) if stage == 1 else (40000, 50000, 60000)
    for n in lengths:
        path = tmp_path / f'rss-{n}.json'
        subprocess.run([sys.executable, '-c', worker, str(n), str(path), str(stage)],
                       env={**os.environ, 'JAX_PLATFORMS': 'cpu'},
                       check=True, capture_output=True, timeout=120)
        measurements.append(json.loads(path.read_text())['increment_bytes'])
    assert max(measurements) - min(measurements) <= 10 * 2**20


def test_completion_refuses_with_identical_message(monkeypatch):
    y = record()
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 1)
    with pytest.raises(rd.IdentificationMemoryRefused) as direct:
        rd.identify(y, .0001, 20000, 40000, freq_max=.5)
    with pytest.raises(rd.IdentificationMemoryRefused) as completion:
        rd._judge_record(y, .0001, [.4], 40000, 20000, rd.RingdownSpec(),
                         ref_hz=.5, observable=lambda value: value)
    assert str(completion.value) == str(direct.value)
    with pytest.raises(rd.IdentificationMemoryRefused) as two_window:
        rd.two_window_witness(y, .0001, [.4], 40000, 20000, freq_max=.5)
    assert str(two_window.value) == str(direct.value)


def test_run_completion_does_not_swallow_refusal(monkeypatch):
    from tests.unit.sparams.test_ringdown_run import _box, _run

    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 1)
    with pytest.raises(rd.IdentificationMemoryRefused) as caught:
        _run(_box('uniform'), 600, ringdown=rd.RingdownSpec())
    exc = caught.value
    assert exc.stage == 1
    assert str(exc).startswith('settling identification would need ')
    assert str(exc).endswith('raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES')


def test_forward_callback_does_not_swallow_refusal(monkeypatch):
    import jax
    import jax.numpy as jnp
    from rfx import _ringdown_jax as rj

    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 1)
    y = record(200).real
    with pytest.raises(rd.IdentificationMemoryRefused) as direct:
        rd.identify(y, .0001, 0, len(y), freq_max=.5)

    def identify_window(window):
        return rd.identify(window, .0001, 0, len(window), freq_max=.5).s

    # JAX wraps exceptions crossing pure_callback; the original text survives.
    with pytest.raises(Exception) as caught:
        jax.block_until_ready(rj.host_poles(identify_window, jnp.asarray(y), 8, .0001, [.4]))
    assert str(direct.value) in str(caught.value)
