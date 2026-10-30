"""Two-stage refusal before allocation, with independent byte estimates."""



import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

import rfx.measurement.ringdown_tail as ringdown_tail
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


@pytest.mark.parametrize('stage', [1, 2], ids=['stage1', 'stage2'])
def test_refusing_rss_is_flat_in_fresh_processes(tmp_path, stage):
    # Stage 1 would allocate about 194/395/601 MiB, all below 1 GiB.
    # The 10 MiB envelope is checked before status/text, including in mutations.
    lengths = (2100, 3000, 3700) if stage == 1 else (40000, 50000, 60000)
    if stage == 1:
        for n in lengths:
            L = int(.33 * n)
            M, k = n - L, L
            expected = 1.87 * 16 * (3*M*L + M*k + 2*k*L + 4*k*k)
            assert 180 * 2**20 < expected < 2**30
    worker = r"""
import json, resource, sys
from pathlib import Path
from tests.contracts.test_identification_memory_bound import rd, record, witness, np
n, stage = int(sys.argv[1]), int(sys.argv[3])
if stage == 1:
    y = np.exp((-.001 + 2j*np.pi*.1) * np.arange(n))
    rd.IDENTIFY_MEMORY_BOUND_BYTES = 64 * 2**20
else:
    y = record(n)
    n_win = n // 2
    nd = ((n_win + 9)//10 - 20 + 9)//10 - 20
    L = int(.33 * nd)
    M, k = nd - L, L
    pencil = 1.87 * 16 * (3*M*L + M*k + 2*k*L + 4*k*k)
    residue = 1.71 * 16 * 5 * n_win * 2
    assert pencil < residue
    rd.IDENTIFY_MEMORY_BOUND_BYTES = (pencil + residue)/2
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
reason = ''
if stage == 1:
    try:
        rd.identify(y, 1., 0, n, freq_max=.4)
    except rd.IdentificationMemoryRefused as exc:
        reason = str(exc)
else:
    result = witness(y)
    reason = result.reason if result.status == 'undetermined' else ''
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
increment = (after - before) * (1 if sys.platform == 'darwin' else 1024)
Path(sys.argv[2]).write_text(json.dumps({'stage': stage, 'n': n, 'increment_bytes': increment, 'reason': reason}))
"""
    measurements = []
    for n in lengths:
        path = tmp_path / f'rss-{n}.json'
        subprocess.run([sys.executable, '-c', worker, str(n), str(path), str(stage)],
                       env={**os.environ, 'JAX_PLATFORMS': 'cpu'},
                       check=True, capture_output=True, timeout=240)
        measurements.append(json.loads(path.read_text()))
    increments = [row['increment_bytes'] for row in measurements]
    assert max(increments) - min(increments) <= 10 * 2**20, increments
    assert max(increments) <= 10 * 2**20, increments
    assert all(f'stage {stage}' in row['reason'] for row in measurements)


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
    assert str(exc).startswith('identification would need ')
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


def test_we_refusal_preserves_synthetic_completion(monkeypatch):
    # Main Nd=178, L=58, M=120, rank=1. WE Nd=278, L=91, M=187.
    main_pencil = 1.87 * 16 * (3*120*58 + 120*58 + 2*58*58 + 4*58**2)
    main_fit = 1.71 * 16 * 5 * 20000
    long_pencil = 1.87 * 16 * (3*187*91 + 187*91 + 2*91*91 + 4*91**2)
    bound = 3_000_000
    assert max(main_pencil, main_fit) < bound < long_pencil
    y, spec = record(), rd.RingdownSpec()
    w2 = rd.two_window_witness(y, .0001, [.4], 40000, 20000, freq_max=.5)
    assert w2.model.rank == 1
    expected = w2.spectra.tobytes()
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    completed = []
    original = rd.early_start_witness

    def capture(*args, **kwargs):
        completed.append(kwargs['spectra'].tobytes())
        return original(*args, **kwargs)

    monkeypatch.setattr(rd, 'early_start_witness', capture)
    judge = rd._judge_record(y, .0001, [.4], 40000, 20000, spec,
                             ref_hz=.5, observable=lambda value: value)
    assert np.isnan(judge['we']) and not judge['we_ok']
    assert completed == [expected]
    plan = object.__new__(rd.RingdownRun)
    plan.spec, plan.n_steps, plan.source_ratio_long = spec, 40000, 0.
    witnesses = plan._error_witnesses(y, .0001, [.4], .5, lambda value: value,
                                     ringdown_tail.plain_dft(y, .0001, [.4]), w2, {})
    we = witnesses[0]
    assert we.name == 'WE' and np.isnan(we.value) and not we.ok
    assert w2.spectra.tobytes() == expected
    for note in (judge['note'], we.note):
        assert f'{long_pencil / 2**30:.2f} GiB at stage 1' in note
        assert f'above the {bound / 2**30:.2f} GiB bound' in note


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_we_refusal_keeps_cavity_main_s(monkeypatch, entry):
    from tests.unit.sparams.test_ringdown_run import _box, FREQS
    from rfx import _ringdown_jax as rj

    # At 1200 steps both windows reject FIR q=6 for insufficient capacity:
    # main: n=600, C=2, L=198, M=402, k=396; WE: n=900, L=297, M=603, k=594.
    main = 1.87 * 16 * (3*402*198*2 + 402*396 + 2*396*198*2 + 4*396**2)
    long = 1.87 * 16 * (3*603*297*2 + 603*594 + 2*594*297*2 + 4*594**2)
    bound = (main + long) / 2
    assert 1.71 * 16 * 5 * 600 * 396 < bound

    def call():
        sim = _box('uniform')
        kw = dict(n_steps=1200, skip_preflight=True, ringdown=rd.RingdownSpec())
        return (sim.run(compute_s_params=True, s_param_freqs=FREQS, **kw) if entry == 'run'
                else sim.forward(port_s11_freqs=FREQS, **kw)).ringdown

    reference = np.asarray(call().s_params)
    assert np.isfinite(reference).all()
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    result = call()
    assert np.asarray(result.s_params).tobytes() == reference.tobytes()
    if entry == 'forward':
        assert np.isnan(np.asarray(result.s_params_long)).all()
        assert np.array_equal(np.asarray(result._status), [rj.STATUS_OK, rj.STATUS_FAILED])
    we = next(w for w in result.report.witnesses if w.name == 'WE')
    assert np.isnan(we.value) and not we.ok
    assert f'{long / 2**30:.2f} GiB at stage 1' in we.note
    assert f'above the {bound / 2**30:.2f} GiB bound' in we.note


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_prerun_refuses_before_any_step_with_same_estimate(monkeypatch, entry):
    from importlib import import_module
    from tests.unit.sparams.test_ringdown_run import _box, FREQS, MM

    sim = _box('uniform')
    step = Mock(side_effect=AssertionError('runner took a step'))
    if entry == 'run':
        monkeypatch.setattr(import_module('rfx.runners.uniform'), 'run_uniform', step)
    else:
        monkeypatch.setattr(sim, '_forward_from_materials', step)
    spec = rd.RingdownSpec(freq_max=1e9, identification_probes=(((8*MM, 7*MM, 2*MM), 'ez'),))
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 1)
    kw = dict(n_steps=4000, skip_preflight=True, ringdown=spec)
    with pytest.raises(rd.IdentificationMemoryRefused) as before:
        if entry == 'run':
            sim.run(compute_s_params=True, s_param_freqs=FREQS, **kw)
        else:
            sim.forward(port_s11_freqs=FREQS, **kw)
    step.assert_not_called()
    with pytest.raises(rd.IdentificationMemoryRefused) as after:
        rd.identify(np.ones((4000, 3)), float(sim._build_grid().dt), 2000, 4000,
                    freq_max=18e9)
    # dt=1.9065748695310057e-12, bin max 18 GHz overrides spec's 1 GHz:
    # q=7, Nd=ceil(2000/7)-20=266, L=87, M=179, k=min(179,261)=179.
    needed = 1.87 * 16 * (3*179*87*3 + 179*179 + 2*179*87*3 + 4*179**2)
    for exc in (before.value, after.value):
        assert (exc.n_win, exc.Nd, exc.C, exc.L, exc.needed_bytes) == (2000, 266, 3, 87, needed)
    assert str(before.value) == str(after.value)


def test_stop_keeps_in_run_memory_checks(monkeypatch):
    from tests.unit.sparams.test_ringdown_run import _box

    sim = _box('uniform')
    check = Mock(side_effect=AssertionError('fixed-run precheck reached'))
    monkeypatch.setattr(rd, '_check_run_identification_memory', check)
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 1)
    plan = rd.RingdownStop(sim, rd.RingdownSpec(), lane='uniform', n_max=4000, grid=sim._build_grid())
    step = Mock(side_effect=RuntimeError('in-run sentinel'))
    with pytest.raises(RuntimeError, match='in-run sentinel'):
        plan.run(step)
    step.assert_called_once()
    check.assert_not_called()


@pytest.mark.parametrize('stage,bound', [(1, 1), (2, (STAGE1 + STAGE2)/2)])
def test_refusal_pickle_round_trip(monkeypatch, stage, bound):
    import pickle

    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    y = record()
    with pytest.raises(rd.IdentificationMemoryRefused) as caught:
        rd.identify(y-y[20000:].mean(), .0001, 20000, 40000, freq_max=.5)
    exc = pickle.loads(pickle.dumps(caught.value))
    assert type(exc) is rd.IdentificationMemoryRefused
    assert type(exc).__module__ == 'rfx.ringdown'
    assert exc.__dict__ == caught.value.__dict__ and str(exc) == str(caught.value)
    assert exc.stage == stage and str(exc).startswith('identification would need ')


@pytest.mark.parametrize('bound', [None, '4 GiB', 'inf', 0, -1, -np.inf, np.nan,
                                  True, False, 1+0j, np.complex128(1), [], np.array([1])])
def test_invalid_bound_is_an_undetermined_value_error(monkeypatch, bound):
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    pencil = Mock(side_effect=AssertionError('pencil allocated'))
    monkeypatch.setattr(rd, '_pencil', pencil)
    result = witness(record())
    assert result.status == 'undetermined'
    assert 'rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES must be a positive finite number or math.inf' in result.reason
    with pytest.raises(ValueError, match='rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES'):
        rd.identify(record(), .0001, 20000, 40000, freq_max=.5)
    pencil.assert_not_called()


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_invalid_bound_is_rejected_before_runner(monkeypatch, entry):
    from tests.unit.sparams.test_ringdown_run import _box

    sim = _box('uniform')
    cls = rd.RingdownRun if entry == 'run' else rd.RingdownForward
    plan = cls(sim, rd.RingdownSpec(), lane='uniform', n_steps=1200, grid=sim._build_grid())
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 'bad')
    step = Mock(side_effect=AssertionError('runner took a step'))
    with pytest.raises(ValueError, match='rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES'):
        plan.run(step)
    step.assert_not_called()


@pytest.mark.parametrize('bound', [4 * 2**30, float(4 * 2**30), np.float64(4 * 2**30), np.inf])
def test_valid_bound_keeps_verdict(monkeypatch, bound):
    baseline = detail(witness(record()))
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', bound)
    assert detail(witness(record())) == baseline


def test_run_settling_warning_retains_memory_remedies(monkeypatch):
    from tests.unit.sparams.test_ringdown_run import _box, FREQS

    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', 1)
    with pytest.warns(UserWarning) as caught:
        result = _box('uniform').run(n_steps=600, compute_s_params=True,
                                     s_param_freqs=FREQS, skip_preflight=True)
    assert result.settling_witness['status'] == 'undetermined'
    messages = [str(w.message) for w in caught if 'ring-down settling witness NOT PASSED' in str(w.message)]
    assert len(messages) == 1
    message = messages[0]
    assert result.settling_witness['reason'] in message
    assert 'GiB at stage 1' in message and 'GiB bound' in message
    assert 'shorten the identification window, identify on fewer channels, or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES' in message
    assert 'increase n_steps' not in message.lower()
