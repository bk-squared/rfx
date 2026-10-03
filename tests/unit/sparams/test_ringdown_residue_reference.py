"""Residue column references and growing-pole amplitudes."""
from pathlib import Path

import numpy as np
import pytest

from rfx import ringdown as rd
from tests.unit.sparams import test_ringdown_oracles as oracle


def start_referenced_fit(y, s, dt, *, column_reference=False):
    if not s.size:
        return np.zeros((0, y.shape[1]), dtype=np.complex128)
    m = np.arange(len(y), dtype=np.float64) * float(dt)
    basis = np.exp(np.outer(m, s))
    norms = np.linalg.norm(basis, axis=0)
    norms = np.where(norms > 0, norms, 1.)
    coef = np.linalg.lstsq(basis / norms, y.astype(np.complex128), rcond=None)[0]
    result = coef / norms[:, None]
    if column_reference:
        growing = s.real > 0
        result[growing] *= np.exp(m[-1] * s[growing])[:, None]
    return result


@pytest.mark.parametrize('cavity', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_existing_oracle_poles_and_residues_are_bit_identical(monkeypatch, cavity, dtype):
    n, dt, fmax, modes = ((26000, oracle.DT_CAV, oracle.F_MAX_CAV, oracle.MODES_CAV)
                         if cavity else (oracle.N_RECORD, oracle.DT, oracle.F_MAX, oracle.MODES))
    y = oracle._record(n, dt, modes).astype(dtype)
    actual = rd.identify(y, dt, n//2, n, freq_max=fmax)
    monkeypatch.setattr(rd, '_fit_residues', start_referenced_fit)
    old = rd.identify(y, dt, n//2, n, freq_max=fmax)
    assert np.all(actual.s.real <= 0)
    assert actual.s.tobytes() == old.s.tobytes()
    assert actual.c.tobytes() == old.c.tobytes()


def test_fast_growing_waveguide_pole_has_finite_end_amplitude():
    path = Path(__file__).parents[2] / 'fixtures/settling_witness/waveguide_worst.npz'
    with np.load(path) as data:
        y = data['selected_record']
    model = rd.identify(y, 4.842700168608755e-12, 503, 713, freq_max=11.6e9)
    assert len(model.s_growing) > 0
    assert np.max(model.s_growing.real) * model.dt * 209 > 709
    assert np.isfinite(model.growing_amplitude).all()
    assert np.max(model.growing_amplitude) > 1e-2


def test_kept_positive_pole_residue_remains_start_referenced():
    s = np.array([.002 + .3j, -.03 + .1j])
    c = np.array([[.4 + .1j], [.8 - .2j]])
    y = np.exp(np.outer(np.arange(100), s)) @ c
    np.testing.assert_allclose(rd._fit_residues(y, s, 1.), c, rtol=1e-13)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
def test_existing_box_record_keeps_poles_and_residues_bit_identical(monkeypatch, lane):
    from tests.unit.sparams.test_ringdown_run import _box, FREQS

    result = _box(lane).forward(n_steps=1500, port_s11_freqs=FREQS, skip_preflight=True)
    y = np.asarray(result.sparam_time_records[0])[:, :2]
    actual = rd.identify(y, result.dt, 750, 1500, freq_max=20e9)
    monkeypatch.setattr(rd, '_fit_residues', start_referenced_fit)
    old = rd.identify(y, result.dt, 750, 1500, freq_max=20e9)
    assert np.all(actual.s.real <= 0)
    assert actual.s.tobytes() == old.s.tobytes()
    assert actual.c.tobytes() == old.c.tobytes()
