"""Independent formed-Hankel oracles for adaptive partial identification."""

from dataclasses import replace
import pickle
import sys

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.measurement import _hankel_svd as hs
from rfx.measurement.matrix_pencil import _pencil as core_pencil
from rfx.sparams import _tail_witness as tw
from rfx.sparams._settling_warning import _memory_only_reason


def formed(y, columns, shift=0):
    rows = len(y) - columns
    indices = np.arange(rows)[:, None] + np.arange(columns)[None, :] + shift
    return np.concatenate([y[indices, ch] for ch in range(y.shape[1])], axis=1)


def rounding(y):
    n = 1 << (len(y) - 1).bit_length()
    u = np.finfo(np.float64).eps / 2
    eta = 2*u + 4*u/(1 - 4*u)*(np.sqrt(2) + 2*u)
    g = np.log2(n)*eta/(1 - np.log2(n)*eta)
    beta = 2*g + np.sqrt(2)*2*u/(1 - 2*u) + (y.shape[1] - 1)*u
    phi = np.linalg.norm(np.max(np.abs(np.fft.fft(y, n=n, axis=0)), axis=0))
    return beta*phi


def periodic(amplitudes):
    columns = len(amplitudes)
    phase = np.exp(2j*np.pi*np.arange(columns)/columns)
    return (phase[None, :]**np.arange(3*columns)[:, None] @ amplitudes)[:, None]


def gap_window():
    amplitudes = np.concatenate((np.geomspace(1, 2e-6, 17),
                                 [1.045e-6, .95e-6], np.linspace(.8e-6, 1e-7, 237)))
    return periodic(amplitudes)


def noise_window(wide=False):
    rng = np.random.default_rng(4)
    shape = (2048, 64) if wide else (780, 1)
    t = np.arange(shape[0])
    return (np.exp((-.001 + .4j)*t)[:, None]
            + 1e-4*(rng.standard_normal(shape) + 1j*rng.standard_normal(shape)))


def report(label, result):
    print(f"{label}: q={result.q}, b={result.b}, cause={result.cause or 'certified'}, "
          f"counter/F_full={result.counter/result.full_cost:.17g}", file=sys.stderr)


@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('nd', [63, 64])
@pytest.mark.parametrize('complex_window', [False, True])
@pytest.mark.parametrize('shift', [0, 1])
def test_T1_products(channels, nd, complex_window, shift):
    rng = np.random.default_rng(8)
    y = rng.standard_normal((nd, channels)).astype(np.complex128)
    if complex_window:
        y += 1j*rng.standard_normal(y.shape)
    columns = nd//3
    op = hs._Hankel(y, columns)
    matrix = formed(y, columns, shift)
    bound = rounding(y)
    for adjoint, expected in ((False, matrix), (True, matrix.conj().T)):
        x = rng.standard_normal((expected.shape[1], 3)) + 1j*rng.standard_normal((expected.shape[1], 3))
        actual = op.product(x, adjoint=adjoint, shift=shift)
        errors = np.linalg.norm(actual - expected @ x, axis=0)
        assert np.all(errors <= bound*np.linalg.norm(x, axis=0))


def test_T2_two_sided_residual_and_rank():
    y = gap_window()
    matrix = formed(y, 256)
    full = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(full > 1e-6*full[0]))
    assert full[rank - 1]/full[rank] == pytest.approx(1.10)
    result = hs._partial_svd(y, 256, 1e-6, lambda b: True)
    report('T2 gap', result)
    assert not result.cause
    assert result.q > 3
    assert np.count_nonzero(result.s > 1e-6*result.s[0]) == rank
    j = rank + 1
    v = result.vh[:j].conj().T
    left = matrix @ v - result.u[:, :j]*result.s[:j]
    right = matrix.conj().T @ result.u[:, :j] - v*result.s[:j]
    rho = np.sqrt(np.sum(np.abs(left)**2, axis=0) + np.sum(np.abs(right)**2, axis=0))
    assert np.max(rho) <= 2*np.sqrt(2)*rounding(y)


def test_T2_reorthonormalization_on_floor():
    y = periodic(np.concatenate((np.geomspace(1, 2e-6, 18), np.full(238, 1e-7))))
    matrix = formed(y, 256)
    full = np.linalg.svd(matrix, compute_uv=False)
    result = hs._partial_svd(y, 256, 1e-6, lambda b: True)
    report('T2 floor', result)
    assert not result.cause
    assert np.count_nonzero(result.s > 1e-6*result.s[0]) == np.count_nonzero(full > 1e-6*full[0])


def test_T3_block_doubles():
    y = periodic(np.concatenate((np.ones(40), np.zeros(216))))
    full = np.linalg.svd(formed(y, 256), compute_uv=False)
    result = hs._partial_svd(y, 256, 1e-6, lambda b: True)
    assert not result.cause
    assert result.b == 64
    assert np.count_nonzero(result.s > 1e-6*result.s[0]) == np.count_nonzero(full > 1e-6*full[0]) == 40


@pytest.mark.parametrize('wide', [False, True])
def test_T4_cost_stop_and_bit_exact_fallback(monkeypatch, wide):
    y = noise_window(wide)
    if wide:
        monkeypatch.setattr(rd, 'PENCIL_PARAMETER', 2016/2048)
    columns = int(len(y)*rd.PENCIL_PARAMETER)
    normalized = y/np.sqrt(np.mean(np.abs(y)**2, axis=0))
    result = hs._partial_svd(normalized, columns, 1e-6, lambda b: True)
    report('T4 noise', result)
    rows, width = len(y) - columns, columns*y.shape[1]
    k = min(rows, width)
    full_cost = 4*(6*max(rows, width)*k*k + 20*k**3)
    assert result.cause == 'count'
    assert result.full_cost == full_cost
    assert result.counter <= full_cost
    assert result.b <= k
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', np.inf)
    actual = rd._pencil(y, 1e-6, 1e-6, svd='partial', n_win=len(y))
    matrix = formed(normalized, columns)
    shifted = formed(normalized, columns, 1)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = max(int(np.count_nonzero(s > 1e-6*s[0])), 1)
    lam = np.linalg.eigvals((u[:, :rank].conj().T @ shifted @ vh[:rank].conj().T)/s[:rank, None])
    valid = np.isfinite(lam) & (np.abs(lam) > 0)
    n_invalid = int(np.sum(~valid))
    lam = lam[valid]
    growing = np.abs(lam) > 1 + 1e-6
    expected = (lam[~growing], rank, lam[growing], n_invalid, s)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize('cause', ['count', 'block'])
def test_T5_unresolved_refusal_and_pickle(monkeypatch, cause):
    y = noise_window(wide=cause == 'count')
    if cause == 'count':
        monkeypatch.setattr(rd, 'PENCIL_PARAMETER', 2016/2048)
    columns = int(len(y)*rd.PENCIL_PARAMETER)
    rows, width = len(y) - columns, columns*y.shape[1]
    k = min(rows, width)
    needed = 1.87*16*(3*rows*width + rows*k + 2*k*width + 4*k*k)
    monkeypatch.setattr(rd, 'IDENTIFY_MEMORY_BOUND_BYTES', needed - 1)
    with pytest.raises(rd.IdentificationMemoryRefused) as caught:
        rd._pencil(y, 1e-6, 1e-6, svd='partial', n_win=len(y))
    exc = caught.value
    assert exc.unresolved and exc.stage == 1
    assert exc.needed_bytes == needed
    assert exc.cause == cause
    clause = ("the partial SVD reached the full SVD's operation count" if cause == 'count'
              else f"a partial-SVD block of b={exc.block} exceeded the bound")
    assert clause in str(exc)
    assert _memory_only_reason(str(exc))
    restored = pickle.loads(pickle.dumps(exc))
    assert str(restored) == str(exc)
    assert restored.unresolved == exc.unresolved
    assert restored.cause == exc.cause and restored.block == exc.block


def test_T6_deterministic_identification_and_settling(monkeypatch):
    y = periodic(np.concatenate((np.ones(40), np.zeros(472))))
    kw = dict(freq_max=.49, guard=None, svd='partial')
    models = [rd._identify(y, 1., 0, len(y), **kw) for _ in range(2)]
    np.testing.assert_array_equal(models[0].s, models[1].s)
    np.testing.assert_array_equal(models[0].c, models[1].c)
    captured: list = []
    identify = tw._IDENTIFICATION.identify

    def capture(*args, **kwargs):
        model = identify(*args, **kwargs)
        captured.append(model)
        return model

    monkeypatch.setattr(tw, '_IDENTIFICATION', replace(tw._IDENTIFICATION, identify=capture))
    for _ in range(2):
        tw.tail_share_witness([('mode', y)], 1., 0, [.1], freq_max=.49)
    assert len(captured) == 4
    for first, second in zip(captured[:2], captured[2:]):
        np.testing.assert_array_equal(first.s, second.s)
        np.testing.assert_array_equal(first.c, second.c)


def test_public_core_stays_full():
    y = noise_window()
    explicit = core_pencil(y, 1e-6, 1e-6, pencil_parameter=rd.PENCIL_PARAMETER, svd='full')
    default = core_pencil(y, 1e-6, 1e-6, pencil_parameter=rd.PENCIL_PARAMETER)
    for a, e in zip(default, explicit):
        np.testing.assert_array_equal(a, e)
