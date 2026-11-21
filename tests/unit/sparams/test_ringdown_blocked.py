"""Small independent dense oracles for time-blocked completion and host fits."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import ringdown_blocked as blocked
from rfx.ringdown import _jax as rj
from rfx.measurement import matrix_pencil as pencil
from tests._x64_compat import enable_x64

DT = .01
FREQS = np.linspace(.7, 3.1, 13)


def _record(length):
    s = np.array([-.11 + 2j*np.pi, -.11 - 2j*np.pi,
                  -.23 + 5.4j*np.pi, -.23 - 5.4j*np.pi])
    c = np.array([[.4+.2j, .2-.1j], [.4-.2j, .2+.1j],
                  [.1-.3j, .3+.2j], [.1+.3j, .3-.2j]])
    ds = np.array([.03+.8j, .03-.8j, -.02-.3j, -.02+.3j])
    t = np.arange(length) * DT
    b = np.exp(t[:, None] * s)
    y = (b @ c).real + .002 * np.sin(2*np.pi*4.3*t[:, None] + np.array([.2, .7]))
    dy = (b @ (.2*c) + (t[:, None]*b*ds) @ c).real
    dy += .04 * np.cos(2*np.pi*3.9*t[:, None] + np.array([.1, .8]))
    y[-1] += np.array([.2, -.3])
    dy[-1] += np.array([.7, -.4])
    return y, dy, s


def _dense(y, dy, s):
    """Today's dense projection, residue derivative and tail, all in NumPy float64."""
    n = len(y)
    t = np.arange(n) * DT
    b = np.exp(t[:, None] * s)
    norms = np.linalg.norm(b, axis=0)
    q, r = np.linalg.qr(b / norms)
    c = np.linalg.solve(r, q.conj().T @ y) / norms[:, None]
    w = 1 / np.sqrt(np.mean(abs(y)**2, axis=0))
    d = (t[:, None]*b)[:, None, :] * (c.T*w[:, None])[None, :, :]
    pd = d - np.einsum('mk,kcj->mcj', q, np.einsum('mk,mcj->kcj', q.conj(), d))
    ds = np.linalg.lstsq(pd.reshape(-1, len(s)), (dy*w).reshape(-1), rcond=None)[0]
    db = t[:, None] * b * ds
    residual = y - b @ c
    rhs = b.conj().T @ (dy - db @ c) + db.conj().T @ residual
    dc = np.linalg.solve(b.conj().T @ b, rhs)
    z = np.exp(-2j*np.pi*FREQS*DT)
    arg = s[None, :]*DT - 2j*np.pi*FREQS[:, None]*DT
    den = -np.expm1(arg)
    phase = np.exp(s * DT * n)[None, :] * z[:, None]**n
    tail = DT * (phase / den) @ c
    dtail = DT * ((phase / den) @ dc
                 + (phase * ds[None, :]*DT*(n/den + np.exp(arg)/den**2)) @ c)
    transform = DT * z[:, None]**np.arange(n)[None, :]
    return (transform @ y + tail, s, c, transform @ dy + dtail, ds, dc)


def _call(y, dy, poles, slots):
    s = np.full(slots, np.log(.5)/DT, dtype=np.complex128)
    s[:len(poles)] = poles
    mask = np.arange(slots) < len(poles)
    arg = s[None, :]*DT - 2j*np.pi*FREQS[:, None]*DT
    sj, mj, aj = jnp.asarray(s), jnp.asarray(mask, dtype=jnp.int32), jnp.asarray(arg)

    def complete(a):
        spectra, aux = rj.completion(a, DT, FREQS, 0, sj, mj, aj, return_aux=True)
        return spectra, aux['s'], aux['c'], aux['delta_s']

    value, tangent = jax.jit(lambda a, v: jax.jvp(complete, (a,), (v,)))(jnp.asarray(y), jnp.asarray(dy))
    value, tangent = jax.tree_util.tree_map(np.asarray, (value, tangent))
    assert all(np.isfinite(a).all() for a in (*value, *tangent)), 'non-finite value or derivative'
    assert np.all(value[3] == 0), 'pole step value must stay zero'
    loss = lambda a: jnp.real(jnp.sum(complete(a)[0]))
    gradient = np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(y)))
    forward = np.real(np.sum(tangent[0]))
    reverse = np.sum(gradient * dy)
    bar = 1e-10 if jax.config.read('jax_enable_x64') else 1e-4
    assert abs(reverse-forward) <= bar*max(abs(forward), np.max(abs(tangent[0]))), 'reverse product differs from JVP'
    return value[0], value[1], value[2], tangent[0], tangent[1], tangent[2]


def _assert_bound(actual, reference, baseline=None):
    names = ('spectra', 'poles', 'residues', 'spectra_jvp', 'ds_dot', 'dc')
    for i, (name, got, ref) in enumerate(zip(names, actual, reference)):
        got = got[:len(ref)] if name in ('poles', 'residues', 'ds_dot', 'dc') else got
        peak = np.max(np.abs(ref))
        error = np.max(np.abs(got-ref))
        old_error = 0 if baseline is None else np.max(np.abs(baseline[i]-ref))
        bound = max(1e-4*peak, 1.1*old_error)
        print(f'{name}: peak={peak:.12g} error={error:.12g} one_block_error={old_error:.12g} bound={bound:.12g}')
        assert error <= bound, (name, error, bound)


@pytest.mark.parametrize('x64', [False, True], ids=['float32', 'x64'])
@pytest.mark.parametrize('count,size', [(3, 400), (30, 40), (300, 4)])
def test_blocked_matches_dense(monkeypatch, x64, count, size):
    """Three actual scan counts, independent float64 value and derivative oracles."""
    y, dy, s = _record(1199)
    reference = _dense(y, dy, s)
    with enable_x64() if x64 else nullcontext():
        if not x64:
            y, dy = y.astype(np.float32), dy.astype(np.float32)
        width = 2*len(s) + y.shape[1]
        itemsize = 16 if x64 else 8
        monkeypatch.setattr(blocked, '_BLOCK_TARGET_BYTES', len(y)*width*itemsize)
        baseline = _call(y, dy, s, len(s))
        monkeypatch.setattr(blocked, '_BLOCK_TARGET_BYTES', size*width*itemsize)
        assert blocked._block_size(len(y), width, jnp.complex128 if x64 else jnp.complex64) == size
        assert (len(y)+size-1)//size == count
        actual = _call(y, dy, s, len(s))
    _assert_bound(actual, reference, None if x64 else baseline)
    jax.clear_caches()


@pytest.mark.parametrize('x64', [False, True], ids=['float32', 'x64'])
def test_padded_512_is_finite(monkeypatch, x64):
    """Both augmented sets of unit rows keep 508 padded poles and tangents finite."""
    y, dy, s = _record(39)
    reference = _dense(y, dy, s)
    with enable_x64() if x64 else nullcontext():
        if not x64:
            y, dy = y.astype(np.float32), dy.astype(np.float32)
        itemsize = 16 if x64 else 8
        monkeypatch.setattr(blocked, '_BLOCK_TARGET_BYTES', 20*(1024+2)*itemsize)
        actual = _call(y, dy, s, 512)
    assert all(np.isfinite(a).all() for a in actual), 'padded columns must remain finite'
    assert np.all(actual[2][4:] == 0) and np.all(actual[4][4:] == 0)
    assert np.all(actual[5][4:] == 0), 'padded residue tangents must be zero'
    if x64:
        _assert_bound(actual, reference)
    jax.clear_caches()


@pytest.mark.parametrize('rank_deficient', [False, True])
@pytest.mark.parametrize('column_reference', [False, True])
def test_host_stage2_minimum_norm(monkeypatch, rank_deficient, column_reference):
    """The triangle SVD uses the whole matrix cutoff, including end-referenced growth."""
    length, dt = 313, .03
    s = np.array([-.2+2j, -.1-3j, .8+1j, -.4+4j])
    if rank_deficient:
        s[1] = s[0]
    t = np.arange(length)*dt
    exponents = t[:, None]*s
    growing = s.real > 0
    exponents[:, growing] = (t-t[-1])[:, None]*s[growing]
    basis = np.exp(exponents)
    rng = np.random.default_rng(2874)
    y = rng.normal(size=(length, 2)) + 1j*rng.normal(size=(length, 2))
    y[-1] += 9
    norms = np.linalg.norm(basis, axis=0)
    reference = np.linalg.lstsq(basis/norms, y, rcond=None)[0]/norms[:, None]
    if not column_reference:
        reference[growing] *= np.exp(-t[-1]*s[growing])[:, None]
    monkeypatch.setattr(pencil, '_RESIDUE_BLOCK_TARGET_BYTES', 17*16*(len(s)+2))
    actual = pencil._fit_residues(y, s, dt, column_reference=column_reference)
    error = np.max(np.abs(actual-reference))
    assert error <= 1e-10*np.max(np.abs(reference)), ('host residue error', error)
