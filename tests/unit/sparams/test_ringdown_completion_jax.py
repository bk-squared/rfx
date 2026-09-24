"""The traced ring-down completion (``rfx._ringdown_jax``) against an analytic oracle (issue #1254).

Physics. A record made of damped oscillations whose frequencies, dampings and
amplitudes depend on a parameter ``p`` has a known infinite-record DFT and a
known derivative of it::

    Y_inf(p)   = dt * sum_k r_k / (1 - lambda_k z)
    dY_inf/dp  = dt * sum_k [ r_k' / (1 - lambda_k z)
                              + r_k z lambda_k' / (1 - lambda_k z)**2 ],
    lambda_k = exp(s_k dt),   lambda_k' = lambda_k s_k' dt.

Each mode's frequency moves as ``f(p) = f0 exp(-p/2)`` (``p = ln eps`` moving a
cavity mode as eps**-1/2), its Q as ``Q(p) = Q0 (1 + q p)`` and each residue as
``r(p) = r0 (1 + a p)``, so the tangent of the record carries the amplitude part
and the frequency-shift part ``r n dt s' lambda**n`` -- the term a cut record's
derivative lacks and the completion must supply. A 9.7 ns record stops while
the modes still ring: its plain DFT's derivative is more than 50 % off.

Checked here (float64 by a scoped x64 context, never at module level):

(i)   value: the traced map equals ``rfx.ringdown``'s numpy completion
      (``plain_dft + tail_dft(identify(...))``, the one ``run(ringdown=...)``
      computes) to 1e-12, and its poles are the host's bit for bit;
(ii)  derivative: the JVP along the exact tangent is the analytic
      infinite-record derivative to 1e-6 of its peak;
(iii) reverse mode is the transpose of forward mode, to 1e-10;
(iv)  the two revived defects each turn (ii) red, the value unchanged: the pole
      derivative dropped (poles held constant: the frequency-shift term is
      lost), and the residues held constant;
plus: padding to a static pole count is inert; the host sees the window bit for
bit and a failed or over-budget identification comes back as a status with an
empty mask; without x64 the map runs in complex64 and stays finite.

Moved from the research lane's E3' map
(``validation/research/fdtd_acc/e3_completion_jax.py`` and its tests, rfx-archive
``20260924_acc_E3_reverse_mode_completion_results.md``), with the lane's own
identification replaced by ``rfx.ringdown.identify``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import _ringdown_jax as rj
from rfx import ringdown as rd
from tests._x64_compat import enable_x64

DT = 1.2148552052451076e-12            # a patch board's realized dt (s)
N_WINDOWED = 8000                      # 9.72 ns: the slow modes still ring
N_START = N_WINDOWED // 2              # the window [T/2, T]
FREQS = 1.5e9 + 1.0e7 * np.arange(201)
F_REF = 4.0e9                          # decimation reference, above every mode
GUARD = 0.9
K_TEST = 12                            # 8 real poles + 4 inert ones

# (f0 Hz, Q0, dQ/Q per unit p, residue ch0, residue ch1, amplitude slope ch0, ch1)
MODES = (
    (2.03e9, 60.0, 0.4, 0.50 * np.exp(0.3j), 0.15 * np.exp(-1.1j), 0.3, -0.2),
    (2.51e9, 100.0, -0.3, 0.40 * np.exp(-0.7j), 0.50 * np.exp(2.0j), -0.5, 0.4),
    (2.95e9, 80.0, 0.2, 0.20 * np.exp(1.9j), 0.25 * np.exp(0.4j), 0.2, 0.1),
    (3.40e9, 40.0, -0.5, 0.25 * np.exp(-2.5j), 0.10 * np.exp(1.2j), -0.1, 0.6),
)


def _modes(p=0.0):
    """Poles s, residues r (K, 2) at parameter p, and their p-derivatives."""
    s, r, ds, dr = [], [], [], []
    for f0, q0, qs, r0, r1, a0, a1 in MODES:
        w = 2.0 * np.pi * f0 * np.exp(-p / 2.0)
        q = q0 * (1.0 + qs * p)
        alpha = w / (2.0 * q)
        dw = -w / 2.0
        dq = q0 * qs
        dalpha = dw / (2.0 * q) - w * dq / (2.0 * q * q)
        slope = np.array([a0, a1])
        for sign in (+1, -1):
            base = np.array([r0, r1]) if sign > 0 else np.conj([r0, r1])
            s.append(-alpha + sign * 1j * w)
            ds.append(-dalpha + sign * 1j * dw)
            r.append(base * (1.0 + slope * p))
            dr.append(base * slope)
    return np.asarray(s), np.asarray(r), np.asarray(ds), np.asarray(dr)


def _series(n):
    """Record y (n, 2) at p = 0 and its exact tangent dy/dp."""
    s, r, ds, dr = _modes()
    t = np.arange(n) * DT
    e = np.exp(np.outer(t, s))
    y = e @ r
    v = e @ dr + (t[:, None] * e * ds[None, :]) @ r
    assert np.max(np.abs(y.imag)) < 1e-12 * np.max(np.abs(y.real))
    return y.real, v.real


def _infinite(p=0.0):
    """Analytic infinite-record DFT and its p-derivative."""
    s, r, ds, dr = _modes(p)
    lam = np.exp(s * DT)
    z = np.exp(-2j * np.pi * FREQS * DT)
    den = 1.0 - lam[None, :] * z[:, None]
    y_inf = DT * (1.0 / den) @ r
    dlam = lam * ds * DT
    dy_inf = DT * ((1.0 / den) @ dr + ((z[:, None] / den ** 2) * dlam[None, :]) @ r)
    return y_inf, dy_inf


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


def _identify_window(w):
    return rd.identify(w, DT, 0, w.shape[0], freq_max=F_REF, guard=GUARD).s


def _completion(y, *, k_max=K_TEST, mutation=None, return_aux=False):
    s0, mask, _status = rj.host_poles(_identify_window, y[N_START:], k_max)
    return rj.completion(y, DT, FREQS, N_START, s0, mask, mutation=mutation,
                         return_aux=return_aux)


def _jvp(y, v, **kw):
    import jax
    import jax.numpy as jnp
    f = lambda a: _completion(a, **kw)  # noqa: E731
    p, t = jax.jit(lambda a, b: jax.jvp(f, (a,), (b,)))(jnp.asarray(y), jnp.asarray(v))
    return np.asarray(p), np.asarray(t)


def test_the_oracle_is_its_own_derivative():
    """The oracle's dY_inf/dp is the derivative of its Y_inf (central difference
    in p at 1e-6), and Q moves with p, not only f."""
    _y, dy = _infinite()
    hp = 1e-6
    fd = (_infinite(hp)[0] - _infinite(-hp)[0]) / (2.0 * hp)
    assert _rel(fd, dy) < 1e-7, _rel(fd, dy)
    s0, _r, ds, _dr = _modes()
    q_rate = [np.imag(d) / np.imag(s) - np.real(d) / np.real(s) for s, d in zip(s0, ds)]
    assert min(abs(q) for q in q_rate) > 0.1


def test_i_value_is_run_s_completion_and_the_host_poles():
    """(i) The traced value equals ``rfx.ringdown``'s numpy completion of the
    same record to 1e-12 of its peak; its poles are the host's bit for bit and
    its pole step is exactly zero."""
    import jax
    import jax.numpy as jnp
    y, _v = _series(N_WINDOWED)
    model = rd.identify(y, DT, N_START, N_WINDOWED, freq_max=F_REF, guard=GUARD)
    C_np = rd.plain_dft(y, DT, FREQS) + rd.tail_dft(model, N_WINDOWED - 1, FREQS)
    with enable_x64():
        C, aux = jax.jit(lambda a: _completion(a, return_aux=True))(jnp.asarray(y))
    C = np.asarray(C)
    assert C.dtype == np.complex128
    k = int(np.sum(np.asarray(aux["mask"])))
    assert k == model.s.size == 2 * len(MODES)
    assert np.array_equal(np.asarray(aux["s_pencil"])[:k], model.s)
    assert np.array_equal(np.asarray(aux["s"]), np.asarray(aux["s_pencil"]))
    assert np.all(np.asarray(aux["delta_s"]) == 0.0)
    assert _rel(C, C_np) < 1e-12, _rel(C, C_np)


def test_ii_jvp_is_the_infinite_record_derivative():
    """(ii) From a record whose plain derivative is > 50 % off, the JVP along
    the exact tangent is the infinite-record derivative to 1e-6 of its peak."""
    y, v = _series(N_WINDOWED)
    y_inf, dy_inf = _infinite()
    assert _rel(rd.plain_dft(v, DT, FREQS), dy_inf) > 0.5
    with enable_x64():
        C, dC = _jvp(y, v)
    print(f"\n[analytic oracle] value {_rel(C, y_inf):.2e}, JVP {_rel(dC, dy_inf):.2e} "
          "of the peak")
    assert _rel(C, y_inf) < 1e-9, _rel(C, y_inf)
    assert _rel(dC, dy_inf) < 1e-6, _rel(dC, dy_inf)


@pytest.mark.parametrize("mutation", ["poles_constant", "hold_residues"])
def test_iv_a_revived_defect_turns_the_derivative_red(mutation):
    """(iv) Dropping the pole derivative (the frequency-shift term is lost) or
    holding the residues constant fails (ii)'s bar by orders of magnitude;
    the value is unchanged by either."""
    y, v = _series(N_WINDOWED)
    y_inf, dy_inf = _infinite()
    with enable_x64():
        C, dC = _jvp(y, v, mutation=mutation)
    err = _rel(dC, dy_inf)
    print(f"\n[{mutation}] JVP {err:.2e} of the peak off the analytic derivative")
    assert _rel(C, y_inf) < 1e-9
    assert err > 1e-2, err


def test_iii_reverse_mode_is_the_transpose_of_forward_mode():
    """(iii) ``jax.grad`` (inside ``jax.jit``) of a real scalar of the completed
    spectrum, dotted with v, equals its JVP; ``jax.vjp`` with a complex
    cotangent u satisfies <vjp(u), v> = Re sum(u * jvp(v)). Both to 1e-10."""
    import jax
    import jax.numpy as jnp
    y, v = _series(N_WINDOWED)
    rng = np.random.default_rng(20260924)
    u = rng.standard_normal((FREQS.size, 2)) + 1j * rng.standard_normal((FREQS.size, 2))
    with enable_x64():
        yj, vj, uj = jnp.asarray(y), jnp.asarray(v), jnp.asarray(u)
        c_t = jnp.asarray(_infinite()[0]) * 0.8

        def loss(a):
            return jnp.sum(jnp.abs(_completion(a) - c_t) ** 2)

        g = np.asarray(jax.jit(jax.grad(loss))(yj))
        _l, dl = jax.jit(lambda a, b: jax.jvp(loss, (a,), (b,)))(yj, vj)
        lhs, rhs = float(np.sum(g * v)), float(dl)
        assert abs(lhs - rhs) <= 1e-10 * abs(rhs), (lhs, rhs)

        _C, vjp_fn = jax.vjp(_completion, yj)
        (ct,) = vjp_fn(uj)
        _C2, dC = jax.jvp(_completion, (yj,), (vj,))
        lhs = float(np.sum(np.asarray(ct) * v))
        rhs = float(np.real(np.sum(u * np.asarray(dC))))
        assert abs(lhs - rhs) <= 1e-10 * abs(rhs), (lhs, rhs)


def test_padding_is_inert():
    """Padded residues and pole derivatives are exactly 0.0 in value and in
    derivative; the unpadded (8), padded (12) and static-bound maps agree to
    1e-13 in value and JVP."""
    import jax
    import jax.numpy as jnp
    y, v = _series(N_WINDOWED)
    k_static = rd.static_pole_bound(N_WINDOWED - N_START, 2, DT, F_REF)
    assert k_static > K_TEST
    out = {}
    with enable_x64():
        for k in (2 * len(MODES), K_TEST, k_static):
            out[k] = _jvp(y, v, k_max=k)
        aux_f = lambda a: _completion(a, return_aux=True)[1]  # noqa: E731
        _a, daux = jax.jvp(lambda a: {kk: aux_f(a)[kk] for kk in ("c", "s")},
                           (jnp.asarray(y),), (jnp.asarray(v),))
        aux = aux_f(jnp.asarray(y))
    base = out[2 * len(MODES)]
    for k in (K_TEST, k_static):
        assert _rel(out[k][0], base[0]) < 1e-13
        assert _rel(out[k][1], base[1]) < 1e-13
    pad = np.asarray(aux["mask"]) == 0
    assert pad.sum() == K_TEST - 2 * len(MODES)
    assert np.all(np.asarray(aux["c"])[pad] == 0.0)
    assert np.all(np.asarray(daux["c"])[pad] == 0.0)
    assert np.all(np.asarray(daux["s"])[pad] == 0.0)


def test_the_static_bound_covers_the_pencil():
    """``static_pole_bound`` is at least the pencil's rank on this window, and
    at least the kept count (the rank bounds the eigenvalue count)."""
    y, _v = _series(N_WINDOWED)
    m = rd.identify(y, DT, N_START, N_WINDOWED, freq_max=F_REF, guard=GUARD)
    bound = rd.static_pole_bound(N_WINDOWED - N_START, 2, DT, F_REF)
    print(f"\n[bound] rank {m.rank}, kept {m.s.size}, bound {bound}, decimated "
          f"{m.n_decimated} samples")
    assert m.s.size <= m.rank <= bound


@pytest.mark.parametrize("x64", [False, True])
def test_the_host_sees_the_window_bit_for_bit(x64):
    """The window crosses to the host as raw words: the host function gets the
    traced window's dtype, shape and bits exactly, under x64 and without."""
    import contextlib

    import jax
    import jax.numpy as jnp
    seen = {}

    def spy(w):
        seen["w"] = np.array(w)
        return np.array([-1.0e8 + 2j * np.pi * 2.0e9, -1.0e8 - 2j * np.pi * 2.0e9])

    y, _v = _series(600)
    with (enable_x64() if x64 else contextlib.nullcontext()):
        dtype = jnp.float64 if x64 else jnp.float32
        w = jnp.asarray(y, dtype=dtype) * 1.000001
        s, mask, status = jax.block_until_ready(
            jax.jit(lambda a: rj.host_poles(spy, a, 4))(w))   # the callback has run
        w_host = np.asarray(w)
    assert seen["w"].dtype == w_host.dtype and seen["w"].shape == w_host.shape
    assert np.array_equal(seen["w"].view(np.uint8), w_host.view(np.uint8))
    assert int(status) == rj.STATUS_OK and np.asarray(mask).tolist() == [1, 1, 0, 0]
    assert np.asarray(s)[0] == pytest.approx(-1.0e8 + 2j * np.pi * 2.0e9, rel=1e-6)


@pytest.mark.parametrize("case", ["raises", "over_budget"])
def test_a_failed_identification_comes_back_as_a_status(case):
    """A host identification that raises, or keeps more poles than the static
    count, returns its status and an all-zero mask: nothing is raised inside
    the traced program, and no pole is silently dropped."""
    import jax
    import jax.numpy as jnp

    def host(w):
        if case == "raises":
            raise ValueError("a window of 3 decimated samples is too short")
        return np.full(5, -1.0e8 + 1j, dtype=np.complex128)

    s, mask, status = jax.jit(lambda a: rj.host_poles(host, a, 4))(
        jnp.ones((50, 2), jnp.float32))
    want = rj.STATUS_FAILED if case == "raises" else rj.STATUS_OVER_BUDGET
    assert int(status) == want
    assert np.all(np.asarray(mask) == 0.0)
    assert np.all(np.isfinite(np.asarray(s)))


def test_float32_path_is_finite_complex64():
    """Without x64 the map runs in complex64; value and JVP are finite and the
    value is within 1e-4 of the infinite-record spectrum."""
    import jax
    import jax.numpy as jnp
    if bool(jax.config.read("jax_enable_x64")):
        pytest.skip("x64 is on in this process; the float32 path is not reachable")
    y, v = _series(N_WINDOWED)
    y_inf, _dy = _infinite()
    C, dC = jax.jit(lambda a, b: jax.jvp(_completion, (a,), (b,)))(
        jnp.asarray(y, dtype=jnp.float32), jnp.asarray(v, dtype=jnp.float32))
    C, dC = np.asarray(C), np.asarray(dC)
    assert C.dtype == np.complex64
    assert np.all(np.isfinite(C)) and np.all(np.isfinite(dC))
    assert _rel(C, y_inf) < 1e-4, _rel(C, y_inf)
