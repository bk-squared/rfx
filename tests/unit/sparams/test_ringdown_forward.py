"""``Simulation.forward(..., ringdown=RingdownSpec())``: the completed S inside the differentiable program (issue #1254).

Physics. The 12 x 11 x 5 mm metal box filled with eps_r 2.2 of
``test_ringdown_run.py`` rings in its TM110 mode near 12.4 GHz with a loaded Q
of about 230, an amplitude decay time of 5.9 ns. Scaling the fill's
permittivity by ``p`` moves the resonance as ``p**-1/2``, so the gradient of
``sum_f |S11(f)|^2`` over 8-18 GHz is dominated by the resonance's shift. A
record cut while the box rings has no record of where the resonance moves to:
the plain 2.86 ns (1500-step) record's gradient is -2.2 where the settled
answer is -90.36, and a 7.6 ns record's is +0.27. The completion adds the
unrecorded tail of the port voltage and current in closed form from the poles
of the record's second half, and its derivative carries the shift: the
completed 2.86 ns gradient is 0.16 % from the settled one.

What is pinned here:

* the completed S of ``forward()`` is ``run(ringdown=...)``'s completion of the
  same record, on both lanes: the host identification sees the same float64
  channels (poles and witnesses equal bit for bit), and the traced value,
  computed in float32, is within float32 rounding of the float64 one;
* every other output of ``forward()`` is byte-identical with and without
  ``ringdown=``, and ``ringdown=None`` stages no host callback;
* the report is ``None`` while traced and ``run()``'s report (W0, W1, WE ...)
  plus the traced-vs-float64 witness on a concrete result, also one returned
  by ``jax.jit``; a failed identification gives NaN, not an exception;
* ``jax.grad`` through the completion agrees with a central difference of the
  completed value (a consistency check on ONE record length, not the
  record-length witness);
* the completed gradient of a short record is the long record's within 1 %,
  the plain one is not; the gradient witness ``WE_gradient`` (the gradient
  through the [T/4, T] completion against the one through [T/2, T]) reads the
  actual gradient error, and says when it cannot be formed;
* what the completion does not cover is refused.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse
from rfx import _ringdown_jax as rj
from rfx import ringdown as rd
from rfx.ringdown import RingdownSpec, gradient_witness
from rfx.sources.sources import CWSource
from tests._x64_compat import enable_x64
from tests.unit.sparams.test_ringdown_run import FREQS, MM, N_SHORT, _box, _digest

#: The traced float32 completion against run()'s float64 one, max |S| over
#: every bin and entry: measured 3.0e-6 / 1.7e-6 (uniform, 1500 steps; JAX
#: 0.10.2 / 0.4.33), 1.5e-6 / 1.8e-6 (graded, 2500 steps) and at most 3.0e-6
#: on a three-cell port, where run()'s own complex64 accumulator round-off in
#: Result.s_params is 2.4e-6 (uniform) and 4.7e-6 (graded). Bar: 5x the largest.
BAR_VALUE_F32 = 1.5e-5
#: The same under scoped x64 (the map in complex128), 600-step records:
#: measured 3.8e-8 on the uniform lane (the complex64 rounding of run()'s
#: (1, 1, nf) S container there) and 0.0 on the graded lane. Bar: about 5x.
BAR_VALUE_X64 = 2.0e-7
#: The completed gradient's reference record and the short one.
N_GRAD_SHORT, N_GRAD_LONG = 1500, 4000


def _kw(lane):
    return {"port_s11_freqs": FREQS} if lane == "uniform" else {}


def _run_kw(lane):
    return {"s_param_freqs": FREQS if lane == "uniform" else None}


def _as_run_layout(lane, s):
    s = np.asarray(s)
    return s.reshape(1, 1, -1) if lane == "uniform" else s


# ---------------------------------------------------------------------------
# The value: run()'s completion, and nothing else moves
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _pair(lane):
    """run(ringdown=) and forward(ringdown=) of the short record, and the plain forward()."""
    n = N_SHORT[lane]
    r = _box(lane).run(n_steps=n, compute_s_params=True, skip_preflight=True,
                       ringdown=RingdownSpec(), **_run_kw(lane))
    plain = _box(lane).forward(n_steps=n, skip_preflight=True, **_kw(lane))
    f = _box(lane).forward(n_steps=n, skip_preflight=True, ringdown=RingdownSpec(),
                           **_kw(lane))
    return r, plain, f


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_the_completed_value_is_run_s(lane):
    r, _plain, f = _pair(lane)
    S_run = r.ringdown.s_params.astype(np.complex128)
    S_fwd = _as_run_layout(lane, f.ringdown.s_params).astype(np.complex128)
    assert f.ringdown.s_params.shape == np.asarray(f.s_params).shape
    assert f.ringdown.s_params.dtype == np.asarray(f.s_params).dtype
    assert np.array_equal(np.asarray(f.ringdown.freqs), np.asarray(f.freqs))
    d = float(np.max(np.abs(S_fwd - S_run)))
    rep_r, rep_f = r.ringdown.report, f.ringdown.report
    print(f"\n[{lane}, {N_SHORT[lane]} steps] |S_forward - S_run| = {d:.2e}; run()'s "
          f"accumulator round-off {rep_r.s_accumulator_roundoff:.2e}; WE {rep_f.witness('WE').value:.2e}")
    assert d <= BAR_VALUE_F32, d
    # one identification: the host saw the same channels run() saw
    assert rep_f.poles == rep_r.poles
    for name in ("W0", "W1", "WE", "W2"):
        assert rep_f.witness(name).value == rep_r.witness(name).value, name
    assert rep_f.witness("traced").value == pytest.approx(d, rel=1e-12)
    assert rep_f.witness("traced").ok and rep_f.witness("W0").value <= rd.W0_REL_BAR


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_a_three_cell_port_completes_its_whole_gap_voltage(lane):
    """A port three cells long: forward()'s traced rebuild must use the
    whole-gap voltage (the sum over the live edges), as run()'s float64 one
    does; a completion fed the midpoint cell's voltage is 0.9 off in |S|."""
    n = 1000
    r = _box(lane, cells=3).run(n_steps=n, compute_s_params=True, skip_preflight=True,
                                ringdown=RingdownSpec(), **_run_kw(lane))
    f = _box(lane, cells=3).forward(n_steps=n, skip_preflight=True,
                                    ringdown=RingdownSpec(), **_kw(lane))
    d = float(np.max(np.abs(_as_run_layout(lane, f.ringdown.s_params).astype(np.complex128)
                            - r.ringdown.s_params.astype(np.complex128))))
    print(f"\n[{lane}, 3-cell port, {n} steps] |S_forward - S_run| = {d:.2e}")
    assert r.ringdown.report.completed
    assert d <= BAR_VALUE_F32, d


def test_the_value_under_x64_is_tighter():
    """Under scoped x64 the map computes in complex128: forward() against
    run() on both lanes, 600-step records."""
    out = {}
    with enable_x64():
        for lane in ("uniform", "graded"):
            r = _box(lane).run(n_steps=600, compute_s_params=True, skip_preflight=True,
                               ringdown=RingdownSpec(), **_run_kw(lane))
            f = _box(lane).forward(n_steps=600, skip_preflight=True,
                                   ringdown=RingdownSpec(), **_kw(lane))
            out[lane] = float(np.max(np.abs(
                _as_run_layout(lane, f.ringdown.s_params).astype(np.complex128)
                - r.ringdown.s_params.astype(np.complex128))))
            assert f.ringdown.report.witness("W0").value <= rd.W0_REL_BAR
    print(f"\n[x64, 600 steps] |S_forward - S_run| {out}")
    assert max(out.values()) <= BAR_VALUE_X64, out


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_every_other_output_is_byte_identical(lane):
    _r, plain, f = _pair(lane)
    fields = [x for x in plain._fields if x != "ringdown"]
    a, b = {}, {}
    for x in fields:
        _digest(getattr(plain, x), x, a)
        _digest(getattr(f, x), x, b)
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
    differ = [k for k in a if a[k] != b[k]]
    assert not differ, differ
    assert plain.settling_db == f.settling_db
    assert plain.ringdown is None
    assert np.asarray(f.time_series).shape == (N_SHORT[lane], 1)


def test_a_model_without_probes_keeps_its_port_cell_series():
    """forward()'s uniform lane records each port's own cell when the model
    declares no probe; with ``ringdown=`` those columns are unchanged."""
    def build():
        sim = _box("uniform")
        sim._probes.clear()
        return sim

    plain = build().forward(n_steps=600, skip_preflight=True, port_s11_freqs=FREQS)
    f = build().forward(n_steps=600, skip_preflight=True, port_s11_freqs=FREQS,
                        ringdown=RingdownSpec())
    assert np.asarray(plain.time_series).shape == (600, 1)
    assert np.array_equal(np.asarray(plain.time_series), np.asarray(f.time_series))
    assert f.ringdown.report.completed


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_ringdown_none_stages_the_same_program(lane):
    """``ringdown=None`` traces to the program ``forward()`` traces without the
    argument, and it holds no host callback. (Against main 7b2fc25d the jaxpr
    text and every output were compared by hash on this box and the #1229
    patch boards, both lanes: identical.)"""
    sim = _box(lane)
    shape = tuple((sim._build_grid() if lane == "uniform"
                   else sim._build_nonuniform_grid()).shape)

    def f(e, **kw):
        r = sim.forward(n_steps=600, skip_preflight=True, eps_override=e,
                        **_kw(lane), **kw)
        return r.time_series, r.s_params

    e = jnp.full(shape, 2.2, jnp.float32)
    j_none = str(jax.make_jaxpr(lambda x: f(x, ringdown=None))(e))
    j_bare = str(jax.make_jaxpr(f)(e))
    j_ring = str(jax.make_jaxpr(lambda x: f(x, ringdown=RingdownSpec()))(e))
    assert j_none == j_bare
    assert "callback" not in j_none and "callback" in j_ring


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_is_none_while_traced():
    sim = _box("uniform")
    seen = {}

    def f(p):
        r = sim.forward(n_steps=600, skip_preflight=True, port_s11_freqs=FREQS,
                        eps_override=jnp.full(tuple(sim._build_grid().shape), 2.2,
                                              jnp.float32) * p,
                        ringdown=RingdownSpec())
        seen["report"] = r.ringdown.report
        return r.ringdown.s_params

    jax.jit(f)(jnp.float32(1.0))
    assert seen["report"] is None


#: Bins across TM110 (12.42 GHz), where the port accumulators are largest.
DENSE = np.linspace(12.30e9, 12.55e9, 101)


@pytest.mark.parametrize("lane, bins, n", [("uniform", "8-18 GHz", 4000),
                                           ("graded", "lane", 4000),
                                           ("uniform", "across TM110", 16000)])
def test_a_jitted_long_record_reads_w0_ok(lane, bins, n):
    """The ``ringdown`` output a jitted function returns is concrete, and its
    report is judged as an eager call's. The user's program rounds the port
    accumulators differently with the probe series bit-identical: measured
    (JAX 0.10.2) 26 ULP / 1.9e-6 of the peak at 4,000 steps (uniform), and
    182 ULP / 1.1e-5 at 16,000 steps with the bins across the resonance --
    over the earlier 1e-5 bar, inside W0's 1e-4 (PI decision 2026-09-25)."""
    sim = _box(lane)
    grid = sim._build_grid() if lane == "uniform" else sim._build_nonuniform_grid()
    eps = jnp.full(tuple(grid.shape), 2.2, jnp.float32)
    kw = ({} if lane == "graded" else
          {"port_s11_freqs": DENSE if bins == "across TM110" else FREQS})

    def f(p):
        return sim.forward(n_steps=n, skip_preflight=True, eps_override=eps * p,
                           ringdown=RingdownSpec(), **kw).ringdown

    out = jax.jit(f)(jnp.float32(1.0))
    assert isinstance(out, rd.RingdownForwardResult)
    rep = out.report
    w0 = rep.witness("W0")
    print(f"\n[{lane}, {bins}, jitted, {n} steps] W0 {w0.value:.3g} of the peak; ULPs "
          f"{ {k: round(v, 1) for k, v in rep.w0_ulps.items()} }")
    assert out.report is rep                             # computed once
    assert rep.completed and w0.ok and w0.value <= rd.W0_REL_BAR
    assert rep.witness("W1").ok and rep.witness("traced").ok


def test_the_report_of_a_batched_result_says_to_take_one_element():
    sim = _box("uniform")
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)

    def f(p):
        return sim.forward(n_steps=600, skip_preflight=True, port_s11_freqs=FREQS,
                           eps_override=eps * p, ringdown=RingdownSpec()).ringdown

    out = jax.vmap(f)(jnp.asarray([1.0, 1.001], jnp.float32))
    assert np.shape(out.s_params) == (2, FREQS.size)
    with pytest.raises(ValueError, match="batched.*tree_map"):
        out.report
    one = jax.tree_util.tree_map(lambda a: a[1], out)
    assert one.report.completed and one.report.witness("W0").ok


def test_a_failed_identification_gives_nan_not_an_exception(monkeypatch):
    def _no_pencil(*_a, **_k):
        raise ValueError("a window of 3 decimated samples is too short for a "
                         "matrix pencil")

    monkeypatch.setattr(rd, "_pencil", _no_pencil)
    f = _box("uniform").forward(n_steps=600, skip_preflight=True,
                                port_s11_freqs=FREQS, ringdown=RingdownSpec())
    assert np.all(np.isnan(np.asarray(f.ringdown.s_params)))
    assert np.all(np.isnan(np.asarray(f.ringdown.s_params_long)))
    assert np.all(np.isfinite(np.asarray(f.s_params)))
    rep = f.ringdown.report
    assert not rep.completed and rep.failure.startswith("identification")


def _no_pencil(*_a, **_k):
    raise ValueError("a window of 3 decimated samples is too short for a matrix pencil")


def _long_window_fails(monkeypatch):
    """The identification on [T/4, T] raises; the one on [T/2, T] runs."""
    orig = rd.identify

    def ident(series, dt, n_start, n_stop, **kw):
        if int(n_stop) - int(n_start) > 0.6 * N_GRAD_SHORT:
            raise ValueError("a window of 3 decimated samples is too short")
        return orig(series, dt, n_start, n_stop, **kw)

    monkeypatch.setattr(rd, "identify", ident)


def _value_and_grads(sim, n, *, jit=True):
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)

    def losses(p):
        r = sim.forward(n_steps=n, skip_preflight=True, port_s11_freqs=FREQS,
                        eps_override=eps * p, ringdown=RingdownSpec())
        return jnp.stack([_loss(r.ringdown.s_params),
                          _loss(r.ringdown.s_params_long)]), r.ringdown

    f = jax.jacrev(losses, has_aux=True)
    jac, rd_out = (jax.jit(f) if jit else f)(jnp.float32(1.0))
    return np.asarray(jac), rd_out


@pytest.mark.parametrize("jit", [False, True])
def test_a_failed_identification_makes_the_gradient_nan(jit, monkeypatch):
    """A failed completion reaches the gradient too: NaN, not 0.0 (a jitted
    value_and_grad used to return (nan, 0.0), an optimiser step on nothing)."""
    monkeypatch.setattr(rd, "_pencil", _no_pencil)
    sim = _box("uniform")
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)

    def loss(p):
        r = sim.forward(n_steps=600, skip_preflight=True, port_s11_freqs=FREQS,
                        eps_override=eps * p, ringdown=RingdownSpec())
        return _loss(r.ringdown.s_params)

    step = jax.value_and_grad(loss)
    v, g = (jax.jit(step) if jit else step)(jnp.float32(1.0))
    assert np.isnan(float(v)) and np.isnan(float(g)), (float(v), float(g))


def test_the_gradient_witness_is_not_formed_when_both_completions_failed(monkeypatch):
    monkeypatch.setattr(rd, "_pencil", _no_pencil)
    jac, rd_out = _value_and_grads(_box("uniform"), N_GRAD_SHORT)
    assert np.all(np.isnan(jac))
    w = gradient_witness(float(jac[0]), float(jac[1]), ringdown=rd_out)
    print(f"\n[both failed] {w.note}")
    assert not w.judged and not w.ok and math.isnan(w.value)
    assert w.note.startswith("not formed") and "against='longer_record'" in w.note
    assert "[window_start T, T]" in w.note and "[window_start/2 T, T]" in w.note


def test_the_gradient_witness_is_not_formed_when_the_long_completion_failed(monkeypatch):
    """Only [T/4, T] fails. The witness says which window and the fallback.
    An objective on ``s_params`` alone keeps a finite gradient (the long window
    is not in its program). The two-cotangent pattern of the witness (jacrev,
    or vjp with (1, 0) and (0, 1)) reads NaN in both gradients: the failed
    window's multiplier is NaN, and its zero cotangent times NaN reaches the
    channels both windows share -- loud, as the witness is not formed anyway."""
    _long_window_fails(monkeypatch)
    jac, rd_out = _value_and_grads(_box("uniform"), N_GRAD_SHORT)
    assert np.all(np.isnan(jac))
    sim = _box("uniform")
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)
    g_main = float(jax.jit(jax.grad(lambda p: _loss(sim.forward(
        n_steps=N_GRAD_SHORT, skip_preflight=True, port_s11_freqs=FREQS,
        eps_override=eps * p, ringdown=RingdownSpec()).ringdown.s_params)))(jnp.float32(1.0)))
    assert np.isfinite(g_main), g_main
    w = gradient_witness(float(jac[0]), float(jac[1]), ringdown=rd_out)
    print(f"\n[long window failed] {w.note}")
    assert not w.judged and not w.ok and math.isnan(w.value)
    assert "[window_start/2 T, T]: the identification failed" in w.note
    assert "[window_start T, T]:" not in w.note and "against='longer_record'" in w.note


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def _case_distributed():
    return _box("graded"), {"distributed": True}, "distributed=True"


def _case_no_time_series():
    return _box("graded"), {"emit_time_series": False}, "emit_time_series=False"


def _case_uniform_without_bins():
    return _box("uniform"), {"port_s11_freqs": None}, "needs port_s11_freqs"


def _case_uniform_two_ports():
    return _box("uniform", second_port=True), {}, "supports one wire port"


def _case_lumped():
    sim = _box("uniform")
    sim._ports.clear()
    sim.add_port(position=(4 * MM, 4 * MM, 2 * MM), component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=13e9, bandwidth=0.8))
    return sim, {}, r"forward\(ringdown=...\) completes wire-port S-parameters only"


def _case_source_on():
    sim = _box("uniform", pulse=CWSource(f0=10e9))
    return sim, {}, r"forward\(ringdown=...\): the wire port .* is still on"


def _case_stencil_order_4():
    return (_box("uniform", stencil_order=4), {},
            r"forward\(ringdown=...\) is not supported with stencil_order=4.*grid's dt")


def _case_not_a_spec():
    return _box("uniform"), {"ringdown": {"window_start": 0.5}}, "RingdownSpec"


@pytest.mark.parametrize("case", [_case_distributed, _case_no_time_series,
                                  _case_uniform_without_bins, _case_uniform_two_ports,
                                  _case_lumped, _case_source_on, _case_stencil_order_4,
                                  _case_not_a_spec])
def test_what_the_completion_does_not_cover_is_refused(case):
    sim, kw, match = case()
    kw.setdefault("port_s11_freqs", FREQS)
    spec = kw.pop("ringdown", RingdownSpec())
    with pytest.raises((NotImplementedError, ValueError, TypeError), match=match):
        sim.forward(n_steps=400, skip_preflight=True, ringdown=spec, **kw)


def test_a_run_that_stepped_at_another_dt_is_not_completed(monkeypatch):
    """With the stencil refusal removed, the (2,4) box steps at 0.857 of the
    grid's dt while the completion would work in the grid's: the realized
    step is checked, the completed S is NaN (not a number built on the wrong
    dt, which a jitted objective would differentiate unseen), and the report
    names the two steps."""
    monkeypatch.setattr(rd, "_refuse_stencil_order", lambda sim, caller: None)
    f = _box("uniform", stencil_order=4).forward(
        n_steps=1500, skip_preflight=True, port_s11_freqs=FREQS, ringdown=RingdownSpec())
    assert float(f.dt) != float(f.grid.dt)
    assert np.all(np.isnan(np.asarray(f.ringdown.s_params)))
    assert np.all(np.isnan(np.asarray(f.ringdown.s_params_long)))
    rep = f.ringdown.report
    print(f"\n[stencil_order=4, refusal removed] {rep.failure}")
    assert not rep.completed and rep.failure.startswith("solver_dt")
    assert rep.witness("solver_dt").value == pytest.approx(float(f.dt) / float(f.grid.dt))
    # and its gradient is NaN, not 0.0
    sim = _box("uniform", stencil_order=4)
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)
    g = jax.jit(jax.grad(lambda p: _loss(sim.forward(
        n_steps=1500, skip_preflight=True, port_s11_freqs=FREQS, eps_override=eps * p,
        ringdown=RingdownSpec()).ringdown.s_params)))(jnp.float32(1.0))
    assert np.isnan(float(g)), float(g)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def _loss(S):
    return jnp.sum(jnp.abs(S) ** 2)


@functools.lru_cache(maxsize=None)
def _grads(n):
    """d/dp of sum |S11|^2 at p = 1 (the fill's permittivity scaled by p),
    through the main completion, the early-start completion and the plain
    record, from one forward pass; and the call's ``ringdown`` output."""
    sim = _box("uniform")
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)

    def losses(p):
        r = sim.forward(n_steps=n, skip_preflight=True, port_s11_freqs=FREQS,
                        eps_override=eps * p, ringdown=RingdownSpec())
        return jnp.stack([_loss(r.ringdown.s_params), _loss(r.ringdown.s_params_long),
                          _loss(r.s_params)]), r.ringdown

    jac, rd_out = jax.jit(jax.jacrev(losses, has_aux=True))(jnp.float32(1.0))
    g_main, g_long, g_plain = (float(x) for x in np.asarray(jac))
    return g_main, g_long, g_plain, rd_out


def test_a_short_record_completed_gradient_is_the_long_one():
    """The 2.86 ns record's completed gradient is within 1 % of the 7.6 ns
    record's completed gradient; its plain gradient is not within 50 %."""
    g_s, _gl, g_p, _r = _grads(N_GRAD_SHORT)
    g_ref, _gl2, g_ref_plain, _r2 = _grads(N_GRAD_LONG)
    err_c = abs(g_s - g_ref) / abs(g_ref)
    err_p = abs(g_p - g_ref) / abs(g_ref)
    print(f"\n[gradient of sum|S11|^2 in the fill permittivity scale] completed "
          f"{N_GRAD_LONG}-step {g_ref:.5f} (plain {g_ref_plain:.5f}); {N_GRAD_SHORT}-step "
          f"completed {g_s:.5f} ({err_c:.2e} off), plain {g_p:.5f} ({err_p:.2e} off)")
    assert err_c < 1.0e-2, err_c
    assert err_p > 0.5, err_p


def test_the_gradient_witness_reads_the_actual_gradient_error():
    """WE on the gradient against the actual error of the completed gradient
    (the completed 4000-step one as reference): at 600 and 800 steps, where the
    completed gradient is 9 % and 5 % off, it reads within 2x and fails its
    1e-2 bar; at 1500 steps (0.16 % off) it passes."""
    g_ref = _grads(N_GRAD_LONG)[0]
    rows = []
    for n in (600, 800, N_GRAD_SHORT):
        g, g_long, _gp, rd_out = _grads(n)
        w = gradient_witness(g, g_long, ringdown=rd_out)
        actual = abs(g - g_ref) / abs(g_ref)
        rows.append((n, w, actual))
        print(f"\n[{n} steps] WE_gradient {w.value:.3e} ({'ok' if w.ok else 'FAILED'}), "
              f"actual {actual:.3e}, ratio {w.value / actual:.2f}")
    for n, w, actual in rows[:2]:
        assert w.judged and not w.ok, (n, w)
        assert 0.5 < w.value / actual < 2.0, (n, w.value, actual)
    n, w, actual = rows[2]
    assert w.judged and w.ok and actual < w.bar, (n, w, actual)


def test_the_gradient_witness_says_when_it_is_not_formed():
    """The pulse delayed to peak at 0.71 ns is off over [T/2, T] of a 1500-step
    record but at its peak at T/4: WE cannot be formed, and the witness says
    so -- not judged, NaN, pointing at the record 1.5-2x longer."""
    pulse = GaussianPulse(f0=13.0e9, bandwidth=0.8, cutoff=24.0)
    f = _box("uniform", pulse=pulse).forward(
        n_steps=1500, skip_preflight=True, port_s11_freqs=FREQS, ringdown=RingdownSpec())
    w = gradient_witness(1.0, 1.0, ringdown=f.ringdown)
    assert not w.judged and not w.ok and math.isnan(w.value)
    assert w.note.startswith("not formed") and "against='longer_record'" in w.note
    w2 = gradient_witness(1.0, 1.001, against="longer_record")
    assert w2.judged and w2.ok and w2.name == "record_length_gradient"


def test_the_gradient_witness_reads_every_leaf_and_fails_on_nan():
    g = {"a": np.array([1.0, -4.0]), "b": np.array([[0.5]])}
    other = {"a": np.array([1.0, -4.0]), "b": np.array([[0.46]])}
    f = _box("uniform").forward(n_steps=600, skip_preflight=True, port_s11_freqs=FREQS,
                                ringdown=RingdownSpec())
    w = gradient_witness(g, other, ringdown=f.ringdown)
    assert w.value == pytest.approx(0.04 / 4.0) and w.ok
    bad = {"a": np.array([1.0, np.nan]), "b": np.array([[0.5]])}
    w = gradient_witness(g, bad, ringdown=f.ringdown)
    assert math.isnan(w.value) and not w.ok
    with pytest.raises(TypeError, match="needs ringdown="):
        gradient_witness(g, other)
    with pytest.raises(ValueError, match="different structures"):
        gradient_witness(g, {"a": g["a"]}, ringdown=f.ringdown)


def test_the_gradient_is_the_derivative_of_the_completed_value():
    """``jax.grad`` through the completion against a central difference of the
    completed forward value, both from the SAME 1000-step record (float64
    fields under scoped x64; Richardson-extrapolated steps 2.5e-5 and 5e-5).
    A consistency check of the derivative, not the record-length witness:
    the derivative is the frozen Gauss-Newton one, the value the pencil's, and
    the two agree to the fit's residual. Measured 1.1e-4 at 1000 steps, 7e-6
    at 3000 and 1.2e-3 at 1500."""
    with enable_x64():
        sim = _box("uniform", precision="float64")
        eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float64)

        def loss(p):
            r = sim.forward(n_steps=1000, skip_preflight=True, port_s11_freqs=FREQS,
                            eps_override=eps * p, ringdown=RingdownSpec())
            return _loss(r.ringdown.s_params)

        p = jnp.float64(1.0)
        g = float(jax.jit(jax.grad(loss))(p))
        f = jax.jit(loss)
        fd = {h: (float(f(p + h)) - float(f(p - h))) / (2.0 * h) for h in (2.5e-5, 5e-5)}
    fd0 = fd[2.5e-5] - (fd[5e-5] - fd[2.5e-5]) / 3.0
    rel = abs(g - fd0) / abs(fd0)
    print(f"\n[1000 steps, float64] AD {g:.8f}, central differences {fd}, extrapolated "
          f"{fd0:.8f}: {rel:.2e}")
    assert rel < 1.0e-3, rel


def test_the_poles_derivative_is_what_carries_the_shift(monkeypatch):
    """The same short-record gradient with the pole derivative dropped (the
    revived defect: poles held constant) is far from the long record's."""
    g_ref = _grads(N_GRAD_LONG)[0]
    monkeypatch.setattr(rj, "completion",
                        functools.partial(rj.completion, mutation="poles_constant"))
    sim = _box("uniform")
    eps = jnp.full(tuple(sim._build_grid().shape), 2.2, jnp.float32)

    def loss(p):
        r = sim.forward(n_steps=N_GRAD_SHORT, skip_preflight=True, port_s11_freqs=FREQS,
                        eps_override=eps * p, ringdown=RingdownSpec())
        return _loss(r.ringdown.s_params)

    g = float(jax.jit(jax.grad(loss))(jnp.float32(1.0)))
    err = abs(g - g_ref) / abs(g_ref)
    print(f"\n[poles held constant] {N_GRAD_SHORT}-step gradient {g:.5f}, {err:.2e} off")
    assert err > 0.1, err
