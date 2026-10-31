"""Solve warnings name this caller despite runner, decorator and JAX frames."""
import functools
import inspect
import warnings

import jax
import jax.numpy as jnp
import pytest

from rfx import Box, Simulation
from rfx._warning_origin import user_stacklevel
from rfx.measurement.decay import DecayStop


def _sim(*, boundary="cpml", waveform=None, amplitude_kind="current", **kwargs):
    sim = Simulation(freq_max=10e9, domain=(.006, .006, .006), dx=.001,
                     boundary=boundary, cpml_layers=2, **kwargs)
    sim.add_source((.003, .003, .003), "ez", amplitude_kind=amplitude_kind, waveform=waveform)
    return sim


def _assert_origin(caught, text):
    hits = [w for w in caught if text in str(w.message)]
    assert hits, [str(w.message) for w in caught]
    assert all(w.filename == __file__ for w in hits), [
        (w.filename, w.lineno, str(w.message)) for w in hits]


def test_stacklevel_counts_from_warn_caller():
    # This function itself is outside rfx: level 1 must name the warn line.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        warnings.warn("caller is level one", stacklevel=user_stacklevel())
    warning, = caught
    assert warning.filename == __file__
    assert warning.lineno == line


@pytest.mark.parametrize("case,text", [
    ("min-steps", "zero energy checks"),
    ("threshold", "still rising"),
    ("report-every", "goes on without progress lines"),
    ("dc-truncated", "truncation-dominated"),
    ("dc-intrinsic", "it deposits a static charge"),
    ("float32-floor", "single-precision rounding level"),
    ("remnant-no-dc", "No soft drive has relative DC"),
    ("remnant-dc", "a static charge left by a soft drive"),
])
def test_run_warning_origin(monkeypatch, case, text):
    """Each original fixed-count site is emitted by a real plain run().

    DC census and the floor rule are deterministic inputs to their emitters;
    the warning functions and the run/runner/loop call chain stay intact.
    A one-step cap leaves E from injection and H=0 for both remnant branches.
    """
    sim = (_sim(solver="adi", boundary="pec", amplitude_kind="field") if case == "report-every"
           else _sim(waveform=jnp.ones_like))
    kwargs = dict(n_steps=1, compute_s_params=False, skip_preflight=True)
    if case == "report-every":
        kwargs["report_every"] = 1
    else:
        kwargs.update(until_decay=.001, decay_min_steps=0,
                      decay_max_steps=1, decay_check_interval=1)
    rows = []
    if case == "min-steps":
        kwargs["decay_min_steps"] = 2
    elif case == "threshold":
        kwargs["until_decay"] = 2.
    elif case in {"dc-truncated", "dc-intrinsic", "remnant-dc"}:
        rows = [("source[0]", "test pulse", .1, .01,
                 1. if case == "dc-truncated" else None)]
    elif case == "float32-floor":
        rule = DecayStop(1e-4, 0, 10, 2, 20)
        for count, value in enumerate((1e12, 1., .5, .5, .5), 1):
            rule.check(count * 10, lambda: value)
        assert rule.criterion == "float32_floor"
        rule.interval = 1000  # Preserve the prepared outcome at the one-step boundary.
        monkeypatch.setattr("rfx.stepping.uniform_loop.make_decay_stop",
                            lambda *a, **k: (rule, lambda *a: 0.))
    monkeypatch.setattr("rfx.measurement.decay.dc_census", lambda *a, **k: rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(**kwargs)
    _assert_origin(caught, text)


@pytest.mark.parametrize("entry", ["run", "forward", "jit-lambda", "jit-partial"])
def test_preflight_warning_origin(entry):
    sim = Simulation(domain=(.02, .02, .02), dx=.001, freq_max=10e9,
                     boundary="pec", snap="declared")
    sim.add_source((.002, .002, .002), "ez", amplitude_kind="current")
    sim.add_thin_conductor(Box((.004, .004, .006), (.014, .014, .006)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entry == "jit-lambda":
            jax.jit(lambda: sim.run(n_steps=1, compute_s_params=False).time_series)()
        elif entry == "jit-partial":
            # The advisory is emitted while tracing, before the existing Grid
            # output refusal; keeping the bound method is the frame regression.
            with pytest.raises(TypeError, match="not a valid JAX type"):
                jax.jit(functools.partial(sim.run, n_steps=1, compute_s_params=False))()
        else:
            getattr(sim, entry)(n_steps=1)
    _assert_origin(caught, "preflight found")


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_nonfinite_result_warning_origin(monkeypatch, entry):
    # Supply a health finding without depending on numerical divergence.
    monkeypatch.setattr("rfx.api._spec._nonfinite_fields", lambda result: [("time_series", 1)])
    sim = _sim()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(sim, entry)(n_steps=1, skip_preflight=True)
    _assert_origin(caught, "result contains non-finite values")


@pytest.mark.parametrize("with_probe", [False, True])
def test_zero_energy_warning_origin(with_probe):
    sim = _sim(waveform=jnp.zeros_like)
    if with_probe:
        sim.add_probe((.002, .002, .002), "ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)
    _assert_origin(caught, "no field energy was recorded")
