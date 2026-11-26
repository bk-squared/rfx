"""Witness harness for issue #169 — ``run_until_decay`` stop quality on
low-group-velocity guided geometries. **RESOLVED** by Criterion A (total
interior-domain-energy decay); this harness keeps both the historical bug
signature and the fix honest.

Historical root cause (the PRE-FIX predicate): the old decay stopper compared
the *instantaneous squared point field at one cell*
(``val_sq = float(monitor_val) ** 2``) against ``decay_by * running_peak``.
On a dielectric guide (cv03, eps=12) the point ``ez`` at the flux_out cell
dips through a transient null between wave packets while the flux DFT — the
quantity the decay run is meant to gate — is still integrating the slow tail.
That point ratio is NON-MONOTONE (it climbs back up after the null), so it was
not a convergence witness at all, and the run stopped ~7% short of the
converged transmission. The FIX (now shipped in ``rfx/simulation.py``) makes
the absorbing-boundary stop gate on the whole-domain energy
``U = Σ(E² + H²)`` over the non-CPML interior, declared decayed once
``U < decay_by * peak_U`` on ``decay_energy_consecutive`` consecutive checks;
whole-domain energy does not pass through per-cell interference nulls, so the
stop now lands at the flux-converged value.

Two witnesses live here:

1. ``test_issue169_recorded_divergence_witness`` — FAST (<1s, no FDTD). Replays
   the OLD point-field predicate against the committed diagnostic JSON (the
   full cv03 trace produced by
   ``scripts/diagnostics/issue169_decay/reproduce_decay_vs_flux.py``) and pins
   the historical bug signature: the predicate fires at the recorded quiet
   step, the point ratio is non-monotone, and the flux at that stop is
   materially below the converged value. This is an always-pass regression
   SENTINEL that locks the recorded evidence of *why the point-field stopper
   was wrong*; it is collectable in the fast PR lane. It does NOT exercise
   ``simulation.py`` (pure JSON replay) and so cannot itself flip — proving the
   live fix is test 2's job.

2. ``test_issue169_decay_reaches_flux_converged_value`` — the ACCEPTANCE test.
   It drives the real ``Simulation.run(until_decay=...)`` path on the smallest
   faithful cv03-class guided geometry that showed the under-run, and asserts
   the decay stop now reaches the flux-converged transmission (within tol of
   the fixed-duration truth) AND lands well past the old ~2151-step
   point-field under-run. It PASSES on current main under Criterion A — it is a
   plain gate, NOT ``xfail``. It is ``pytest.mark.gpu`` because the decay path
   JIT-dispatches each step in a Python loop, so a faithful repro is ~50s+ (the
   cost floor is the ~2150-step stop, set by source cutoff + guide transit — it
   cannot be shrunk below the fast-lane budget without destroying the signature;
   measured across four geometries). It therefore lives in the gpu suite,
   matching ``test_decay_convergence.py``.

NOTE: this harness does NOT itself modify ``rfx/simulation.py``; the Criterion A
change landed in the core separately.
The geometry, source frequency and truth duration use units of a, c/a and a/c0, respectively.
"""

import json
import math
import os

import numpy as np
import pytest

try:
    from jax import enable_x64 as _enable_x64
except ImportError:
    from tests._x64_compat import enable_x64 as _enable_x64


@pytest.fixture(autouse=True)
def _scoped_x64():
    """Enable x64 PER-TEST via the context manager, NOT a module-level
    ``jax.config.update`` (which permanently flips x64 for the whole pytest
    process and leaks into downstream same-process tests — they then fail
    with lax.scan carry-dtype TypeErrors mid-suite; see #171 commit 8e4ed44
    and tests/unit/autodiff/test_waveguide_sparam_ad.py). The eps=12 guided flux DFT
    underflows to NaN in complex64, so the acceptance test needs float64 to
    measure ``flux_spectrum``; the context manager restores the prior setting
    on exit, keeping x64 scoped to this file. (The fast JSON-replay sentinel
    is unaffected — it is pure float64 numpy already.)"""
    with _enable_x64(True):
        yield



_DUMP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "scripts", "diagnostics", "issue169_decay", "issue169_decay_vs_flux.json",
)

_DECAY_BY = 1e-5
_MIN_STEPS = 2000


@pytest.mark.skipif(
    not os.path.exists(_DUMP),
    reason="issue169 diagnostic JSON not present; run "
    "scripts/diagnostics/issue169_decay/reproduce_decay_vs_flux.py",
)
def test_issue169_recorded_divergence_witness():
    """Pin the #169 bug signature from the committed cv03 diagnostic trace.

    This replays the CURRENT instantaneous-point-field predicate against the
    recorded point-field ratio curve and confirms three things, each of which
    is exactly what makes the point-field stopper the wrong tool:

      (a) the predicate fires at the recorded quiet step (~2200);
      (b) the point ratio is NON-MONOTONE — it climbs back up well above the
          decay threshold AFTER firing (the R5 transient-null witness);
      (c) the flux DFT at the stop is materially below the converged value
          (a >10% transmission deficit) — i.e. the run stopped far too early.

    Always-pass sentinel: it locks the recorded evidence so a future edit to
    the diagnostic, the geometry, or the predicate that silently erased the
    under-run would trip here.
    The recorded quiet step is subsampled every 50 steps, so the replayed fire step allows one sample interval.
    """
    with open(_DUMP) as fh:
        d = json.load(fh)

    ud = d["until_decay_run"]
    pf = d["point_field_decay"]
    conv = d["converged"]
    curve = d["fixed_convergence_curve"]

    steps = np.asarray(pf["step"], dtype=int)
    ratio = np.asarray(pf["ratio_sq_to_running_peak"], dtype=float)

    fired = None
    for s, r in zip(steps, ratio):
        if s >= _MIN_STEPS and 0.0 <= r < _DECAY_BY:
            fired = int(s)
            break
    assert fired is not None, "current predicate never fires on the cv03 trace"
    assert abs(fired - ud["stop_step"]) <= 50, (
        f"replayed predicate fires at {fired}, recorded stop_step "
        f"{ud['stop_step']}"
    )
    assert abs(fired - pf["quiet_step_ratio_sq_below_1e-5"]) <= 50

    after = ratio[steps >= fired]
    max_after = float(np.max(after))
    assert max_after > 1e-3, (
        f"point ratio after the stop only reached {max_after:.2e}; the #169 "
        "non-monotone transient-null signature is gone"
    )
    assert max_after > 100.0 * _DECAY_BY

    T_stop = ud["T_bandmean_at_stop"]
    T_conv = conv["final_band_1200units"]
    deficit = (T_conv - T_stop) / T_conv
    assert deficit > 0.10, (
        f"flux deficit at the decay stop is only {deficit:.1%}; expected the "
        "#169 large under-run (>10%)"
    )

    assert conv["converged_n_steps"] > 1.5 * ud["stop_step"], (
        "flux converges before the decay stop -> no under-run, witness stale"
    )
    assert curve[-1]["T_peak"] - ud["T_peak_at_stop"] > 0.10



pytestmark_acceptance = [pytest.mark.gpu]

_C0 = 2.998e8

_RES = 8
_SX = 9.0
_PAD = 1.75
_DPML = 1.5
_EPS_WG = 12.0
_WG_W = 1.0
_FCEN = 0.15
_DF = 0.1
_NF = 21
_SRC_MEEP = -3.5
_FIN_MEEP = -2.5
_FOUT_MEEP = 2.5
_TRUTH_UNITS = 300.0


def _build_guided(probe=False):
    """Build the minimal eps=12 guided sim (cv03-class) + 2 flux monitors.

    Returns (sim, dt_s, a_m, meep_freqs, monitor_pos_rfx).
    The source x coordinate uses the domain centre as its origin.
    """
    import jax.numpy as jnp

    from rfx import Box, Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources.sources import ModulatedGaussian

    a = 1.0e-6
    dx = a / _RES
    sy = 2.0 * (_PAD + _DPML + _WG_W / 2.0)
    interior_x = _SX
    interior_y = sy - 2.0 * _DPML
    domain_x = interior_x * a
    domain_y = interior_y * a
    cpml_n = int(_DPML * _RES)
    ox = interior_x / 2.0
    oy = interior_y / 2.0

    bw = _DF / (_FCEN * math.pi * math.sqrt(2))
    fcen_hz = _FCEN * _C0 / a
    src_x = (_SRC_MEEP + ox) * a
    fin_x = (_FIN_MEEP + ox) * a
    fout_x = (_FOUT_MEEP + ox) * a
    meep_freqs = np.linspace(_FCEN - _DF / 2, _FCEN + _DF / 2, _NF)

    sim = Simulation(
        freq_max=0.25 * _C0 / a,
        domain=(domain_x, domain_y, dx), dx=dx,
        boundary=BoundarySpec.uniform("upml"),
        cpml_layers=cpml_n, mode="2d_tmz",
    )
    sim.add_material("wg", eps_r=_EPS_WG)
    wg_lo = (oy - _WG_W / 2) * a
    wg_hi = (oy + _WG_W / 2) * a
    sim.add(Box((0, wg_lo, 0), (domain_x, wg_hi, dx)), material="wg")
    for i in range(int(_WG_W * _RES)):
        y = wg_lo + (i + 0.5) * dx
        sim.add_source(
            position=(src_x, y, 0), component="ez",
            waveform=ModulatedGaussian(
                f0=fcen_hz, bandwidth=bw,
                amplitude=1.0 / (_WG_W * _RES),
                cutoff=5.0 / math.sqrt(2),
            ),
        )
    mon_pos = (fout_x, oy * a, 0.0)
    if probe:
        sim.add_probe(position=mon_pos, component="ez")
    freqs_rfx = jnp.asarray(meep_freqs * _C0 / a)
    flux_size = (2 * _WG_W * a, 10 * dx)
    flux_center = (oy * a, dx / 2)
    sim.add_flux_monitor(axis="x", coordinate=fin_x, freqs=freqs_rfx,
                         name="flux_in", size=flux_size, center=flux_center)
    sim.add_flux_monitor(axis="x", coordinate=fout_x, freqs=freqs_rfx,
                         name="flux_out", size=flux_size, center=flux_center)

    dt_s = dx / (_C0 * math.sqrt(2)) * 0.99
    return sim, dt_s, a, meep_freqs, mon_pos


def _T_peak(res, meep_freqs):
    from rfx import flux_spectrum

    fin = np.asarray(flux_spectrum(res.flux_monitors["flux_in"]))
    fout = np.asarray(flux_spectrum(res.flux_monitors["flux_out"]))
    eps = float(np.max(np.abs(fin))) * 1e-6
    T = fout / np.where(np.abs(fin) > eps, fin, eps)
    peak_idx = int(np.argmax(np.abs(fin)))
    return float(T[peak_idx]), peak_idx


@pytest.mark.gpu
def test_issue169_decay_reaches_flux_converged_value(monkeypatch):
    """run_until_decay on a guided eps=12 sim must reach the converged flux.

    RESOLVED by Criterion A (total interior-domain-energy decay). On absorbing
    boundaries (upml here) the stop now gates on the whole-domain energy
    ``U = sum(E^2 + H^2)`` over the non-CPML interior, declared decayed once
    ``U < decay_by * peak_U`` on ``decay_energy_consecutive`` (default 2)
    consecutive checks. Unlike the old single-cell point-field stop — which
    fired at a slow-tail transient null (~step 2151, T~0.857 vs converged
    ~0.921, a ~7% under-run) — the whole-domain energy does not pass through
    per-cell interference nulls, so the stop now lands at the flux-converged
    transmission.

    SUCCESS CRITERION: the decay run's flux-DFT transmission at its stop step
    must be within ``_TOL`` of the fixed-duration converged truth. With
    Criterion A this PASSES (measured 2026-09-16 on main + #1078: fires at step
    2051 with T 0.9074101 against truth 0.9073967, 1.5e-5 relative).

    The stop step moved from ~4351 to ~2051 with two changes, and neither is a
    weakening of this gate. #1057 continued the eps=12 slab through its own
    absorber pad, so the guided mode now leaves through the PML instead of
    reflecting off a vacuum facet at the interior/pad seam, and the domain
    empties an order of magnitude sooner. #1078 then fixed the criterion's
    reference: ``peak_U`` was tracked only from ``decay_min_steps``, so on a
    domain that empties before that step the reference WAS the residual and
    the run went to the cap; the peak is now tracked from the first check,
    like the flux-stop branch beside it.
    The energy criterion must still hold by step 2500. The stop also waits
    for the echo window from the realized physical x/y domain and eps=12,
    rounded up to the next decay check.
    """
    from rfx.measurement.decay import DecayStop

    _TOL = 0.02
    check_interval = 50
    energy_step = None
    check = DecayStop.check

    def observe_check(rule, count, energy):
        nonlocal energy_step
        result = check(rule, count, energy)
        if (energy_step is None and rule.enabled and rule.n_end is not None
                and count % rule.interval == 0
                and count >= max(rule.min_steps, rule.n_d)
                and rule.below_thr >= rule.consecutive):
            energy_step = count
        return result

    monkeypatch.setattr(DecayStop, "check", observe_check)

    sim_truth, dt_s, a, meep_freqs, mon_pos = _build_guided(probe=False)
    n_truth = int(_TRUTH_UNITS * (a / _C0) / dt_s) + 150
    res_truth = sim_truth.run(
        n_steps=n_truth, subpixel_smoothing=True, skip_preflight=True,
    )
    T_truth, _ = _T_peak(res_truth, meep_freqs)

    sim_decay, _, _, _, _ = _build_guided(probe=False)
    decay_cap = int((_TRUTH_UNITS + 400.0) * (a / _C0) / dt_s) + 150
    res_decay = sim_decay.run(
        until_decay=1e-5,
        decay_check_interval=check_interval,
        decay_min_steps=2000,
        decay_max_steps=decay_cap,
        decay_monitor_component="ez",
        decay_monitor_position=mon_pos,
        subpixel_smoothing=True,
        skip_preflight=True,
    )
    stop_step = int(res_decay.time_series.shape[0])
    stop, = (d for d in res_decay.diagnostics if d.code == "until_decay.stop")
    grid = res_decay.grid
    cpml_cells = int(_DPML * _RES)
    physical_x = (grid.nx - 2 * cpml_cells - 1) * float(grid.dx)
    physical_y = (grid.ny - 2 * cpml_cells - 1) * float(grid.dx)
    diagonal = math.hypot(physical_x, physical_y)
    tau_max = 2 * diagonal * math.sqrt(_EPS_WG) / 299_792_458.0
    window_steps = math.ceil(2.05 * tau_max / float(grid.dt))
    drive_end_step = int(stop.values["drive_end_step"])
    echo_ready_step = next(
        step for step in range(1, decay_cap + 1)
        if step - max(drive_end_step, round(0.5 * step)) >= window_steps
    )
    stop_limit = (math.ceil(max(2500, echo_ready_step) / check_interval)
                  * check_interval)
    T_stop, _ = _T_peak(res_decay, meep_freqs)

    assert stop_step < decay_cap, (
        f"decay run hit max_steps cap {decay_cap}; criterion never fired"
    )

    assert energy_step is not None and energy_step <= 2500, (
        f"energy criterion first held on consecutive checks at {energy_step}; "
        "with the slab continued through its absorber pad, Criterion A must "
        "hold by step 2500. A later energy check means the domain is holding "
        "energy it should not."
    )

    assert stop_step <= stop_limit, (
        f"decay stopped at step {stop_step}; expected at most {stop_limit}, "
        "the first check at or after max(2500, echo-ready step "
        f"{echo_ready_step}) for the realized eps=12 domain."
    )

    assert abs(T_stop - T_truth) <= _TOL * abs(T_truth), (
        f"run_until_decay stopped at step {stop_step} with T_peak={T_stop:.4f}, "
        f"but the converged truth is T_peak={T_truth:.4f} "
        f"(deficit {100 * (T_truth - T_stop) / T_truth:.1f}%). The point-field "
        "stopper fired at the slow-tail transient null before the flux DFT "
        "converged (issue #169)."
    )
