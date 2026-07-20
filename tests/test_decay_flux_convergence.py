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
change landed in the core separately. See
``docs/research_notes/20260613_issue169_decay_criterion.md`` for the original
spec and the R2 gate.
"""

import json
import math
import os

import numpy as np
import pytest

try:
    # Modern JAX (scoped x64 context manager promoted to top-level).
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX (< ~0.4.31)
    from jax.experimental import enable_x64 as _enable_x64


@pytest.fixture(autouse=True)
def _scoped_x64():
    """Enable x64 PER-TEST via the context manager, NOT a module-level
    ``jax.config.update`` (which permanently flips x64 for the whole pytest
    process and leaks into downstream same-process tests — they then fail
    with lax.scan carry-dtype TypeErrors mid-suite; see #171 commit 8e4ed44
    and tests/test_waveguide_sparam_ad.py). The eps=12 guided flux DFT
    underflows to NaN in complex64, so the acceptance test needs float64 to
    measure ``flux_spectrum``; the context manager restores the prior setting
    on exit, keeping x64 scoped to this file. (The fast JSON-replay sentinel
    is unaffected — it is pure float64 numpy already.)"""
    with _enable_x64(True):
        yield


# --------------------------------------------------------------------------- #
# (1) FAST recorded-divergence witness (no FDTD; reads the diagnostic JSON).
# --------------------------------------------------------------------------- #

_DUMP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "diagnostics", "issue169_decay", "issue169_decay_vs_flux.json",
)

# The decay predicate, as implemented at rfx/simulation.py:1958-1960, in the
# #169 issue-body configuration. We mirror the exact integer schedule the
# Python loop uses (>= min_steps, on a check_interval boundary, peak>0).
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
    """
    with open(_DUMP) as fh:
        d = json.load(fh)

    ud = d["until_decay_run"]
    pf = d["point_field_decay"]
    conv = d["converged"]
    curve = d["fixed_convergence_curve"]

    steps = np.asarray(pf["step"], dtype=int)
    ratio = np.asarray(pf["ratio_sq_to_running_peak"], dtype=float)

    # (a) replay the current predicate on the recorded ratio trace.
    fired = None
    for s, r in zip(steps, ratio):
        if s >= _MIN_STEPS and 0.0 <= r < _DECAY_BY:
            fired = int(s)
            break
    assert fired is not None, "current predicate never fires on the cv03 trace"
    # The replayed fire step must match the recorded run_until_decay stop and
    # the recorded quiet step (subsampled every 50 steps, so allow one cell).
    assert abs(fired - ud["stop_step"]) <= 50, (
        f"replayed predicate fires at {fired}, recorded stop_step "
        f"{ud['stop_step']}"
    )
    assert abs(fired - pf["quiet_step_ratio_sq_below_1e-5"]) <= 50

    # (b) NON-MONOTONE: after firing, the point ratio climbs back far above the
    # decay threshold (catches a transient null, not true decay).
    after = ratio[steps >= fired]
    max_after = float(np.max(after))
    assert max_after > 1e-3, (
        f"point ratio after the stop only reached {max_after:.2e}; the #169 "
        "non-monotone transient-null signature is gone"
    )
    assert max_after > 100.0 * _DECAY_BY  # >> the threshold it just crossed

    # (c) the flux at the stop is far below the converged transmission. The
    # converged band-mean is ~0.966; the stop reads ~0.781 -> ~19% deficit.
    T_stop = ud["T_bandmean_at_stop"]
    T_conv = conv["final_band_1200units"]
    deficit = (T_conv - T_stop) / T_conv
    assert deficit > 0.10, (
        f"flux deficit at the decay stop is only {deficit:.1%}; expected the "
        "#169 large under-run (>10%)"
    )

    # And the flux DFT was still CLIMBING past the stop step: the converged
    # fixed run is much longer than the decay stop.
    assert conv["converged_n_steps"] > 1.5 * ud["stop_step"], (
        "flux converges before the decay stop -> no under-run, witness stale"
    )
    # T_peak monotonic-enough sanity: the last (longest) fixed run beats the
    # stop transmission by a wide margin.
    assert curve[-1]["T_peak"] - ud["T_peak_at_stop"] > 0.10


# --------------------------------------------------------------------------- #
# (2) ACCEPTANCE test — xfail(strict) until the criterion is made flux-aware.
#     Drives the real Simulation.run(until_decay=...) path; gpu-marked because
#     a faithful guided repro is ~50s (Python-loop per-step JIT dispatch).
# --------------------------------------------------------------------------- #

pytestmark_acceptance = [pytest.mark.gpu]

_C0 = 2.998e8

# Smallest cv03-class guided geometry that still shows the under-run. Measured
# (CPU): point stop ~2151 steps -> T_peak ~0.857, fixed-duration truth ~0.921
# (~7% deficit). Shrinking further (sx<9 or res<8) collapses the under-run.
_RES = 8
_SX = 9.0          # interior length (units of a); guide spans the whole x
_PAD = 1.75        # transverse air pad (units of a) each side
_DPML = 1.5        # UPML thickness (units of a) each side
_EPS_WG = 12.0
_WG_W = 1.0        # guide width (units of a)
_FCEN = 0.15       # c/a
_DF = 0.1
_NF = 21
_SRC_MEEP = -3.5   # source x (meep coords, origin at domain center)
_FIN_MEEP = -2.5   # input flux plane
_FOUT_MEEP = 2.5   # output flux plane
_TRUTH_UNITS = 300.0  # fixed-duration truth length (a/c0); T converged by ~300


def _build_guided(probe=False):
    """Build the minimal eps=12 guided sim (cv03-class) + 2 flux monitors.

    Returns (sim, dt_s, a_m, meep_freqs, monitor_pos_rfx).
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
def test_issue169_decay_reaches_flux_converged_value():
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
    Criterion A this PASSES (de-risk at res=8: fires near step ~4351, well past
    the old ~2151 under-run point and below the cap, with T within ~0.13% of
    truth).
    """
    _TOL = 0.02  # 2% of the converged transmission

    # Truth: fixed-duration run long enough that the flux DFT has converged.
    sim_truth, dt_s, a, meep_freqs, mon_pos = _build_guided(probe=False)
    n_truth = int(_TRUTH_UNITS * (a / _C0) / dt_s) + 150
    res_truth = sim_truth.run(
        n_steps=n_truth, subpixel_smoothing=True, skip_preflight=True,
    )
    T_truth, _ = _T_peak(res_truth, meep_freqs)

    # Decay run: the #169 configuration (until_decay=1e-5, ez at flux_out).
    sim_decay, _, _, _, _ = _build_guided(probe=False)
    decay_cap = int((_TRUTH_UNITS + 400.0) * (a / _C0) / dt_s) + 150
    res_decay = sim_decay.run(
        until_decay=1e-5,
        decay_check_interval=50,
        decay_min_steps=2000,
        decay_max_steps=decay_cap,
        decay_monitor_component="ez",
        decay_monitor_position=mon_pos,
        subpixel_smoothing=True,
        skip_preflight=True,
    )
    stop_step = int(res_decay.time_series.shape[0])
    T_stop, _ = _T_peak(res_decay, meep_freqs)

    # The decay run must NOT have hit the cap (otherwise it isn't testing the
    # stop criterion at all).
    assert stop_step < decay_cap, (
        f"decay run hit max_steps cap {decay_cap}; criterion never fired"
    )

    # Positive witness (Criterion A): the energy stop must land in the
    # CONVERGED range — well PAST the old single-cell point-field under-run
    # (~2151 steps), where the flux DFT had not yet integrated the slow tail.
    # The interior-energy criterion fires only after the energy has actually
    # left the domain, which on this geometry is ~4351 steps (de-risk). We pin
    # a generous lower bound that still excludes the old under-run point.
    assert stop_step > 3000, (
        f"decay stopped at step {stop_step}; Criterion A should fire well past "
        "the old ~2151-step point-field under-run, in the flux-converged range"
    )

    # The fix criterion: decay-stop transmission within tol of the converged
    # truth. PASSES under Criterion A (interior-energy decay lands at the
    # flux-converged stop); the pre-fix point-field stop under-ran here.
    assert abs(T_stop - T_truth) <= _TOL * abs(T_truth), (
        f"run_until_decay stopped at step {stop_step} with T_peak={T_stop:.4f}, "
        f"but the converged truth is T_peak={T_truth:.4f} "
        f"(deficit {100 * (T_truth - T_stop) / T_truth:.1f}%). The point-field "
        "stopper fired at the slow-tail transient null before the flux DFT "
        "converged (issue #169)."
    )
