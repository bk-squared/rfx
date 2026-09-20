"""A lossless patch in an open domain must not gain energy late in the ring-down.

THE INVARIANT.  The fixture is a lossless dielectric, perfect conductors and a CPML.  After
the source is over, every physical path removes energy and none adds any, so the late-time
envelope can only decay.  A run whose envelope turns and climbs is not a resonance, a beat or
a truncated transient -- it is the update operator with an eigenvalue outside the unit circle,
and the only question is how long you have to wait to see it.

WHY THIS FIXTURE.  An isolated patch on a grounded substrate with the lateral domain padded and
a thin absorber is the configuration that produced rfx's longest unexplained growth record.  The
diagnosis (branch ``diag/801-patch-ringdown-padding``, artifacts under
``scripts/diagnostics/_artifacts/patch_pad_cpml_ringdown/``) attributed it by bisect to
``a3e4dba4`` -- the #931 lattice-ownership contract -- and then to one measurable difference:
the pre-#931 edge rule shorted a one-edge-wide ring of TANGENTIAL edges one node past each
conductor's hi-face footprint (500 of them on the reference arm), and that overhang ring plus a
thin absorber grew.  Neither alone does.  Nothing pinned any of it; the growth stopped as a side
effect of a contract change made for other reasons.  This is that pin.

WHY THE FULL RECORD, AND WHY THE GPU LANE.  The unstable mode is seeded at round-off, so it only
becomes visible once it has overtaken the decaying physical field.  How long that takes was
MEASURED on this arm rather than estimated -- truncating the same record and scoring it with this
file's own metrics.  THIS TABLE IS THE 2026-09-16 RECORD, taken on the (250, 189, 81) realization
that existed before #1136; its ``mutated`` column is that grid's and the mutated arm no longer
behaves this way (see WHAT THE MUTATION DOES NOW, below).  On today's (249, 189, 81) grid only the
150-period row has been re-measured: shipped -43.30, mutated -35.41.

    periods   steps    shipped    mutated     gate fires on the mutated arm?
       40      7109    -11.32     -14.69      no -- and the UNSTABLE arm scores BETTER
      100     17773    -29.04     -28.22      no
      120     21327    -34.87     -18.40      no (rate +2.66e-5, under the bar)
      150     26659    -43.37       0.00      YES

150 periods is therefore the FLOOR, not a margin: at 120 the SHIPPED arm has not reached the -40 dB
bar either (-34.87), on the old grid the mutated arm missed both halves of the gate there, and at
40 it looked HEALTHIER than the shipped one.  A shortened record does not make this test cheaper,
it makes it wrong.  Since #1136 the record length is bounded from ABOVE as well, for the falsifier:
see THE FALSIFIER NOW DEPENDS ON WHERE THE RECORD STOPS, below.  (An analytic crossover estimate
for the OLD grid's growing arm is deliberately not quoted here: the growth and decay rates come
from different estimators unless both are taken at this file's own blocking, and mixing them gives
a step count that disagrees with the table above.)

The full arm is 26 s on rtx4090 against ~53 min on a CPU pod, so the gate lives on the GPU lane.

TWO-SIDED, because a one-sided reflection bar has already pinned a diverging run in this repo:
the assertion is ``settling_db <= -40`` AND a negative fitted decay rate on every probe.  The
rate is fitted log-linearly on BLOCK MAXIMA -- the series oscillates at ~f0, so ``env[::k]``
aliases and can render a growing envelope flat -- over the last half of the record.

A FITTED RATE RATHER THAN A MIN-TO-END RATIO, and the reason is blocking, so the blocking is
quoted with it.  That ratio on the shipped (healthy) arm depends entirely on how coarsely the
envelope is blocked: 1.00 at 20 blocks, 1.00 at 40 (this file's ``n_blocks``), 1.36 at 80, 2.28
at 160, 9.94 at 400.  At this file's own setting a bare no-upturn check would in fact
discriminate -- 1.00 healthy against 41 mutated -- so it is not chosen because it fails here.
It is not chosen because its verdict moves with a parameter that has nothing to do with the
physics, and a finer blocking turns the TM010/TM001 beat into an "upturn" on a perfectly healthy
run.  A least-squares slope over many blocks does not have that sensitivity.

THE FALSIFIER IS IN THIS FILE.  ``test_the_gate_is_red_under_the_pre_931_edge_rule`` runs the
same arm with one mutation -- ``rfx.boundaries.pec._volume_edge_masks`` replaced by the
``a3e4dba4^`` body -- and requires THIS GATE'S OWN PREDICATE to come out red.  A gate whose red
state has never been observed is not known to measure anything (this repo has been bitten by
exactly that; see the "a physics gate can bind an artifact" lesson).  Keep the two together: if
the mutation test stops being red, this gate has stopped discriminating and the green one means
nothing.

WHAT THE MUTATION DOES NOW, AND WHAT IT DID (measured 2026-09-20, VESSL 369367262373, rtx4090,
float32, this file's ``__main__`` run unchanged at two commits; logs
``bk-workspace/.1141-ringdown/20260920T163029Z/``).  Until #1136 the falsifier required GROWTH,
a positive rate on every probe.  #1136 (#1070) sizes the grid from the declared length: this
fixture's x ratio is ``232.00000000000003`` and ``ceil`` had bought a 233rd cell, which no Box
filled and which the CPML pad extension replicated as a vacuum column through the x-hi pad.
With that cell gone the mutated arm no longer grows:

    commit      realized grid     arm       worst log rate/step   settling    gate
    eb9efeae    (250, 189, 81)    shipped        -1.934e-04       -43.37 dB   green
    eb9efeae    (250, 189, 81)    mutated        +3.002e-04         0.00 dB   red (grows)
    a6d6fce1    (249, 189, 81)    shipped        -1.926e-04       -43.30 dB   green
    a6d6fce1    (249, 189, 81)    mutated        -8.379e-05       -35.41 dB   red (settling only)

Read it exactly this far and no further.  (1) The growth record needed BOTH the pre-#931 edge rule
AND the 250-cell realization; the 250-cell realization carried the x-hi vacuum pad facet (#1070),
and the extra cell and the facet were not separated -- no run has the cell without the facet.  So
"overhang ring plus a thin absorber grew" above is the 2026-09-16 reading of a rig that also had
the facet, and it has not been re-derived without it.  (2) On the grid rfx builds today the
mutation does not make the operator unstable on this record; it slows the decay (worst rate
-8.4e-05 against -1.9e-04) and leaves the run 4.59 dB short of the -40 dB bar, so the gate is still
red under it, through its SETTLING half alone -- the rate half passes (-8.4e-05 is below the
-2.0e-05 bar).  The mutated row agrees across two GPU runs (369367262302 and 369367262373) to the
four figures the two logs print.  (3) The shipped arm moved by 0.07 dB between the two grids.  (4)
Whether the pre-#931 rule grows on some other facet-free rig is not answered here.

THE FALSIFIER NOW DEPENDS ON WHERE THE RECORD STOPS.  While the mutated arm grew, a longer
record only made it redder.  It now DECAYS, so it is red only because it is still 4.59 dB short
of the bar at step 26659, and a longer record would carry it under.  DERIVED, not measured
(review of PR #1142): with the settling increment ``(20/ln 10) * rate * d(steps)`` applied from
the tail window's start, a model that lands within ~1 dB on the shipped arm, the mutated arm
crosses -40 dB at about 170 periods if the settling-worst probe is the fastest-decaying one
(-1.501e-04) and about 187 if it is the slowest (-8.379e-05).  So ``NUM_PERIODS`` sits in a
window: at least 150 for the healthy arm to settle, and not raised without re-measuring the
mutated arm.  rfx's own truncated-ring-down advisory fires on the mutated run ("run ended at
-36.8 dB of peak ... consider until_decay or more periods") and must NOT be followed here.  With
a long enough record both arms pass the settling half, and the rate half does not separate them
(-8.4e-05 passes the -2.0e-05 bar), so on today's grid this gate tells the two edge rules apart
by a 7.89 dB difference in how far the ring-down has got by step 26659, and by nothing else.

PRECISION.  The seed argument above is a float32 round-off argument, and the lane that produced
this gate's red/green evidence pins ``JAX_ENABLE_X64=0``.  The production GPU lanes do not pin
it, so an x64 session would change the seed amplitude and could move the step at which the
mutated arm turns up.  That is covered rather than assumed: ``conftest.py``'s ``_no_x64_leak``
(#646) fails a session that has flipped x64 globally, and an x64 run could move the mutated arm's
settling figure; if that takes it under the bar the FALSIFIER fails loudly -- which is the right
failure, because it says the red state was not reproduced.

WHAT THIS GATE DOES NOT COVER.  It pins ONE point: ``n = 4`` (dx = h/4) with
``cpml_layers = 8``, the arm #931 fixed.  It is not a statement about other resolutions or other
absorber depths, and it should not be read as one -- thinner absorbers on this same fixture are
a live, separately tracked question (#801).

Related: #801 (the growth record and the live thin-absorber class), #931 (the contract change
that ended the n = 4 / 8-layer growth), #1070 / #1136 (the absorber-pad defect found on the same
fixture, whose fix removed the cell the growth record depended on), #1141 (the decision to
record this rather than re-pin the old grid).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Fixture geometry, in units of the substrate thickness, so every length lands on a lattice
# node at every resolution (the refinement-ladder property the original rig was built around).
H = 0.787e-3                  # substrate thickness; the unit of every length
EPS_R = 3.38                  # RO4003C
L_H, W_H = 11, 13             # patch 11h x 13h
DOMX_H, DOMY_H, DOMZ_H = 38, 23, 16
ZGND_H, XP0_H = 5, 16
PAD_H = 10                    # lateral padding each side, in h -- the growing arm's value
N_CELLS_PER_H = 4             # dx = h / N_CELLS_PER_H; the resolution the record was taken at
F0, BW = 8.5e9, 1.6

# 150 periods = 26659 steps.  A FLOOR for the gate (the healthy arm is at -34.87 dB at 120
# periods) and, since #1136, close to a CEILING for the falsifier: the mutated arm now decays and
# is derived to cross -40 dB at about 170-187 periods.  Do not change it without re-measuring
# both arms.  See the docstring.
NUM_PERIODS = 150.0

# Gate, two-sided.
# (1) The shipped ring-down witness's own bar.  Measured on this arm, shipped / mutated:
#     -43.37 / 0.00 dB on the (250, 189, 81) grid before #1136; -43.30 / -35.41 dB on today's
#     (249, 189, 81).
SETTLING_DB_BAR = -40.0
# (2) The late-time decay rate must be negative.  Measured worst rate per step, shipped /
#     mutated: -1.934e-4 / +3.002e-4 before #1136; -1.926e-4 / -8.379e-5 today.  (An earlier
#     revision quoted "+4.4e-4 mutated", a figure this file's harness produces on neither grid.)
#     The bar sits an order of magnitude above the shipped rate, so a rate that is merely "not
#     very negative" fails too.  It does NOT separate today's mutated arm, which passes it; the
#     settling half does.
MAX_LOG_RATE_PER_STEP = -2.0e-5


def _build(n=N_CELLS_PER_H, pad_h=PAD_H, cpml=None):
    """The isolated patch: ground plane, substrate, patch, interior dipole, parity probe quad.

    Ground and patch are declared as one-cell PEC Boxes, which is what the growth record used
    and what #931 realizes as filled slabs with walls on both faces.  Every coordinate carries
    the same -0.1 cell nudge the original rig used so that no declared bound sits exactly on a
    node (float32 half-open Box bounds are otherwise not deterministic).
    """
    from rfx import Box, Simulation
    from rfx.sources import GaussianPulse

    dx = H / n
    cpml = 2 * n if cpml is None else cpml
    nudge = -0.1 * dx
    px = py = pad_h * H
    domx, domy, domz = DOMX_H * H + 2 * px, DOMY_H * H + 2 * py, DOMZ_H * H
    L, W = L_H * H, W_H * H
    x0 = XP0_H * H + px + nudge
    y0 = (DOMY_H / 2 - W_H / 2) * H + py + nudge
    z_gnd = ZGND_H * H + nudge
    z_sub_lo = z_gnd + dx
    z_sub_hi = z_sub_lo + H
    z_patch_hi = z_sub_hi + dx

    sim = Simulation(freq_max=15e9, domain=(domx, domy, domz), dx=dx,
                     cpml_layers=cpml, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((nudge, nudge, z_gnd), (domx + nudge, domy + nudge, z_sub_lo)), material="pec")
    sim.add(Box((nudge, nudge, z_gnd), (domx + nudge, domy + nudge, z_sub_hi)),
            material="ro4003c")
    sim.add(Box((x0, y0, z_sub_hi), (x0 + L, y0 + W, z_patch_hi)), material="pec")

    z_mid = 0.5 * (z_sub_lo + z_sub_hi)
    sim.add_source(position=(x0 + 0.31 * L, y0 + W / 2 - 0.27 * W, z_mid),
                   component="ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=BW))
    xc, yc = x0 + L / 2, y0 + W / 2
    for qx, qy in ((xc - 3.5 * H, yc - 3.5 * H), (xc + 3.5 * H, yc - 3.5 * H),
                   (xc - 3.5 * H, yc + 3.5 * H), (xc + 3.5 * H, yc + 3.5 * H)):
        sim.add_probe(position=(qx, qy, z_mid), component="ez")
    return sim


def _legacy_volume_edge_masks(cell_mask, periodic):
    """``a3e4dba4^:rfx/boundaries/pec.py::tangential_edge_masks``.

    A component is PEC iff the body has an occupied neighbour along THAT COMPONENT'S OWN axis.
    Copied from the same historical body already carried by
    ``scripts/diagnostics/slow_931_farfield_attribution.py``.  On a one-cell-thick Box this
    selects only the in-plane components -- and, on the Box's hi rim, one node further out than
    the shipped incidence rule, which is the overhang ring this gate exists for.
    """
    from rfx.boundaries import pec

    return tuple(cell_mask & (pec._shift(cell_mask, axis, periodic, +1)
                              | pec._shift(cell_mask, axis, periodic, -1))
                 for axis in range(3))


def _late_time_log_rate_per_step(time_series, n_blocks=40, fit_fraction=0.5):
    """Per-probe exponential rate of the late-time envelope, from block maxima.

    Returns one rate per probe: the slope of ``log(block max)`` against step number over the
    last ``fit_fraction`` of the record.  Negative = decaying.  A non-finite sample makes the
    rate ``+inf`` for that probe rather than NaN, because a NaN reaching a ``< 0`` comparison
    reads as a pass.
    """
    raw = np.asarray(time_series, dtype=float)
    env = np.abs(np.where(np.isfinite(raw), raw, 0.0))
    blown = (~np.isfinite(raw)).any(axis=0)
    n_steps = env.shape[0]
    block = max(1, n_steps // n_blocks)
    n_full = n_steps // block
    maxima = np.array([env[i * block:(i + 1) * block].max(axis=0) for i in range(n_full)])
    centres = (np.arange(n_full) + 0.5) * block
    first = int(n_full * (1.0 - fit_fraction))
    rates = []
    for p in range(maxima.shape[1]):
        if blown[p]:
            rates.append(float("inf"))
            continue
        y, t = maxima[first:, p], centres[first:]
        ok = y > 0
        rates.append(float(np.polyfit(t[ok], np.log(y[ok]), 1)[0]) if ok.sum() > 2
                     else float("inf"))
    return rates


def _settling_db(time_series):
    """Worst-probe end-of-run envelope against peak, the shipped witness's arithmetic.

    A probe that reached inf/NaN scores ``+inf``, never NaN: a NaN reaching a ``<= -40``
    comparison reads as settled, which is the failure #885 closed.
    """
    raw = np.asarray(time_series, dtype=float)
    env = np.abs(np.where(np.isfinite(raw), raw, 0.0))
    blown = (~np.isfinite(raw)).any(axis=0)
    tail = env[int(env.shape[0] * 0.95):].max(axis=0)
    peak = env.max(axis=0)
    return max(float("inf") if bad
               else 20 * math.log10(max(float(t), 1e-300) / max(float(p), 1e-300))
               for t, p, bad in zip(tail, peak, blown))


def _run_rates(monkeypatch=None, legacy=False):
    sim = _build()
    if legacy:
        from rfx.boundaries import pec
        monkeypatch.setattr(pec, "_volume_edge_masks", _legacy_volume_edge_masks)
    # The preflight is part of the result, so it is READ rather than skipped blind: this
    # fixture is a deliberate anti-pattern (one-cell PEC volumes, a lossless dielectric in an
    # open domain, a thin absorber) and must keep saying so. An empty preflight here would mean
    # the board changed under the test, which is the thing the gate cannot afford to miss.
    findings = [str(v) for v in sim.preflight()]
    assert findings, (
        "preflight reported NOTHING on a fixture built to trip it (one-cell PEC volumes, a "
        "lossless dielectric in an open CPML domain). The board this test scores is not the "
        "board it was written for.")
    # Re-run without the advisory pass; it has already been read.
    result = sim.run(num_periods=NUM_PERIODS, skip_preflight=True)
    series = np.asarray(result.time_series)
    assert series.ndim == 2 and series.shape[1] == 4, f"unexpected probe record {series.shape}"
    return _late_time_log_rate_per_step(series), _settling_db(series), findings


@pytest.mark.gpu
@pytest.mark.slow_physics
def test_lossless_open_domain_ringdown_decays_on_every_probe():
    """The gate: a passive lossless open-domain ring-down must not gain energy.

    Two-sided on purpose. A one-sided bar has pinned a diverging run in this repo before.
    """
    rates, settling, preflight = _run_rates()
    worst = max(rates)
    assert settling <= SETTLING_DB_BAR, (
        f"ring-down did not settle: worst-probe end/peak {settling:.2f} dB against the bar "
        f"{SETTLING_DB_BAR:.0f} dB, per-probe late-time log rates {rates} per step. "
        f"Preflight on this arm, verbatim: {preflight}. See issue #801.")
    assert worst < MAX_LOG_RATE_PER_STEP, (
        f"ring-down settled but the late-time envelope is not decaying: per-probe log rates "
        f"{rates} per step, worst {worst:.3e} against the bar {MAX_LOG_RATE_PER_STEP:.1e} "
        f"(settling {settling:.2f} dB). A lossless structure in an open domain cannot gain "
        f"energy, so a non-negative rate is the update operator, not the physics. "
        f"Preflight on this arm, verbatim: {preflight}. See issue #801.")


@pytest.mark.gpu
@pytest.mark.slow_physics
def test_the_gate_is_red_under_the_pre_931_edge_rule(monkeypatch):
    """The falsifier: with the pre-#931 edge rule the gate above must be RED.

    One mutation, and it is the one the bisect landed on. It asserts the gate's own predicate,
    not growth: since #1136 the mutated arm decays, slowly, and misses the settling bar by
    4.59 dB (module docstring, 2026-09-20 table). If this ever passes quietly, the gate above
    has stopped discriminating and its green tells you nothing.
    """
    rates, settling, _preflight = _run_rates(monkeypatch=monkeypatch, legacy=True)
    gate_green = settling <= SETTLING_DB_BAR and max(rates) < MAX_LOG_RATE_PER_STEP
    assert not gate_green, (
        f"the pre-#931 edge rule no longer turns the gate red: per-probe log rates {rates} per "
        f"step (worst {max(rates):.3e} against {MAX_LOG_RATE_PER_STEP:.1e}), settling "
        f"{settling:.2f} dB against {SETTLING_DB_BAR:.0f} dB. Recorded red state on the "
        f"(249, 189, 81) grid: worst -8.379e-05, settling -35.41 dB. Either the mutation stopped "
        f"reaching the solve (check that rfx.boundaries.pec._volume_edge_masks is still what "
        f"realized_pec_edge_masks calls), or the realized fixture moved again, or NUM_PERIODS "
        f"was raised (the mutated arm decays and is derived to cross the bar at about 170-187 "
        f"periods) -- in every case the companion gate is no longer known to measure anything.")


if __name__ == "__main__":  # measurement helper, not part of the suite
    import sys
    from contextlib import contextmanager

    @contextmanager
    def _patched(legacy):
        from rfx.boundaries import pec
        original = pec._volume_edge_masks
        if legacy:
            pec._volume_edge_masks = _legacy_volume_edge_masks
        try:
            yield
        finally:
            pec._volume_edge_masks = original

    for use_legacy in (False, True):
        with _patched(use_legacy):
            sim_ = _build()
            res_ = sim_.run(num_periods=NUM_PERIODS, skip_preflight=True)
            ts_ = np.asarray(res_.time_series)
        r = _late_time_log_rate_per_step(ts_)
        settle = _settling_db(ts_)
        print(f"n={N_CELLS_PER_H} pad={PAD_H} periods={NUM_PERIODS} steps={ts_.shape[0]} "
              f"legacy={use_legacy!s:5s} -> rates {[f'{v:+.3e}' for v in r]} "
              f"worst {max(r):+.3e} settling {settle:.2f} dB", flush=True)
    sys.exit(0)
