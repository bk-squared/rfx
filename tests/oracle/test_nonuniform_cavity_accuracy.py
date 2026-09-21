"""Analytic-gated NonUniformGrid accuracy test (air PEC cavity, TM111).

Closes the gap flagged in ``project_mesh_strategy_decision`` (P2b: "build ONE
analytic-gated NU accuracy test before promoting any NU port out of shadow") and
confirmed by the 2026-06-18 scoping: the only committed NU-grid *physics* test on
a graded mesh — ``test_nonuniform_convergence.py::test_nonuniform_z_convergence``
— is explicitly ORACLE-FREE (a Cauchy self-consistency check on a partial-
dielectric cavity with "no simple closed-form resonance"). No committed CPU test
pinned NU-grid physics against a CLOSED-FORM analytic value via a full FDTD run.
The ``stage1_nu_cavity_physics_gate`` uses TM110, whose analytic frequency is
``p=0`` and therefore z-INDEPENDENT — so the *graded z axis* was never gated
against a number it actually moves.

This test fills that hole: an AIR-filled (eps_r=1, so the analytic ``f_mnp`` is
EXACT) rectangular PEC cavity on a genuinely z-GRADED Yee mesh, gated against the
closed form for **TM111** — a mode whose frequency depends on the graded z extent
``d`` (``cos(pi z/d)``, p=1), measured by Harminv ring-down.

SCOPE (explicit): this validates the NonUniformGrid's GRID PHYSICS only — that a
genuinely graded mesh, time-stepped with the global-min-cell dt and the CORE-C2
local-cell-width / mean-spacing curl metrics, reproduces a closed-form 3D cavity
resonance whose analytic value depends on the graded axis, to a measured
tolerance. It does NOT promote any NU PORT extractor out of shadow (the MSL/coax/
multimode-waveguide/Floquet NU port paths remain shadow/blocked). It is the
sanctioned prerequisite foundation, not a port promotion.

Convergence-to-limit is covered by the sibling oracle-free Cauchy test; this test
adds the ANALYTIC anchor the sibling explicitly lacks. Measured convergence
DIRECTION (independent evidence the agreement is genuine, not coincidental):
TM111 error 3.46% @ dx=2mm -> 2.66% @ dx=1mm (shrinks with refinement;
2026-06-18). dx=1mm is used here because a=40mm and b=35mm are then exact integer
cell counts (no in-plane dimension-snapping confound), so the residual is
dominated by the graded-z effective-d offset + coarse-mesh dispersion — exactly
what a graded-mesh accuracy gate should bound.

WHAT THE "effective-d offset" WAS (#562, 2026-08-04): the NU grid allocated one
E-node per profile cell, but N cells need N+1 bounding nodes, so the realized
z extent was ``sum(dz) - dz[-1]`` — one coarse cell short of the ``d`` this test
feeds the closed form. That was most of the residual this docstring bounded:
after the missing bounding node was restored, **TM111 error 2.66% -> 0.025%**
(same geometry, same 4:1 grading, same harminv extraction) — the remainder is
coarse-mesh dispersion, which is what the gate was meant to bound in the first
place. The 4% gate below is therefore now ~160x looser than the measured
residual; this PR makes that tightening, re-measured across resolutions and
grading ratios first rather than divided down (see the gate block in the body).

TWO ARMS, one shared rig (``_graded_cavity_tm111``): the original single fine
band, and a small-large-small-large z profile with TWO fine bands separated by a
coarse one. The two-band mesh crosses four fine/coarse transitions instead of
two and places its fine cells at different phases of the TM111 standing wave; it
was compared against a closed form only inside a cross-validation case until now
(the gap issue #810 names). Both arms use the same gate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest


class _TM111(NamedTuple):
    """What one graded-cavity arm measured. ``d`` is the REALIZED z extent."""
    d: float
    f_tm110: float
    f_tm111: float
    f_sim: float
    d_near: float
    d_second: float
    err: float
    sim_freqs: list
    mode: object


def _finest_cell_runs(dz_profile, rel_tol=1e-9):
    """Lengths of the consecutive runs of cells at the profile's FINEST size.

    ``[8]`` = one fine band; ``[8, 8]`` = two fine bands with something coarser
    between them. This is what lets the two-band arm assert that its mesh really
    has two separated fine bands, the way the single-band arm asserts
    ``grading_ratio > 2.0`` — a profile that collapsed into one band, or whose
    fine cells were smoothed away, returns a different list.
    """
    sizes = np.asarray(dz_profile, dtype=float)
    finest = float(sizes.min())
    runs, cur = [], 0
    for size in sizes:
        if abs(size - finest) <= rel_tol * finest:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _graded_cavity_tm111(dz_profile, a, b, dx, tag):
    """Run the air PEC cavity on ``dz_profile`` and extract TM111 by harminv.

    Shared by both graded arms so they cannot drift apart: same mode, same
    source/probe placement, same 8000-step harminv window, same closed form,
    and the same REALIZED-extent convention — ``d = sum(dz_profile)``, never a
    declared or raw value.

    Returns a ``_TM111`` record and prints the full extracted spectrum
    (R5: the trace, not a headline).
    """
    from rfx import Simulation, GaussianPulse
    from rfx.grid import C0

    d = float(np.sum(dz_profile))              # actual graded z extent (NOT hardcoded)

    def f_mnp(m, n, p):
        return (C0 / 2) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)

    f_tm110 = f_mnp(1, 1, 0)   # z-INDEPENDENT (p=0) — what stage1's gate uses
    f_tm111 = f_mnp(1, 1, 1)   # z-DEPENDENT (p=1) — what THIS gate uses

    sim = Simulation(
        freq_max=2 * f_tm111,
        domain=(a, b),
        boundary="pec",          # closed cavity, no CPML (cheap, no absorber phantoms)
        dx=dx,
        dz_profile=dz_profile,
    )
    # ez soft source + ez probe couple to TM-family modes (those carrying Ez).
    # Both at z=d/4 (NOT d/2, a TM111 node): cos(pi/4)=0.707 keeps TM111 visible.
    # x,y at thirds avoid the sin-nodes of TM110 and TM111.
    sim.add_source((a / 3, b / 3, d / 4), "ez",
                   waveform=GaussianPulse(f0=f_tm111, bandwidth=0.8))
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 4), "ez")

    result = sim.run(n_steps=8000)
    modes = result.find_resonances(freq_range=(0.6 * f_tm111, 1.5 * f_tm111))

    # --- R5: dump the full extracted spectrum + analytic anchors, not a headline ---
    sim_freqs = sorted(m.freq for m in modes)
    grading_ratio = max(dz_profile) / min(dz_profile)
    print(f"\n[NU-cavity/{tag}] d (graded z) = {d*1e3:.3f} mm, "
          f"grading ratio = {grading_ratio:.2f}, "
          f"finest-cell runs = {_finest_cell_runs(dz_profile)}")
    print(f"[NU-cavity/{tag}] analytic TM110(z-indep) = {f_tm110/1e9:.4f} GHz, "
          f"TM111(z-dep) = {f_tm111/1e9:.4f} GHz")
    print(f"[NU-cavity/{tag}] sim modes (GHz): {[round(f/1e9, 4) for f in sim_freqs]}")
    print(f"[NU-cavity/{tag}] sim Q: {[round(m.Q, 1) for m in sorted(modes, key=lambda m: m.freq)]}")
    print(f"[NU-cavity/{tag}] sim amplitude: "
          f"{[float(f'{m.amplitude:.4g}') for m in sorted(modes, key=lambda m: m.freq)]}")

    assert modes, "no resonances found in the TM111 band"

    # Mode-ID robustness (separation RATIO, not an absolute window — robust to small
    # cross-machine frequency shifts): TM111 = the sim mode nearest the analytic
    # value, and the match must be UNAMBIGUOUS — the 2nd-nearest sim mode must be
    # several times farther.
    by_dist = sorted(modes, key=lambda m: abs(m.freq - f_tm111))
    mode = by_dist[0]
    f_sim = mode.freq
    d_near = abs(f_sim - f_tm111)
    d_second = abs(by_dist[1].freq - f_tm111) if len(by_dist) > 1 else float("inf")
    err = d_near / f_tm111
    print(f"[NU-cavity/{tag}] TM111 f_sim = {f_sim/1e9:.4f} GHz, err = {err*100:.3f}% "
          f"(Q = {mode.Q:.1f}, amplitude = {mode.amplitude:.4g}, "
          f"2nd-nearest delta {d_second/1e9:.3f} GHz)")
    return _TM111(d, f_tm110, f_tm111, f_sim, d_near, d_second, err,
                  sim_freqs, mode)


@pytest.mark.slow
def test_nonuniform_z_graded_cavity_tm111_accuracy():
    """A genuinely z-graded NU mesh reproduces the closed-form TM111 cavity
    resonance to within a measured tolerance.

    Gate: |f_sim - f_analytic(actual d)| / f_analytic < 0.04%, DERIVED from the
    measured envelope via the shared envelope multiplier (see the gate block in
    the body for the scan and the reasoning). It was < 4% while the #562 extent
    defect dominated the residual (measured 2.66% on a
    0.25mm fine band / dx=1mm air cavity, 2026-06-18; ~1.5x margin for cross-
    machine float). Mode-ID is robust: TM111 sits >1 GHz from its nearest sim
    neighbour (measured 1.08 GHz), so the nearest-to-analytic match is unambiguous.

    LEVERAGE (honest power of this gate): the graded-z (p=1) term contributes only
    ~29% of f^2 here (the (1/d)^2 term vs (1/a)^2+(1/b)^2), so f's sensitivity to the
    graded-z extent is diluted ~sqrt. Even so, at the 0.04% gate this catches
    sub-percent graded-z metric errors (an effective-d wrong by ~0.14% =
    0.04/0.2884 already trips it) — graded-STENCIL correctness is the job of the
    CORE-C2 per-stencil guard (test_review_tier1_validation_battery.py::test_corec2_*).
    This test's role is the end-to-end full-FDTD analytic anchor the oracle-free
    sibling lacks, not a high-resolution stencil probe.

    The 2nd-nearest separation gate below also doubles as a spurious-mode guard: a
    Harminv split/ghost line inside the band would shrink the separation and trip it.
    """
    from rfx.auto_config import smooth_grading

    # Air-filled PEC cavity. a,b chosen as exact integer cell counts at dx=1mm
    # (a=40 cells, b=35 cells) so there is NO in-plane dimension-snapping bias;
    # TM111 ~7 GHz is isolated >1.3 GHz from the nearest ez-coupling TM mode
    # (TM110) — see the geometry design note in the module docstring.
    a, b = 40e-3, 35e-3
    dx = 1e-3

    # Genuinely z-GRADED profile: coarse 1mm end bands + a fine 0.25mm middle band
    # (raw 4:1 ratio), smoothed to <=1.3 per step. x,y stay uniform so the grading
    # is unambiguously on z (the axis TM111's frequency depends on).
    fine = 0.25e-3
    n_fine = int(round(2e-3 / fine))           # 2mm-wide fine band
    n_coarse = int(round(17e-3 / dx))          # ~17mm coarse band each side
    dz_raw = [dx] * n_coarse + [fine] * n_fine + [dx] * n_coarse
    dz_profile = list(smooth_grading(dz_raw, max_ratio=1.3))

    grading_ratio = max(dz_profile) / min(dz_profile)
    got = _graded_cavity_tm111(dz_profile, a, b, dx, "single-band")
    f_tm111, f_sim, err = got.f_tm111, got.f_sim, got.err

    assert grading_ratio > 2.0, (
        f"mesh not genuinely graded (max/min dz ratio {grading_ratio:.2f} <= 2) — "
        "the test would be a vacuous uniform-mesh check"
    )
    assert _finest_cell_runs(dz_profile) == [n_fine], (
        f"expected ONE fine band of {n_fine} cells, got runs "
        f"{_finest_cell_runs(dz_profile)}"
    )

    # Mode-ID robustness (separation RATIO, not an absolute window — robust to small
    # cross-machine frequency shifts). Measured: nearest delta ~0.18 GHz,
    # 2nd-nearest ~0.90 GHz (~5x), so a 3x + 0.5 GHz separation gate has
    # comfortable margin.
    assert got.d_second > 3 * got.d_near and got.d_second > 0.5e9, (
        f"TM111 mode-ID ambiguous: nearest {f_sim/1e9:.4f} GHz "
        f"(delta {got.d_near/1e9:.3f}), 2nd-nearest delta "
        f"{got.d_second/1e9:.3f} GHz — "
        "not unambiguously separated from a neighbouring mode"
    )

    # The ANALYTIC accuracy gate (the gap this test fills): the z-graded NU mesh
    # reproduces the z-DEPENDENT closed-form TM111 to within the measured tolerance.
    # DERIVED gate: measured envelope x the repo-wide envelope multiplier,
    # quantized up by the shared helper (#528/#539), in percent units.
    # Envelope 0.0252 % at this test's configuration -> 0.04 %.
    #
    # It was 4 % until #562 removed the one-cell z-extent defect that dominated
    # the residual (2.66 % -> 0.025 %, ~106x). A 4 % gate on a 0.025 % residual
    # is 160x loose — it could not catch the defect class it exists for.
    #
    # Measured surrounding sensitivity (single machine; percent error of TM111
    # vs the closed form):
    #
    #     dx = 1.0 mm, grading 4:1  ->  0.0252   <- this test
    #     dx = 1.0 mm, grading 2:1  ->  0.0164
    #     dx = 1.0 mm, uniform      ->  0.0011
    #     dx = 0.5 mm, grading 4:1  ->  0.0032
    #     dx = 2.0 mm, grading 4:1  ->  1.2567   <- 50x worse; do NOT
    #                                              re-parameterize to 2 mm
    #                                              without re-deriving the gate
    #
    # Domain-SIZE sensitivity, measured by the #573 reviewer at fixed dx and
    # fixed 4:1 grading while sweeping cavity size (z, percent error):
    #
    #     0.0373 / 0.0306 / 0.0252 / 0.0264 / 0.0213
    #
    # The residual tracks f0, so the gate binds the dispersion floor rather than
    # a fixture artifact — but the worst case is inside the gate ON THE #573
    # REVIEWER'S (unrecorded) size grid; the committed producer's documented
    # grid (validation/research/nu_cavity_gates/nu_cavity_gate_scan.py, #596)
    # measures its smallest-cavity point ABOVE this gate (0.0715% vs 0.04%) —
    # direct evidence that the "do NOT re-parameterize without re-deriving"
    # warning covers cavity SIZE as well as dx.
    #
    # Harminv window sensitivity, same source: over n_steps in {4k, 8k, 12k, 16k}
    # the error spans 0.0071 pt (xy) / 0.0110 pt (z), and the committed 8000-step
    # point is each family's MAXIMUM — so the envelope already covers estimator
    # scatter by construction rather than by luck. z's margin over that scatter
    # is ~1.35x, which makes z the more fragile of the two gates.
    #
    # The envelope is taken at THIS configuration, not over the scan: the test
    # runs one configuration, so the gate is a regression lock on it and the
    # neighbours are evidence. Both cavity tests are `slow`-marked and so run
    # only in the weekly slow lane; the envelope is therefore single-machine.
    # If another runner reds, re-measure and widen with the new datum recorded
    # — do not blanket-loosen.
    from tests._gate_policy import gate_from_envelope
    _MEASURED_ENVELOPE_PCT = 0.0252
    GATE = gate_from_envelope(_MEASURED_ENVELOPE_PCT, quantum=100) / 100.0

    # In-test FALSIFIER, added with the tightening: a gate is worth only what it
    # rejects. +-0.5 % is 12x the new gate and 8x BELOW the old one, so this
    # demonstrates discrimination the 4 % gate did not have.
    _FALSIFIER = 0.005
    # And the gate must reject the REAL historical regression, not only a
    # synthetic perturbation: the residual this configuration measured while the
    # #562 extent defect was live was 2.66 %, which the new gate rejects
    # by 66x. The old gate accepted it.
    assert 0.0266 > GATE, (
        "the derived gate would NOT have caught the #562-era residual "
        f"(2.66 % vs gate {GATE*100:.3f} %) — tightening bought nothing")

    for _sign in (+1.0, -1.0):
        _f_wrong = f_tm111 * (1.0 + _sign * _FALSIFIER)
        _err_wrong = abs(f_sim - _f_wrong) / _f_wrong
        assert _err_wrong > GATE, (
            f"falsifier failed: a {_sign*_FALSIFIER*100:+.2f}% frequency error "
            f"(anchor {_f_wrong/1e9:.4f} GHz) gives err {_err_wrong*100:.4f}% "
            f"which does NOT exceed the {GATE*100:.3f}% gate"
        )

    assert err < GATE, (
        f"NU-graded TM111 error {err*100:.4f}% >= {GATE*100:.3f}% — the "
        f"non-uniform grid does not reproduce the closed-form z-dependent "
        f"cavity resonance "
        f"(f_sim={f_sim/1e9:.4f} vs analytic={f_tm111/1e9:.4f} GHz)"
    )


@pytest.mark.slow
def test_nonuniform_z_two_fine_band_cavity_tm111_accuracy():
    """Same cavity, same closed form, but the graded z axis carries TWO fine
    bands separated by a coarse one (small-large-small-large).

    The sibling above grades z once: one fine band with a coarse band on each
    side, so the mesh has a single fine-to-coarse transition pair. A z profile
    that returns to the fine size a second time makes the solver cross four
    transitions instead of two, and the fine bands then sit at different phases
    of the TM111 ``cos(pi z/d)`` standing wave rather than symmetric about the
    node. That configuration was compared against a closed form only inside a
    cross-validation case (issue #810 names the gap); this arm keeps it.

    Everything else is the sibling's: a=40mm, b=35mm, dx=1mm, 0.25mm fine cells,
    ``smooth_grading(max_ratio=1.3)``, ``d`` from the realized profile, an ez
    source and probe at the same relative positions, 8000 steps, harminv over
    0.6-1.5 f_TM111.

    GATE: the sibling's, unchanged — ``gate_from_envelope(0.0252, quantum=100)``
    = 0.04 %. That envelope was measured on the SINGLE-band configuration, not
    on this one; this arm reports its own measured error against it and does not
    derive a gate of its own.
    """
    from rfx.auto_config import smooth_grading

    a, b = 40e-3, 35e-3
    dx = 1e-3

    # Two 2mm fine bands, 9mm of coarse cells before, between and after, so the
    # bands are genuinely separated and neither touches a PEC wall. The raw
    # extent (31mm) is smaller than the sibling's 36mm because smooth_grading
    # inserts a transition chain on BOTH sides of each fine band — twice as many
    # here — and the realized extent lands at 41.75mm, within 0.4mm of the
    # sibling's 41.37mm so both arms sit at the same TM111 (~6.7 GHz) and the
    # same cavity-size point of the sibling's recorded size sweep.
    fine = 0.25e-3
    n_fine = int(round(2e-3 / fine))           # 2mm-wide fine band, each
    n_end = int(round(9e-3 / dx))              # 9mm coarse band at each wall
    n_mid = int(round(9e-3 / dx))              # 9mm coarse band BETWEEN the bands
    dz_raw = ([dx] * n_end + [fine] * n_fine + [dx] * n_mid
              + [fine] * n_fine + [dx] * n_end)
    dz_profile = list(smooth_grading(dz_raw, max_ratio=1.3))

    grading_ratio = max(dz_profile) / min(dz_profile)
    got = _graded_cavity_tm111(dz_profile, a, b, dx, "two-band")
    f_tm111, f_sim, err = got.f_tm111, got.f_sim, got.err

    assert grading_ratio > 2.0, (
        f"mesh not genuinely graded (max/min dz ratio {grading_ratio:.2f} <= 2) — "
        "the test would be a vacuous uniform-mesh check"
    )

    # The property that distinguishes this arm from the sibling, asserted on the
    # REALIZED profile rather than on the raw one: two runs of finest cells, with
    # coarser cells between them. If smooth_grading ever merged the two bands, or
    # a future edit dropped one, the runs list changes and this reds — which is
    # also the (b)-mutation this arm was measured against.
    runs = _finest_cell_runs(dz_profile)
    assert runs == [n_fine, n_fine], (
        f"expected TWO separated fine bands of {n_fine} cells each, got runs "
        f"{runs} — the realized mesh is not small-large-small-large"
    )
    # ... and the gap between them must be real coarse cells, not one transition
    # cell: the coarse plateau survives the smoothing.
    sizes = np.asarray(dz_profile)
    finest = sizes.min()
    at_finest = np.flatnonzero(np.abs(sizes - finest) <= 1e-9 * finest)
    gap = sizes[at_finest[n_fine - 1] + 1:at_finest[n_fine]]
    assert float(gap.max()) == float(sizes.max()), (
        f"the band separation never reaches the coarsest cell "
        f"({gap.max()*1e3:.4f} mm vs {sizes.max()*1e3:.4f} mm) — the two fine "
        "bands are joined by a transition chain, not separated by a coarse band"
    )

    assert got.d_second > 3 * got.d_near and got.d_second > 0.5e9, (
        f"TM111 mode-ID ambiguous: nearest {f_sim/1e9:.4f} GHz "
        f"(delta {got.d_near/1e9:.3f}), 2nd-nearest delta "
        f"{got.d_second/1e9:.3f} GHz — "
        "not unambiguously separated from a neighbouring mode"
    )

    # SIBLING GATE, UNCHANGED. Measured on this configuration: 0.0237 %
    # (f_sim 6.72715 GHz vs closed form 6.72874 GHz, realized d = 41.749 mm),
    # i.e. inside the sibling's 0.04 % with ~1.7x margin. No envelope of this
    # arm's own is derived here.
    from tests._gate_policy import gate_from_envelope
    _SIBLING_ENVELOPE_PCT = 0.0252
    GATE = gate_from_envelope(_SIBLING_ENVELOPE_PCT, quantum=100) / 100.0

    _FALSIFIER = 0.005
    for _sign in (+1.0, -1.0):
        _f_wrong = f_tm111 * (1.0 + _sign * _FALSIFIER)
        _err_wrong = abs(f_sim - _f_wrong) / _f_wrong
        assert _err_wrong > GATE, (
            f"falsifier failed: a {_sign*_FALSIFIER*100:+.2f}% frequency error "
            f"(anchor {_f_wrong/1e9:.4f} GHz) gives err {_err_wrong*100:.4f}% "
            f"which does NOT exceed the {GATE*100:.3f}% gate"
        )

    assert err < GATE, (
        f"two-fine-band NU TM111 error {err*100:.4f}% >= {GATE*100:.3f}% — the "
        f"non-uniform grid does not reproduce the closed-form z-dependent "
        f"cavity resonance on a small-large-small-large z profile "
        f"(f_sim={f_sim/1e9:.4f} vs analytic={f_tm111/1e9:.4f} GHz, "
        f"realized d={got.d*1e3:.3f} mm)"
    )
