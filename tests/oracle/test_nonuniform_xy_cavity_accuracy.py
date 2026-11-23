"""Analytic-gated in-plane (x/y) NonUniformGrid accuracy test (air PEC cavity, TM110).

Closes the top-ranked blind spot in issue #403: in-plane NU grading
(``dx_profile`` / ``dy_profile``) shipped with NO binding pytest physics oracle.
``test_nonuniform_xy.py::test_end_to_end_nonuniform_xy_runs`` asserts only finite
fields + a nonzero probe amplitude (a CONTRACT smoke test), and the AD sentinels
assert only that gradients "flow" — exactly the ungated-graded-axis trap class
that the sibling ``test_nonuniform_cavity_accuracy.py`` was created to close for
the *z* axis (TM111). That fix landed for ``dz_profile``; ``dx_profile`` /
``dy_profile`` never got the analog. This module is that analog.

WHY TM110 (and not TM111): a rectangular PEC cavity's analytic resonance is
``f_mnp = (c/2)*sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)``. The z-sibling had to reach for
TM111 (p=1) because TM110's frequency is z-INDEPENDENT — so a graded *z* axis
would never move it. The situation inverts for the in-plane axes: TM110 (p=0) is
z-independent but its frequency depends on BOTH in-plane extents ``a = sum(dx)``
and ``b = sum(dy)``. Keeping z uniform+thin and grading x AND y therefore puts
the ENTIRE mode frequency (100% of f^2) on the graded axes — full leverage on the
axes under test, cleaner than the z-sibling whose graded term was only ~29% of
f^2 (diluted ~sqrt). An ``ez`` soft source + ``ez`` probe couple to the TM (Ez)
family; TM110 is isolated ~2.6 GHz below the next Ez mode (TM210), so mode-ID is
unambiguous.

SCOPE (explicit): this validates the NonUniformGrid's in-plane GRID PHYSICS only —
that a genuinely x- and y-graded mesh, time-stepped with the global-min-cell dt
and the CORE-C2 local-cell-width / mean-spacing curl metrics, reproduces a
closed-form 3D cavity resonance whose analytic value is set entirely by the graded
in-plane extents, to a measured tolerance. It does NOT promote any NU port
extractor out of shadow. It is the in-plane analog of the sanctioned z-axis
foundation gate, not a port promotion.

MEASURED ERROR + FAITHFULNESS FINDING (recorded on main @ 199aafc, 2026-07-20,
float32, harminv via ``find_resonances``; NOT raw rfft-argmax — the #396 audit
proved argmax gates are FFT-bin-blind, harminv resolves <0.05%):

  * x&y-GRADED (ratio 4:1 -> a=45.375mm, b=39.375mm): TM110 f_sim = 5.1627 GHz vs
    analytic 5.0404 GHz -> signed error **+2.43%** (stable to 0.002% across
    8000/12000/16000 steps; the mode is high-Q clean).
  * UNIFORM mesh, SAME extents (grading OFF): TM110 signed error **+2.47%**.

The +2.4% bias was therefore NOT introduced by grading — it was present, to
within 0.04 pt, on the ungraded profile of the same coarse resolution. That
comparison was worth making, but read it for what it shows: **both legs shared a
baseline that #562 dominated**, so it established that grading did not ADD to a
2.4% defect, not that grading is free.

With the defect gone the same comparison resolves the real grading term, measured
at IDENTICAL extents on this harness: **graded 4:1 = 0.0282%, ungraded = 0.0084%
— grading adds ~0.02 pt, a factor 3.35x above the uniform floor.** So grading is
now the DOMINANT contribution rather than a null effect; both numbers remain
negligible against the 0.05% gate, which is the honest form of the original
claim. (Caveat checked, not assumed: matching the graded extents forces uniform
cells of 1.0083/1.0096 mm, +0.83%/+0.96% over the nominal 1 mm — far too small to
account for 3.35x. Corroborating points at the ~0.01% floor: the round-1 uniform
leg 0.0108% and #564's committed -0.007%.)

Convergence direction (independent evidence the agreement is genuine): the error
is dominated by the coarse 1mm bulk, not the 0.25mm fine band.

WHAT THAT SHARED BASELINE ACTUALLY WAS — correction (#562, 2026-08-04). This
docstring attributed the +2.4% to "the standard coarse-Yee PEC-cavity
extent-convention bias (~1 cell)". **There is no such Yee/PEC convention bias,
and that attribution is withdrawn.** Both legs above ran through the NU grid
builder (a *uniform profile* still routes to ``make_nonuniform_grid``), which
allocated one E-node per profile cell where N cells need N+1 bounding nodes — so
both realized ``sum(dx) - dx[-1]``, one cell narrow, which is exactly the ~2.3%
seen. Measured, same PEC box at integer cell counts (45 x 39 cells, dx=1mm,
TM110 by harminv), before the fix: uniform builder **-0.007%**, NU builder
**+2.469%** (and -0.008% against the closed form at the extent it had actually
built — both solvers were accurate; only one built the requested cavity). After
restoring the bounding node the two builders are bit-identical on an identical
mesh, and THIS test reads **0.028%** (was 2.43%) on the same graded geometry.
That left the then-3.5% gate ~125x looser than the measured residual; this PR
makes the tightening, deriving 0.05% from the measured envelope (see the gate
block in the body).

LEVERAGE (honest power of this gate): because TM110's frequency is set 100% by the
graded in-plane extents, this gate catches graded-metric errors on x/y directly
(no sqrt-dilution). It is a full-FDTD end-to-end analytic anchor — the coverage
the #403 blind-spot lacked. Since the gate became 0.05% it IS a sub-percent
probe, and because there is no dilution here the numbers follow the gate almost
directly: TM110's f^2 shares are 0.4296 (a-axis) and 0.5704 (b-axis), so tripping
0.05% takes an effective-extent error of 0.05% (both axes together) to 0.12%
(a-axis alone), where the old 3.5% gate needed 3.5-8.1%. (An earlier revision of
this paragraph imported the z-sibling's sqrt-diluted 12-17% figure into this
file, two sentences after asserting there is no dilution, and rescaled it by the
z gate's ratio — corrected here from the closed form.) (The per-stencil CORE-C2 guard,
test_review_tier1_validation_battery.py::test_corec2_*, still covers the
stencil-level question this end-to-end test cannot isolate.)
"""

from __future__ import annotations

import numpy as np
import pytest


def test_nonuniform_xy_graded_cavity_tm110_accuracy():
    """A genuinely x- AND y-graded NU mesh reproduces the closed-form TM110 cavity
    resonance to within a measured tolerance.

    Gate: |f_sim - f_analytic(actual a,b)| / f_analytic < 0.05%, DERIVED from the
    measured envelope via the shared envelope multiplier (see the gate block in
    the body for the scan, the reasoning, and the limits). It was < 3.5% while
    the #562 one-cell extent defect dominated the residual (2.43% then, 0.028%
    now — an ~87x improvement on unchanged geometry).

    The error remains grading-INVARIANT, which is the finding this test exists
    for. What it bounds is the coarse-mesh DISPERSION floor: the "coarse-Yee
    PEC-extent floor" this paragraph used to name was the #562 grid defect, and
    the module docstring above withdraws that attribution — both legs of the
    2.43% == 2.47% comparison had gone through the NU builder.

    Mode-ID is robust: TM110 sits >2.5 GHz from its nearest Ez-coupling neighbour
    (TM210), so the nearest-to-analytic match is unambiguous. The separation gate
    doubles as a spurious-mode guard — a harminv split/ghost line inside the band
    would shrink the separation and trip it.

    Discrimination is proven IN-TEST (not asserted by construction): a solver or
    analytic wrong by +/-0.5% is shown to trip the accuracy gate (it was +/-10%
    against the old 3.5% gate; +/-10% would clear a 0.05% gate by 200x and prove
    nothing about it). Anti-vacuity is
    enforced by a genuine-grading assert (max/min cell ratio > 2 on both in-plane
    axes) — verified externally that a near-uniform profile (ratio 1.11) FAILS it,
    so the gate cannot silently degrade into a uniform-mesh check.
    The envelope is deliberately taken at the committed configuration rather than over the whole scan, so the gate is a regression lock on that configuration.
    """
    from rfx import Simulation, GaussianPulse
    from rfx.auto_config import smooth_grading
    from rfx.grid import C0

    BOUNDARY_CELL_SIZE_M = 1e-3
    dx = BOUNDARY_CELL_SIZE_M
    fine = 0.25e-3

    def graded_profile(coarse_mm, fine_mm):
        """Coarse 1mm end bands + a fine 0.25mm middle band (raw 4:1), smoothed to
        <=1.3 per step. The final max/min ratio stays 4 (coarse cells untouched)."""
        n_coarse = int(round(coarse_mm * 1e-3 / dx))
        n_fine = int(round(fine_mm * 1e-3 / fine))
        raw = [dx] * n_coarse + [fine] * n_fine + [dx] * n_coarse
        return list(smooth_grading(raw, max_ratio=1.3))

    dx_prof = graded_profile(coarse_mm=19.0, fine_mm=2.0)
    dy_prof = graded_profile(coarse_mm=16.0, fine_mm=2.0)
    dz_prof = [dx] * 10

    a = float(np.sum(dx_prof))
    b = float(np.sum(dy_prof))
    d = float(np.sum(dz_prof))
    ratio_x = max(dx_prof) / min(dx_prof)
    ratio_y = max(dy_prof) / min(dy_prof)

    def f_mnp(m, n, p):
        return (C0 / 2) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)

    f_tm110 = f_mnp(1, 1, 0)
    f_tm210 = f_mnp(2, 1, 0)
    f_tm120 = f_mnp(1, 2, 0)

    sim = Simulation(
        freq_max=2 * f_tm110,
        domain=(0, 0, 0),
        boundary="pec",
        dx=dx,
        dz_profile=np.asarray(dz_prof),
        dx_profile=np.asarray(dx_prof),
        dy_profile=np.asarray(dy_prof),
    )
    sim.add_source((a / 3, b / 3, d / 2), "ez",
                   waveform=GaussianPulse(f0=f_tm110, bandwidth=0.8))
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), "ez")

    result = sim.run(n_steps=8000)
    modes = result.find_resonances(freq_range=(0.6 * f_tm110, 1.5 * f_tm110))

    sim_freqs = sorted(m.freq for m in modes)
    print(f"\n[NU-xy-cavity] a(x graded)={a*1e3:.3f} mm, b(y graded)={b*1e3:.3f} mm, "
          f"d(z unif)={d*1e3:.3f} mm; grading ratio x={ratio_x:.2f} y={ratio_y:.2f}")
    print(f"[NU-xy-cavity] analytic TM110={f_tm110/1e9:.4f} TM210={f_tm210/1e9:.4f} "
          f"TM120={f_tm120/1e9:.4f} GHz")
    print(f"[NU-xy-cavity] sim modes (GHz, Q): "
          f"{[(round(m.freq/1e9, 4), round(m.Q, 0)) for m in sorted(modes, key=lambda x: x.freq)]}")

    assert modes, "no resonances found in the TM110 band"

    assert ratio_x > 2.0 and ratio_y > 2.0, (
        f"mesh not genuinely graded in-plane (max/min ratio x={ratio_x:.2f} "
        f"y={ratio_y:.2f}; need >2) — the test would be a vacuous uniform check"
    )

    by_dist = sorted(sim_freqs, key=lambda f: abs(f - f_tm110))
    f_sim = by_dist[0]
    d_near = abs(f_sim - f_tm110)
    d_second = abs(by_dist[1] - f_tm110) if len(by_dist) > 1 else float("inf")
    assert d_second > 3 * d_near and d_second > 0.5e9, (
        f"TM110 mode-ID ambiguous: nearest {f_sim/1e9:.4f} GHz (delta {d_near/1e9:.3f}), "
        f"2nd-nearest {by_dist[1]/1e9:.4f} GHz (delta {d_second/1e9:.3f}) — "
        "not unambiguously separated from a neighbouring mode"
    )
    err = d_near / f_tm110
    print(f"[NU-xy-cavity] TM110 f_sim={f_sim/1e9:.4f} GHz, err={err*100:.3f}% "
          f"(2nd-nearest delta {d_second/1e9:.3f} GHz, sep ratio {d_second/d_near:.1f}x)")

    from tests._gate_policy import gate_from_envelope
    _MEASURED_ENVELOPE_PCT = 0.0282
    GATE = gate_from_envelope(_MEASURED_ENVELOPE_PCT, quantum=100) / 100.0

    _FALSIFIER = 0.005
    assert 0.0243 > GATE, (
        "the derived gate would NOT have caught the #562-era residual "
        f"(2.43 % vs gate {GATE*100:.3f} %) — tightening bought nothing")

    for sign in (+1.0, -1.0):
        f_wrong = f_tm110 * (1.0 + sign * _FALSIFIER)
        err_wrong = abs(f_sim - f_wrong) / f_wrong
        assert err_wrong > GATE, (
            f"falsifier failed: a {sign*_FALSIFIER*100:+.2f}% frequency error "
            f"(anchor {f_wrong/1e9:.4f} GHz) gives err {err_wrong*100:.4f}% which "
            f"does NOT exceed the {GATE*100:.3f}% gate — the gate would not catch it"
        )

    assert err < GATE, (
        f"NU in-plane-graded TM110 error {err*100:.4f}% >= {GATE*100:.3f}% — the "
        f"non-uniform x/y grid does not reproduce the closed-form cavity resonance "
        f"whose frequency is set by the graded in-plane extents "
        f"(f_sim={f_sim/1e9:.4f} vs analytic={f_tm110/1e9:.4f} GHz)"
    )
