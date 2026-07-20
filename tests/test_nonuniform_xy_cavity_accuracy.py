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

The +2.4% bias is therefore NOT introduced by grading — it is present, to within
0.04 pt, on the uniform mesh of the same coarse resolution. It is the standard
coarse-Yee PEC-cavity extent-convention bias (~1 cell: the tangential-Ez cavity
is ~1 cell narrower than the nominal ``sum(dx)`` extent; on a ~45mm/1mm cavity a
1-cell narrowing is ~2.3%, which is what we see). The graded run reproducing the
uniform run's accuracy is the POSITIVE finding: in-plane ``dx_profile`` /
``dy_profile`` grading is faithful — it does not corrupt the resonance beyond the
uniform-mesh discretization floor. Convergence direction (independent evidence the
agreement is genuine): the error is dominated by the coarse 1mm bulk, not the
0.25mm fine band, exactly what a graded-mesh accuracy gate should bound.

LEVERAGE (honest power of this gate): because TM110's frequency is set 100% by the
graded in-plane extents, this gate catches graded-metric errors on x/y directly
(no sqrt-dilution). It is a full-FDTD end-to-end analytic anchor — the coverage
the #403 blind-spot lacked — not a sub-percent stencil probe (that is the CORE-C2
per-stencil guard's job, test_review_tier1_validation_battery.py::test_corec2_*).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_nonuniform_xy_graded_cavity_tm110_accuracy():
    """A genuinely x- AND y-graded NU mesh reproduces the closed-form TM110 cavity
    resonance to within a measured tolerance.

    Gate: |f_sim - f_analytic(actual a,b)| / f_analytic < 3.5% (measured 2.43% on a
    4:1 graded x/y air cavity, main @199aafc 2026-07-20; ~1.44x margin for
    cross-machine float32 drift, matching the z-sibling's ~1.5x discipline). The
    error is grading-INVARIANT (uniform-mesh same extents = 2.47%), so the gate
    bounds the coarse-Yee PEC-extent floor, and a grading regression that corrupted
    the in-plane metric beyond that floor would trip it.

    Mode-ID is robust: TM110 sits >2.5 GHz from its nearest Ez-coupling neighbour
    (TM210), so the nearest-to-analytic match is unambiguous. The separation gate
    doubles as a spurious-mode guard — a harminv split/ghost line inside the band
    would shrink the separation and trip it.

    Discrimination is proven IN-TEST (not asserted by construction): a solver or
    analytic wrong by +/-10% is shown to trip the accuracy gate. Anti-vacuity is
    enforced by a genuine-grading assert (max/min cell ratio > 2 on both in-plane
    axes) — verified externally that a near-uniform profile (ratio 1.11) FAILS it,
    so the gate cannot silently degrade into a uniform-mesh check.
    """
    from rfx import Simulation, GaussianPulse
    from rfx.auto_config import smooth_grading
    from rfx.grid import C0

    dx = 1e-3
    fine = 0.25e-3

    def graded_profile(coarse_mm, fine_mm):
        """Coarse 1mm end bands + a fine 0.25mm middle band (raw 4:1), smoothed to
        <=1.3 per step. The final max/min ratio stays 4 (coarse cells untouched)."""
        n_coarse = int(round(coarse_mm * 1e-3 / dx))
        n_fine = int(round(fine_mm * 1e-3 / fine))
        raw = [dx] * n_coarse + [fine] * n_fine + [dx] * n_coarse
        return list(smooth_grading(raw, max_ratio=1.3))

    # Grade BOTH x and y (a != b so TM110 is isolated from TM210/TM120); keep z
    # uniform+thin (TM110 has p=0 -> z-independent, so z carries no frequency and
    # 10 uniform cells are enough for a valid 3D Yee cavity).
    dx_prof = graded_profile(coarse_mm=19.0, fine_mm=2.0)
    dy_prof = graded_profile(coarse_mm=16.0, fine_mm=2.0)
    dz_prof = [dx] * 10

    a = float(np.sum(dx_prof))              # actual graded x extent (NOT hardcoded)
    b = float(np.sum(dy_prof))              # actual graded y extent (NOT hardcoded)
    d = float(np.sum(dz_prof))
    ratio_x = max(dx_prof) / min(dx_prof)
    ratio_y = max(dy_prof) / min(dy_prof)

    def f_mnp(m, n, p):
        return (C0 / 2) * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)

    f_tm110 = f_mnp(1, 1, 0)   # z-INDEPENDENT (p=0), set 100% by in-plane a,b
    f_tm210 = f_mnp(2, 1, 0)   # nearest Ez-coupling neighbour
    f_tm120 = f_mnp(1, 2, 0)

    sim = Simulation(
        freq_max=2 * f_tm110,
        domain=(0, 0, 0),        # overridden by the three axis profiles
        boundary="pec",          # closed cavity, no CPML (energy-conserving, cheap)
        dx=dx,                   # required boundary cell size for in-plane profiles
        dz_profile=np.asarray(dz_prof),
        dx_profile=np.asarray(dx_prof),
        dy_profile=np.asarray(dy_prof),
    )
    # ez soft source + ez probe couple to the TM (Ez) family. Source at (a/3, b/3),
    # probe at (2a/3, 2b/3): sin(pi/3)=sin(2pi/3)=0.866 keeps TM110 (and TM210)
    # visible while avoiding the wall nodes. z at d/2 is arbitrary (p=0, uniform-z).
    sim.add_source((a / 3, b / 3, d / 2), "ez",
                   waveform=GaussianPulse(f0=f_tm110, bandwidth=0.8))
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), "ez")

    result = sim.run(n_steps=8000)
    modes = result.find_resonances(freq_range=(0.6 * f_tm110, 1.5 * f_tm110))

    # --- R5: dump the full extracted spectrum + analytic anchors, not a headline ---
    sim_freqs = sorted(m.freq for m in modes)
    print(f"\n[NU-xy-cavity] a(x graded)={a*1e3:.3f} mm, b(y graded)={b*1e3:.3f} mm, "
          f"d(z unif)={d*1e3:.3f} mm; grading ratio x={ratio_x:.2f} y={ratio_y:.2f}")
    print(f"[NU-xy-cavity] analytic TM110={f_tm110/1e9:.4f} TM210={f_tm210/1e9:.4f} "
          f"TM120={f_tm120/1e9:.4f} GHz")
    print(f"[NU-xy-cavity] sim modes (GHz, Q): "
          f"{[(round(m.freq/1e9, 4), round(m.Q, 0)) for m in sorted(modes, key=lambda x: x.freq)]}")

    assert modes, "no resonances found in the TM110 band"

    # Anti-vacuity: the mesh must be GENUINELY graded on both in-plane axes, else
    # this silently becomes a uniform-mesh check (covered elsewhere) with zero
    # dx_profile/dy_profile coverage — the exact vacuity #403 flags.
    assert ratio_x > 2.0 and ratio_y > 2.0, (
        f"mesh not genuinely graded in-plane (max/min ratio x={ratio_x:.2f} "
        f"y={ratio_y:.2f}; need >2) — the test would be a vacuous uniform check"
    )

    # Mode-ID robustness (separation RATIO, not an absolute window, and NOT
    # amplitude rank): TM110 = the sim mode nearest the analytic value, and the
    # match must be UNAMBIGUOUS — the 2nd-nearest sim mode must be several times
    # farther. Measured: nearest delta ~0.12 GHz, 2nd-nearest ~2.76 GHz (~22x).
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

    # In-test FALSIFIER (proves the gate has discrimination power, not vacuous):
    # a solver/analytic wrong by +/-10% MUST trip the accuracy gate. +/-10% clears
    # the gate (3.5%) plus the base +2.4% bias in either sign, so both directions
    # fail unambiguously (measured: +10% -> 6.9% err, -10% -> 13.8% err).
    GATE = 0.035
    for sign in (+1.0, -1.0):
        f_wrong = f_tm110 * (1.0 + sign * 0.10)
        err_wrong = abs(f_sim - f_wrong) / f_wrong
        assert err_wrong > GATE, (
            f"falsifier failed: a {sign*10:+.0f}% frequency error (anchor "
            f"{f_wrong/1e9:.4f} GHz) gives err {err_wrong*100:.3f}% which does NOT "
            f"exceed the {GATE*100:.1f}% gate — the gate would not catch it"
        )

    # The ANALYTIC accuracy gate (the coverage #403 flags as missing): the x&y
    # graded NU mesh reproduces the closed-form in-plane-dependent TM110 to within
    # the measured tolerance (2.43% graded == 2.47% uniform -> grading faithful).
    assert err < GATE, (
        f"NU in-plane-graded TM110 error {err*100:.3f}% >= {GATE*100:.1f}% — the "
        f"non-uniform x/y grid does not reproduce the closed-form cavity resonance "
        f"whose frequency is set by the graded in-plane extents "
        f"(f_sim={f_sim/1e9:.4f} vs analytic={f_tm110/1e9:.4f} GHz)"
    )
