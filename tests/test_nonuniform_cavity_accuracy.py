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
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_nonuniform_z_graded_cavity_tm111_accuracy():
    """A genuinely z-graded NU mesh reproduces the closed-form TM111 cavity
    resonance to within a measured tolerance (the graded axis sets the analytic).

    Gate: |f_sim - f_analytic(actual d)| / f_analytic < 4% (measured 2.66% on a
    0.25mm fine band / dx=1mm air cavity, 2026-06-18; ~1.5x margin for cross-
    machine float). Mode-ID is robust: TM111 sits >1 GHz from its nearest sim
    neighbour (measured 1.08 GHz), so the nearest-to-analytic match is unambiguous.
    """
    from rfx import Simulation, GaussianPulse
    from rfx.auto_config import smooth_grading
    from rfx.grid import C0

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

    d = float(np.sum(dz_profile))              # actual graded z extent (NOT hardcoded)
    grading_ratio = max(dz_profile) / min(dz_profile)

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
    print(f"\n[NU-cavity] d (graded z) = {d*1e3:.3f} mm, grading ratio = {grading_ratio:.2f}")
    print(f"[NU-cavity] analytic TM110(z-indep) = {f_tm110/1e9:.4f} GHz, "
          f"TM111(z-dep) = {f_tm111/1e9:.4f} GHz")
    print(f"[NU-cavity] sim modes (GHz): {[round(f/1e9, 4) for f in sim_freqs]}")

    assert modes, "no resonances found in the TM111 band"
    assert grading_ratio > 2.0, (
        f"mesh not genuinely graded (max/min dz ratio {grading_ratio:.2f} <= 2) — "
        "the test would be a vacuous uniform-mesh check"
    )

    # Mode-ID robustness (separation RATIO, not an absolute window — robust to small
    # cross-machine frequency shifts): TM111 = the sim mode nearest the analytic
    # value, and the match must be UNAMBIGUOUS — the 2nd-nearest sim mode must be
    # several times farther. Measured: nearest delta ~0.18 GHz, 2nd-nearest ~0.90 GHz
    # (~5x), so a 3x + 0.5 GHz separation gate has comfortable margin.
    by_dist = sorted(sim_freqs, key=lambda f: abs(f - f_tm111))
    f_sim = by_dist[0]
    d_near = abs(f_sim - f_tm111)
    d_second = abs(by_dist[1] - f_tm111) if len(by_dist) > 1 else float("inf")
    assert d_second > 3 * d_near and d_second > 0.5e9, (
        f"TM111 mode-ID ambiguous: nearest {f_sim/1e9:.4f} GHz (delta {d_near/1e9:.3f}), "
        f"2nd-nearest {by_dist[1]/1e9:.4f} GHz (delta {d_second/1e9:.3f}) — "
        "not unambiguously separated from a neighbouring mode"
    )
    err = d_near / f_tm111
    print(f"[NU-cavity] TM111 f_sim = {f_sim/1e9:.4f} GHz, err = {err*100:.3f}% "
          f"(2nd-nearest delta {d_second/1e9:.3f} GHz)")

    # The ANALYTIC accuracy gate (the gap this test fills): the z-graded NU mesh
    # reproduces the z-DEPENDENT closed-form TM111 to within the measured tolerance.
    assert err < 0.04, (
        f"NU-graded TM111 error {err*100:.3f}% >= 4% — the non-uniform grid does "
        f"not reproduce the closed-form z-dependent cavity resonance "
        f"(f_sim={f_sim/1e9:.4f} vs analytic={f_tm111/1e9:.4f} GHz)"
    )
